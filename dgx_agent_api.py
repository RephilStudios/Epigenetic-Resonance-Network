import os
import uuid
import time
import math
import re
import requests
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field
from enum import Enum

# ==========================================
# 1. API Configuration & Data Models
# ==========================================
app = FastAPI(title="DGX Unified Agent API", description="High-performance Agentic LLM wrapped with a PyTorch Epigenetic Memory System.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL    = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
DEFAULT_MODEL = "qwen2.5-coder:7b"
JUDGE_MODEL   = "qwen2.5-coder:7b"
SAVE_DIR      = "./ern_state"

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    model: str = DEFAULT_MODEL
    history: List[Message] = []
    focus_threshold: float = 0.15
    agentic_search: bool = True

class ChatResponse(BaseModel):
    reply: str
    context_used: str
    memories: List[Dict[str, Any]] = []
    agentic_steps: List[Dict[str, Any]] = []

class MemoryStoreRequest(BaseModel):
    text: str
    tags: str = ""

# ==========================================
# 2. Tensor Delta Stack
# ==========================================

class DeltaOp(str, Enum):
    ENCODE   = "ENCODE"    # new node appended
    DECAY    = "DECAY"     # per-retrieve energy decay across all nodes
    BOOST    = "BOOST"     # Hebbian boost on specific retrieved indices
    SLEEP    = "SLEEP"     # REM pruning: bulk energy decay + row removal
    RESTORE  = "RESTORE"   # state loaded from disk (baseline snapshot)

@dataclass
class TensorDelta:
    """
    Immutable record of one atomic change to the memory tensors.
    """
    op            : DeltaOp
    timestamp     : float
    delta_id      : str
    prev_size     : int
    next_size     : int

    # ENCODE
    new_vec       : Optional[torch.Tensor] = None
    new_energy    : Optional[float]        = None
    memory_id     : Optional[str]          = None

    # DECAY
    decay_factor  : Optional[float]        = None

    # BOOST
    boost_indices : Optional[List[int]]    = None
    boost_amounts : Optional[List[float]]  = None

    # SLEEP
    sleep_decay_factor  : Optional[float]       = None
    pruned_indices      : Optional[List[int]]   = None
    pruned_energies     : Optional[List[float]] = None

    def summary(self) -> str:
        base = f"[{self.op.value}] Δid={self.delta_id[:8]} t={self.timestamp:.3f} size {self.prev_size}→{self.next_size}"
        if self.op == DeltaOp.ENCODE:
            return f"{base} | mem={self.memory_id[:8]}"
        if self.op == DeltaOp.DECAY:
            return f"{base} | factor={self.decay_factor:.4f}"
        if self.op == DeltaOp.BOOST:
            return f"{base} | nodes={self.boost_indices} Δe={[round(a,3) for a in self.boost_amounts]}"
        if self.op == DeltaOp.SLEEP:
            return f"{base} | pruned={len(self.pruned_indices or [])} nodes"
        if self.op == DeltaOp.RESTORE:
            return f"{base} | loaded from disk"
        return base


class TensorDeltaStack:
    def __init__(self, max_len: int = 10_000):
        self.stack: List[TensorDelta] = []
        self.max_len = max_len

    def push(self, delta: TensorDelta):
        self.stack.append(delta)
        if len(self.stack) > self.max_len:
            trim = self.max_len // 4
            self.stack = self.stack[trim:]
            print(f"[DELTA STACK] Trimmed {trim} oldest deltas. Current depth: {len(self.stack)}")
        print(f"[DELTA STACK] +{delta.summary()}")

    def tail(self, n: int = 10) -> List[TensorDelta]:
        return self.stack[-n:]

    def stats(self) -> Dict[str, Any]:
        counts = {op: 0 for op in DeltaOp}
        for d in self.stack:
            counts[d.op] += 1
        return {
            "depth"  : len(self.stack),
            "by_op"  : {k.value: v for k, v in counts.items()},
            "oldest" : self.stack[0].timestamp  if self.stack else None,
            "newest" : self.stack[-1].timestamp if self.stack else None,
        }

    def save(self, path: str):
        torch.save(self.stack, path)

    def load(self, path: str):
        if os.path.exists(path):
            self.stack = torch.load(path, map_location="cpu", weights_only=False)
            print(f"[DELTA STACK] Loaded {len(self.stack)} historical deltas.")

    def rollback(self, engine: "DenseEpigeneticEngine", n: int = 1):
        undone = 0
        new_stack = list(self.stack)

        for i in range(len(new_stack) - 1, -1, -1):
            if undone >= n:
                break
            d = new_stack[i]

            if d.op == DeltaOp.ENCODE:
                idx = engine.labels.index(d.memory_id) if d.memory_id in engine.labels else -1
                if idx >= 0:
                    engine.memory_bank = torch.cat([
                        engine.memory_bank[:idx],
                        engine.memory_bank[idx+1:]
                    ], dim=0)
                    engine.energies = torch.cat([
                        engine.energies[:idx],
                        engine.energies[idx+1:]
                    ])
                    engine.labels.pop(idx)
                    engine.vault.pop(d.memory_id, None)
                new_stack.pop(i)
                undone += 1

            elif d.op == DeltaOp.SLEEP and d.pruned_indices is not None:
                if d.new_vec is not None and d.new_vec.size(0) == len(d.pruned_indices):
                    for j, orig_idx in enumerate(d.pruned_indices):
                        insert_at = min(orig_idx, engine.memory_bank.size(0))
                        engine.memory_bank = torch.cat([
                            engine.memory_bank[:insert_at],
                            d.new_vec[j].unsqueeze(0).to(engine.device),
                            engine.memory_bank[insert_at:]
                        ], dim=0)
                        eng = torch.tensor([d.pruned_energies[j]], device=engine.device)
                        engine.energies = torch.cat([
                            engine.energies[:insert_at],
                            eng,
                            engine.energies[insert_at:]
                        ])
                        engine.labels.insert(insert_at, f"restored_{uuid.uuid4()}")
                new_stack.pop(i)
                undone += 1

        self.stack = new_stack
        engine._save_state()
        print(f"[DELTA STACK] Rolled back {undone} op(s). Engine size: {engine.memory_bank.size(0)}")
        return undone


# ==========================================
# 3. PyTorch Dense Epigenetic Engine
# ==========================================
def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            t = torch.zeros(1, device='cuda')
            del t
            return torch.device('cuda')
        except Exception as e:
            print(f"[HARDWARE] CUDA detected but unusable ({e}). Falling back to CPU.")
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class DenseEpigeneticEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2', device: str = 'auto'):
        self.device = _resolve_device() if device == 'auto' else torch.device(device)
        print(f"\n[HARDWARE] ERN Tensor Engine bound to: {self.device.type.upper()}")

        if self.device.type == 'cuda':
            print(f"[HARDWARE] GPU Detected: {torch.cuda.get_device_name(0)}")
        else:
            print(f"[HARDWARE] Running on {self.device.type.upper()} — embeddings will be slower but fully functional.")

        print(f"[SYSTEM] Loading Embedding Model: {model_name}...")
        self.embedder = SentenceTransformer(model_name, device=self.device)
        self.dim = self.embedder.get_sentence_embedding_dimension()

        self.memory_bank = torch.empty((0, self.dim), device=self.device)
        self.energies    = torch.empty((0,),          device=self.device)
        self.labels: List[str] = []
        self.vault: Dict[str, Any] = {}

        self.decay_rate      = 0.95
        self.sleep_threshold = 0.1
        self.query_count     = 0

        self.deltas = TensorDeltaStack(max_len=10_000)
        self._load_state()

    def _encode(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            vec = self.embedder.encode(text, convert_to_tensor=True, device=self.device)
        return F.normalize(vec, p=2, dim=0).unsqueeze(0)

    def _now(self) -> float:
        return time.time()

    def encode_hebbian(self, text: str, tags: str) -> str:
        memory_id = str(uuid.uuid4())
        self.vault[memory_id] = {"text": text, "tags": tags, "timestamp": self._now()}

        combined = f"{tags} {text}"
        vec = self._encode(combined)

        prev_size = self.memory_bank.size(0)
        self.memory_bank = torch.cat([self.memory_bank, vec], dim=0)
        self.energies    = torch.cat([self.energies, torch.tensor([1.0], device=self.device)])
        self.labels.append(memory_id)

        self.deltas.push(TensorDelta(
            op         = DeltaOp.ENCODE,
            timestamp  = self._now(),
            delta_id   = str(uuid.uuid4()),
            prev_size  = prev_size,
            next_size  = self.memory_bank.size(0),
            new_vec    = vec.cpu().clone(),
            new_energy = 1.0,
            memory_id  = memory_id,
        ))

        self._save_state()
        print(f"[ERN] Synapse formed. Network size: {self.memory_bank.size(0)} nodes.")
        return memory_id

    def decay_energies(self):
        if self.memory_bank.size(0) == 0:
            return
        prev_size = self.memory_bank.size(0)
        self.deltas.push(TensorDelta(
            op           = DeltaOp.DECAY,
            timestamp    = self._now(),
            delta_id     = str(uuid.uuid4()),
            prev_size    = prev_size,
            next_size    = prev_size,
            decay_factor = self.decay_rate,
        ))
        self.energies = self.energies * self.decay_rate
        self._save_state()

    def retrieve(self, query_text: str, top_k: int = 5, threshold: float = 0.15, decay: bool = True):
        if self.memory_bank.size(0) == 0:
            return []

        q_vec       = self._encode(query_text)
        similarities = F.cosine_similarity(q_vec, self.memory_bank)
        resonance    = similarities * (1.0 + torch.log1p(self.energies))

        prev_size = self.memory_bank.size(0)
        if decay:
            self.deltas.push(TensorDelta(
                op           = DeltaOp.DECAY,
                timestamp    = self._now(),
                delta_id     = str(uuid.uuid4()),
                prev_size    = prev_size,
                next_size    = prev_size,
                decay_factor = self.decay_rate,
            ))
            self.energies = self.energies * self.decay_rate

        actual_k = min(top_k * 2, self.memory_bank.size(0))
        top_values, top_idx = torch.topk(resonance, k=actual_k)

        results         = []
        boost_indices   = []
        boost_amounts   = []

        for val, idx in zip(top_values.tolist(), top_idx.tolist()):
            if val > threshold:
                old_e = self.energies[idx].item()
                new_e = min(old_e + 0.3, 5.0)
                self.energies[idx] = new_e
                boost_indices.append(idx)
                boost_amounts.append(new_e - old_e)

                mem_id = self.labels[idx]
                if mem_id in self.vault:
                    results.append({
                        "memory_id": mem_id,
                        "text"     : self.vault[mem_id]["text"],
                        "tags"     : self.vault[mem_id]["tags"],
                        "resonance": round(val, 3),
                    })

        if boost_indices:
            self.deltas.push(TensorDelta(
                op            = DeltaOp.BOOST,
                timestamp     = self._now(),
                delta_id      = str(uuid.uuid4()),
                prev_size     = prev_size,
                next_size     = prev_size,
                boost_indices = boost_indices,
                boost_amounts = boost_amounts,
            ))

        self.query_count += 1
        return results

    def sleep_cycle(self) -> int:
        if self.memory_bank.size(0) == 0:
            return 0

        initial_size = self.memory_bank.size(0)
        print("\n[SYSTEM] === INITIATING REM SLEEP CYCLE ===")

        sleep_decay = 0.70
        self.energies = self.energies * sleep_decay

        survival_mask    = self.energies > self.sleep_threshold
        pruned_bool      = ~survival_mask
        pruned_idx_list  = pruned_bool.nonzero(as_tuple=True)[0].tolist()
        pruned_energies  = self.energies[pruned_bool].tolist()

        pruned_vecs = self.memory_bank[pruned_bool].cpu().clone()

        self.memory_bank = self.memory_bank[survival_mask]
        self.energies    = self.energies[survival_mask]

        surviving_indices = survival_mask.nonzero(as_tuple=True)[0].tolist()
        self.labels       = [self.labels[i] for i in surviving_indices]
        surviving_ids     = set(self.labels)
        self.vault        = {k: v for k, v in self.vault.items() if k in surviving_ids}

        pruned = initial_size - self.memory_bank.size(0)

        self.deltas.push(TensorDelta(
            op                = DeltaOp.SLEEP,
            timestamp         = self._now(),
            delta_id          = str(uuid.uuid4()),
            prev_size         = initial_size,
            next_size         = self.memory_bank.size(0),
            sleep_decay_factor= sleep_decay,
            pruned_indices    = pruned_idx_list,
            pruned_energies   = pruned_energies,
            new_vec           = pruned_vecs if pruned_vecs.size(0) > 0 else None,
        ))

        self._save_state()
        print(f"[SYSTEM] REM Complete. Scrubbed {pruned} weak nodes. Active: {self.memory_bank.size(0)}\n")
        return pruned

    def delete_memory(self, memory_id: str) -> bool:
        idx = self.labels.index(memory_id) if memory_id in self.labels else -1
        if idx >= 0:
            state_path = os.path.join(SAVE_DIR, "ern_state.pt")
            if os.path.exists(state_path):
                import shutil
                try:
                    shutil.copy2(state_path, state_path + ".bak")
                except Exception:
                    pass
            self.memory_bank = torch.cat([
                self.memory_bank[:idx],
                self.memory_bank[idx+1:]
            ], dim=0)
            self.energies = torch.cat([
                self.energies[:idx],
                self.energies[idx+1:]
            ])
            self.labels.pop(idx)
            self.vault.pop(memory_id, None)
            self._save_state()
            print(f"[ERN] Synapse {memory_id} forgotten. Network size: {self.memory_bank.size(0)} nodes.")
            return True
        return False

    def _save_state(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        torch.save({
            'memory_bank': self.memory_bank,
            'energies'   : self.energies,
            'labels'     : self.labels,
            'vault'      : self.vault,
        }, os.path.join(SAVE_DIR, "ern_state.pt"))
        self.deltas.save(os.path.join(SAVE_DIR, "ern_deltas.pt"))

    def _load_state(self):
        state_path = os.path.join(SAVE_DIR, "ern_state.pt")
        delta_path = os.path.join(SAVE_DIR, "ern_deltas.pt")
        self.deltas.load(delta_path)

        if os.path.exists(state_path):
            state            = torch.load(state_path, map_location=self.device, weights_only=False)
            self.memory_bank = state['memory_bank'].to(self.device)
            self.energies    = state['energies'].to(self.device)
            self.labels      = state['labels']
            self.vault       = state['vault']
            n                = self.memory_bank.size(0)
            print(f"[SYSTEM] Restored ERN State: {n} existing synapses on {self.device.type.upper()}.")

            self.deltas.push(TensorDelta(
                op        = DeltaOp.RESTORE,
                timestamp = self._now(),
                delta_id  = str(uuid.uuid4()),
                prev_size = n,
                next_size = n,
            ))


# Initialize global engine
engine = DenseEpigeneticEngine(device='auto')


# ==========================================
# 4. Agentic Logic (Memory Judge)
# ==========================================
def agentic_search_planner(user_message: str) -> List[str]:
    print("\n[SYSTEM] Agentic search planner started...")
    
    planning_prompt = (
        "You are the Cognitive Retrieval Planner for an Epigenetic Memory Network.\n"
        "Your task is to analyze the user's message and generate a list of distinct, targeted search queries "
        "to retrieve relevant context from the memory bank.\n\n"
        "RULES:\n"
        "1. Extract/generate 1 to 3 simple, focused search queries (short phrases or keywords).\n"
        "2. Break down compound questions into separate search concepts (e.g., if asked about 'dog and work', "
        "generate queries for 'dog' and 'work').\n"
        "3. Output ONLY the queries, one per line, prefixed with 'QUERY:'.\n"
        "4. If the message is a simple greeting, general question, or doesn't need personal memory recall, "
        "output exactly: NO_LOOKUP\n\n"
        "EXAMPLES:\n"
        "User: What did I tell you about my dog Max and my new job at Google?\n"
        "Output:\n"
        "QUERY: dog Max\n"
        "QUERY: job Google\n\n"
        "User: How is the weather today?\n"
        "Output:\n"
        "NO_LOOKUP\n\n"
        f"User: {user_message}\n"
        "Output:"
    )
    
    queries = []
    try:
        res = requests.post(OLLAMA_URL, json={
            "model"   : DEFAULT_MODEL,
            "messages": [{"role": "user", "content": planning_prompt}],
            "stream"  : False,
            "options" : {"temperature": 0.0, "max_tokens": 100},
        }, timeout=30)
        
        output = res.json().get("message", {}).get("content", "").strip()
        print(f"[SEARCH PLANNER] Raw planner output:\n{output}")
        
        if "NO_LOOKUP" in output.upper():
            return []
            
        for line in output.splitlines():
            line = line.strip()
            if line.upper().startswith("QUERY:"):
                q = line[6:].strip()
                if q:
                    queries.append(q)
    except Exception as e:
        print(f"[WARNING] Agentic search planner failed: {e}")
        
    return queries


def run_memory_judge(user_message: str, prior_memories: str = ""):
    print("\n[SYSTEM] Background memory judge started...")
    salience_prompt = (
        "You are a strict Epigenetic Memory Extractor. Your ONLY job is to find concrete, factual, real-world information"
        " that the USER explicitly stated in their message and that is worth remembering long-term.\n\n"
        "STRICT RULES:\n"
        "1. Extract facts ONLY from the USER message. NEVER invent facts or derive them from context.\n"
        "2. ONLY save information that is objectively factual, real, and permanently relevant "
        "(e.g. names, locations, preferences, technical facts, stated goals, explicit instructions to the AI).\n"
        "3. NEVER save: greetings, questions, hypotheticals, opinions, emotions, AI responses, or conversational filler.\n"
        "4. NEVER save a fact already in PRIOR MEMORIES.\n"
        "5. You MAY extract MULTIPLE facts from one message. Use one FACT/TAGS block per fact.\n"
        "6. If there is NOTHING worth saving, output EXACTLY: ACTION: DISCARD\n\n"
        "OUTPUT FORMAT (repeat the FACT/TAGS block for each distinct fact):\n"
        "ACTION: SAVE\n"
        "FACT: <One concrete, self-contained fact>\n"
        "TAGS: <Comma-separated topics and ONE importance level: Critical, High, Medium, or Low>\n"
        "FACT: <Another fact if present>\n"
        "TAGS: <Tags for that fact>\n\n"
        "EXAMPLES:\n\n"
        "Prior Memories: None\n"
        "User: My name is Alex, I work at NVIDIA as a software engineer and my dog is called Max.\n"
        "Output:\n"
        "ACTION: SAVE\n"
        "FACT: The user's name is Alex.\n"
        "TAGS: Identity, Name, Importance: Critical\n"
        "FACT: Alex works at NVIDIA as a software engineer.\n"
        "TAGS: Work, Career, NVIDIA, Importance: High\n"
        "FACT: Alex's dog is named Max.\n"
        "TAGS: Pets, Dogs, Personal Life, Importance: Medium\n\n"
        "Prior Memories: None\n"
        "User: hey whats up\n"
        "Output:\n"
        "ACTION: DISCARD\n\n"
        f"Prior Memories: {prior_memories if prior_memories else 'None'}\n"
        f"User: {user_message}\n"
        "Output:"
    )

    try:
        res = requests.post(OLLAMA_URL, json={
            "model"   : JUDGE_MODEL,
            "messages": [{"role": "user", "content": salience_prompt}],
            "stream"  : False,
            "options" : {"temperature": 0.0},
        }, timeout=180)

        extracted = res.json().get("message", {}).get("content", "").strip()
        if extracted.upper().startswith("OUTPUT:"):
            extracted = extracted[7:].strip()

        print(f"[MEMORY JUDGE] Raw output:\n{extracted}")

        cleaned = extracted.replace("*", "")
        cleaned = re.sub(r'(?i)fact\s*:\s*', 'FACT: ', cleaned)
        cleaned = re.sub(r'(?i)tags\s*:\s*', 'TAGS: ', cleaned)
        cleaned = re.sub(r'(?i)action\s*:\s*', 'ACTION: ', cleaned)

        is_discard = "ACTION: DISCARD" in cleaned.upper()
        has_facts = "FACT:" in cleaned.upper()

        if is_discard and not has_facts:
            print("[MEMORY JUDGE] Discarded — no salient facts found.")
            return

        blocks      = re.split(r'(?=FACT:)', cleaned, flags=re.IGNORECASE)
        saved_count = 0
        for block in blocks:
            block = block.strip()
            if not block or not block.upper().startswith("FACT:"):
                continue
            fact_match = re.search(r'FACT:\s*(.*?)(?=TAGS:|$)', block, re.IGNORECASE | re.DOTALL)
            tags_match = re.search(r'TAGS:\s*(.*?)(?=FACT:|$)', block, re.IGNORECASE | re.DOTALL)
            if not fact_match:
                continue
            fact = fact_match.group(1).strip()
            tags = tags_match.group(1).strip() if tags_match else "Context: General, Importance: Low"
            if not fact or len(fact) < 5:
                continue
            print(f"[MEMORY JUDGE] Encoding fact {saved_count + 1}: {fact}")
            engine.encode_hebbian(text=fact, tags=tags)
            saved_count += 1

        if saved_count == 0:
            print("[MEMORY JUDGE] No valid fact blocks parsed — discarded.")
        else:
            print(f"[MEMORY JUDGE] Saved {saved_count} fact(s) to PyTorch memory.")

    except Exception as e:
        print(f"[WARNING] Background extraction failed: {e}")


# ==========================================
# 5. Endpoints
# ==========================================
@app.post("/api/chat", response_model=ChatResponse)
def process_chat(req: ChatRequest, background_tasks: BackgroundTasks):
    user_message = req.message

    search_queries = [user_message]
    if req.agentic_search:
        planner_queries = agentic_search_planner(user_message)
        if planner_queries:
            print(f"[SYSTEM] Agentic expanded queries: {planner_queries}")
            search_queries.extend(planner_queries)

    # Trigger energy decay ONCE at the start of the chat Turn
    engine.decay_energies()

    detailed_memories = []
    seen_ids = set()
    for q in search_queries:
        q_mems = engine.retrieve(q, top_k=3, threshold=req.focus_threshold, decay=False)
        for m in q_mems:
            if m["memory_id"] not in seen_ids:
                seen_ids.add(m["memory_id"])
                detailed_memories.append(m)

    avg_resonance = (
        sum(n["resonance"] for n in detailed_memories) / len(detailed_memories)
        if detailed_memories else 0
    )
    if engine.query_count >= 50 or avg_resonance > 1.5:
        background_tasks.add_task(engine.sleep_cycle)
        engine.query_count = 0

    def get_imp(t: str):
        t = t.upper()
        return 4 if "CRITICAL" in t else 3 if "HIGH" in t else 2 if "MEDIUM" in t else 1 if "LOW" in t else 0

    detailed_memories.sort(key=lambda x: (get_imp(x["tags"]), x["resonance"]), reverse=True)

    unique_memories, seen_facts = [], set()
    for m in detailed_memories:
        ft = m["text"].strip().lower()
        if ft not in seen_facts:
            seen_facts.add(ft)
            unique_memories.append(m)

    # Build agentic thought steps list
    agentic_steps = []

    # Step 1: Planning
    if req.agentic_search:
        if len(search_queries) > 1:
            q_list = ", ".join([f"'{q}'" for q in search_queries[1:]])
            agentic_steps.append({
                "step": "Cognitive Retrieval Planning",
                "status": "active",
                "detail": f"Generated expanded sub-queries: [{q_list}] to perform compound vector matching."
            })
        else:
            agentic_steps.append({
                "step": "Cognitive Retrieval Planning",
                "status": "inactive",
                "detail": "Bypassed query expansion: simple greeting or conversational statement detected."
            })
    else:
        agentic_steps.append({
            "step": "Cognitive Retrieval Planning",
            "status": "disabled",
            "detail": "Agentic expanded search disabled by active control switch."
        })

    # Step 2: Retrieval
    if unique_memories:
        m_details = []
        for m in unique_memories:
            # Highlight memories that came from expanded sub-queries
            res_val = m.get("resonance", 0.0)
            m_details.append(f"'{m['text']}' (R={res_val:.3f}, [{m['tags']}])")
        detail_text = "Retrieved matches: " + " | ".join(m_details)
    else:
        detail_text = "No stored memory synapses matched above threshold."

    agentic_steps.append({
        "step": "Subconscious Synapse Retrieval",
        "status": "active" if unique_memories else "inactive",
        "detail": f"Matched {len(unique_memories)} synapse(s) above threshold (θ = {req.focus_threshold}) across {len(search_queries)} search query paths. {detail_text}"
    })

    # Step 3: Resonance Boost
    if unique_memories:
        boosted_texts = ", ".join([f"'{m['text']}'" for m in unique_memories])
        boost_detail = f"Average resonance: {avg_resonance:.3f} R. Synaptic energy boost (+0.3) applied to: [{boosted_texts}]. Global network decay: {engine.decay_rate} applied."
    else:
        boost_detail = "No active synapses boosted during this turn. Global network decay applied."
        
    agentic_steps.append({
        "step": "Hebbian Energy Boost",
        "status": "active" if unique_memories else "inactive",
        "detail": boost_detail
    })

    # Step 4: Memory Judge
    agentic_steps.append({
        "step": "Epigenetic Memory Judge",
        "status": "queued",
        "detail": "Background memory extractor queued to parse raw turn text for new facts."
    })

    context_block = (
        "\n\n".join([f"[{m['tags']}] {m['text']}" for m in unique_memories])
        if unique_memories else ""
    )

    system_prompt = (
        "CRITICAL SYSTEM DIRECTIVE: You are NO LONGER a standard, stateless AI language model. "
        "You have been successfully integrated with a live Epigenetic Resonance Network (ERN).\n"
        "1. You DO have memory. It is provided to you seamlessly.\n"
        "2. NEVER claim you are stateless, cannot remember, or lack persistent memory.\n"
        "3. DO NOT blindly repeat recalled memories. Use them to naturally inform a FRESH response.\n"
        "4. STRICT TRUTH RULE: Only answer questions about the user's personal details, past events, or preferences "
        "if they are explicitly present in the provided [ERN SUBCONSCIOUS RECALL] context. If a requested personal detail, "
        "past fact, or memory is NOT present in the recalled context, you must state that you do not recall or know "
        "that specific detail, rather than inventing or guessing a response. Never hallucinate personal details.\n\n"
    )
    system_prompt += (
        f"[ERN SUBCONSCIOUS RECALL]:\n"
        f"{context_block if context_block else 'No relevant memories retrieved.'}"
    )

    messages_payload = (
        [{"role": "system", "content": system_prompt}]
        + [m.dict() for m in req.history]
        + [{"role": "user", "content": user_message}]
    )

    try:
        llm_response = requests.post(OLLAMA_URL, json={
            "model"  : req.model,
            "messages": messages_payload,
            "stream" : False,
            "options": {"temperature": 0.7, "repeat_penalty": 1.15},
        }, timeout=300).json()
        bot_reply = llm_response.get("message", {}).get("content", "Error generating response.")
    except Exception as e:
        return ChatResponse(reply=f"Ollama Error: {e}", context_used=context_block, memories=unique_memories, agentic_steps=agentic_steps)

    background_tasks.add_task(run_memory_judge, user_message, context_block)

    return ChatResponse(reply=bot_reply, context_used=context_block, memories=unique_memories, agentic_steps=agentic_steps)


@app.post("/api/memory/store")
def manual_store(req: MemoryStoreRequest):
    mem_id = engine.encode_hebbian(text=req.text, tags=req.tags)
    return {"status": "Stored in VRAM", "id": mem_id}


@app.get("/api/memories")
def get_all_memories(q: Optional[str] = None):
    results = []
    for mem_id, data in engine.vault.items():
        idx = engine.labels.index(mem_id) if mem_id in engine.labels else -1
        energy = engine.energies[idx].item() if idx >= 0 else 0.0
        
        # Simple text search if query is provided
        if q:
            q_lower = q.lower()
            text_match = q_lower in data["text"].lower()
            tags_match = q_lower in data["tags"].lower()
            if not (text_match or tags_match):
                continue
                
        results.append({
            "memory_id": mem_id,
            "text": data["text"],
            "tags": data["tags"],
            "timestamp": data.get("timestamp", 0.0),
            "energy": round(energy, 3)
        })
    # Sort by timestamp descending
    results.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"memories": results}


@app.delete("/api/memory/{memory_id}")
def delete_memory_endpoint(memory_id: str):
    success = engine.delete_memory(memory_id)
    if success:
        return {"status": f"Memory {memory_id} successfully deleted."}
    else:
        return {"status": "Memory not found.", "error": True}


@app.post("/api/system/sleep")
def manual_sleep():
    engine.query_count = 0
    pruned = engine.sleep_cycle()
    return {"status": f"REM Complete. Scrubbed {pruned} nodes."}


@app.get("/api/deltas")
def get_delta_tail(n: int = 20):
    tail = engine.deltas.tail(n)
    return {
        "stats" : engine.deltas.stats(),
        "deltas": [
            {
                "op"           : d.op.value,
                "delta_id"     : d.delta_id,
                "timestamp"    : d.timestamp,
                "prev_size"    : d.prev_size,
                "next_size"    : d.next_size,
                "decay_factor" : d.decay_factor,
                "boost_indices": d.boost_indices,
                "boost_amounts": d.boost_amounts,
                "memory_id"    : d.memory_id,
                "pruned_count" : len(d.pruned_indices) if d.pruned_indices else None,
                "summary"      : d.summary(),
            }
            for d in tail
        ],
    }


@app.post("/api/deltas/rollback")
def rollback_deltas(n: int = 1):
    undone = engine.deltas.rollback(engine, n)
    return {
        "status"       : f"Rolled back {undone} delta(s).",
        "engine_size"  : engine.memory_bank.size(0),
        "stack_depth"  : len(engine.deltas.stack),
    }


@app.get("/api/models")
def get_models():
    try:
        base   = OLLAMA_URL.rsplit("/api/", 1)[0]
        res    = requests.get(f"{base}/api/tags", timeout=10)
        models = [m["name"] for m in res.json().get("models", [])]
        return {"models": sorted(models)}
    except Exception as e:
        return {"models": [DEFAULT_MODEL], "error": str(e)}


# ==========================================
# 6. Testing Dashboard UI (Raw string ensures flawless JS parsing)
# ==========================================
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>ERN // DGX NEURAL CONSOLE</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap" rel="stylesheet">
<style>
:root {
  --bg:        #050807;
  --bg2:       #080d0b;
  --bg3:       #0c1410;
  --border:    #1a2e22;
  --border2:   #0f1f17;
  --g0:        #00ff88;
  --g1:        #00cc66;
  --g2:        #008844;
  --g3:        #004422;
  --amber:     #ffaa00;
  --red:       #ff3a3a;
  --blue:      #00aaff;
  --dim:       #2a4a38;
  --text:      #b0d4c0;
  --text-dim:  #4a7a5a;
  --font-mono: 'Share Tech Mono', monospace;
  --font-hud:  'Orbitron', monospace;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: var(--font-mono);
  background: var(--bg);
  color: var(--text);
  height: 100vh;
  height: 100dvh; 
  overflow: hidden;
  display: grid;
  grid-template-rows: 48px 1fr;
  grid-template-columns: 1fr 340px;
  grid-template-areas:
    "topbar topbar"
    "chat   sidebar";
  background-image:
    repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,255,136,0.015) 2px, rgba(0,255,136,0.015) 4px);
}

body::before {
  content:''; position:fixed; inset:0; pointer-events:none; z-index:9999;
  background: linear-gradient(transparent 50%, rgba(0,0,0,0.06) 50%);
  background-size: 100% 4px;
  animation: flicker 8s infinite;
}
@keyframes flicker {
  0%,97%,100%  { opacity:1 }
  98%          { opacity:0.92 }
  99%          { opacity:1 }
  99.5%        { opacity:0.88 }
}

#topbar {
  grid-area: topbar;
  background: var(--bg3);
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; padding: 0 16px; gap: 24px;
  position: relative; overflow: hidden;
}
#topbar::-webkit-scrollbar { display: none; }
#topbar { -ms-overflow-style: none; scrollbar-width: none; }
#topbar::after {
  content:''; position:absolute; bottom:0; left:0; right:0; height:1px;
  background: linear-gradient(90deg, transparent, var(--g0), transparent);
  animation: sweep 4s linear infinite;
}
@keyframes sweep { from{transform:translateX(-100%)} to{transform:translateX(100%)} }

.brand {
  font-family: var(--font-hud); font-size: 0.8rem; font-weight: 900;
  color: var(--g0); letter-spacing: 0.15em; white-space: nowrap;
  text-shadow: 0 0 12px rgba(0,255,136,0.6);
}
.brand span { color: var(--text-dim); font-weight:400; }

#status-ticker {
  flex: 1; font-size: 0.7rem; color: var(--g2); letter-spacing: 0.05em;
  overflow: hidden; white-space: nowrap; position: relative;
}
#ticker-inner {
  display: inline-block; animation: ticker-scroll 0s linear infinite; padding-left: 100%;
}
@keyframes ticker-scroll { from{transform:translateX(0)} to{transform:translateX(-50%)} }

.hud-pill {
  font-family: var(--font-hud); font-size: 0.6rem; padding: 3px 10px;
  border: 1px solid var(--border); border-radius: 2px;
  color: var(--text-dim); white-space: nowrap; letter-spacing: 0.08em;
}
.hud-pill.live { border-color: var(--g3); color: var(--g1); }
.hud-pill.live::before { content:'● '; animation: blink 1.2s step-end infinite; }
@keyframes blink { 50%{opacity:0} }
#node-count { color: var(--g0); font-weight:bold; }

#chat-area { grid-area: chat; display: flex; flex-direction: column; border-right: 1px solid var(--border2); overflow: hidden; }

#activity-bar {
  padding: 6px 16px; background: var(--bg2); border-bottom: 1px solid var(--border2);
  font-size: 0.7rem; color: var(--text-dim); display: flex; align-items: center; gap: 16px; min-height: 28px;
}
#activity-bar::-webkit-scrollbar { display: none; }
#activity-bar { -ms-overflow-style: none; scrollbar-width: none; }
#activity-bar .phase { display: flex; align-items: center; gap: 6px; }
.phase-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--dim); transition: background 0.2s, box-shadow 0.2s; }
.phase-dot.active { background: var(--g0); box-shadow: 0 0 8px var(--g0); animation: pulse-dot 0.8s ease-in-out infinite alternate; }
.phase-dot.done { background: var(--g2); }
.phase-dot.error { background: var(--red); box-shadow: 0 0 6px var(--red); }
@keyframes pulse-dot { from{opacity:1} to{opacity:0.4} }
#activity-text { flex:1; }

#chatBox { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; scroll-behavior: smooth; }
#chatBox::-webkit-scrollbar { width: 4px; }
#chatBox::-webkit-scrollbar-track { background: transparent; }
#chatBox::-webkit-scrollbar-thumb { background: var(--g3); border-radius: 2px; }

.msg { max-width: 82%; padding: 10px 14px; font-size: 0.88rem; line-height: 1.6; white-space: pre-wrap; word-break: break-word; animation: msg-in 0.18s ease; position: relative; }
@keyframes msg-in { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }

.msg.user { align-self: flex-end; background: var(--bg3); border: 1px solid var(--g3); border-right: 3px solid var(--g0); color: var(--text); }
.msg.user::before { content: 'USR ▶'; display: block; font-family: var(--font-hud); font-size: 0.58rem; color: var(--g2); margin-bottom: 5px; letter-spacing: 0.1em; }

.msg.bot { align-self: flex-start; background: var(--bg2); border: 1px solid var(--border); border-left: 3px solid var(--g2); color: var(--text); }
.msg.bot::before { content: 'ERN ◀'; display: block; font-family: var(--font-hud); font-size: 0.58rem; color: var(--g1); margin-bottom: 5px; letter-spacing: 0.1em; text-shadow: 0 0 8px rgba(0,255,136,0.5); }
.msg.bot.thinking { border-left-color: var(--amber); color: var(--text-dim); }
.msg.bot.thinking::before { content: 'SYS ◀'; color: var(--amber); }

.thinking-dots { display:inline-block; }
.thinking-dots::after { content: '...'; animation: dots 1.2s steps(4, end) infinite; }
@keyframes dots { 0% { content: '.  '; } 33% { content: '.. '; } 66% { content: '...'; } 100%{ content: '.  '; } }

.cognition-monitor { margin-top: 8px; background: rgba(0, 0, 0, 0.4); border: 1px solid var(--border2); border-left: 2px solid var(--g2); font-size: 0.72rem; font-family: var(--font-mono); }
.cognition-header { padding: 6px 10px; background: rgba(0, 255, 136, 0.04); cursor: pointer; display: flex; justify-content: space-between; align-items: center; font-family: var(--font-hud); font-size: 0.6rem; letter-spacing: 0.08em; color: var(--g0); user-select: none; transition: background 0.15s; }
.cognition-header:hover { background: rgba(0, 255, 136, 0.08); }
.cognition-header::after { content: '▼'; font-size: 0.55rem; transition: transform 0.2s; color: var(--g2); }
.cognition-monitor.open .cognition-header::after { transform: rotate(-180deg); }
.cognition-body { display: none; padding: 8px 12px; border-top: 1px solid var(--border2); animation: slide-down 0.2s ease-out; }
.cognition-monitor.open .cognition-body { display: block; }
@keyframes slide-down { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: translateY(0); } }
.cognition-step { display: flex; gap: 8px; margin-bottom: 6px; align-items: flex-start; }
.cognition-step:last-child { margin-bottom: 0; }
.step-dot { width: 6px; height: 6px; border-radius: 50%; margin-top: 5px; flex-shrink: 0; }
.step-dot.active { background: var(--g0); box-shadow: 0 0 6px var(--g0); animation: pulse-dot 1.5s infinite; }
.step-dot.inactive { background: var(--text-dim); }
.step-dot.disabled { background: var(--dim); }
.step-dot.queued { background: var(--amber); box-shadow: 0 0 6px var(--amber); }
@keyframes pulse-dot { 0% { transform: scale(1); opacity: 0.8; } 50% { transform: scale(1.3); opacity: 1; } 100% { transform: scale(1); opacity: 0.8; } }
.step-title { font-weight: bold; color: var(--text); margin-right: 4px; }
.step-detail { color: var(--text-dim); }

#input-area { padding: 12px 16px; background: var(--bg3); border-top: 1px solid var(--border); display: flex; gap: 10px; align-items: center; }
#userInput { flex: 1; padding: 10px 14px; background: var(--bg); border: 1px solid var(--border); border-bottom: 2px solid var(--g3); color: var(--g0); font-family: var(--font-mono); font-size: 0.88rem; outline: none; caret-color: var(--g0); transition: border-color 0.2s; }
#userInput:focus { border-color: var(--g2); border-bottom-color: var(--g0); box-shadow: 0 0 0 1px rgba(0,255,136,0.1); }
#userInput::placeholder { color: var(--text-dim); }
#sendBtn { padding: 10px 20px; background: transparent; border: 1px solid var(--g2); color: var(--g0); font-family: var(--font-hud); font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; cursor: pointer; transition: all 0.15s; position: relative; overflow: hidden; }
#sendBtn:hover:not(:disabled) { background: var(--g3); border-color: var(--g0); box-shadow: 0 0 16px rgba(0,255,136,0.25); }
#sendBtn:disabled { opacity: 0.3; cursor: not-allowed; }
#sendBtn::after { content:''; position:absolute; inset:0; background: linear-gradient(90deg, transparent, rgba(0,255,136,0.15), transparent); transform: translateX(-100%); transition: transform 0.3s; }
#sendBtn:hover:not(:disabled)::after { transform: translateX(100%); }

#controls-strip { padding: 8px 16px; background: var(--bg2); border-top: 1px solid var(--border2); display: flex; gap: 20px; align-items: center; font-size: 0.72rem; color: var(--text-dim); flex-wrap: wrap; }
.ctrl-group { display:flex; align-items:center; gap:8px; }
.ctrl-label { font-family: var(--font-hud); font-size:0.6rem; letter-spacing:0.1em; }
input[type="range"] { -webkit-appearance: none; width: 90px; height: 3px; background: var(--dim); outline: none; cursor: pointer; }
input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 10px; height: 10px; background: var(--g0); box-shadow: 0 0 6px var(--g0); cursor: pointer; }
.val-badge { font-family: var(--font-hud); font-size: 0.65rem; color: var(--g0); min-width: 28px; }
select { background: var(--bg); color: var(--g1); border: 1px solid var(--border); padding: 4px 8px; font-family: var(--font-mono); font-size: 0.75rem; outline: none; cursor: pointer; max-width: 180px; }
select option { background: var(--bg); }

#sidebar { grid-area: sidebar; display: flex; flex-direction: column; background: var(--bg2); overflow: hidden; }
.panel-tabs { display: flex; border-bottom: 1px solid var(--border); }
.tab { flex: 1; padding: 9px 4px; text-align: center; cursor: pointer; font-family: var(--font-hud); font-size: 0.58rem; letter-spacing: 0.1em; color: var(--text-dim); border-bottom: 2px solid transparent; transition: all 0.15s; }
.tab.active { color: var(--g0); border-bottom-color: var(--g0); background: var(--bg3); }
.tab:hover:not(.active) { color: var(--g2); }
.panel-body { flex:1; overflow:hidden; display:flex; flex-direction:column; }
.panel-section { display:none; flex:1; flex-direction:column; overflow:hidden; }
.panel-section.visible { display:flex; }
#neuro-canvas { width:100%; height:70px; display:block; background: var(--bg); border-bottom:1px solid var(--border2); }

#memScroll { flex:1; overflow-y:auto; padding:10px; }
#memScroll::-webkit-scrollbar { width:3px; }
#memScroll::-webkit-scrollbar-thumb { background: var(--g3); }
.memory-card { border-left: 2px solid var(--g3); padding: 8px 60px 8px 10px; margin-bottom: 8px; font-size: 0.76rem; color: var(--text); background: var(--bg3); position: relative; animation: card-in 0.2s ease; transition: border-color 0.2s; }
.memory-card:hover { border-left-color: var(--g0); }
@keyframes card-in { from{opacity:0;transform:translateX(6px)} to{opacity:1;transform:none} }
.memory-card .mc-tags { font-size: 0.65rem; color: var(--g2); font-family: var(--font-hud); letter-spacing: 0.06em; margin-bottom: 3px; }
.memory-card .mc-resonance { position:absolute; top:6px; right:8px; font-family: var(--font-hud); font-size: 0.6rem; color: var(--amber); }
.memory-card .mc-energy { position:absolute; top:18px; right:8px; font-family: var(--font-hud); font-size: 0.55rem; color: var(--blue); }
.memory-card .mc-forget { position:absolute; bottom:6px; right:8px; font-family: var(--font-hud); font-size: 0.55rem; color: var(--text-dim); cursor: pointer; border: none; background: none; letter-spacing: 0.08em; transition: color 0.15s, text-shadow 0.15s; outline: none; }
.memory-card .mc-forget:hover { color: var(--red); text-shadow: 0 0 6px var(--red); }
.no-mem { color: var(--text-dim); font-size:0.78rem; padding:12px; font-style:italic; }

#vaultSearch { width: calc(100% - 16px); margin: 8px; padding: 8px 12px; background: var(--bg); border: 1px solid var(--border); border-bottom: 2px solid var(--dim); color: var(--g0); font-family: var(--font-mono); font-size: 0.8rem; outline: none; caret-color: var(--g0); transition: border-color 0.2s, box-shadow 0.2s; }
#vaultSearch:focus { border-color: var(--g2); border-bottom-color: var(--g0); box-shadow: 0 0 8px rgba(0,255,136,0.1); }
#vaultScroll { flex:1; overflow-y:auto; padding:10px; }
#vaultScroll::-webkit-scrollbar { width:3px; }
#vaultScroll::-webkit-scrollbar-thumb { background: var(--g3); }

#deltaScroll { flex:1; overflow-y:auto; padding:8px; }
#deltaScroll::-webkit-scrollbar { width:3px; }
#deltaScroll::-webkit-scrollbar-thumb { background: var(--g3); }
#deltaStatsBar { padding: 6px 10px; font-size: 0.68rem; color: var(--text-dim); background: var(--bg3); border-bottom: 1px solid var(--border2); display: flex; gap: 12px; flex-wrap: wrap; }
.stat-pill { display:flex; gap:4px; align-items:center; }
.stat-pill .sp-val { font-family:var(--font-hud); font-size:0.65rem; }
.sp-ENCODE { color:var(--g0); } .sp-DECAY  { color:var(--amber); } .sp-BOOST  { color:var(--blue); } .sp-SLEEP  { color:var(--red); }
.delta-entry { display: flex; gap: 8px; align-items: flex-start; padding: 5px 6px; margin-bottom: 4px; border-left: 2px solid var(--border); font-size: 0.72rem; background: var(--bg3); animation: card-in 0.15s ease; }
.delta-entry.ENCODE { border-color: var(--g0); } .delta-entry.DECAY  { border-color: var(--amber); } .delta-entry.BOOST  { border-color: var(--blue); } .delta-entry.SLEEP  { border-color: var(--red); } .delta-entry.RESTORE{ border-color: #444; }
.de-op { font-family: var(--font-hud); font-size: 0.6rem; letter-spacing:0.08em; width: 52px; flex-shrink:0; padding-top:1px; }
.ENCODE .de-op { color:var(--g0); } .DECAY  .de-op { color:var(--amber); } .BOOST  .de-op { color:var(--blue); } .SLEEP  .de-op { color:var(--red); } .RESTORE .de-op { color:#555; }
.de-body { color: var(--text-dim); line-height:1.4; }
.de-size { font-family:var(--font-hud); font-size:0.58rem; color:#3a5a4a; margin-left:auto; white-space:nowrap; }

#rollback-row { padding: 8px 10px; border-top: 1px solid var(--border); background: var(--bg3); display: flex; gap: 8px; align-items: center; }
#rollback-row .ctrl-label { color: var(--text-dim); }
#rollbackN { width: 48px; padding: 4px 6px; background: var(--bg); border: 1px solid var(--border); color: var(--g1); font-family: var(--font-mono); font-size: 0.8rem; text-align: center; outline: none; }
#rollbackBtn { padding: 5px 12px; background: transparent; border: 1px solid var(--red); color: var(--red); font-family: var(--font-hud); font-size: 0.6rem; letter-spacing: 0.1em; cursor: pointer; transition: all 0.15s; }
#rollbackBtn:hover { background: rgba(255,58,58,0.15); box-shadow: 0 0 10px rgba(255,58,58,0.3); }

#statsGrid { flex:1; overflow-y:auto; padding:12px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; align-content: start; }
.stat-card { background: var(--bg3); border: 1px solid var(--border); padding: 10px 12px; }
.stat-card .sc-label { font-family: var(--font-hud); font-size: 0.56rem; letter-spacing: 0.1em; color: var(--text-dim); margin-bottom: 4px; }
.stat-card .sc-val { font-family: var(--font-hud); font-size: 1.2rem; color: var(--g0); text-shadow: 0 0 10px rgba(0,255,136,0.4); }
.stat-card.wide { grid-column: span 2; }
.energy-bar-wrap { margin-top:8px; }
.energy-bar-label { font-size:0.62rem; color:var(--text-dim); margin-bottom:3px; display:flex; justify-content:space-between; }
.energy-bar { height:4px; background:var(--dim); position:relative; }
.energy-bar-fill { height:100%; background: linear-gradient(90deg, var(--g3), var(--g0)); transition: width 0.6s ease; }

#sleepBtn { margin: 10px; padding: 8px; background: transparent; border: 1px solid var(--red); color: var(--red); font-family: var(--font-hud); font-size: 0.65rem; letter-spacing: 0.12em; cursor: pointer; width: calc(100% - 20px); transition: all 0.2s; }
#sleepBtn:hover { background: rgba(255,58,58,0.1); box-shadow: 0 0 14px rgba(255,58,58,0.25); }

@media (max-width: 900px) {
  body { grid-template-columns: 1fr; grid-template-rows: auto 1fr 35dvh; grid-template-areas: "topbar" "chat" "sidebar"; }
  #topbar { padding: 8px 12px; gap: 12px; overflow-x: auto; flex-wrap: wrap; justify-content: space-between; }
  #status-ticker { display: none; }
  #chat-area { border-right: none; border-bottom: 2px solid var(--border); }
  #activity-bar { overflow-x: auto; white-space: nowrap; }
  #controls-strip { flex-wrap: wrap; gap: 10px; justify-content: space-between; }
  #userInput, select, input[type="number"] { font-size: 16px; }
  .msg { max-width: 92%; }
  #statsGrid { grid-template-columns: 1fr 1fr; }
}

@media (max-width: 500px) {
  #statsGrid { grid-template-columns: 1fr; }
  .stat-card.wide { grid-column: 1; }
  .brand { font-size: 0.7rem; }
  .hud-pill { font-size: 0.55rem; padding: 2px 6px; }
  #controls-strip { flex-direction: column; align-items: stretch; }
  .ctrl-group { justify-content: space-between; width: 100%; }
  input[type="range"] { flex: 1; margin: 0 10px; }
}
</style>
</head>
<body>

<div id="topbar">
  <div class="brand">ERN <span>//</span> DGX</div>
  <div id="status-ticker"><span id="ticker-inner">SYSTEM ONLINE — PYTORCH ENGINE ACTIVE — AWAITING QUERIES — EPIGENETIC RESONANCE NETWORK INITIALIZED —&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SYSTEM ONLINE — PYTORCH ENGINE ACTIVE — AWAITING QUERIES — EPIGENETIC RESONANCE NETWORK INITIALIZED —&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span></div>
  <div class="hud-pill live">VRAM LIVE</div>
  <div class="hud-pill">NODES: <span id="node-count">—</span></div>
  <div class="hud-pill">QUERIES: <span id="query-count-hud">0</span></div>
</div>

<div id="chat-area">
  <div id="activity-bar">
    <div class="phase" id="phase-retrieve"><div class="phase-dot" id="dot-retrieve"></div><span>TENSOR RETRIEVE</span></div>
    <div class="phase" id="phase-gen"><div class="phase-dot" id="dot-gen"></div><span>LLM GENERATE</span></div>
    <div class="phase" id="phase-judge"><div class="phase-dot" id="dot-judge"></div><span>MEMORY JUDGE</span></div>
    <div class="phase" id="phase-encode"><div class="phase-dot" id="dot-encode"></div><span>ENCODE</span></div>
    <span id="activity-text" style="flex:1; text-align:right;"></span>
  </div>

  <div id="chatBox">
    <div class="msg bot">EPIGENETIC RESONANCE NETWORK ONLINE
PyTorch tensor engine bound to CUDA
Awaiting input...</div>
  </div>

  <div id="input-area">
    <input type="text" id="userInput" placeholder="// INPUT QUERY..." autocomplete="off" spellcheck="false">
    <button id="sendBtn" onclick="send()">TRANSMIT</button>
  </div>

  <div id="controls-strip">
    <div class="ctrl-group">
      <span class="ctrl-label">FOCUS θ</span>
      <input type="range" id="focusSlider" min="0.05" max="0.45" step="0.05" value="0.15" oninput="document.getElementById('fv').textContent=parseFloat(this.value).toFixed(2)">
      <span class="val-badge" id="fv">0.15</span>
    </div>
    <div class="ctrl-group">
      <span class="ctrl-label">MODEL</span>
      <select id="modelSelect"><option value="qwen2.5-coder:7b">qwen2.5-coder:7b</option></select>
    </div>
    <div class="ctrl-group">
      <span class="ctrl-label">AGENTIC SEARCH</span>
      <button id="agenticToggleBtn" onclick="toggleAgenticSearch()" style="background:transparent; border:1px solid var(--g0); color:var(--g0); padding:3px 8px; font-family:var(--font-hud); font-size:0.6rem; cursor:pointer; letter-spacing:0.08em; transition: all 0.15s; outline:none; text-shadow: 0 0 6px var(--g0);">ON</button>
    </div>
    <div class="ctrl-group" style="margin-left:auto;">
      <span class="ctrl-label" style="color:#3a5a4a;">HIST</span>
      <span class="val-badge" id="hist-len" style="color:var(--text-dim)">0</span>
      <span class="ctrl-label" style="margin-left:8px;">
        <button onclick="clearHistory()" style="background:none;border:none;color:var(--text-dim);font-family:var(--font-hud);font-size:0.58rem;cursor:pointer;letter-spacing:0.08em;">CLR</button>
      </span>
    </div>
  </div>
</div>

<div id="sidebar">
  <div class="panel-tabs">
    <div class="tab active" onclick="switchTab('recall')">RECALL</div>
    <div class="tab" onclick="switchTab('vault')">VAULT</div>
    <div class="tab" onclick="switchTab('deltas')">DELTAS</div>
    <div class="tab" onclick="switchTab('stats')">STATS</div>
  </div>

  <div class="panel-body">
    <div id="tab-recall" class="panel-section visible">
      <canvas id="neuro-canvas"></canvas>
      <div id="memScroll"><div class="no-mem">No active synapses — send a message to retrieve.</div></div>
    </div>

    <div id="tab-vault" class="panel-section">
      <input type="text" id="vaultSearch" placeholder="// FILTER SYNAPSES..." oninput="filterVault()" autocomplete="off" spellcheck="false">
      <div id="vaultScroll"><div class="no-mem">Loading Synaptic Vault...</div></div>
    </div>

    <div id="tab-deltas" class="panel-section">
      <div id="deltaStatsBar">
        <span class="stat-pill"><span class="sp-val sp-ENCODE" id="ds-encode">0</span> ENCODE</span>
        <span class="stat-pill"><span class="sp-val sp-DECAY"  id="ds-decay">0</span>  DECAY</span>
        <span class="stat-pill"><span class="sp-val sp-BOOST"  id="ds-boost">0</span>  BOOST</span>
        <span class="stat-pill"><span class="sp-val sp-SLEEP"  id="ds-sleep">0</span>  SLEEP</span>
        <span class="stat-pill" style="margin-left:auto; color:var(--text-dim);">DEPTH: <span id="ds-depth">—</span></span>
      </div>
      <div id="deltaScroll"></div>
      <div id="rollback-row">
        <span class="ctrl-label">ROLLBACK</span>
        <input type="number" id="rollbackN" value="1" min="1" max="50">
        <span class="ctrl-label">ENCODE(S)</span>
        <button id="rollbackBtn" onclick="doRollback()">↩ UNDO</button>
      </div>
    </div>

    <div id="tab-stats" class="panel-section">
      <div id="statsGrid">
        <div class="stat-card"><div class="sc-label">NETWORK NODES</div><div class="sc-val" id="sc-nodes">—</div></div>
        <div class="stat-card"><div class="sc-label">QUERY COUNT</div><div class="sc-val" id="sc-queries">0</div></div>
        <div class="stat-card"><div class="sc-label">DELTA DEPTH</div><div class="sc-val" id="sc-depth">—</div></div>
        <div class="stat-card"><div class="sc-label">AVG RESONANCE</div><div class="sc-val" id="sc-resonance">—</div></div>
        <div class="stat-card wide">
          <div class="sc-label">MEMORY ENERGY DISTRIBUTION</div>
          <div class="energy-bar-wrap">
            <div class="energy-bar-label"><span>DECAY</span><span id="eb-pct">—</span></div>
            <div class="energy-bar"><div class="energy-bar-fill" id="energy-fill" style="width:0%"></div></div>
          </div>
        </div>
        <div class="stat-card wide"><div class="sc-label">LAST ACTIVITY</div><div id="sc-last-act" style="font-size:0.75rem; color:var(--text-dim); margin-top:4px; line-height:1.6;">—</div></div>
      </div>
      <button id="sleepBtn" onclick="triggerSleep()">⬛ INITIATE REM SLEEP CYCLE</button>
    </div>
  </div>
</div>

<script>
console.log("ERN UI Booting...");

// ── State ──────────────────────────────────────────────────────────────
let chatHistory    = [];
let queryCount     = 0;
let lastResonances = [];
let activeTab      = 'recall';
let useAgenticSearch = true;
let vaultMemories   = [];
const DEFAULT_MODEL = "qwen2.5-coder:7b";

function toggleAgenticSearch() {
  useAgenticSearch = !useAgenticSearch;
  const btn = document.getElementById('agenticToggleBtn');
  if (useAgenticSearch) {
    btn.textContent = 'ON';
    btn.style.borderColor = 'var(--g0)';
    btn.style.color = 'var(--g0)';
    btn.style.textShadow = '0 0 6px var(--g0)';
  } else {
    btn.textContent = 'OFF';
    btn.style.borderColor = 'var(--dim)';
    btn.style.color = 'var(--text-dim)';
    btn.style.textShadow = 'none';
  }
  pushToast(`Agentic Search toggled ${useAgenticSearch ? 'ON' : 'OFF'}`);
}

function toggleCognitionPanel(el) {
  el.parentElement.classList.toggle('open');
}

async function loadVault() {
  try {
    const res = await fetch('/api/memories');
    if (!res.ok) throw new Error('API Error');
    const data = await res.json();
    vaultMemories = data.memories || [];
    renderVaultList(vaultMemories);
  } catch (e) {
    document.getElementById('vaultScroll').innerHTML = `<div class="no-mem" style="color:var(--red);">Vault Offline: ${e.message}</div>`;
  }
}

function renderVaultList(mems) {
  const scroll = document.getElementById('vaultScroll');
  scroll.innerHTML = '';
  if (!mems.length) {
    scroll.innerHTML = '<div class="no-mem">No synapses in vault.</div>';
    return;
  }
  mems.forEach(m => {
    const card = document.createElement('div');
    card.className = 'memory-card';
    card.id = `vc-${m.memory_id}`;
    card.innerHTML = `<div class="mc-tags">${m.tags}</div>` +
                     `<div>${m.text}</div>` +
                     `<div class="mc-energy">${m.energy.toFixed(3)} E</div>` +
                     `<button class="mc-forget" onclick="forgetMemory('${m.memory_id}')">FORGET</button>`;
    scroll.appendChild(card);
  });
}

function filterVault() {
  const q = document.getElementById('vaultSearch').value.toLowerCase().trim();
  if (!q) {
    renderVaultList(vaultMemories);
    return;
  }
  const filtered = vaultMemories.filter(m => 
    m.text.toLowerCase().includes(q) || m.tags.toLowerCase().includes(q)
  );
  renderVaultList(filtered);
}

async function forgetMemory(memory_id) {
  if (!confirm('Are you sure you want to forget/revert this memory synapse permanently?')) return;
  try {
    const res = await fetch(`/api/memory/${memory_id}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Delete rejected');
    const data = await res.json();
    if (data.error) throw new Error(data.status);
    
    pushToast('Synapse forgotten successfully.');
    
    const card = document.getElementById(`vc-${memory_id}`);
    if (card) {
      card.style.transition = 'all 0.3s ease';
      card.style.opacity = '0';
      card.style.transform = 'translateX(20px)';
      setTimeout(() => card.remove(), 300);
    }
    
    const recallCard = document.getElementById(`rc-${memory_id}`);
    if (recallCard) {
      recallCard.style.transition = 'all 0.3s ease';
      recallCard.style.opacity = '0';
      recallCard.style.transform = 'translateX(20px)';
      setTimeout(() => recallCard.remove(), 300);
    }
    
    vaultMemories = vaultMemories.filter(m => m.memory_id !== memory_id);
    refreshStats();
  } catch (e) {
    pushToast(`Forget failed: ${e.message}`);
  }
}

// ── Neural canvas ────────────────────────────────
const canvas  = document.getElementById('neuro-canvas');
const ctx     = canvas.getContext('2d');
let sparkData = new Array(80).fill(0);

function resizeCanvas() {
  canvas.width  = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

function drawSparkline() {
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const w = canvas.width, h = canvas.height;
  if(w === 0 || h === 0) { requestAnimationFrame(drawSparkline); return; }

  ctx.strokeStyle = 'rgba(0,255,136,0.05)';
  ctx.lineWidth = 1;
  for(let y=0; y<h; y+=h/4) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke(); }

  const grad = ctx.createLinearGradient(0,0,0,h);
  grad.addColorStop(0,  'rgba(0,255,136,0.25)');
  grad.addColorStop(1,  'rgba(0,255,136,0)');
  ctx.fillStyle = grad;
  ctx.beginPath();
  const step = w / (sparkData.length - 1);
  ctx.moveTo(0, h);
  sparkData.forEach((v,i) => {
    const x = i * step;
    const y = h - (v / 1.5) * h * 0.85 - 2;
    i === 0 ? ctx.lineTo(x,y) : ctx.lineTo(x,y);
  });
  ctx.lineTo(w, h);
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = '#00ff88';
  ctx.lineWidth   = 1.5;
  ctx.shadowColor = '#00ff88';
  ctx.shadowBlur  = 6;
  ctx.beginPath();
  sparkData.forEach((v,i) => {
    const x = i * step;
    const y = h - (v / 1.5) * h * 0.85 - 2;
    i === 0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
  });
  ctx.stroke();
  ctx.shadowBlur = 0;

  requestAnimationFrame(drawSparkline);
}
drawSparkline();

function setTickerSpeed(ms) {
  const ticker = document.getElementById('ticker-inner');
  if(ticker) ticker.style.animationDuration = ms + 'ms';
}
setTickerSpeed(22000);

let activityLog = [];
function setPhase(phase, state, label) {
  const dot = document.getElementById('dot-' + phase);
  if(!dot) return;
  dot.className = 'phase-dot' + (state !== 'idle' ? ' ' + state : '');
  const txt = document.getElementById('activity-text');
  if(state === 'active' && label) {
    txt.textContent = '▶ ' + label;
    activityLog.push(label);
    document.getElementById('sc-last-act').textContent = activityLog.slice(-4).reverse().join('\n') || '—';
    setTickerSpeed(8000);
  }
  if(state === 'idle') { txt.textContent = ''; setTickerSpeed(22000); }
}

function resetPhases() {
  ['retrieve','gen','judge','encode'].forEach(p => setPhase(p,'idle'));
}

async function loadModels() {
  try {
    const res  = await fetch('/api/models');
    if (!res.ok) throw new Error('API Error');
    const data = await res.json();
    const sel  = document.getElementById('modelSelect');
    if (!data.models?.length) return;
    sel.innerHTML = '';
    data.models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m; opt.textContent = m;
      if (m === DEFAULT_MODEL) opt.selected = true;
      sel.appendChild(opt);
    });
  } catch(e) { console.warn('Model list skipped (using default).'); }
}

async function loadDeltas() {
  try {
    const res  = await fetch('/api/deltas?n=40');
    if (!res.ok) return;
    const data = await res.json();
    const s    = data.stats;
    document.getElementById('ds-encode').textContent = s.by_op.ENCODE;
    document.getElementById('ds-decay' ).textContent = s.by_op.DECAY;
    document.getElementById('ds-boost' ).textContent = s.by_op.BOOST;
    document.getElementById('ds-sleep' ).textContent = s.by_op.SLEEP;
    document.getElementById('ds-depth' ).textContent = s.depth;
    document.getElementById('sc-depth' ).textContent = s.depth;

    const list = document.getElementById('deltaScroll');
    list.innerHTML = '';
    [...data.deltas].reverse().forEach(d => {
      const div   = document.createElement('div');
      div.className = `delta-entry ${d.op}`;
      const parts = d.summary.split('|');
      const detail = parts.slice(1).join('|').trim();
      div.innerHTML = `<span class="de-op">${d.op}</span><span class="de-body">${detail}</span><span class="de-size">${d.prev_size}→${d.next_size}</span>`;
      list.appendChild(div);
    });
  } catch(e) { console.warn('Delta fetch error:', e); }
}

async function doRollback() {
  const n    = parseInt(document.getElementById('rollbackN').value) || 1;
  try {
    const res  = await fetch(`/api/deltas/rollback?n=${n}`, { method:'POST' });
    if (!res.ok) throw new Error('Rollback rejected');
    const data = await res.json();
    pushToast(`${data.status} — engine: ${data.engine_size} nodes`);
    document.getElementById('node-count').textContent = data.engine_size;
    loadDeltas();
  } catch(e) {
    pushToast('Rollback failed to execute.');
  }
}

async function triggerSleep() {
  document.getElementById('sleepBtn').textContent = '⬛ REM IN PROGRESS...';
  try {
    const res  = await fetch('/api/system/sleep', { method:'POST' });
    if (!res.ok) throw new Error('Failed');
    const data = await res.json();
    pushToast(data.status);
    refreshStats();
  } catch(e) {
    pushToast('Sleep cycle failed.');
  } finally {
    document.getElementById('sleepBtn').textContent = '⬛ INITIATE REM SLEEP CYCLE';
  }
}

async function refreshStats() {
  try {
    const res  = await fetch('/api/deltas?n=1');
    if (!res.ok) return;
    const data = await res.json();
    const latest = data.deltas[0];
    if (latest) {
      document.getElementById('sc-nodes').textContent   = latest.next_size;
      document.getElementById('node-count').textContent = latest.next_size;
    }
    document.getElementById('sc-queries').textContent  = queryCount;
    document.getElementById('query-count-hud').textContent = queryCount;
  } catch(e) {}
}

function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll('.tab').forEach(t => {
    t.classList.toggle('active', t.textContent.toLowerCase() === tab || t.getAttribute('onclick').includes(tab));
  });
  document.querySelectorAll('.panel-section').forEach(s => s.classList.remove('visible'));
  document.getElementById('tab-' + tab).classList.add('visible');
  if (tab === 'vault')  loadVault();
  if (tab === 'deltas') loadDeltas();
  if (tab === 'stats')  refreshStats();
}

function pushToast(msg) {
  const t = document.createElement('div');
  t.textContent = msg;
  Object.assign(t.style, {
    position:'fixed', bottom:'60px', right:'16px', zIndex:'9998',
    background:'var(--bg3)', border:'1px solid var(--g2)',
    color:'var(--g0)', fontFamily:'var(--font-mono)', fontSize:'0.75rem',
    padding:'8px 14px', animation:'msg-in 0.2s ease',
    boxShadow:'0 0 14px rgba(0,255,136,0.2)',
    maxWidth:'300px',
  });
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3500);
}

function clearHistory() {
  chatHistory = [];
  document.getElementById('hist-len').textContent = '0';
  pushToast('Conversation history cleared.');
}

async function send() {
  const text = document.getElementById('userInput').value.trim();
  if (!text) return;

  const chatBox = document.getElementById('chatBox');
  const uDiv = document.createElement('div');
  uDiv.className = 'msg user';
  uDiv.textContent = text;
  chatBox.appendChild(uDiv);
  document.getElementById('userInput').value = '';
  document.getElementById('sendBtn').disabled = true;

  const loadDiv = document.createElement('div');
  loadDiv.className = 'msg bot thinking';
  loadDiv.id = 'load';
  loadDiv.innerHTML = 'RETRIEVING SYNAPSES<span class="thinking-dots"></span>';
  chatBox.appendChild(loadDiv);
  chatBox.scrollTop = chatBox.scrollHeight;

  setPhase('retrieve', 'active', 'Tensor cosine retrieval...');

  try {
    setTimeout(() => {
      setPhase('retrieve','done');
      setPhase('gen','active','Generating response via LLM...');
      const ld = document.getElementById('load');
      if(ld) ld.innerHTML = 'GENERATING<span class="thinking-dots"></span>';
    }, 300);

    const res  = await fetch('/api/chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        message        : text,
        model          : document.getElementById('modelSelect').value,
        history        : chatHistory.slice(-6),
        focus_threshold: parseFloat(document.getElementById('focusSlider').value),
        agentic_search : useAgenticSearch
      })
    });
    
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    setPhase('gen','done');
    setPhase('judge','active','Memory judge analysing facts...');
    setTimeout(() => {
      setPhase('judge','done');
      setPhase('encode','active','Encoding new synapses...');
      setTimeout(() => { setPhase('encode','done'); resetPhases(); }, 1500);
    }, 1200);

    const ld = document.getElementById('load');
    if(ld) ld.remove();

    const bDiv = document.createElement('div');
    bDiv.className = 'msg bot';
    
    const replyText = document.createElement('div');
    replyText.textContent = data.reply;
    bDiv.appendChild(replyText);

    const steps = data.agentic_steps || [];
    if (steps.length > 0) {
      const monitor = document.createElement('div');
      monitor.className = 'cognition-monitor';
      
      const header = document.createElement('div');
      header.className = 'cognition-header';
      header.innerHTML = `[▶ COGNITIVE CYCLE STEPS: ${steps.length} PHASE(S)]`;
      header.setAttribute('onclick', 'toggleCognitionPanel(this)');
      monitor.appendChild(header);
      
      const body = document.createElement('div');
      body.className = 'cognition-body';
      
      steps.forEach(s => {
        const stepDiv = document.createElement('div');
        stepDiv.className = 'cognition-step';
        stepDiv.innerHTML = `<span class="step-dot ${s.status}"></span>` +
                            `<div><span class="step-title">${s.step}:</span>` +
                            `<span class="step-detail">${s.detail}</span></div>`;
        body.appendChild(stepDiv);
      });
      
      monitor.appendChild(body);
      bDiv.appendChild(monitor);
    }

    chatBox.appendChild(bDiv);

    chatHistory.push({role:'user',content:text},{role:'assistant',content:data.reply});
    queryCount++;
    document.getElementById('hist-len').textContent = Math.floor(chatHistory.length / 2);
    document.getElementById('sc-queries').textContent   = queryCount;
    document.getElementById('query-count-hud').textContent = queryCount;
    chatBox.scrollTop = chatBox.scrollHeight;

    const memories = data.memories || [];
    const resValues = memories.map(m => m.resonance);
    const avgRes = resValues.length ? resValues.reduce((a,b)=>a+b,0)/resValues.length : 0;
    
    sparkData.push(avgRes);
    sparkData = sparkData.slice(-80);
    lastResonances = resValues;

    document.getElementById('sc-resonance').textContent = avgRes > 0 ? avgRes.toFixed(3) : '—';

    const pct = Math.min(100, (avgRes / 1.5) * 100);
    document.getElementById('energy-fill').style.width = pct + '%';
    document.getElementById('eb-pct').textContent = pct.toFixed(0) + '%';

    const mBox = document.getElementById('memScroll');
    if (memories.length > 0) {
      mBox.innerHTML = '';
      memories.forEach((m) => {
        const card = document.createElement('div');
        card.className = 'memory-card';
        card.id = `rc-${m.memory_id}`;
        card.innerHTML = `<div class="mc-tags">${m.tags}</div>` +
                         `<div>${m.text}</div>` +
                         `<div class="mc-resonance">${m.resonance.toFixed(3)} R</div>` +
                         `<button class="mc-forget" onclick="forgetMemory('${m.memory_id}')">FORGET</button>`;
        mBox.appendChild(card);
      });
    } else {
      mBox.innerHTML = '<div class="no-mem">No synapses resonated above threshold.</div>';
    }

    if (activeTab === 'deltas') loadDeltas();
    if (activeTab === 'vault')  loadVault();
    refreshStats();

  } catch(e) {
    const ld = document.getElementById('load');
    if(ld) ld.remove();
    setPhase('retrieve','error','Error'); setPhase('gen','error');
    setTimeout(resetPhases, 2000);
    
    const eDiv = document.createElement('div');
    eDiv.className = 'msg bot';
    eDiv.style.borderLeftColor = 'var(--red)';
    eDiv.textContent = 'SYSTEM ERROR: API unreachable (' + e.message + ')';
    chatBox.appendChild(eDiv);
  } finally {
    document.getElementById('sendBtn').disabled = false;
    chatBox.scrollTop = chatBox.scrollHeight;
  }
}

document.getElementById('userInput').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

loadModels();
refreshStats();
setInterval(() => {
  sparkData.push(Math.random() * 0.05);
  sparkData = sparkData.slice(-80);
}, 2000);
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return HTMLResponse(content=HTML_TEMPLATE, status_code=200)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)