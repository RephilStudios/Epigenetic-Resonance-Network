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
from typing import List, Optional
from sentence_transformers import SentenceTransformer

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

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
DEFAULT_MODEL = "qwen2.5-coder:7b"  # Main chat model
JUDGE_MODEL   = "qwen2.5-coder:7b"  # Fast 7B model dedicated to memory extraction
SAVE_DIR = "./ern_state"

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    model: str = DEFAULT_MODEL
    history: List[Message] = []
    focus_threshold: float = 0.15 

class ChatResponse(BaseModel):
    reply: str
    context_used: str

class MemoryStoreRequest(BaseModel):
    text: str
    tags: str = ""

# ==========================================
# 2. PyTorch Dense Epigenetic Engine
# ==========================================
class DenseEpigeneticEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"\n[HARDWARE] ERN Tensor Engine bound to: {self.device.type.upper()}")
        
        if self.device.type == 'cuda':
            print(f"[HARDWARE] GPU Detected: {torch.cuda.get_device_name(0)}")
        
        print(f"[SYSTEM] Loading Embedding Model: {model_name}...")
        self.embedder = SentenceTransformer(model_name, device=self.device)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        
        self.memory_bank = torch.empty((0, self.dim), device=self.device)
        self.energies = torch.empty((0,), device=self.device)
        self.labels = []
        self.vault = {}
        
        self.decay_rate = 0.95
        self.sleep_threshold = 0.1
        self.query_count = 0

        self._load_state()

    def _encode(self, text: str):
        with torch.no_grad():
            vec = self.embedder.encode(text, convert_to_tensor=True, device=self.device)
        return F.normalize(vec, p=2, dim=0).unsqueeze(0) 

    def encode_hebbian(self, text: str, tags: str):
        memory_id = str(uuid.uuid4())
        self.vault[memory_id] = {"text": text, "tags": tags, "timestamp": time.time()}
        
        combined_context = f"{tags} {text}"
        vec = self._encode(combined_context)
        
        self.memory_bank = torch.cat([self.memory_bank, vec], dim=0)
        self.energies = torch.cat([self.energies, torch.tensor([1.0], device=self.device)])
        self.labels.append(memory_id)
        
        self._save_state()
        print(f"[ERN] Synapse formed. Network size: {self.memory_bank.size(0)} nodes.")
        return memory_id

    def retrieve(self, query_text: str, top_k: int = 5, threshold: float = 0.15):
        if self.memory_bank.size(0) == 0:
            return []
            
        q_vec = self._encode(query_text)
        similarities = F.cosine_similarity(q_vec, self.memory_bank)
        resonance = similarities * (1.0 + torch.log1p(self.energies))
        
        self.energies = self.energies * self.decay_rate
        
        actual_k = min(top_k * 2, self.memory_bank.size(0))
        top_values, top_idx = torch.topk(resonance, k=actual_k)
        
        results = []
        for val, idx in zip(top_values.tolist(), top_idx.tolist()):
            if val > threshold:
                self.energies[idx] = min(self.energies[idx].item() + 0.3, 5.0) # Hebbian Boost cap
                mem_id = self.labels[idx]
                if mem_id in self.vault:
                    results.append({
                        "memory_id": mem_id,
                        "text": self.vault[mem_id]["text"],
                        "tags": self.vault[mem_id]["tags"],
                        "resonance": round(val, 3)
                    })
        
        self.query_count += 1
        return results

    def sleep_cycle(self):
        if self.memory_bank.size(0) == 0: return 0
        initial_size = self.memory_bank.size(0)
        print("\n[SYSTEM] === INITIATING REM SLEEP CYCLE ===")
        
        self.energies = self.energies * 0.70 
        survival_mask = self.energies > self.sleep_threshold
        
        self.memory_bank = self.memory_bank[survival_mask]
        self.energies = self.energies[survival_mask]
        
        surviving_indices = survival_mask.nonzero(as_tuple=True)[0].tolist()
        self.labels = [self.labels[i] for i in surviving_indices]
        
        # Cleanup vault
        surviving_ids = set(self.labels)
        self.vault = {k: v for k, v in self.vault.items() if k in surviving_ids}
        
        pruned = initial_size - self.memory_bank.size(0)
        self._save_state()
        print(f"[SYSTEM] REM Complete. Scrubbed {pruned} weak nodes. Active: {self.memory_bank.size(0)}\n")
        return pruned

    def _save_state(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        torch.save({
            'memory_bank': self.memory_bank,
            'energies': self.energies,
            'labels': self.labels,
            'vault': self.vault
        }, os.path.join(SAVE_DIR, "ern_state.pt"))

    def _load_state(self):
        state_path = os.path.join(SAVE_DIR, "ern_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)
            self.memory_bank = state['memory_bank'].to(self.device)
            self.energies = state['energies'].to(self.device)
            self.labels = state['labels']
            self.vault = state['vault']
            print(f"[SYSTEM] Restored ERN State: {self.memory_bank.size(0)} existing synapses.")

# Initialize global engine
engine = DenseEpigeneticEngine()

# ==========================================
# 3. Agentic Logic (Memory Judge)
# ==========================================
def run_memory_judge(user_message: str, prior_memories: str = ""):
    """
    Extracts multiple concrete, factual memories ONLY from what the USER said.
    Always uses JUDGE_MODEL (fast 7B) — completely independent of the main chat model.
    """
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
        "Prior Memories: [Identity] The user's name is Alex.\n"
        "User: What is my name again?\n"
        "Output:\n"
        "ACTION: DISCARD\n\n"
        "Prior Memories: None\n"
        "User: Can you write me a poem?\n"
        "Output:\n"
        "ACTION: DISCARD\n\n"
        "Now process this message:\n"
        f"Prior Memories: {prior_memories if prior_memories else 'None'}\n"
        f"User: {user_message}\n"
        "Output:"
    )

    try:
        res = requests.post(OLLAMA_URL, json={
            "model": JUDGE_MODEL,  # Always use the fast dedicated judge model
            "messages": [{"role": "user", "content": salience_prompt}],
            "stream": False,
            "options": {"temperature": 0.0}
        }, timeout=180)

        extracted = res.json().get("message", {}).get("content", "").strip()
        # Strip any leading "Output:" prefix the model may add
        if extracted.upper().startswith("OUTPUT:"): extracted = extracted[7:].strip()

        print(f"[MEMORY JUDGE] Raw output:\n{extracted}")

        if "ACTION: DISCARD" in extracted.upper() or "ACTION: SAVE" not in extracted.upper():
            print("[MEMORY JUDGE] Discarded — no salient facts found.")
            return

        # Parse multiple FACT/TAGS blocks from the output
        # Split on FACT: boundaries to support multiple facts
        blocks = re.split(r'(?=FACT:)', extracted, flags=re.IGNORECASE)
        saved_count = 0
        for block in blocks:
            block = block.strip()
            if not block:
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
# 4. Endpoints
# ==========================================
@app.post("/api/chat", response_model=ChatResponse)
def process_chat(req: ChatRequest, background_tasks: BackgroundTasks):
    user_message = req.message
    
    # 1. FAST INTERNAL TENSOR RETRIEVAL
    detailed_memories = engine.retrieve(user_message, top_k=5, threshold=req.focus_threshold)
    
    # Autonomous Fatigue Trigger
    avg_resonance = sum(n["resonance"] for n in detailed_memories) / len(detailed_memories) if detailed_memories else 0
    if engine.query_count >= 50 or avg_resonance > 1.5:
        background_tasks.add_task(engine.sleep_cycle)
        engine.query_count = 0
    
    # Importance sorting & Deduplication
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
            
    context_block = "\n\n".join([f"[{m['tags']}] {m['text']}" for m in unique_memories]) if unique_memories else ""

    # 2. OLLAMA AGENT GENERATION
    system_prompt = (
        "CRITICAL SYSTEM DIRECTIVE: You are NO LONGER a standard, stateless AI language model. "
        "You have been successfully integrated with a live Epigenetic Resonance Network (ERN). "
        "1. You DO have memory. It is provided to you seamlessly.\n"
        "2. NEVER claim you are stateless, cannot remember, or lack persistent memory.\n"
        "3. DO NOT blindly repeat recalled memories. Use them to naturally inform a FRESH response.\n\n"
    )
    system_prompt += f"[ERN SUBCONSCIOUS RECALL]:\n{context_block if context_block else 'No relevant memories retrieved.'}"

    messages_payload = [{"role": "system", "content": system_prompt}] + [m.dict() for m in req.history] + [{"role": "user", "content": user_message}]

    try:
        llm_response = requests.post(OLLAMA_URL, json={
            "model": req.model, "messages": messages_payload, "stream": False,
            "options": {"temperature": 0.7, "repeat_penalty": 1.15}
        }, timeout=300).json()
        bot_reply = llm_response.get("message", {}).get("content", "Error generating response.")
    except Exception as e:
        return ChatResponse(reply=f"Ollama Error: {e}", context_used=context_block)

    # 3. BACKGROUND MEMORY JUDGE — runs on fast JUDGE_MODEL, never blocks main chat
    background_tasks.add_task(run_memory_judge, user_message, context_block)

    return ChatResponse(reply=bot_reply, context_used=context_block)

@app.post("/api/memory/store")
def manual_store(req: MemoryStoreRequest):
    mem_id = engine.encode_hebbian(text=req.text, tags=req.tags)
    return {"status": "Stored in VRAM", "id": mem_id}

@app.post("/api/system/sleep")
def manual_sleep():
    engine.query_count = 0
    pruned = engine.sleep_cycle()
    return {"status": f"REM Complete. Scrubbed {pruned} nodes."}

@app.get("/api/models")
def get_models():
    """Proxy Ollama's model list so the UI can populate the dropdown dynamically."""
    try:
        base = OLLAMA_URL.rsplit("/api/", 1)[0]
        res = requests.get(f"{base}/api/tags", timeout=10)
        models = [m["name"] for m in res.json().get("models", [])]
        return {"models": sorted(models)}
    except Exception as e:
        return {"models": [DEFAULT_MODEL], "error": str(e)}

# ==========================================
# 5. Testing Dashboard UI
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><title>DGX ERN Unified Agent</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #121212; color: #e0e0e0; margin: 0; display: flex; height: 100vh; }
        .chat-container { flex: 2; display: flex; flex-direction: column; border-right: 1px solid #333; }
        .brain-container { flex: 1; background: #0a0a0a; padding: 20px; overflow-y: auto; }
        .header { padding: 16px 20px; background: #1a1a1a; border-bottom: 1px solid #333; font-weight: bold; color: #0ff; display: flex; justify-content: space-between; align-items: center; }
        .messages { flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 15px; }
        .msg { max-width: 80%; padding: 12px; border-radius: 8px; line-height: 1.4; white-space: pre-wrap; word-break: break-word; }
        .user { align-self: flex-end; background: #2b5c5c; color: #fff; }
        .bot  { align-self: flex-start; background: #222; border: 1px solid #444; }
        .input-area { padding: 16px 20px; background: #1a1a1a; display: flex; gap: 10px; }
        input[type="text"] { flex: 1; padding: 12px; background: #000; border: 1px solid #444; color: #fff; border-radius: 4px; outline: none; }
        button { padding: 12px 24px; background: #0ff; color: #000; border: none; border-radius: 4px; font-weight: bold; cursor: pointer; transition: opacity .15s; }
        button:disabled { opacity: 0.4; cursor: not-allowed; }
        .memory-card { background: #111; border-left: 3px solid #f0f; padding: 10px; margin-bottom: 10px; font-size: 0.85rem; color: #ccc; border-radius: 2px; }
        .controls { display: flex; gap: 12px; align-items: center; font-size: 0.85rem; color: #aaa; flex-wrap: wrap; }
        select#modelSelect {
            background: #000;
            color: #0ff;
            border: 1px solid #444;
            border-radius: 4px;
            padding: 6px 10px;
            font-size: 0.85rem;
            outline: none;
            cursor: pointer;
            min-width: 200px;
        }
        select#modelSelect option { background: #111; color: #0ff; }
        .model-badge { font-size: 0.7rem; color: #555; margin-left: 4px; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="header">
            <span>DGX Agentic API Endpoint</span>
            <div class="controls">
                <label>Focus:</label>
                <input type="range" id="focusSlider" min="0.05" max="0.45" step="0.05" value="0.15" oninput="document.getElementById('fv').innerText=this.value">
                <span id="fv" style="color: #f0f; font-weight:bold;">0.15</span>
                <label>Model:</label>
                <select id="modelSelect">
                    <option value="qwen2.5-coder:7b">qwen2.5-coder:7b</option>
                </select>
            </div>
        </div>
        <div class="messages" id="chatBox"><div class="msg bot">Unified PyTorch Engine Online. Awaiting queries.</div></div>
        <div class="input-area">
            <input type="text" id="userInput" placeholder="Type a message... (Enter to send)">
            <button id="sendBtn" onclick="send()">Send</button>
        </div>
    </div>
    <div class="brain-container">
        <div class="header" style="padding:0 0 20px 0; border:none; color:#f0f;">VRAM Recall</div>
        <div id="memoryBox" style="color:#444; font-style:italic;">Awaiting...</div>
    </div>
    <script>
        let history = [];
        const DEFAULT_MODEL = "qwen2.5-coder:7b";

        // Populate model dropdown from Ollama on load
        async function loadModels() {
            try {
                const res = await fetch('/api/models');
                const data = await res.json();
                const sel = document.getElementById('modelSelect');
                if (!data.models || data.models.length === 0) return;
                sel.innerHTML = '';
                data.models.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m;
                    opt.textContent = m;
                    if (m === DEFAULT_MODEL) opt.selected = true;
                    sel.appendChild(opt);
                });
            } catch(e) {
                console.warn('Could not load model list:', e);
            }
        }

        async function send() {
            const text = document.getElementById('userInput').value.trim();
            if(!text) return;
            const chatBox = document.getElementById('chatBox');
            const uDiv = document.createElement('div');
            uDiv.className = 'msg user';
            uDiv.textContent = text;
            chatBox.appendChild(uDiv);
            document.getElementById('userInput').value = '';
            document.getElementById('sendBtn').disabled = true;
            const loadDiv = document.createElement('div');
            loadDiv.className = 'msg bot';
            loadDiv.id = 'load';
            loadDiv.textContent = 'Processing...';
            chatBox.appendChild(loadDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
            try {
                const res = await fetch('/api/chat', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ 
                        message: text,
                        model: document.getElementById('modelSelect').value,
                        history: history.slice(-6),
                        focus_threshold: parseFloat(document.getElementById('focusSlider').value)
                    })
                });
                const data = await res.json();
                document.getElementById('load').remove();
                const bDiv = document.createElement('div');
                bDiv.className = 'msg bot';
                bDiv.textContent = data.reply;
                chatBox.appendChild(bDiv);
                history.push({role: "user", content: text}, {role: "assistant", content: data.reply});
                chatBox.scrollTop = chatBox.scrollHeight;
                const mBox = document.getElementById('memoryBox');
                if (data.context_used) {
                    mBox.innerHTML = '';
                    data.context_used.split('\\n\\n').forEach(function(m) {
                        const card = document.createElement('div');
                        card.className = 'memory-card';
                        card.textContent = m;
                        mBox.appendChild(card);
                    });
                } else {
                    mBox.innerHTML = '<div style="color:#444;font-style:italic;">No active synapses...</div>';
                }
            } catch (e) {
                const le = document.getElementById('load');
                if (le) le.remove();
                chatBox.innerHTML += `<div class="msg bot" style="color:#f55;">Error: ${e.message}</div>`;
            } finally {
                document.getElementById('sendBtn').disabled = false;
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        }

        document.getElementById('userInput').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
        });

        loadModels();
    </script>
</body>
</html>
"""
@app.get("/", response_class=HTMLResponse)
def serve_ui(): return HTMLResponse(content=HTML_TEMPLATE, status_code=200)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)