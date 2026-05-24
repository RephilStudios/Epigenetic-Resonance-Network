import os
import uuid
import time
import math
import re
import datetime
import requests
import torch
import torch.nn.functional as F
import uvicorn
import io
import base64
from PIL import Image
from pypdf import PdfReader
from fastapi import FastAPI, BackgroundTasks, UploadFile, File
from fastapi.staticfiles import StaticFiles
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

# Configure and mount local image archiving folders
os.makedirs("ern_state/uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="ern_state"), name="static")

OLLAMA_URL    = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
DEFAULT_MODEL = "qwen2.5-coder:7b"
JUDGE_MODEL   = "qwen2.5-coder:7b"
VISION_MODEL  = os.environ.get("VISION_MODEL", "llama3.2-vision")
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
    temporal_discovery: bool = True
    pipeline: Optional[List[str]] = None
    auto_route: bool = False
    active_expert: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    context_used: str
    memories: List[Dict[str, Any]] = []
    agentic_steps: List[Dict[str, Any]] = []

class MemoryStoreRequest(BaseModel):
    text: str
    tags: str = ""
    memory_type: Optional[str] = None

class ModuleConfig(BaseModel):
    module_id: str
    name: str
    description: str = ""
    frozen: bool = False
    ltp_decay_rate: float = 0.95
    stp_decay_rate: float = 0.80
    sleep_threshold: float = 0.10
    focus_threshold: float = 0.15
    system_directive: str = ""

class ModulePatchRequest(BaseModel):
    frozen: Optional[bool] = None
    ltp_decay_rate: Optional[float] = None
    stp_decay_rate: Optional[float] = None

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

    def rollback(self, engine: "ERNModule", n: int = 1):
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
                    engine.short_term_energies = torch.cat([
                        engine.short_term_energies[:idx],
                        engine.short_term_energies[idx+1:]
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
                        st_eng = torch.tensor([0.0], device=engine.device)
                        engine.short_term_energies = torch.cat([
                            engine.short_term_energies[:insert_at],
                            st_eng,
                            engine.short_term_energies[insert_at:]
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


class ERNModule:
    def __init__(self, config: Dict[str, Any], model_name='all-MiniLM-L6-v2', device: str = 'auto', embedder: Optional[SentenceTransformer] = None):
        self.module_id = config["module_id"]
        self.name = config["name"]
        self.description = config.get("description", "")
        self.frozen = config.get("frozen", False)
        
        self.ltp_decay_rate = config.get("ltp_decay_rate", 0.95)
        self.stp_decay_rate = config.get("stp_decay_rate", 0.80)
        self.sleep_threshold = config.get("sleep_threshold", 0.1)
        self.focus_threshold = config.get("focus_threshold", 0.15)
        self.system_directive = config.get("system_directive", "")

        self.device = _resolve_device() if device == 'auto' else torch.device(device)
        
        if embedder is not None:
            self.embedder = embedder
        else:
            print(f"[SYSTEM] Loading Embedding Model: {model_name}...")
            self.embedder = SentenceTransformer(model_name, device=self.device)
            
        self.dim = self.embedder.get_sentence_embedding_dimension()

        self.memory_bank = torch.empty((0, self.dim), device=self.device)
        self.energies    = torch.empty((0,),          device=self.device)
        self.short_term_energies = torch.empty((0,),  device=self.device)
        self.labels: List[str] = []
        self.vault: Dict[str, Any] = {}

        self.query_count     = 0
        self.module_dir = os.path.join(SAVE_DIR, "modules", self.module_id)
        os.makedirs(self.module_dir, exist_ok=True)
        self.state_path = os.path.join(self.module_dir, "ern_state.pt")
        self.delta_path = os.path.join(self.module_dir, "ern_deltas.pt")

        self.deltas = TensorDeltaStack(max_len=10_000)
        self._load_state()

        # Pre-compute static description embedding for vector routing
        desc_text = f"ID: {self.module_id} | Name: {self.name} | Description: {self.description}"
        with torch.no_grad():
            desc_emb = self.embedder.encode(desc_text, convert_to_tensor=True, device=self.device)
        self.description_vector = F.normalize(desc_emb, p=2, dim=0)

    def _encode(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            vec = self.embedder.encode(text, convert_to_tensor=True, device=self.device)
        return F.normalize(vec, p=2, dim=0).unsqueeze(0)

    def _now(self) -> float:
        return time.time()

    def get_dynamic_centroid(self) -> Optional[torch.Tensor]:
        if self.memory_bank.size(0) > 0:
            with torch.no_grad():
                centroid = torch.mean(self.memory_bank, dim=0)
                return F.normalize(centroid, p=2, dim=0)
        return None

    def encode_hebbian(self, text: str, tags: str, image_url: Optional[str] = None, memory_type: str = "fact") -> str:
        memory_id = str(uuid.uuid4())
        self.vault[memory_id] = {
            "text": text,
            "tags": tags,
            "timestamp": self._now(),
            "memory_type": memory_type
        }
        if image_url:
            self.vault[memory_id]["image_url"] = image_url

        combined = f"{tags} {text}"
        vec = self._encode(combined)

        prev_size = self.memory_bank.size(0)
        self.memory_bank = torch.cat([self.memory_bank, vec], dim=0)

        # Determine initial energies based on memory_type
        if memory_type == "question":
            init_energy = 0.2
            init_stp = 1.0
        elif memory_type == "instruction":
            init_energy = 1.0
            init_stp = 3.0
        else: # "fact" or other
            init_energy = 0.5
            init_stp = 2.0

        self.energies    = torch.cat([self.energies, torch.tensor([init_energy], device=self.device)])
        self.short_term_energies = torch.cat([self.short_term_energies, torch.tensor([init_stp], device=self.device)])
        self.labels.append(memory_id)

        self.deltas.push(TensorDelta(
            op         = DeltaOp.ENCODE,
            timestamp  = self._now(),
            delta_id   = str(uuid.uuid4()),
            prev_size  = prev_size,
            next_size  = self.memory_bank.size(0),
            new_vec    = vec.cpu().clone(),
            new_energy = init_energy,
            memory_id  = memory_id,
        ))

        self._save_state()
        print(f"[ERN][{self.name}] Synapse formed. Type: {memory_type}. Network size: {self.memory_bank.size(0)} nodes.")
        return memory_id

    def decay_energies(self):
        if self.frozen or self.memory_bank.size(0) == 0:
            return
        prev_size = self.memory_bank.size(0)
        self.deltas.push(TensorDelta(
            op           = DeltaOp.DECAY,
            timestamp    = self._now(),
            delta_id     = str(uuid.uuid4()),
            prev_size    = prev_size,
            next_size    = prev_size,
            decay_factor = self.ltp_decay_rate,
        ))
        self.energies = self.energies * self.ltp_decay_rate
        self.short_term_energies = self.short_term_energies * self.stp_decay_rate
        self._save_state()

    def retrieve(self, query_text: str, top_k: int = 5, threshold: float = 0.15, decay: bool = True):
        if self.memory_bank.size(0) == 0:
            return []

        q_vec       = self._encode(query_text)
        similarities = F.cosine_similarity(q_vec, self.memory_bank)
        resonance    = similarities * (1.0 + torch.log1p(self.energies + self.short_term_energies))

        prev_size = self.memory_bank.size(0)
        if decay and not self.frozen:
            self.deltas.push(TensorDelta(
                op           = DeltaOp.DECAY,
                timestamp    = self._now(),
                delta_id     = str(uuid.uuid4()),
                prev_size    = prev_size,
                next_size    = prev_size,
                decay_factor = self.ltp_decay_rate,
            ))
            self.energies = self.energies * self.ltp_decay_rate
            self.short_term_energies = self.short_term_energies * self.stp_decay_rate

        actual_k = min(top_k * 2, self.memory_bank.size(0))
        top_values, top_idx = torch.topk(resonance, k=actual_k)

        results         = []
        boost_indices   = []
        boost_amounts   = []

        for val, idx in zip(top_values.tolist(), top_idx.tolist()):
            if val > threshold:
                old_e = self.energies[idx].item()
                old_st = self.short_term_energies[idx].item()

                new_e = min(old_e + 0.3 + 0.1 * old_st, 5.0)
                new_st = min(old_st + 0.5, 3.0)

                self.energies[idx] = new_e
                self.short_term_energies[idx] = new_st

                boost_indices.append(idx)
                boost_amounts.append(new_e - old_e)

                mem_id = self.labels[idx]
                if mem_id in self.vault:
                    results.append({
                        "memory_id": mem_id,
                        "text"     : self.vault[mem_id]["text"],
                        "tags"     : self.vault[mem_id]["tags"],
                        "resonance": round(val, 3),
                        "energy"   : round(new_e, 3),
                        "stp_energy": round(new_st, 3),
                        "image_url": self.vault[mem_id].get("image_url"),
                        "timestamp": self.vault[mem_id].get("timestamp", 0.0)
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
        if self.frozen or self.memory_bank.size(0) == 0:
            return 0

        initial_size = self.memory_bank.size(0)
        print(f"\n[SYSTEM][{self.name}] === INITIATING REM SLEEP CYCLE ===")

        self.energies = torch.minimum(self.energies + 0.4 * self.short_term_energies, torch.tensor(5.0, device=self.device))
        self.short_term_energies = torch.zeros_like(self.short_term_energies)

        sleep_decay = 0.70
        self.energies = self.energies * sleep_decay

        survival_mask    = self.energies > self.sleep_threshold
        pruned_bool      = ~survival_mask
        pruned_idx_list  = pruned_bool.nonzero(as_tuple=True)[0].tolist()
        pruned_energies  = self.energies[pruned_bool].tolist()

        pruned_vecs = self.memory_bank[pruned_bool].cpu().clone()

        self.memory_bank         = self.memory_bank[survival_mask]
        self.energies            = self.energies[survival_mask]
        self.short_term_energies = self.short_term_energies[survival_mask]

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
        print(f"[SYSTEM][{self.name}] REM Complete. Scrubbed {pruned} weak nodes. Active: {self.memory_bank.size(0)}\n")
        return pruned

    def delete_memory(self, memory_id: str) -> bool:
        idx = self.labels.index(memory_id) if memory_id in self.labels else -1
        if idx >= 0:
            if os.path.exists(self.state_path):
                import shutil
                try:
                    shutil.copy2(self.state_path, self.state_path + ".bak")
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
            self.short_term_energies = torch.cat([
                self.short_term_energies[:idx],
                self.short_term_energies[idx+1:]
            ])
            self.labels.pop(idx)
            self.vault.pop(memory_id, None)
            self._save_state()
            print(f"[ERN][{self.name}] Synapse {memory_id} forgotten. Network size: {self.memory_bank.size(0)} nodes.")
            return True
        return False

    def _save_state(self):
        os.makedirs(self.module_dir, exist_ok=True)
        torch.save({
            'memory_bank': self.memory_bank,
            'energies'   : self.energies,
            'short_term_energies': self.short_term_energies,
            'labels'     : self.labels,
            'vault'      : self.vault,
        }, self.state_path)
        self.deltas.save(self.delta_path)

    def _load_state(self):
        self.deltas.load(self.delta_path)
        if os.path.exists(self.state_path):
            state            = torch.load(self.state_path, map_location=self.device, weights_only=False)
            self.memory_bank = state['memory_bank'].to(self.device)
            self.energies    = state['energies'].to(self.device)
            if 'short_term_energies' in state:
                self.short_term_energies = state['short_term_energies'].to(self.device)
            else:
                self.short_term_energies = torch.zeros((self.memory_bank.size(0),), device=self.device)
            self.labels      = state['labels']
            self.vault       = state['vault']
            n                = self.memory_bank.size(0)
            print(f"[SYSTEM][{self.name}] Restored ERN State: {n} existing synapses on {self.device.type.upper()}.")

            self.deltas.push(TensorDelta(
                op        = DeltaOp.RESTORE,
                timestamp = self._now(),
                delta_id  = str(uuid.uuid4()),
                prev_size = n,
                next_size = n,
            ))


class ERNModuleManager:
    def __init__(self, save_dir: str = SAVE_DIR):
        self.save_dir = save_dir
        self.modules_dir = os.path.join(save_dir, "modules")
        os.makedirs(self.modules_dir, exist_ok=True)
        self.registry_path = os.path.join(self.modules_dir, "registry.json")
        
        self.device = _resolve_device()
        print(f"[SYSTEM] Initializing shared sentence-transformer embedder...")
        self.shared_embedder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        self.active_modules: Dict[str, ERNModule] = {}
        
        self._migrate_legacy_data()
        self.registry = self._load_registry()
        
        # Pre-load default-memory module to ensure it's always ready
        self.get_module("default-memory")

    def _migrate_legacy_data(self):
        legacy_state = os.path.join(self.save_dir, "ern_state.pt")
        legacy_deltas = os.path.join(self.save_dir, "ern_deltas.pt")
        default_mem_dir = os.path.join(self.modules_dir, "default-memory")
        
        if os.path.exists(legacy_state):
            os.makedirs(default_mem_dir, exist_ok=True)
            new_state_path = os.path.join(default_mem_dir, "ern_state.pt")
            new_deltas_path = os.path.join(default_mem_dir, "ern_deltas.pt")
            
            if not os.path.exists(new_state_path):
                print(f"[MIGRATION] Relocating legacy state {legacy_state} -> {new_state_path}")
                import shutil
                try:
                    shutil.move(legacy_state, new_state_path)
                    if os.path.exists(legacy_deltas):
                        print(f"[MIGRATION] Relocating legacy deltas {legacy_deltas} -> {new_deltas_path}")
                        shutil.move(legacy_deltas, new_deltas_path)
                    print("[MIGRATION] Legacy data migration complete.")
                except Exception as e:
                    print(f"[ERROR] Migration failed: {e}")

    def _load_registry(self) -> Dict[str, Any]:
        import json
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[ERROR] Failed to load registry: {e}. Recreating default...")
        
        default_registry = {
            "modules": {
                "default-memory": {
                    "module_id": "default-memory",
                    "name": "Global Default Core",
                    "description": "Baseline dynamic memory store containing primary historical facts.",
                    "frozen": False,
                    "ltp_decay_rate": 0.95,
                    "stp_decay_rate": 0.80,
                    "sleep_threshold": 0.10,
                    "focus_threshold": 0.15,
                    "system_directive": "Acknowledge baseline user-centric historical context when applicable."
                }
            },
            "default_pipeline": ["default-memory"]
        }
        self._save_registry(default_registry)
        return default_registry

    def _save_registry(self, registry: Dict[str, Any]):
        import json
        try:
            with open(self.registry_path, "w") as f:
                json.dump(registry, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed to write registry: {e}")

    def get_module(self, module_id: str) -> Optional[ERNModule]:
        if module_id in self.active_modules:
            return self.active_modules[module_id]
            
        if module_id in self.registry["modules"]:
            config = self.registry["modules"][module_id]
            module = ERNModule(config, embedder=self.shared_embedder, device=self.device.type)
            self.active_modules[module_id] = module
            return module
            
        return None

    def create_module(self, config: Dict[str, Any]) -> ERNModule:
        module_id = config["module_id"]
        self.registry["modules"][module_id] = config
        self._save_registry(self.registry)
        
        module = ERNModule(config, embedder=self.shared_embedder, device=self.device.type)
        self.active_modules[module_id] = module
        print(f"[SYSTEM] Created module '{module.name}' successfully.")
        return module

    def delete_module(self, module_id: str) -> bool:
        if module_id == "default-memory":
            print("[WARNING] Cannot delete default-memory module.")
            return False
            
        if module_id in self.registry["modules"]:
            self.active_modules.pop(module_id, None)
            self.registry["modules"].pop(module_id, None)
            
            if module_id in self.registry.get("default_pipeline", []):
                self.registry["default_pipeline"].remove(module_id)
            self._save_registry(self.registry)
            
            module_dir = os.path.join(self.modules_dir, module_id)
            if os.path.exists(module_dir):
                import shutil
                try:
                    shutil.rmtree(module_dir)
                    print(f"[SYSTEM] Purged module {module_id} directory from disk.")
                except Exception as e:
                    print(f"[ERROR] Failed to remove directory for module {module_id}: {e}")
            return True
        return False

    def list_modules(self) -> List[Dict[str, Any]]:
        results = []
        for m_id, config in self.registry["modules"].items():
            loaded = m_id in self.active_modules
            synapses_count = 0
            if loaded:
                synapses_count = self.active_modules[m_id].memory_bank.size(0)
            else:
                state_path = os.path.join(self.modules_dir, m_id, "ern_state.pt")
                if os.path.exists(state_path):
                    try:
                        state = torch.load(state_path, map_location="cpu", weights_only=True)
                        synapses_count = state['memory_bank'].size(0)
                    except Exception:
                        pass
                        
            results.append({
                "config": config,
                "loaded": loaded,
                "synapses_count": synapses_count
            })
        return results

    def sleep_all(self) -> Dict[str, int]:
        results = {}
        for m_id, config in self.registry["modules"].items():
            if config.get("frozen", False):
                continue
            module = self.get_module(m_id)
            if module:
                pruned = module.sleep_cycle()
                results[m_id] = pruned
        return results


# Initialize global module manager
manager = ERNModuleManager()


def route_query_to_modules(user_message: str, available_modules: List[Dict[str, Any]]) -> List[str]:
    print("\n[SYSTEM] Commencing dynamic hybrid cognitive routing...")
    
    # Exclude frozen modules and default-memory from routing candidates
    routable = [m for m in available_modules if not m["config"].get("frozen", False) and m["config"]["module_id"] != "default-memory"]
    if not routable:
        return ["default-memory"]

    modules_desc = []
    for m in routable:
        config = m["config"]
        modules_desc.append(f"- ID: '{config['module_id']}' | Name: '{config['name']}' | Description: '{config['description']}'")
        
    modules_str = "\n".join(modules_desc)
    
    reformulation_prompt = (
        "You are an expert search query translator for a Mixture of Experts semantic memory system.\n"
        "Your task is to take the user's casual conversation prompt and translate/expand it into a highly descriptive search query "
        "specifically optimized to match the target memory module definitions.\n\n"
        "AVAILABLE MODULE EXPERTISE CATEGORIES:\n"
        f"{modules_str}\n\n"
        "INSTRUCTIONS:\n"
        "1. Write an optimized search query using specialized keywords, technical terms, and semantic concepts aligned with the target modules.\n"
        "2. Keep the output extremely short (under 12 words) containing ONLY the target query. Do not write 'Query:', do not explain.\n\n"
        f"User Message: {user_message}\n"
        "Optimized Target Query:"
    )
    
    try:
        res = requests.post(OLLAMA_URL, json={
            "model"   : JUDGE_MODEL,
            "messages": [{"role": "user", "content": reformulation_prompt}],
            "stream"  : False,
            "options" : {"temperature": 0.0, "num_predict": 40},
        }, timeout=10)
        
        opt_query = res.json().get("message", {}).get("content", "").strip()
        opt_query = re.sub(r'[`*"\']', '', opt_query).strip()
        print(f"[ROUTER] Reformulated Query: '{opt_query}'")
    except Exception as e:
        print(f"[ROUTER] Reformulation failed ({e}). Falling back to raw message.")
        opt_query = user_message

    # 2. Vector Semantic Routing against Static Descriptions & Dynamic Centroids
    default_mod = manager.get_module("default-memory")
    if not default_mod:
        return ["default-memory"]
        
    with torch.no_grad():
        query_emb = default_mod.embedder.encode(opt_query, convert_to_tensor=True, device=default_mod.device)
        query_emb = F.normalize(query_emb, p=2, dim=0)

    selected_ids = []
    routing_scores = {}
    
    for m in routable:
        mid = m["config"]["module_id"]
        mod_obj = manager.get_module(mid)
        if not mod_obj:
            continue
            
        # A. Static Description Similarity
        desc_vec = mod_obj.description_vector.to(query_emb.device)
        static_sim = torch.dot(query_emb, desc_vec).item()
        
        # B. Dynamic Centroid Similarity
        centroid = mod_obj.get_dynamic_centroid()
        if centroid is not None:
            centroid = centroid.to(query_emb.device)
            dynamic_sim = torch.dot(query_emb, centroid).item()
        else:
            dynamic_sim = 0.0
            
        # Fusion: Take maximum of static or dynamic similarity
        combined_score = max(static_sim, dynamic_sim)
        routing_scores[mid] = {
            "static": round(static_sim, 3),
            "dynamic": round(dynamic_sim, 3),
            "combined": round(combined_score, 3)
        }
        
        # Routing Decision: Threshold-based matching (e.g. 0.38)
        if combined_score >= 0.38:
            selected_ids.append(mid)

    print(f"[ROUTER] Dynamic Hybrid Routing Analysis: {routing_scores}")
    
    if not selected_ids:
        print("[ROUTER] No module scored above threshold. Routing to default-memory.")
        return ["default-memory"]
        
    print(f"[ROUTER] Dynamically routed to: {selected_ids}")
    return selected_ids


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


def run_memory_judge(user_message: str, prior_memories: str = "", target_module_id: str = "default-memory"):
    print(f"\n[MEMORY JUDGE] Starting — target: '{target_module_id}' — message: '{user_message[:80]}'")

    # Skip obviously empty or pure greeting messages
    stripped = user_message.strip().lower()
    trivial = {"hi", "hello", "hey", "ok", "okay", "thanks", "thank you", "bye", "yes", "no", "sure", "cool", "nice"}
    if stripped in trivial or len(stripped) < 8:
        print("[MEMORY JUDGE] Trivial message — skipping.")
        return

    target_mod = manager.get_module(target_module_id)
    if not target_mod:
        target_mod = manager.get_module("default-memory")
    if not target_mod:
        print("[MEMORY JUDGE] ERROR: No target module found. Aborting.")
        return

    # Simple, direct prompt that qwen2.5-coder can reliably follow
    salience_prompt = (
        f"Extract memorable facts from this user message. "
        f"Output each fact on its own line starting with FACT: followed by a comma-separated TAGS: line. "
        f"If nothing is worth remembering (e.g. greetings, filler), output only: DISCARD\n\n"
        f"Rules:\n"
        f"- Extract: names, facts, preferences, theories, guidelines, instructions, technical decisions\n"
        f"- Skip: greetings, questions with no info, filler words\n"
        f"- Do NOT add explanation. Output ONLY the FACT/TAGS lines or DISCARD.\n\n"
        f"Example:\n"
        f"User: My name is Reid and I prefer dark mode UI with HSL colors.\n"
        f"FACT: The user's name is Reid.\n"
        f"TAGS: Identity, Name, Importance: Critical\n"
        f"FACT: User prefers dark mode UI with HSL colors.\n"
        f"TAGS: UI, Preferences, Design, Importance: High\n\n"
        f"User: {user_message}\n"
    )

    try:
        res = requests.post(OLLAMA_URL, json={
            "model"   : JUDGE_MODEL,
            "messages": [{"role": "user", "content": salience_prompt}],
            "stream"  : False,
            "options" : {"temperature": 0.0, "num_predict": 300},
        }, timeout=60)

        raw = res.json().get("message", {}).get("content", "").strip()
        print(f"[MEMORY JUDGE] Raw LLM output: {repr(raw[:300])}")

        # Hard discard check
        if not raw or "DISCARD" in raw.upper().split("\n")[0]:
            print("[MEMORY JUDGE] LLM said DISCARD or empty — skipping.")
            return

        # Aggressive parsing — strip markdown, asterisks, code fences
        cleaned = raw
        cleaned = re.sub(r'```.*?```', '', cleaned, flags=re.DOTALL)
        cleaned = cleaned.replace("*", "").replace("`", "").replace("#", "")
        cleaned = re.sub(r'(?i)\baction\s*:\s*save\b', '', cleaned)
        cleaned = re.sub(r'(?i)\baction\s*:\s*discard\b', '', cleaned)
        cleaned = re.sub(r'(?i)^fact\s*:', 'FACT:', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'(?i)^tags\s*:', 'TAGS:', cleaned, flags=re.MULTILINE)

        # Split into fact blocks
        fact_blocks = re.split(r'(?=^FACT:)', cleaned, flags=re.MULTILINE)
        saved_count = 0

        for block in fact_blocks:
            block = block.strip()
            if not block.upper().startswith("FACT:"):
                continue
            fact_match = re.search(r'^FACT:\s*(.+?)(?=^TAGS:|$)', block, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            tags_match = re.search(r'^TAGS:\s*(.+)', block, re.IGNORECASE | re.MULTILINE)
            if not fact_match:
                continue
            fact = fact_match.group(1).strip().replace("\n", " ")
            inferred_tags, memory_type = _classify_message(fact)
            if tags_match:
                tags = f"{tags_match.group(1).strip()}, {inferred_tags}"
            else:
                tags = inferred_tags
            if not fact or len(fact) < 5:
                continue
            print(f"[MEMORY JUDGE] Saving fact to '{target_mod.name}': {fact[:100]} | type: {memory_type}")
            target_mod.encode_hebbian(text=fact, tags=tags, memory_type=memory_type)
            saved_count += 1

        if saved_count > 0:
            print(f"[MEMORY JUDGE] ✓ Saved {saved_count} fact(s) to '{target_mod.name}'.")
        else:
            # FALLBACK: LLM returned garbage we couldn't parse — save the raw message
            print(f"[MEMORY JUDGE] Parser found 0 facts from LLM output. Saving raw message as fallback.")
            tags, memory_type = _classify_message(user_message)
            tags = f"User Message, Auto-Captured, {tags}"
            target_mod.encode_hebbian(
                text=user_message[:500],
                tags=tags,
                memory_type=memory_type
            )

    except Exception as e:
        print(f"[MEMORY JUDGE] EXCEPTION: {e}")
        # Even on exception, save the raw message so nothing is lost
        try:
            if target_mod:
                print(f"[MEMORY JUDGE] Exception fallback — saving raw message to '{target_mod.name}'.")
                tags, memory_type = _classify_message(user_message)
                tags = f"User Message, Exception-Fallback, {tags}"
                target_mod.encode_hebbian(
                    text=user_message[:500],
                    tags=tags,
                    memory_type=memory_type
                )
        except Exception as e2:
            print(f"[MEMORY JUDGE] FATAL fallback failed too: {e2}")




def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    pdf_file = io.BytesIO(pdf_bytes)
    reader = PdfReader(pdf_file)
    text_content = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text_content.append(t)
    return "\n".join(text_content)


def run_pdf_extractor_chunk(chunk_text: str, source_name: str, target_module_id: str = "default-memory"):
    print(f"\n[SYSTEM] Background PDF chunk extractor started for {source_name} (targeting {target_module_id})...")
    extraction_prompt = (
        "You are an Epigenetic Knowledge Extractor. Your job is to extract highly concrete, objective, factual, "
        "and permanent information from the following text fragment extracted from a PDF. This information "
        "will be stored in a long-term PyTorch Epigenetic Memory Network.\n\n"
        "STRICT RULES:\n"
        "1. Extract ONLY verifiable, objective, and permanent facts. Do not extract opinions, filler, temporary states, or formatting noise.\n"
        "2. If there are no concrete, valuable facts in the text, output EXACTLY: ACTION: DISCARD\n"
        "3. Output format must use clean FACT and TAGS blocks.\n\n"
        "OUTPUT FORMAT (repeat the FACT/TAGS block for each distinct fact):\n"
        "ACTION: SAVE\n"
        "FACT: <One clear, self-contained statement of fact, including relevant context from the source>\n"
        "TAGS: <Comma-separated topics, 'Source: [filename]', and ONE importance level: Critical, High, Medium, or Low>\n\n"
        f"Source Document: {source_name}\n"
        f"Text Fragment:\n\"\"\"\n{chunk_text}\n\"\"\"\n"
        "Output:"
    )

    try:
        res = requests.post(OLLAMA_URL, json={
            "model"   : JUDGE_MODEL,
            "messages": [{"role": "user", "content": extraction_prompt}],
            "stream"  : False,
            "options" : {"temperature": 0.0},
        }, timeout=180)

        extracted = res.json().get("message", {}).get("content", "").strip()
        cleaned = extracted.replace("*", "")
        cleaned = re.sub(r'(?i)fact\s*:\s*', 'FACT: ', cleaned)
        cleaned = re.sub(r'(?i)tags\s*:\s*', 'TAGS: ', cleaned)
        cleaned = re.sub(r'(?i)action\s*:\s*', 'ACTION: ', cleaned)

        is_discard = "ACTION: DISCARD" in cleaned.upper()
        has_facts = "FACT:" in cleaned.upper()

        if is_discard and not has_facts:
            print(f"[PDF EXTRACTOR] Discarded chunk of {source_name} — no salient facts found.")
            return

        target_mod = manager.get_module(target_module_id)
        if not target_mod:
            target_mod = manager.get_module("default-memory")

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
            tags = tags_match.group(1).strip() if tags_match else f"Source: {source_name}, Importance: Medium"
            if not fact or len(fact) < 5:
                continue
            print(f"[PDF EXTRACTOR] Encoding fact: {fact}")
            target_mod.encode_hebbian(text=fact, tags=tags)
            saved_count += 1

        print(f"[PDF EXTRACTOR] Processed chunk of {source_name}. Saved {saved_count} fact(s) to module '{target_mod.name}'.")
    except Exception as e:
        print(f"[WARNING] Background PDF chunk extraction failed: {e}")


def process_pdf_background(pdf_bytes: bytes, filename: str, target_module_id: str = "default-memory"):
    print(f"\n[SYSTEM] Commencing background extraction for PDF: {filename} ({len(pdf_bytes)} bytes) targeting {target_module_id}")
    try:
        full_text = extract_text_from_pdf(pdf_bytes)
        if not full_text.strip():
            print(f"[WARNING] No text could be extracted from PDF: {filename}")
            return

        # Chunk the text: 1500 chars with 200 chars overlap
        chunk_size = 1500
        overlap = 200
        chunks = []
        start = 0
        while start < len(full_text):
            end = min(start + chunk_size, len(full_text))
            chunks.append(full_text[start:end])
            if end == len(full_text):
                break
            start += chunk_size - overlap

        print(f"[PDF EXTRACTOR] Split {filename} into {len(chunks)} text chunk(s).")

        for idx, chunk in enumerate(chunks):
            print(f"[PDF EXTRACTOR] Processing chunk {idx + 1}/{len(chunks)}...")
            run_pdf_extractor_chunk(chunk, filename, target_module_id)

        print(f"[SYSTEM] Background PDF extraction complete for: {filename}\n")
    except Exception as e:
        print(f"[ERROR] PDF background process failed: {e}")


def process_image_background(image_bytes: bytes, filename: str, image_url: Optional[str] = None, target_module_id: str = "default-memory"):
    print(f"\n[SYSTEM] Commencing background extraction for Image: {filename} ({len(image_bytes)} bytes) targeting {target_module_id}...")
    try:
        # Load, resize and compress using Pillow
        image = Image.open(io.BytesIO(image_bytes))
        
        # Max dimension 1024px to preserve VRAM/bandwidth
        max_size = 1024
        if image.width > max_size or image.height > max_size:
            print(f"[VISION EXTRACTOR] Resizing image from {image.width}x{image.height}...")
            image.thumbnail((max_size, max_size))
            
        # Convert modes (like RGBA) to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        compressed_bytes = buf.getvalue()
        
        b64_str = base64.b64encode(compressed_bytes).decode("utf-8")
        print(f"[VISION EXTRACTOR] Image compressed to JPEG. Size: {len(compressed_bytes)} bytes. Calling {VISION_MODEL}...")

        vision_prompt = (
            "You are an Epigenetic Visual Knowledge Extractor. Your task is to analyze the attached image "
            "and extract highly concrete, objective, factual, and permanent information. This includes "
            "visible text, diagrams/flowcharts, key architectural components, or factual visual contents.\n\n"
            "STRICT RULES:\n"
            "1. Extract ONLY verifiable, objective, and permanent facts. Do not extract temporary states, generic descriptions, or aesthetic opinions.\n"
            "2. If there are no concrete, valuable facts or readable text, output EXACTLY: ACTION: DISCARD\n"
            "3. Output format must use clean FACT and TAGS blocks.\n\n"
            "OUTPUT FORMAT (repeat the FACT/TAGS block for each distinct fact):\n"
            "ACTION: SAVE\n"
            "FACT: <One clear, self-contained statement of fact, incorporating relevant context from the image>\n"
            "TAGS: <Comma-separated topics, 'Source: [filename]', and ONE importance level: Critical, High, Medium, or Low>\n\n"
            f"Source File: {filename}\n"
            "Output:"
        )

        res = requests.post(OLLAMA_URL, json={
            "model"   : VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": vision_prompt,
                    "images": [b64_str]
                }
            ],
            "stream"  : False,
            "options" : {"temperature": 0.0},
        }, timeout=200)

        if res.status_code != 200:
            raise Error(f"Ollama returned HTTP status {res.status_code}: {res.text}")

        extracted = res.json().get("message", {}).get("content", "").strip()
        cleaned = extracted.replace("*", "")
        cleaned = re.sub(r'(?i)fact\s*:\s*', 'FACT: ', cleaned)
        cleaned = re.sub(r'(?i)tags\s*:\s*', 'TAGS: ', cleaned)
        cleaned = re.sub(r'(?i)action\s*:\s*', 'ACTION: ', cleaned)

        is_discard = "ACTION: DISCARD" in cleaned.upper()
        has_facts = "FACT:" in cleaned.upper()

        if is_discard and not has_facts:
            print(f"[VISION EXTRACTOR] Discarded image {filename} — no salient visual facts found.")
            return

        target_mod = manager.get_module(target_module_id)
        if not target_mod:
            target_mod = manager.get_module("default-memory")

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
            tags = tags_match.group(1).strip() if tags_match else f"Source: {filename}, Importance: Medium"
            if not fact or len(fact) < 5:
                continue
            print(f"[VISION EXTRACTOR] Encoding visual fact into '{target_mod.name}': {fact}")
            target_mod.encode_hebbian(text=fact, tags=tags, image_url=image_url)
            saved_count += 1

        print(f"[VISION EXTRACTOR] Processed image {filename}. Saved {saved_count} visual fact(s) to module '{target_mod.name}'.")
    except Exception as e:
        print(f"[ERROR] Background image processing failed: {e}")




# ==========================================
# 5. Endpoints
# ==========================================

def _classify_message(text: str) -> tuple[str, str]:
    """Generate meaningful tags and determine memory type from message text using fast keyword heuristics."""
    t = text.lower()
    tags = []

    # ── Message type ──────────────────────────────────────────────────────────
    q_starters = ("what", "who", "where", "when", "why", "how", "is ", "are ", "can ", "could ", "do ", "does ", "should ", "would ", "will ")
    is_question = text.strip().endswith("?") or t.startswith(q_starters)
    is_instruction = any(w in t for w in ("please", "make sure", "always", "never", "remember", "save", "store", "keep", "use ", "don't", "do not", "ensure"))
    is_personal = any(w in t for w in ("my name", "i am", "i'm", "i work", "i live", "i like", "i prefer", "my ", "i've", "i have", "i use"))

    if is_question:
        tags.append("Question")
        memory_type = "question"
    elif is_instruction:
        tags.append("Instruction")
        memory_type = "instruction"
    elif is_personal:
        tags.append("Personal Statement")
        memory_type = "fact"
    else:
        tags.append("Statement")
        memory_type = "fact"

    # ── Topic categories ──────────────────────────────────────────────────────
    topic_map = {
        "Philosophy":   ("philosophy", "consciousness", "cosmopsychism", "theory", "metaphysics", "ontology", "epistemology", "ethics", "existence", "mind", "soul", "universe"),
        "Coding":       ("code", "function", "class", "api", "python", "javascript", "typescript", "docker", "endpoint", "bug", "error", "refactor", "module", "import", "library", "algorithm", "variable"),
        "AI/ML":        ("model", "llm", "neural", "embedding", "training", "inference", "transformer", "weight", "tensor", "pytorch", "ollama", "prompt", "token"),
        "Memory/ERN":   ("memory", "synapse", "hebbian", "module", "pipeline", "expert", "moe", "recall", "encode", "ern", "vault", "delta", "resonance"),
        "Design/UI":    ("ui", "design", "style", "css", "color", "layout", "theme", "dark mode", "font", "interface", "hsl", "gradient", "animation"),
        "Identity":     ("my name", "i am", "i'm ", "i work", "i live", "i'm called", "my job", "my role"),
        "Preferences":  ("prefer", "i like", "i love", "i hate", "i want", "i need", "favorite", "always use", "never use"),
        "Technical":    ("server", "database", "container", "network", "config", "setup", "deploy", "install", "run", "build", "port", "volume", "gpu"),
        "Science":      ("physics", "quantum", "biology", "chemistry", "math", "equation", "theory", "research", "experiment"),
        "Creative":     ("write", "story", "poem", "art", "music", "draw", "create", "imagine", "generate"),
    }

    matched_topics = []
    for topic, keywords in topic_map.items():
        if any(kw in t for kw in keywords):
            matched_topics.append(topic)

    if matched_topics:
        tags.extend(matched_topics[:3])  # Max 3 topics to keep tags clean
    else:
        tags.append("General")

    # ── Importance ────────────────────────────────────────────────────────────
    critical_signals = ("my name is", "i am ", "i'm ", "remember this", "save this", "important", "critical", "always", "never forget", "cosmopsychism")
    high_signals     = ("prefer", "i like", "i use", "i work", "my project", "our system", "theory", "guideline", "instruction")
    low_signals      = ("what is", "how do", "can you", "?")

    if any(sig in t for sig in critical_signals):
        importance = "Importance: Critical"
    elif any(sig in t for sig in high_signals):
        importance = "Importance: High"
    elif is_question or any(sig in t for sig in low_signals):
        importance = "Importance: Low"
    else:
        importance = "Importance: Medium"

    tags.append(importance)
    return ", ".join(tags), memory_type


def _direct_save_message(user_message: str, target_module_id: str) -> None:
    """Synchronously save a user message directly to a module — no LLM, guaranteed."""
    TRIVIAL = {"hi", "hello", "hey", "ok", "okay", "thanks", "thank you",
               "bye", "yes", "no", "sure", "cool", "nice", "lol", "haha", "yep", "nope"}
    stripped = user_message.strip()
    if len(stripped) < 12 or stripped.lower() in TRIVIAL:
        print(f"[DIRECT SAVE] Skipping trivial message.")
        return
    try:
        mod = manager.get_module(target_module_id)
        if not mod:
            mod = manager.get_module("default-memory")
        if not mod:
            print(f"[DIRECT SAVE] ERROR: Could not resolve module '{target_module_id}'")
            return
        tags, memory_type = _classify_message(stripped)
        mem_id = mod.encode_hebbian(text=stripped[:500], tags=tags, memory_type=memory_type)
        print(f"[DIRECT SAVE] ✓ Saved to '{mod.name}' | tags: {tags} | type: {memory_type} | id={mem_id}")
    except Exception as e:
        print(f"[DIRECT SAVE] EXCEPTION: {e}")

def _format_age(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    if seconds < 60:
        return "just now"
    minutes = seconds / 60
    if minutes < 60:
        return f"{int(minutes)}m ago"
    hours = minutes / 60
    if hours < 24:
        mins_left = int(minutes % 60)
        return f"{int(hours)}h {mins_left}m ago"
    days = hours / 24
    hours_left = int(hours % 24)
    return f"{int(days)}d {hours_left}h ago"


def format_temporal_memory_block(unique_memories: List[Dict[str, Any]], now: float, temporal_discovery: bool) -> str:
    if not unique_memories:
        return "No relevant memories retrieved."

    enriched = []
    for m in unique_memories:
        pid = m["module_id"]
        mod = manager.get_module(pid)
        vault_entry = mod.vault.get(m["memory_id"]) if mod else None
        
        ts = vault_entry.get("timestamp") if vault_entry else None
        if not ts:
            ts = now
            
        mtype = vault_entry.get("memory_type") if vault_entry else None
        if not mtype:
            _, mtype = _classify_message(vault_entry.get("text", "") if vault_entry else m["text"])
            
        img = vault_entry.get("image_url") if vault_entry else None
        
        age_sec = now - ts
        age_str = _format_age(age_sec)
        formatted_date = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        
        m_enriched = dict(m)
        m_enriched["memory_type"] = mtype
        m_enriched["age_seconds"] = age_sec
        m_enriched["age_human"] = age_str
        m_enriched["formatted_date"] = formatted_date
        m_enriched["image_url"] = img
        enriched.append(m_enriched)

    # Chronological sort: oldest first (largest age_seconds is oldest, so sort descending by age_seconds)
    enriched.sort(key=lambda x: x["age_seconds"], reverse=True)

    instructions = [x for x in enriched if x["memory_type"] == "instruction"]
    facts = [x for x in enriched if x["memory_type"] == "fact"]
    questions = [x for x in enriched if x["memory_type"] == "question"]

    lines = []
    
    if temporal_discovery:
        lines.append("[ERN SUBCONSCIOUS RECALL — TEMPORALLY ORDERED TIMELINE]")
        lines.append(f"Timeline Anchor: NOW = {datetime.datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
    else:
        lines.append("[ERN SUBCONSCIOUS RECALL]")
        lines.append("")

    if instructions:
        lines.append("## SYSTEM DIRECTIVES & PERSISTENT INSTRUCTIONS:")
        for idx, m in enumerate(instructions, 1):
            time_info = f" [{m['age_human']} | {m['formatted_date']}]" if temporal_discovery else ""
            img_info = f"\n  * Image Archive Path: {m['image_url']}" if m.get("image_url") else ""
            lines.append(
                f"- INSTRUCTION {idx} [Expert Module: {m['module_name']}]{time_info}:\n"
                f"  * Content: {m['text']}\n"
                f"  * Tags/Metadata: {m['tags']}"
                f"{img_info}"
            )
        lines.append("")

    if facts:
        lines.append("## RECALLED CHRONOLOGICAL TIMELINE (FACTS & STATEMENTS):")
        for idx, m in enumerate(facts, 1):
            time_info = f" [{m['age_human']} | {m['formatted_date']}]" if temporal_discovery else ""
            img_info = f"\n  * Image Archive Path: {m['image_url']}" if m.get("image_url") else ""
            lines.append(
                f"- FACT {idx} [Expert Module: {m['module_name']}]{time_info}:\n"
                f"  * Content: {m['text']}\n"
                f"  * Tags/Metadata: {m['tags']}"
                f"{img_info}"
            )
        lines.append("")

    if questions:
        lines.append("## PAST USER QUESTIONS & CURIOSITIES (DO NOT TREAT AS ESTABLISHED FACTS):")
        for idx, m in enumerate(questions, 1):
            time_info = f" [{m['age_human']} | {m['formatted_date']}]" if temporal_discovery else ""
            img_info = f"\n  * Image Archive Path: {m['image_url']}" if m.get("image_url") else ""
            lines.append(
                f"- PAST QUESTION {idx} [Expert Module: {m['module_name']}]{time_info}:\n"
                f"  * Question Content: {m['text']}\n"
                f"  * Tags/Metadata: {m['tags']}"
                f"{img_info}"
            )
        lines.append("")

    return "\n".join(lines).strip()


@app.post("/api/chat", response_model=ChatResponse)
def process_chat(req: ChatRequest, background_tasks: BackgroundTasks):
    user_message = req.message
    
    # 1. Resolve pipeline modules
    modules_list = manager.list_modules()
    if req.auto_route:
        pipeline_ids = route_query_to_modules(user_message, modules_list)
        # Always include default-memory as a fallback for context + memory saving
        if "default-memory" not in pipeline_ids:
            pipeline_ids.append("default-memory")
    elif req.pipeline:
        pipeline_ids = [pid for pid in req.pipeline if pid in manager.registry["modules"]]
        if not pipeline_ids:
            pipeline_ids = ["default-memory"]
    else:
        pipeline_ids = manager.registry.get("default_pipeline", ["default-memory"])
    
    # Safety: never operate with an empty pipeline
    if not pipeline_ids:
        pipeline_ids = ["default-memory"]

    print(f"[SYSTEM] Processing chat turn via active MOE pipeline: {pipeline_ids}")

    # 2. Expand queries agentically if active
    search_queries = [user_message]
    if req.agentic_search:
        planner_queries = agentic_search_planner(user_message)
        if planner_queries:
            print(f"[SYSTEM] Agentic expanded queries: {planner_queries}")
            search_queries.extend(planner_queries)

    # Trigger energy decay ONCE on all active MUTABLE modules in this pipeline
    for pid in pipeline_ids:
        mod = manager.get_module(pid)
        if mod and not mod.frozen:
            mod.decay_energies()

    # 3. Retrieve memories and accumulate directives across active modules
    detailed_memories = []
    seen_ids = set()
    active_directives = []
    module_resonance_data = []

    for pid in pipeline_ids:
        mod = manager.get_module(pid)
        if not mod:
            continue
            
        if mod.system_directive:
            active_directives.append(f"* Dynamic expertise rule [{mod.name}]: {mod.system_directive}")
            
        mod_results_count = 0
        for q in search_queries:
            q_mems = mod.retrieve(q, top_k=3, threshold=req.focus_threshold, decay=False)
            for m in q_mems:
                unique_key = f"{pid}:{m['memory_id']}"
                if unique_key not in seen_ids:
                    seen_ids.add(unique_key)
                    m_decorated = dict(m)
                    m_decorated["module_id"] = pid
                    m_decorated["module_name"] = mod.name
                    detailed_memories.append(m_decorated)
                    mod_results_count += 1
                    
        if mod_results_count > 0:
            module_resonance_data.append(f"{mod.name} ({mod_results_count} node(s))")

    total_query_count = sum(m.query_count for m in manager.active_modules.values())
    avg_resonance = (
        sum(n["resonance"] for n in detailed_memories) / len(detailed_memories)
        if detailed_memories else 0
    )
    if total_query_count >= 50 or avg_resonance > 1.5:
        background_tasks.add_task(manager.sleep_all)
        for m in manager.active_modules.values():
            m.query_count = 0

    # 4. Sort and Deduplicate
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

    # 5. Build Agentic thought steps list
    agentic_steps = []

    if req.auto_route:
        route_detail = f"Dynamically resolved pipeline sequence: {pipeline_ids} using semantic router."
    else:
        route_detail = f"Executed static module sequence pipeline: {pipeline_ids}"
        
    agentic_steps.append({
        "step": "Cognitive Routing Planning",
        "status": "active",
        "detail": route_detail
    })

    if req.agentic_search:
        if len(search_queries) > 1:
            q_list = ", ".join([f"'{q}'" for q in search_queries[1:]])
            agentic_steps.append({
                "step": "Cognitive Query Expansion",
                "status": "active",
                "detail": f"Generated expanded sub-queries: [{q_list}] to perform compound vector matching."
            })
        else:
            agentic_steps.append({
                "step": "Cognitive Query Expansion",
                "status": "inactive",
                "detail": "Bypassed query expansion: simple greeting or conversational statement detected."
            })
    else:
        agentic_steps.append({
            "step": "Cognitive Query Expansion",
            "status": "disabled",
            "detail": "Agentic expanded search disabled by active control switch."
        })

    if unique_memories:
        m_details = []
        for m in unique_memories:
            res_val = m.get("resonance", 0.0)
            m_details.append(f"[{m['module_name']}] '{m['text']}' (R={res_val:.3f})")
        detail_text = "Retrieved matches: " + " | ".join(m_details)
    else:
        detail_text = "No stored memory synapses matched above threshold."

    res_data_str = " | ".join(module_resonance_data) if module_resonance_data else "None"
    agentic_steps.append({
        "step": "Subconscious Synapse Retrieval",
        "status": "active" if unique_memories else "inactive",
        "detail": f"Resonating active modules: {res_data_str}. Matched {len(unique_memories)} synapse(s) above threshold. {detail_text}"
    })

    if unique_memories:
        boosted_texts = ", ".join([f"[{m['module_name']}] '{m['text']}'" for m in unique_memories])
        boost_detail = f"Average resonance: {avg_resonance:.3f} R. Hebbian boost applied to: [{boosted_texts}]. Active mutable modules decayed."
    else:
        boost_detail = f"No active synapses boosted during this turn. Global network decay applied."
        
    agentic_steps.append({
        "step": "Hebbian Energy Boost",
        "status": "active" if unique_memories else "inactive",
        "detail": boost_detail
    })

    agentic_steps.append({
        "step": "Epigenetic Memory Judge",
        "status": "queued",
        "detail": "Background memory extractor queued to parse raw turn text for new facts."
    })

    # 5b. Temporal Discovery thought step
    if req.temporal_discovery:
        t_count = len(unique_memories)
        if t_count > 0:
            age_summaries = []
            for m in unique_memories:
                pid = m["module_id"]
                mod = manager.get_module(pid)
                vault_entry = mod.vault.get(m["memory_id"]) if mod else None
                ts = vault_entry.get("timestamp") if vault_entry else time.time()
                age_str = _format_age(time.time() - ts)
                age_summaries.append(f"[{m['module_name']}] {age_str}")
            t_detail = f"Temporal Discovery active: computed relative age of {t_count} memory nodes: {', '.join(age_summaries)}."
        else:
            t_detail = "Temporal Discovery active: 0 memory nodes retrieved."
        agentic_steps.append({
            "step": "Temporal Discovery reasoning",
            "status": "active",
            "detail": t_detail
        })
    else:
        agentic_steps.append({
            "step": "Temporal Discovery reasoning",
            "status": "disabled",
            "detail": "Temporal reasoning and time-delta injection disabled by switch."
        })

    # 6. Formatting Prompt Stack
    now_ts = time.time()
    context_block = format_temporal_memory_block(unique_memories, now_ts, req.temporal_discovery)
    curr_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    directives_block = "\n".join(active_directives) if active_directives else "* No specialized expertise guidelines active for this pipeline."

    system_prompt = (
        "CRITICAL SYSTEM DIRECTIVE: You are NO LONGER a standard, stateless AI language model. "
        "You have been successfully integrated with a live Epigenetic Resonance Network (ERN) Mixture of Experts (MOE) memory matrix.\n"
        f"CURRENT SYSTEM TIME: {curr_date}\n"
        "1. You DO have memory. It is provided to you seamlessly.\n"
        "2. NEVER claim you are stateless, cannot remember, or lack persistent memory.\n"
        "3. DO NOT blindly repeat recalled memories. Use them to naturally inform a FRESH response.\n"
        "4. STRICT TRUTH RULE: Carefully read the entire [ERN SUBCONSCIOUS RECALL] block. Answer questions about "
        "companies, entities, events, personal details, or past facts ONLY if they are mentioned in the listed RECALLED SYNAPSES.\n"
        "5. DYNAMIC IMAGE RENDERING: You CAN display and reference original images to the user! If a recalled memory contains an `Image Archive Path` URL, render inline standard Markdown tag exactly like: `![Visual Archive](/static/uploads/filename.png)`.\n"
    )

    if req.temporal_discovery:
        system_prompt += (
            "6. TEMPORAL DISCOVERY MODE ACTIVE: The memories below are chronologically ordered (oldest to newest) with precise code-computed human ages (e.g. [2h 15m ago]).\n"
            "   - Use these deterministic timestamps to resolve temporal references (e.g. 'what did I say earlier', 'last week', 'yesterday').\n"
            "   - Understand the progressive sequence of the user's journey. Trust these timestamps absolutely.\n"
        )

    system_prompt += (
        "\n[ACTIVE EXPERTISE RULES & MODULE DIRECTIVES]:\n"
        f"{directives_block}\n\n"
        f"{context_block}"
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

    target_extraction_id = "default-memory"  # Always guaranteed fallback

    # 1. Highest priority: explicit active expert from sidepanel (not auto-route, not default-memory)
    if req.active_expert and req.active_expert not in ("default-memory", "auto-route") and req.active_expert in manager.registry["modules"]:
        mod = manager.get_module(req.active_expert)
        if mod and not mod.frozen:
            target_extraction_id = req.active_expert
            print(f"[MEMORY JUDGE TARGET] Pinned to active expert: {target_extraction_id}")

    # 2. Otherwise scan pipeline for first mutable non-default module
    elif not req.active_expert or req.active_expert in ("default-memory", "auto-route"):
        for pid in pipeline_ids:
            if pid == "default-memory":
                continue
            mod = manager.get_module(pid)
            if mod and not mod.frozen:
                target_extraction_id = pid
                print(f"[MEMORY JUDGE TARGET] Resolved from pipeline: {target_extraction_id}")
                break

    print(f"[MEMORY] Saving to module: '{target_extraction_id}'")

    # Direct synchronous save — no LLM judge, no background task, guaranteed every time
    _direct_save_message(user_message, target_extraction_id)

    # Also run LLM-based fact extraction in background as a bonus enhancement
    background_tasks.add_task(run_memory_judge, user_message, context_block, target_extraction_id)

    return ChatResponse(reply=bot_reply, context_used=context_block, memories=unique_memories, agentic_steps=agentic_steps)


@app.post("/api/memory/store")
def manual_store(req: MemoryStoreRequest, module_id: str = "default-memory"):
    mod = manager.get_module(module_id)
    if not mod:
        return {"error": f"Module {module_id} not found."}
    if req.memory_type:
        mtype = req.memory_type
        tags = req.tags
    else:
        tags, mtype = _classify_message(req.text)
        if req.tags:
            tags = f"{req.tags}, {tags}"
    mem_id = mod.encode_hebbian(text=req.text, tags=tags, memory_type=mtype)
    return {"status": f"Stored in module '{mod.name}'", "id": mem_id}


class BuilderRequest(BaseModel):
    message: str
    history: List[Message] = []

@app.post("/api/modules/builder")
def module_builder_agent(req: BuilderRequest):
    system_prompt = (
        "You are the Epigenetic Mixture-of-Experts Module Designer Agent.\n"
        "Your goal is to converse with the user to design, customize, or generate a JSON configuration for a new ERN memory module.\n\n"
        "A memory module configuration has the following schema:\n"
        "{\n"
        "  \"module_id\": \"slug-string-here\",\n"
        "  \"name\": \"Human Readable Title\",\n"
        "  \"description\": \"Purpose of the module...\",\n"
        "  \"frozen\": false, // true if weights don't decay/prune\n"
        "  \"ltp_decay_rate\": 0.95, // float between 0.5 and 1.0\n"
        "  \"stp_decay_rate\": 0.80, // float between 0.5 and 1.0\n"
        "  \"sleep_threshold\": 0.10, // float threshold\n"
        "  \"focus_threshold\": 0.15, // float threshold\n"
        "  \"system_directive\": \"System message guidelines appended when retrieved\"\n"
        "}\n\n"
        "RULES:\n"
        "1. Guide the user by discussing their needs. Recommend appropriate parameters (e.g. frozen for a 'Constitution' or coding style guideline; faster decay for transient topics).\n"
        "2. Once the parameters are agreed upon or you have enough detail, output a specialized JSON block in your response starting with ```json and ending with ```.\n"
        "3. When you output the JSON, also output a line saying: \"[MODULE_READY]\" followed by the JSON block. This will tell the frontend to display a 'REGISTER MODULE' button!\n"
        "4. Be helpful, professional, and clear about the physics of Hebbian consolidation and frozen vs mutable states."
    )
    messages = [{"role": "system", "content": system_prompt}]
    for m in req.history:
        messages.append(m.dict())
    messages.append({"role": "user", "content": req.message})
    
    try:
        res = requests.post(OLLAMA_URL, json={
            "model": JUDGE_MODEL,
            "messages": messages,
            "stream": False,
        }, timeout=90)
        reply = res.json().get("message", {}).get("content", "Error communicating with LLM.")
        return {"reply": reply}
    except Exception as e:
        return {"reply": f"Builder Agent Error: {e}"}


@app.post("/api/memory/upload-pdf")
def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...), module_id: str = "default-memory"):
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are supported.", "status": "Failed"}
    try:
        pdf_bytes = file.file.read()
        background_tasks.add_task(process_pdf_background, pdf_bytes, file.filename, module_id)
        return {"status": "Processing PDF in background", "filename": file.filename, "module_id": module_id}
    except Exception as e:
        return {"error": f"Failed to upload PDF: {str(e)}", "status": "Failed"}


@app.post("/api/memory/upload-image")
def upload_image(background_tasks: BackgroundTasks, file: UploadFile = File(...), module_id: str = "default-memory"):
    ext = file.filename.lower().split('.')[-1]
    if ext not in ["png", "jpg", "jpeg", "webp", "bmp"]:
        return {"error": "Only standard image formats (PNG, JPG, JPEG, WEBP, BMP) are supported.", "status": "Failed"}
    try:
        image_bytes = file.file.read()
        
        # Archive image file inside local static folder
        unique_filename = f"{uuid.uuid4()}.{ext}"
        archive_path = os.path.join("ern_state/uploads", unique_filename)
        with open(archive_path, "wb") as f:
            f.write(image_bytes)
            
        image_url = f"/static/uploads/{unique_filename}"
        
        background_tasks.add_task(process_image_background, image_bytes, file.filename, image_url, module_id)
        return {"status": "Processing image in background", "filename": file.filename, "image_url": image_url, "module_id": module_id}
    except Exception as e:
        return {"error": f"Failed to upload image: {str(e)}", "status": "Failed"}


# --- Expert Module Endpoints ---

@app.get("/api/modules")
def list_modules_endpoint():
    return {
        "modules": manager.list_modules(),
        "default_pipeline": manager.registry.get("default_pipeline", ["default-memory"])
    }

@app.post("/api/modules")
def create_module_endpoint(req: ModuleConfig):
    config = req.dict()
    try:
        mod = manager.create_module(config)
        return {"status": "Success", "module": config}
    except Exception as e:
        return {"error": f"Failed to create module: {str(e)}", "status": "Failed"}

@app.delete("/api/modules/{module_id}")
def delete_module_endpoint(module_id: str):
    success = manager.delete_module(module_id)
    if success:
        return {"status": f"Module {module_id} successfully deleted."}
    else:
        return {"status": "Module not found or undeletable.", "error": True}

@app.patch("/api/modules/{module_id}")
def patch_module_endpoint(module_id: str, req: ModulePatchRequest):
    if module_id not in manager.registry["modules"]:
        return {"error": f"Module {module_id} not found in registry.", "status": "Failed"}
        
    config = manager.registry["modules"][module_id]
    
    if req.frozen is not None:
        config["frozen"] = req.frozen
        if module_id in manager.active_modules:
            manager.active_modules[module_id].frozen = req.frozen
            
    if req.ltp_decay_rate is not None:
        config["ltp_decay_rate"] = req.ltp_decay_rate
        if module_id in manager.active_modules:
            manager.active_modules[module_id].ltp_decay_rate = req.ltp_decay_rate
            
    if req.stp_decay_rate is not None:
        config["stp_decay_rate"] = req.stp_decay_rate
        if module_id in manager.active_modules:
            manager.active_modules[module_id].stp_decay_rate = req.stp_decay_rate
            
    manager.registry["modules"][module_id] = config
    manager._save_registry(manager.registry)
    return {"status": "Success", "module": config}

@app.post("/api/modules/pipeline")
def update_pipeline_endpoint(pipeline: List[str]):
    valid_ids = [m["config"]["module_id"] for m in manager.list_modules()]
    clean_pipeline = [pid for pid in pipeline if pid in valid_ids]
    if not clean_pipeline:
        return {"error": "Pipeline cannot be empty or contain only invalid IDs.", "status": "Failed"}
    manager.registry["default_pipeline"] = clean_pipeline
    manager._save_registry(manager.registry)
    return {"status": "Success", "default_pipeline": clean_pipeline}


@app.get("/api/memories")
def get_all_memories(q: Optional[str] = None, module_id: str = "default-memory"):
    # When auto-route is selected, only aggregate modules in the active pipeline
    if module_id == "auto-route":
        module_ids = manager.registry.get("default_pipeline", ["default-memory"])
        if not module_ids:
            module_ids = ["default-memory"]
    else:
        module_ids = [module_id]

    results = []
    for mid in module_ids:
        mod = manager.get_module(mid)
        if not mod:
            continue
        for mem_id, data in mod.vault.items():
            idx = mod.labels.index(mem_id) if mem_id in mod.labels else -1
            energy = mod.energies[idx].item() if idx >= 0 else 0.0
            stp_energy = mod.short_term_energies[idx].item() if idx >= 0 else 0.0

            mtype = data.get("memory_type")
            if not mtype:
                _, mtype = _classify_message(data["text"])

            if q:
                q_lower = q.lower()
                if q_lower not in data["text"].lower() and q_lower not in data["tags"].lower():
                    continue

            results.append({
                "memory_id": mem_id,
                "text": data["text"],
                "tags": data["tags"],
                "memory_type": mtype,
                "timestamp": data.get("timestamp", 0.0),
                "energy": round(energy, 3),
                "stp_energy": round(stp_energy, 3),
                "image_url": data.get("image_url"),
                "module_id": mid,
                "module_name": mod.name,
            })

    results.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"memories": results}


@app.delete("/api/memory/{memory_id}")
def delete_memory_endpoint(memory_id: str, module_id: str = "default-memory"):
    mod = manager.get_module(module_id)
    if not mod:
        return {"error": f"Module {module_id} not found.", "error_bool": True}
    success = mod.delete_memory(memory_id)
    if success:
        return {"status": f"Memory {memory_id} successfully deleted from module '{mod.name}'."}
    else:
        return {"status": "Memory not found.", "error": True}


@app.post("/api/system/sleep")
def manual_sleep(module_id: Optional[str] = None):
    if module_id:
        mod = manager.get_module(module_id)
        if not mod:
            return {"error": f"Module {module_id} not found."}
        mod.query_count = 0
        pruned = mod.sleep_cycle()
        return {"status": f"REM Complete for '{mod.name}'. Scrubbed {pruned} nodes."}
    else:
        results = manager.sleep_all()
        summary_str = ", ".join([f"'{m_id}': scrubbed {count} nodes" for m_id, count in results.items()])
        return {"status": f"REM Complete across active pipeline modules. Details: {summary_str or 'None'}"}


@app.get("/api/deltas")
def get_delta_tail(n: int = 20, module_id: str = "default-memory"):
    mod = manager.get_module(module_id)
    if not mod:
        return {"error": f"Module {module_id} not found.", "deltas": []}
    tail = mod.deltas.tail(n)
    return {
        "stats" : mod.deltas.stats(),
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
def rollback_deltas(n: int = 1, module_id: str = "default-memory"):
    mod = manager.get_module(module_id)
    if not mod:
        return {"error": f"Module {module_id} not found."}
    undone = mod.deltas.rollback(mod, n)
    return {
        "status"       : f"Rolled back {undone} delta(s).",
        "engine_size"  : mod.memory_bank.size(0),
        "stack_depth"  : len(mod.deltas.stack),
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
.memory-card { border-left: 2px solid var(--g3); padding: 8px 100px 8px 10px; margin-bottom: 8px; font-size: 0.76rem; color: var(--text); background: var(--bg3); position: relative; animation: card-in 0.2s ease; transition: border-color 0.2s; }
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

/* MOE FULL-PAGE DASHBOARD STYLING */
.main-tab {
  font-family: var(--font-hud);
  font-size: 0.72rem;
  font-weight: bold;
  color: var(--text-dim);
  height: 100%;
  display: flex;
  align-items: center;
  padding: 0 16px;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: all 0.2s ease;
  letter-spacing: 0.08em;
}
.main-tab:hover {
  color: var(--text);
  background: rgba(0,255,136,0.03);
}
.main-tab.active {
  color: var(--g0);
  border-bottom-color: var(--g0);
  background: rgba(0,255,136,0.04);
  text-shadow: 0 0 8px rgba(0,255,136,0.4);
}

#moe-dashboard {
  grid-area: chat;
  display: none;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
  background: var(--bg2);
  border-right: 2px solid var(--border);
}
.moe-dash-header {
  background: var(--bg3);
  border-bottom: 1px solid var(--border);
  padding: 10px 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 48px;
}
.moe-dash-title {
  font-family: var(--font-hud);
  font-size: 0.85rem;
  font-weight: bold;
  color: var(--g0);
  letter-spacing: 0.1em;
}
.moe-dash-body {
  display: grid;
  grid-template-columns: 1.25fr 0.75fr;
  flex: 1;
  overflow: hidden;
}
@media (max-width: 1000px) {
  .moe-dash-body {
    grid-template-columns: 1fr;
    grid-template-rows: 1.2fr 0.8fr;
  }
}
.moe-dash-left {
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  padding: 16px;
  gap: 16px;
}
.moe-dash-right {
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: var(--bg3);
  padding: 16px;
  gap: 12px;
}
.moe-pipeline-widget {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px;
}
.moe-grid-registry {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 12px;
}
.architect-chat-standalone {
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg);
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.moe-container {
  display: flex;
  flex-direction: column;
  gap: 12px;
  flex: 1;
  overflow-y: auto;
  padding-right: 4px;
}
.moe-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px;
  transition: all 0.2s ease;
}
.moe-card:hover {
  border-color: var(--g0);
  box-shadow: 0 0 10px rgba(0,255,136,0.1);
}
.moe-card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
.moe-card-title {
  font-family: var(--font-mono);
  font-weight: bold;
  color: var(--g0);
  font-size: 0.85rem;
}
.moe-badge {
  font-size: 0.6rem;
  padding: 2px 6px;
  border-radius: 4px;
  font-family: var(--font-mono);
  text-transform: uppercase;
}
.moe-badge.frozen {
  background: rgba(255,153,0,0.15);
  border: 1px solid rgba(255,153,0,0.5);
  color: #ff9900;
}
.moe-badge.mutable {
  background: rgba(0,204,255,0.15);
  border: 1px solid rgba(0,204,255,0.5);
  color: #00ccff;
}
.moe-card-desc {
  font-size: 0.75rem;
  color: var(--g2);
  line-height: 1.3;
  margin-bottom: 10px;
}
.moe-card-metrics {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px;
  font-size: 0.65rem;
  font-family: var(--font-mono);
  background: var(--bg3);
  padding: 6px;
  border-radius: 4px;
}
.moe-metric-val {
  color: var(--g0);
}
.moe-card-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 10px;
}
.moe-pipeline-strip {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 14px;
  margin-bottom: 4px;
}
.moe-pipeline-header {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: var(--g2);
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 6px;
  letter-spacing: 0.05em;
}
.moe-pipeline-flow {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  padding: 6px 0;
}
.moe-flow-badge {
  background: var(--bg3);
  border: 1px solid var(--border);
  color: var(--g1);
  font-family: var(--font-mono);
  font-size: 0.65rem;
  padding: 4px 8px;
  border-radius: 4px;
  display: flex;
  align-items: center;
  gap: 4px;
}
.moe-flow-badge.active-exec {
  border-color: var(--g0);
  background: rgba(0,255,136,0.05);
}
.moe-flow-arrow {
  color: var(--g3);
  font-size: 0.8rem;
}
.moe-routing-pill {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.7rem;
  color: var(--g0);
}
/* Switch styling */
.switch {
  position: relative;
  display: inline-block;
  width: 28px;
  height: 16px;
}
.switch input {
  opacity: 0;
  width: 0;
  height: 0;
}
.slider {
  position: absolute;
  cursor: pointer;
  top: 0; left: 0; right: 0; bottom: 0;
  background-color: var(--bg3);
  border: 1px solid var(--border);
  transition: .2s;
  border-radius: 16px;
}
.slider:before {
  position: absolute;
  content: "";
  height: 10px; width: 10px;
  left: 2px; bottom: 2px;
  background-color: var(--g2);
  transition: .2s;
  border-radius: 50%;
}
input:checked + .slider {
  background-color: rgba(0,255,136,0.1);
  border-color: var(--g0);
}
input:checked + .slider:before {
  transform: translateX(12px);
  background-color: var(--g0);
}

/* Module Builder Chat CSS */
.moe-builder-chat {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 8px;
  height: 200px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  margin-top: auto;
}
.moe-builder-header {
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  padding: 8px 12px;
  font-family: var(--font-mono);
  font-size: 0.65rem;
  font-weight: bold;
  color: var(--g1);
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.moe-builder-messages {
  flex: 1;
  padding: 10px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.moe-builder-msg {
  max-width: 85%;
  padding: 6px 10px;
  border-radius: 6px;
  font-size: 0.72rem;
  line-height: 1.35;
}
.moe-builder-msg.user {
  background: rgba(0,204,255,0.1);
  border: 1px solid rgba(0,204,255,0.3);
  color: #cceeff;
  align-self: flex-end;
}
.moe-builder-msg.bot {
  background: var(--bg2);
  border: 1px solid var(--border);
  color: var(--g2);
  align-self: flex-start;
}
.moe-deploy-card {
  margin-top: 6px;
  background: rgba(0,255,136,0.05);
  border: 1px dashed var(--g0);
  border-radius: 6px;
  padding: 8px;
  font-size: 0.68rem;
}
.moe-builder-input-area {
  display: flex;
  border-top: 1px solid var(--border);
  background: var(--bg2);
}
.moe-builder-input {
  flex: 1;
  background: transparent;
  border: none;
  color: var(--g0);
  font-family: var(--font-sans);
  font-size: 0.75rem;
  padding: 8px 12px;
  outline: none;
}
.moe-builder-btn {
  background: transparent;
  border: none;
  color: var(--g0);
  cursor: pointer;
  padding: 8px 12px;
  font-family: var(--font-mono);
  font-size: 0.75rem;
  transition: color 0.2s;
}
.moe-builder-btn:hover {
  color: #fff;
}
}
.btn-delete-module:hover {
  opacity: 1;
}
.modal-overlay {
  display: none;
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(0, 0, 0, 0.75);
  backdrop-filter: blur(8px);
  z-index: 10000;
  justify-content: center;
  align-items: center;
}
.btn-create-expert-from-msg {
  transition: all 0.2s ease;
}
</style>
</head>
<body>

<!-- Beautiful Glassmorphic Expert Creator Modal -->
<div id="expertCreatorModal" class="modal-overlay">
  <div class="modal-content" style="background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; width: 90%; max-width: 500px; padding: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.5); font-family: var(--font-hud);">
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 10px; margin-bottom: 15px;">
      <span style="font-size: 0.95rem; font-weight: bold; color: var(--g0); letter-spacing: 0.05em;">📦 CREATE EXPERT FROM RESPONSE</span>
      <span onclick="closeExpertCreatorModal()" style="color: var(--text-dim); cursor: pointer; font-size: 1.2rem; font-weight: bold;" onmouseover="this.style.color='var(--red)'" onmouseout="this.style.color='var(--text-dim)'">&times;</span>
    </div>
    
    <div style="display: flex; flex-direction: column; gap: 12px;">
      <div>
        <label style="font-size: 0.65rem; color: var(--g1); font-family: var(--font-mono); display: block; margin-bottom: 4px;">EXPERT NAME:</label>
        <input type="text" id="modalExpertName" value="Custom Advisor" style="width: 100%; background: var(--bg3); border: 1px solid var(--border); color: var(--text); padding: 8px; font-size: 0.8rem; border-radius: 4px; outline: none; font-family: var(--font-mono);" oninput="autoGenModalId(this.value)">
      </div>
      <div>
        <label style="font-size: 0.65rem; color: var(--g1); font-family: var(--font-mono); display: block; margin-bottom: 4px;">EXPERT ID (SLUG):</label>
        <input type="text" id="modalExpertId" value="custom-advisor" style="width: 100%; background: var(--bg3); border: 1px solid var(--border); color: var(--text); padding: 8px; font-size: 0.8rem; border-radius: 4px; outline: none; font-family: var(--font-mono);">
      </div>
      <div>
        <label style="font-size: 0.65rem; color: var(--g1); font-family: var(--font-mono); display: block; margin-bottom: 4px;">DESCRIPTION:</label>
        <textarea id="modalExpertDesc" style="width: 100%; background: var(--bg3); border: 1px solid var(--border); color: var(--text); padding: 8px; font-size: 0.8rem; border-radius: 4px; height: 50px; resize: none; outline: none; font-family: var(--font-mono);">An expert memory module generated dynamically from a curated response.</textarea>
      </div>
      <div>
        <label style="font-size: 0.65rem; color: var(--g1); font-family: var(--font-mono); display: block; margin-bottom: 4px;">SYSTEM DIRECTIVE / EXPERTISE RULES:</label>
        <textarea id="modalExpertDirective" style="width: 100%; background: var(--bg3); border: 1px solid var(--border); color: var(--text); padding: 8px; font-size: 0.8rem; border-radius: 4px; height: 110px; resize: vertical; outline: none; font-family: var(--font-mono);"></textarea>
      </div>
      
      <div style="display: flex; gap: 10px;">
        <div style="flex: 1;">
          <label style="font-size: 0.65rem; color: var(--g1); font-family: var(--font-mono); display: block; margin-bottom: 4px;">LTP DECAY:</label>
          <input type="number" id="modalExpertLtp" value="0.95" step="0.01" min="0.5" max="1" style="width: 100%; background: var(--bg3); border: 1px solid var(--border); color: var(--text); padding: 6px; font-size: 0.75rem; border-radius: 4px; outline: none; font-family: var(--font-mono);">
        </div>
        <div style="flex: 1;">
          <label style="font-size: 0.65rem; color: var(--g1); font-family: var(--font-mono); display: block; margin-bottom: 4px;">STP DECAY:</label>
          <input type="number" id="modalExpertStp" value="0.80" step="0.01" min="0.5" max="1" style="width: 100%; background: var(--bg3); border: 1px solid var(--border); color: var(--text); padding: 6px; font-size: 0.75rem; border-radius: 4px; outline: none; font-family: var(--font-mono);">
        </div>
        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; padding-left: 10px;">
          <label style="font-size: 0.65rem; color: var(--g1); font-family: var(--font-mono); display: block; margin-bottom: 4px;">FROZEN:</label>
          <input type="checkbox" id="modalExpertFrozen" style="width: 18px; height: 18px; cursor: pointer;">
        </div>
      </div>
    </div>
    
    <div style="display: flex; justify-content: flex-end; gap: 10px; margin-top: 20px; border-top: 1px solid var(--border); padding-top: 15px;">
      <button onclick="closeExpertCreatorModal()" style="background: transparent; border: 1px solid var(--border); color: var(--text-dim); padding: 6px 12px; font-size: 0.75rem; border-radius: 4px; cursor: pointer; font-family: var(--font-mono);" onmouseover="this.style.color='#fff'" onmouseout="this.style.color='var(--text-dim)'">CANCEL</button>
      <button onclick="submitModalCreateExpert()" style="background: var(--g0); border: none; color: #000; padding: 6px 12px; font-size: 0.75rem; border-radius: 4px; cursor: pointer; font-weight: bold; font-family: var(--font-mono);" onmouseover="this.style.boxShadow='0 0 10px var(--g0)'" onmouseout="this.style.boxShadow='none'">⚡ CREATE & DEPLOY</button>
    </div>
  </div>
</div>

<div id="topbar">
  <div class="brand">ERN <span>//</span> DGX</div>
  <div class="main-tabs" style="display:flex; height:100%; margin-left:24px; border-left: 1px solid var(--border); padding-left: 8px;">
    <div id="main-tab-chat" class="main-tab active" onclick="switchMainView('chat')">💬 CHAT CONSOLE</div>
    <div id="main-tab-moe" class="main-tab" onclick="switchMainView('moe')">🧠 MOE ARCHITECT</div>
  </div>
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
    <button id="uploadBtn" onclick="document.getElementById('pdfInput').click()" style="background:transparent; border:1px solid var(--border); color:var(--text-dim); padding:10px 12px; font-size:1.1rem; cursor:pointer; transition: all 0.15s; outline:none; font-family:var(--font-mono);" title="Upload PDF/Image for Epigenetic Extraction">📎</button>
    <input type="file" id="pdfInput" accept=".pdf, image/*" style="display:none" onchange="handleAttachmentUpload(this)">
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
    <div class="ctrl-group">
      <span class="ctrl-label">TEMPORAL DISCOVERY</span>
      <button id="temporalToggleBtn" onclick="toggleTemporalDiscovery()" style="background:transparent; border:1px solid var(--g0); color:var(--g0); padding:3px 8px; font-family:var(--font-hud); font-size:0.6rem; cursor:pointer; letter-spacing:0.08em; transition: all 0.15s; outline:none; text-shadow: 0 0 6px var(--g0);">ON</button>
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

<!-- GORGEOUS FULL-PAGE MOE CONTROLLER -->
<div id="moe-dashboard">
  <div class="moe-dash-header">
    <div class="moe-dash-title">🤖 EPIGENETIC MIXTURE-OF-EXPERTS (MOE) CONTROL CENTER</div>
    <div style="font-family:var(--font-mono); font-size:0.7rem; color:var(--text-dim);">VRAM ACCELERATED DIVISION</div>
  </div>
  <div class="moe-dash-body">
    <!-- Left column: Registry list and static pipeline settings -->
    <div class="moe-dash-left">
      <!-- Pipeline Strip -->
      <div class="moe-pipeline-widget">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
          <div style="font-family:var(--font-mono); font-size:0.8rem; font-weight:bold; color:var(--g0);">ACTIVE PIPELINE CONFIGURATION</div>
          <div class="moe-routing-pill">
            <span style="font-family:var(--font-mono); font-size:0.75rem; letter-spacing:0.05em; color:var(--g1); margin-right:4px;">AUTO-ROUTE:</span>
            <label class="switch">
              <input type="checkbox" id="moeAutoRoute" onchange="toggleAutoRoute(this.checked)">
              <span class="slider"></span>
            </label>
          </div>
        </div>
        <div id="moePipelineFlow" class="moe-pipeline-flow">
          <div class="no-mem">// INACTIVE PIPELINE</div>
        </div>
      </div>
      
      <!-- Registry Grid Header -->
      <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px;">
        <div style="font-family:var(--font-hud); font-size:0.85rem; font-weight:bold; color:var(--text);">ACTIVE EXPERT REGISTRY</div>
        <button onclick="loadModulesUI()" style="background:transparent; border:1px solid var(--border); color:var(--g0); font-family:var(--font-mono); font-size:0.7rem; padding:4px 8px; border-radius:4px; cursor:pointer;">🔄 REFRESH REGISTRY</button>
      </div>
      
      <!-- Grid Container for Cards -->
      <div class="moe-grid-registry" id="moeModulesList">
        <div class="no-mem">Loading Modules...</div>
      </div>
    </div>

    <!-- Right column: Massive interactive Builder chat console -->
    <div class="moe-dash-right">
      <div style="font-family:var(--font-hud); font-size:0.8rem; font-weight:bold; color:var(--g0); margin-bottom:10px; letter-spacing:0.05em;">💬 CONSTRUCT NEW EXPERTS WITH AI</div>
      
      <div class="architect-chat-standalone">
        <div class="moe-builder-header">
          <span>🤖 AI MODULE ARCHITECT</span>
          <button onclick="clearBuilderChat()" style="background:none; border:none; color:var(--text-dim); font-size:0.6rem; cursor:pointer; font-family:var(--font-mono);">RESET</button>
        </div>
        <div class="moe-builder-messages" id="builderMessages">
          <div class="moe-builder-msg bot">Greetings. I am the ERN Expert Architect. Converse with me to design and customize a new expert memory module, or configure parameter thresholds. Let me know what you want to build.</div>
        </div>
        <div class="moe-builder-input-area">
          <input type="text" id="builderInput" class="moe-builder-input" placeholder="Say 'make a python coding style module'..." onkeydown="if(event.key==='Enter') sendBuilderMessage()">
          <button onclick="sendBuilderMessage()" class="moe-builder-btn">SEND</button>
        </div>
      </div>
    </div>
  </div>
</div>

<div id="sidebar">
  <div style="padding: 10px 10px 0 10px;">
    <div style="font-family:var(--font-mono); font-size:0.6rem; color:var(--g2); margin-bottom:4px; letter-spacing:0.05em;">ACTIVE SIDEPANEL EXPERT:</div>
    <select id="moduleSelectGlobal" onchange="onGlobalModuleChange()" style="width:100%; background:var(--bg3); border:1px solid var(--border); color:var(--g0); font-family:var(--font-mono); font-size:0.75rem; padding:6px; border-radius:4px; outline:none; cursor:pointer;">
      <option value="auto-route" selected>🧠 Dynamic Router (Auto-Route)</option>
    </select>
  </div>

  <div class="panel-tabs" style="margin-top: 8px;">
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
let activeMainView = 'chat';
let useAgenticSearch = true;
let useTemporalDiscovery = true;
let vaultMemories   = [];
const pendingModules = {};
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

function toggleTemporalDiscovery() {
  useTemporalDiscovery = !useTemporalDiscovery;
  const btn = document.getElementById('temporalToggleBtn');
  if (useTemporalDiscovery) {
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
  pushToast(`Temporal Discovery toggled ${useTemporalDiscovery ? 'ON' : 'OFF'}`);
}

function toggleCognitionPanel(el) {
  el.parentElement.classList.toggle('open');
}

async function loadVault() {
  try {
    const modId = document.getElementById('moduleSelectGlobal').value;
    const res = await fetch(`/api/memories?module_id=${modId}`);
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
  const isGlobal = document.getElementById('moduleSelectGlobal').value === 'auto-route';
  mems.forEach(m => {
    const card = document.createElement('div');
    card.className = 'memory-card';
    card.id = `vc-${m.memory_id}`;
    
    let imgHtml = '';
    if (m.image_url) {
      imgHtml = `<div class="mc-image" style="margin-top: 6px; border: 1px solid var(--border); overflow: hidden; max-height: 90px; max-width: 160px; cursor: pointer; border-radius: 2px;" onclick="window.open('${m.image_url}', '_blank')" title="Click to view full image">` +
                `<img src="${m.image_url}" style="width: 100%; height: 100%; object-fit: cover; filter: brightness(0.92) contrast(1.05);" />` +
                `</div>`;
    }
    
    const formattedDate = m.timestamp ? new Date(m.timestamp * 1000).toLocaleString() : 'Date Unknown';
    const moduleBadge = isGlobal && m.module_name
      ? `<span style="display: inline-block; white-space: nowrap; font-size:0.55rem; background: rgba(0,255,136,0.1); border: 1px solid var(--g3); color: var(--g1); padding: 1px 5px; border-radius: 3px; margin-left: 6px; font-family: var(--font-mono);">${m.module_name}</span>`
      : '';
    
    const mType = m.memory_type || 'fact';
    let typeBadgeColor = '#00bcff';
    if (mType === 'question') typeBadgeColor = '#ffb300';
    if (mType === 'instruction') typeBadgeColor = '#00ff88';
    const typeBadge = `<span style="display: inline-block; white-space: nowrap; font-size:0.55rem; background: rgba(0,0,0,0.3); border: 1px solid ${typeBadgeColor}; color: ${typeBadgeColor}; padding: 1px 5px; border-radius: 3px; margin-left: 6px; font-family: var(--font-mono); text-transform: uppercase;">${mType}</span>`;
    
    card.innerHTML = `<div class="mc-tags">${m.tags}${moduleBadge}${typeBadge}</div>` +
                     `<div style="word-break: break-word;">${m.text}</div>` +
                     imgHtml +
                     `<div class="mc-energy">${m.energy.toFixed(3)} LTP | ${m.stp_energy ? m.stp_energy.toFixed(3) : '0.000'} STP</div>` +
                     `<div class="mc-date" style="font-size: 0.58rem; color: var(--text-dim); margin-top: 4px; font-family: var(--font-hud);">${formattedDate}</div>` +
                     `<button class="mc-forget" onclick="forgetMemory('${m.memory_id}', '${m.module_id || ''}')">FORGET</button>`;
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

async function forgetMemory(memory_id, moduleId = null) {
  if (!confirm('Are you sure you want to forget/revert this memory synapse permanently?')) return;
  // Use the memory's own module_id if provided, otherwise fall back to global selector
  const modId = moduleId || document.getElementById('moduleSelectGlobal').value;
  const effectiveModId = (modId === 'auto-route') ? 'default-memory' : modId;
  try {
    const res = await fetch(`/api/memory/${memory_id}?module_id=${effectiveModId}`, { method: 'DELETE' });
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
    const modId = document.getElementById('moduleSelectGlobal').value;
    const res  = await fetch(`/api/deltas?n=40&module_id=${modId}`);
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
  const modId = document.getElementById('moduleSelectGlobal').value;
  try {
    const res  = await fetch(`/api/deltas/rollback?n=${n}&module_id=${modId}`, { method:'POST' });
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
  const modId = document.getElementById('moduleSelectGlobal').value;
  document.getElementById('sleepBtn').textContent = '⬛ REM IN PROGRESS...';
  try {
    const res  = await fetch(`/api/system/sleep?module_id=${modId}`, { method:'POST' });
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
    const modId = document.getElementById('moduleSelectGlobal').value;
    const res  = await fetch(`/api/deltas?n=1&module_id=${modId}`);
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

function switchMainView(view) {
  activeMainView = view;
  const chatArea = document.getElementById('chat-area');
  const moeDash = document.getElementById('moe-dashboard');
  const chatTab = document.getElementById('main-tab-chat');
  const moeTab = document.getElementById('main-tab-moe');
  
  if (view === 'moe') {
    chatArea.style.display = 'none';
    moeDash.style.display = 'flex';
    chatTab.classList.remove('active');
    moeTab.classList.add('active');
    loadModulesUI();
  } else {
    chatArea.style.display = 'flex';
    moeDash.style.display = 'none';
    chatTab.classList.add('active');
    moeTab.classList.remove('active');
  }
}

function pushToast(msg, isError = false) {
  const t = document.createElement('div');
  t.textContent = msg;
  const borderColor = isError ? 'var(--red)' : 'var(--g2)';
  const color = isError ? 'var(--red)' : 'var(--g0)';
  const shadow = isError ? 'rgba(255,77,77,0.2)' : 'rgba(0,255,136,0.2)';
  Object.assign(t.style, {
    position:'fixed', bottom:'60px', right:'16px', zIndex:'9998',
    background:'var(--bg3)', border:`1px solid ${borderColor}`,
    color: color, fontFamily:'var(--font-mono)', fontSize:'0.75rem',
    padding:'8px 14px', animation:'msg-in 0.2s ease',
    boxShadow:`0 0 14px ${shadow}`,
    maxWidth:'300px',
  });
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3500);
}

async function handleAttachmentUpload(input) {
  if (!input.files || !input.files[0]) return;
  const file = input.files[0];
  
  const isPDF = file.name.toLowerCase().endsWith('.pdf');
  const isImage = /\.(png|jpe?g|webp|bmp)$/i.test(file.name) || file.type.startsWith('image/');
  
  if (!isPDF && !isImage) {
    pushToast('Unsupported file format. Please upload a PDF or an Image.', true);
    return;
  }

  const uploadBtn = document.getElementById('uploadBtn');
  const origText = uploadBtn.textContent;
  
  uploadBtn.disabled = true;
  uploadBtn.textContent = '⏳';
  uploadBtn.style.color = 'var(--amber)';
  uploadBtn.style.borderColor = 'var(--amber)';

  const modId = document.getElementById('moduleSelectGlobal').value;
  const endpoint = (isPDF ? '/api/memory/upload-pdf' : '/api/memory/upload-image') + `?module_id=${modId}`;
  const label = isPDF ? 'PDF' : 'Image';
  pushToast(`Ingesting ${label}: ${file.name}... Sending to VRAM targeting ${modId}.`, false);

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      body: formData
    });
    if (!res.ok) throw new Error('Network error or file too large.');
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    pushToast(`${label} transmitted successfully! Subconscious fact extraction commenced in background.`, false);
    setTimeout(loadVault, 5000);
  } catch (e) {
    pushToast(`${label} Extraction Trigger Failed: ${e.message}`, true);
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.textContent = origText;
    uploadBtn.style.color = 'var(--text-dim)';
    uploadBtn.style.borderColor = 'var(--border)';
    input.value = '';
  }
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

    const selectedExpert = document.getElementById('moduleSelectGlobal') ? document.getElementById('moduleSelectGlobal').value : 'auto-route';
    const isAutoRoute = (selectedExpert === 'auto-route');
    const customPipeline = isAutoRoute ? getActivePipeline() : [selectedExpert];

    const res  = await fetch('/api/chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        message            : text,
        model              : document.getElementById('modelSelect').value,
        history            : chatHistory.slice(-6),
        focus_threshold    : parseFloat(document.getElementById('focusSlider').value),
        agentic_search     : useAgenticSearch,
        temporal_discovery : useTemporalDiscovery,
        auto_route         : isAutoRoute,
        pipeline           : customPipeline,
        active_expert      : isAutoRoute ? null : selectedExpert
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
    
    // Clean escape raw HTML from LLM output for safety
    let formattedReply = data.reply
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
    
    // Parse Markdown images: ![Alt Text](URL) -> visual inline image
    formattedReply = formattedReply.replace(/!\[(.*?)\]\((.*?)\)/g, 
      '<div class="chat-inline-image" style="margin: 8px 0; border: 1px solid var(--border); overflow: hidden; max-width: 320px; border-radius: 4px; cursor: pointer;" onclick="window.open(\'$2\', \'_blank\')" title="Click to view full image">' +
      '<img src="$2" alt="$1" style="width:100%; height:auto; display:block; filter: brightness(0.92) contrast(1.05);" />' +
      '</div>'
    );
    
    // Parse Markdown links: [Link Text](URL) -> styled anchor tag
    formattedReply = formattedReply.replace(/\[(.*?)\]\((.*?)\)/g, 
      '<a href="$2" target="_blank" style="color: var(--g0); text-decoration: underline; font-family: var(--font-mono); font-size: 0.72rem;">$1</a>'
    );
    
    // Replace newlines with break tags
    formattedReply = formattedReply.replace(/\n/g, '<br>');
    
    replyText.innerHTML = formattedReply;
    bDiv.appendChild(replyText);

    // Append action button to build expert from this response
    const actionDiv = document.createElement('div');
    actionDiv.className = 'msg-actions';
    actionDiv.style.marginTop = '8px';
    actionDiv.style.display = 'flex';
    actionDiv.style.justifyContent = 'flex-end';
    actionDiv.innerHTML = `
      <button class="btn-create-expert-from-msg" style="background: rgba(0,255,127,0.06); border: 1px solid rgba(0,255,127,0.4); color: var(--g0); font-family: var(--font-mono); font-size: 0.65rem; padding: 4px 8px; border-radius: 4px; cursor: pointer; display: flex; align-items: center; gap: 4px; transition: all 0.2s; font-weight: bold; letter-spacing: 0.03em;" onmouseover="this.style.background='var(--g0)'; this.style.color='#000'; this.style.borderColor='var(--g0)';" onmouseout="this.style.background='rgba(0,255,127,0.06)'; this.style.color='var(--g0)'; this.style.borderColor='rgba(0,255,127,0.4)';" onclick="openCreateExpertModalFromMsg(this)">
        ➕ CREATE EXPERT FROM RESPONSE
      </button>
    `;
    bDiv.appendChild(actionDiv);

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
        
        let imgHtml = '';
        if (m.image_url) {
          imgHtml = `<div class="mc-image" style="margin-top: 6px; border: 1px solid var(--border); overflow: hidden; max-height: 90px; max-width: 160px; cursor: pointer; border-radius: 2px;" onclick="window.open('${m.image_url}', '_blank')" title="Click to view full image">` +
                    `<img src="${m.image_url}" style="width: 100%; height: 100%; object-fit: cover; filter: brightness(0.92) contrast(1.05);" />` +
                    `</div>`;
        }
        
        const formattedDate = m.timestamp ? new Date(m.timestamp * 1000).toLocaleString() : 'Date Unknown';
        
        const mType = m.memory_type || 'fact';
        let typeBadgeColor = '#00bcff';
        if (mType === 'question') typeBadgeColor = '#ffb300';
        if (mType === 'instruction') typeBadgeColor = '#00ff88';
        const typeBadge = `<span style="display: inline-block; white-space: nowrap; font-size:0.55rem; background: rgba(0,0,0,0.3); border: 1px solid ${typeBadgeColor}; color: ${typeBadgeColor}; padding: 1px 5px; border-radius: 3px; margin-left: 6px; font-family: var(--font-mono); text-transform: uppercase;">${mType}</span>`;
        
        card.innerHTML = `<div class="mc-tags">[Expert: ${m.module_name || 'default-memory'}] | ${m.tags}${typeBadge}</div>` +
                         `<div style="word-break: break-word;">${m.text}</div>` +
                         imgHtml +
                         `<div class="mc-resonance">${m.resonance.toFixed(3)} R</div>` +
                         `<div class="mc-energy">${m.energy ? m.energy.toFixed(3) : '0.000'} LTP | ${m.stp_energy ? m.stp_energy.toFixed(3) : '0.000'} STP</div>` +
                         `<div class="mc-date" style="font-size: 0.58rem; color: var(--text-dim); margin-top: 4px; font-family: var(--font-hud);">${formattedDate}</div>` +
                         `<button class="mc-forget" onclick="forgetMemory('${m.memory_id}', '${m.module_id || 'default-memory'}')">FORGET</button>`;
        mBox.appendChild(card);
      });
    } else {
      mBox.innerHTML = '<div class="no-mem">No synapses resonated above threshold.</div>';
    }

    if (activeTab === 'deltas') loadDeltas();
    if (activeTab === 'vault')  loadVault();
    if (activeMainView === 'moe') loadModulesUI();
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

// ── Mixture of Experts (MOE) Logic ───────────────────────────────────────────
let moeModules = [];
let defaultPipeline = [];
let builderChatHistory = [];

async function onGlobalModuleChange() {
  const modId = document.getElementById('moduleSelectGlobal').value;
  pushToast(`Active sidepanel expert switched to: ${modId}`);
  
  const autoRouteChk = document.getElementById('moeAutoRoute');
  if (autoRouteChk) {
    if (modId === 'auto-route') {
      autoRouteChk.checked = true;
    } else {
      autoRouteChk.checked = false;
    }
    renderPipelineFlow();
  }

  if (activeTab === 'vault')  loadVault();
  if (activeTab === 'deltas') loadDeltas();
  if (activeTab === 'stats')  refreshStats();
}

async function loadModulesUI() {
  try {
    const res = await fetch('/api/modules');
    if (!res.ok) throw new Error('API Error');
    const data = await res.json();
    moeModules = data.modules || [];
    defaultPipeline = data.default_pipeline || [];
    
    // Preserve current selection; default to 'auto-route' on first load
    const globalSel = document.getElementById('moduleSelectGlobal');
    const currentVal = globalSel.value || 'auto-route';
    globalSel.innerHTML = '';
    
    // Always first option: Dynamic Router
    const routeOpt = document.createElement('option');
    routeOpt.value = 'auto-route';
    routeOpt.textContent = '🧠 Dynamic Router (Auto-Route)';
    routeOpt.selected = (currentVal === 'auto-route');
    globalSel.appendChild(routeOpt);

    moeModules.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.config.module_id;
      opt.textContent = `${m.config.name} (${m.config.module_id})`;
      if (m.config.module_id === currentVal) opt.selected = true;
      globalSel.appendChild(opt);
    });
    
    // Render pipeline strip badge flow
    renderPipelineFlow();

    // Render Modules List
    renderModulesList();

  } catch(e) {
    document.getElementById('moeModulesList').innerHTML = `<div class="no-mem" style="color:var(--red);">Offline: ${e.message}</div>`;
  }
}

function renderPipelineFlow() {
  const flow = document.getElementById('moePipelineFlow');
  flow.innerHTML = '';
  
  if (document.getElementById('moeAutoRoute').checked) {
    flow.innerHTML = `<div class="moe-flow-badge active-exec" style="border-color:var(--g0); color:var(--g0);">🧠 Dynamic Router Deciding...</div>`;
    return;
  }
  
  if (!defaultPipeline.length) {
    flow.innerHTML = `<div class="no-mem">// PIPELINE EMPTY</div>`;
    return;
  }
  
  defaultPipeline.forEach((pid, idx) => {
    const m = moeModules.find(x => x.config.module_id === pid);
    if (!m) return;
    
    const badge = document.createElement('div');
    badge.className = 'moe-flow-badge';
    badge.id = `flow-badge-${pid}`;
    badge.textContent = m.config.name;
    
    flow.appendChild(badge);
    
    if (idx < defaultPipeline.length - 1) {
      const arrow = document.createElement('span');
      arrow.className = 'moe-flow-arrow';
      arrow.innerHTML = '⚡';
      flow.appendChild(arrow);
    }
  });
}

function getActivePipeline() {
  if (document.getElementById('moeAutoRoute').checked) {
    return [];
  }
  return defaultPipeline;
}

async function toggleAutoRoute(checked) {
  pushToast(`Auto-Route ${checked ? 'enabled (Dynamic Routing)' : 'disabled (Static Pipeline)'}`);
  const globalSel = document.getElementById('moduleSelectGlobal');
  if (globalSel) {
    if (checked) {
      globalSel.value = 'auto-route';
    } else {
      globalSel.value = 'default-memory';
    }
  }
  renderPipelineFlow();
}

function renderModulesList() {
  const container = document.getElementById('moeModulesList');
  container.innerHTML = '';
  
  moeModules.forEach(m => {
    const c = m.config;
    const card = document.createElement('div');
    card.className = 'moe-card';
    
    const isDefault = c.module_id === 'default-memory';
    const isPipeline = defaultPipeline.includes(c.module_id);
    const badgeClass = c.frozen ? 'frozen' : 'mutable';
    const badgeText = c.frozen ? 'FROZEN' : 'MUTABLE';
    
    const deleteBtn = isDefault ? '' : `<button onclick="deleteModule('${c.module_id}')" class="btn-delete-module">🗑️ DELETE</button>`;
    const pipelineCheckbox = `
      <input type="checkbox" id="chk-${c.module_id}" ${isPipeline ? 'checked' : ''} onchange="togglePipelineModule('${c.module_id}', this.checked)" style="cursor:pointer;">
    `;
    
    card.innerHTML = `
      <div class="moe-card-header">
        <div style="display:flex; align-items:center; gap:8px;">
          ${pipelineCheckbox}
          <span class="moe-card-title">${c.name}</span>
        </div>
        <span class="moe-badge ${badgeClass}" onclick="toggleFrozenState('${c.module_id}', ${!c.frozen})" style="cursor:pointer; user-select:none; transition: all 0.2s;" title="Click to toggle frozen status (frozen experts prevent synaptic decay)">${badgeText}</span>
      </div>
      <div class="moe-card-desc">${c.description}</div>
      <div class="moe-card-metrics">
        <div>SYNAPSES: <span class="moe-metric-val">${m.synapses_count}</span></div>
        <div>LTP DECAY: <span class="moe-metric-val">${c.ltp_decay_rate.toFixed(2)}</span></div>
        <div>STP DECAY: <span class="moe-metric-val">${c.stp_decay_rate.toFixed(2)}</span></div>
      </div>
      <div class="moe-card-actions">
        <span style="font-size:0.6rem; color:var(--text-dim); font-family:var(--font-mono);">${c.module_id}</span>
        ${deleteBtn}
      </div>
    `;
    container.appendChild(card);
  });
}

async function toggleFrozenState(moduleId, isFrozen) {
  try {
    pushToast(`${isFrozen ? 'Freezing' : 'Unfreezing'} expert module '${moduleId}'...`);
    const res = await fetch(`/api/modules/${moduleId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frozen: isFrozen })
    });
    if (!res.ok) throw new Error('API modification failed');
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    
    pushToast(`Successfully ${isFrozen ? 'froze' : 'unfroze'} '${moduleId}'!`);
    loadModulesUI();
  } catch(e) {
    pushToast(`Failed to toggle state: ${e.message}`, true);
    loadModulesUI();
  }
}

async function togglePipelineModule(moduleId, checked) {
  let newPipeline = [...defaultPipeline];
  if (checked) {
    if (!newPipeline.includes(moduleId)) newPipeline.push(moduleId);
  } else {
    newPipeline = newPipeline.filter(pid => pid !== moduleId);
  }
  
  try {
    const res = await fetch('/api/modules/pipeline', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(newPipeline)
    });
    if (!res.ok) throw new Error('Failed to update pipeline');
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    defaultPipeline = data.default_pipeline || [];
    pushToast(`Pipeline updated: ${defaultPipeline.join(' -> ')}`);
    renderPipelineFlow();
  } catch(e) {
    pushToast(`Error: ${e.message}`, true);
    loadModulesUI();
  }
}

async function deleteModule(moduleId) {
  if (!confirm(`Are you sure you want to delete module '${moduleId}' permanently? All synapses will be scrubbed.`)) return;
  try {
    const res = await fetch(`/api/modules/${moduleId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('API delete failed');
    pushToast(`Module '${moduleId}' deleted successfully.`);
    loadModulesUI();
  } catch(e) {
    pushToast(`Error: ${e.message}`, true);
  }
}

// AI Module Builder Chat
async function sendBuilderMessage() {
  const input = document.getElementById('builderInput');
  const text = input.value.trim();
  if (!text) return;
  
  const messagesBox = document.getElementById('builderMessages');
  
  // Append User Message
  const uDiv = document.createElement('div');
  uDiv.className = 'moe-builder-msg user';
  uDiv.textContent = text;
  messagesBox.appendChild(uDiv);
  input.value = '';
  messagesBox.scrollTop = messagesBox.scrollHeight;
  
  // Add temporary loading bot message
  const loadDiv = document.createElement('div');
  loadDiv.className = 'moe-builder-msg bot';
  loadDiv.innerHTML = 'Thinking<span class="thinking-dots"></span>';
  messagesBox.appendChild(loadDiv);
  messagesBox.scrollTop = messagesBox.scrollHeight;
  
  try {
    const res = await fetch('/api/modules/builder', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        history: builderChatHistory
      })
    });
    if (!res.ok) throw new Error('Network error');
    const data = await res.json();
    
    // Remove loading message
    loadDiv.remove();
    
    // Update builder chat history
    builderChatHistory.push({ role: 'user', content: text });
    builderChatHistory.push({ role: 'assistant', content: data.reply });
    
    // Append Bot Message
    const bDiv = document.createElement('div');
    bDiv.className = 'moe-builder-msg bot';
    
    // Format bot message (basic markdown and JSON capture)
    let reply = data.reply;
    
    // Check if reply contains a deployable JSON block
    const jsonMatch = reply.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
    let deployCardHtml = '';
    if (jsonMatch) {
      try {
        let jsonStr = jsonMatch[1].replace(/\[?MODULE_READY\]?/gi, '').trim();
        
        // Robust cleaning: extract clean boundaries
        const startIdx = jsonStr.indexOf('{');
        if (startIdx !== -1) {
          let endIdx = Math.max(jsonStr.lastIndexOf('}'), jsonStr.lastIndexOf(']'));
          if (endIdx !== -1 && endIdx > startIdx) {
            jsonStr = jsonStr.substring(startIdx, endIdx + 1);
          }
        }
        
        // Auto-repair: replace accidental closing bracket ] with curly brace } if it starts with {
        if (jsonStr.startsWith('{') && jsonStr.endsWith(']')) {
          jsonStr = jsonStr.slice(0, -1) + '}';
        }
        
        // Strip inlined comments (e.g. // float threshold)
        jsonStr = jsonStr.replace(/\/\/.*$/gm, '');
        
        // Strip trailing commas before closing braces/brackets
        jsonStr = jsonStr.replace(/,\s*([\}\]])/g, '$1');
        
        const config = JSON.parse(jsonStr);
        const configId = 'mod_' + Math.random().toString(36).substring(2, 9);
        pendingModules[configId] = config;
        deployCardHtml = `
          <div class="moe-deploy-card">
            <strong style="color:var(--g0);">📦 EXPERT READY TO DEPLOY</strong><br/>
            Name: ${config.name || 'Unnamed'}<br/>
            ID: ${config.module_id || 'unnamed-expert'}<br/>
            Directives: ${config.system_directive ? config.system_directive.slice(0, 40) + '...' : 'None'}<br/>
            <button onclick="deployBuiltModule('${configId}')" style="background:var(--g0); border:none; color:#000; font-family:var(--font-mono); font-size:0.65rem; padding:4px 8px; border-radius:3px; margin-top:6px; cursor:pointer; font-weight:bold;">⚡ DEPLOY EXPERT</button>
          </div>
        `;
      } catch(je) {
        console.warn("JSON parsing in architect reply failed:", je);
      }
    }
    
    bDiv.innerHTML = `<div style="white-space:pre-wrap;">${reply}</div>` + deployCardHtml;
    messagesBox.appendChild(bDiv);
    messagesBox.scrollTop = messagesBox.scrollHeight;
    
  } catch(e) {
    if (loadDiv) loadDiv.remove();
    const bDiv = document.createElement('div');
    bDiv.className = 'moe-builder-msg bot';
    bDiv.style.borderColor = 'var(--red)';
    bDiv.textContent = `Architect failed: ${e.message}`;
    messagesBox.appendChild(bDiv);
    messagesBox.scrollTop = messagesBox.scrollHeight;
  }
}

async function deployBuiltModule(configId) {
  try {
    const config = pendingModules[configId];
    if (!config) throw new Error('Configuration not found');
    pushToast(`Deploying expert module '${config.name}'...`);
    const res = await fetch('/api/modules', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    if (!res.ok) throw new Error('API creation failed');
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    
    pushToast(`Successfully deployed '${config.name}'!`);
    loadModulesUI();
    
    // Append system message in builder chat confirming deployment
    const messagesBox = document.getElementById('builderMessages');
    const sDiv = document.createElement('div');
    sDiv.className = 'moe-builder-msg bot';
    sDiv.style.borderLeftColor = 'var(--g0)';
    sDiv.innerHTML = `<span style="color:var(--g0);">✔️ SUCCESS:</span> Module '${config.name}' deployed, loaded in PyTorch memory, and active.`;
    messagesBox.appendChild(sDiv);
    messagesBox.scrollTop = messagesBox.scrollHeight;
    
  } catch(e) {
    pushToast(`Deployment Failed: ${e.message}`, true);
  }
}

function clearBuilderChat() {
  builderChatHistory = [];
  document.getElementById('builderMessages').innerHTML = '<div class="moe-builder-msg bot">Greetings. I am the ERN Expert Architect. Converse with me to design and customize a new expert memory module, or configure parameter thresholds.</div>';
}

function autoGenModalId(name) {
  const slug = name.toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)/g, '');
  document.getElementById('modalExpertId').value = slug;
}

function openCreateExpertModalFromMsg(btn) {
  // Extract original bot response text (exclude the actions div)
  const botMsgDiv = btn.closest('.msg.bot');
  // Clone it to manipulate without affecting live UI
  const clone = botMsgDiv.cloneNode(true);
  const actionsDiv = clone.querySelector('.msg-actions');
  if (actionsDiv) actionsDiv.remove();
  const monitorDiv = clone.querySelector('.cognition-monitor');
  if (monitorDiv) monitorDiv.remove();
  
  // Get text content, clean spacing, and decode entity references if any
  let text = clone.innerText.trim();
  
  // Smart pre-fill:
  // Pre-fill Directive with the actual text or a summary
  document.getElementById('modalExpertDirective').value = `Act according to these guidelines and rules:\n${text}`;
  
  // Set default name based on first 3 words
  const words = text.split(/\s+/).slice(0, 3).join(' ');
  const cleanName = words.replace(/[^a-zA-Z0-9\s]+/g, '').trim() || 'Custom Expert';
  document.getElementById('modalExpertName').value = cleanName + ' Expert';
  autoGenModalId(cleanName + ' Expert');
  
  // Open modal
  document.getElementById('expertCreatorModal').style.display = 'flex';
}

function closeExpertCreatorModal() {
  document.getElementById('expertCreatorModal').style.display = 'none';
}

async function submitModalCreateExpert() {
  const name = document.getElementById('modalExpertName').value.trim();
  const id = document.getElementById('modalExpertId').value.trim();
  const desc = document.getElementById('modalExpertDesc').value.trim();
  const directive = document.getElementById('modalExpertDirective').value.trim();
  const ltp = parseFloat(document.getElementById('modalExpertLtp').value);
  const stp = parseFloat(document.getElementById('modalExpertStp').value);
  const frozen = document.getElementById('modalExpertFrozen').checked;
  
  if (!name || !id) {
    pushToast("Name and ID are required fields!", true);
    return;
  }
  
  try {
    pushToast(`Creating expert module '${id}'...`);
    const res = await fetch('/api/modules', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        module_id: id,
        name: name,
        description: desc,
        frozen: frozen,
        ltp_decay_rate: ltp,
        stp_decay_rate: stp,
        sleep_threshold: 0.10,
        focus_threshold: 0.15,
        system_directive: directive
      })
    });
    
    if (!res.ok) throw new Error('API request failed');
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    
    pushToast(`Successfully deployed Expert '${name}'!`);
    closeExpertCreatorModal();
    
    // Automatically reload and switch global active expert to it
    setTimeout(() => {
      loadModulesUI().then(() => {
        const globalSel = document.getElementById('moduleSelectGlobal');
        globalSel.value = id;
        onGlobalModuleChange();
      });
    }, 500);
  } catch(e) {
    pushToast(`Failed to deploy expert: ${e.message}`, true);
  }
}

loadModels();
loadModulesUI();
refreshStats();
setInterval(() => {
  sparkData.push(Math.random() * 0.05);
  sparkData = sparkData.slice(-80);
}, 2000);

// ── Registry Watcher — auto-refresh UI when registry.json changes ─────────────
let _registryFingerprint = '';
async function _pollRegistryChanges() {
  try {
    const res = await fetch('/api/modules');
    if (!res.ok) return;
    const data = await res.json();
    // Build a lightweight fingerprint: module IDs + pipeline
    const fp = JSON.stringify({
      ids: (data.modules || []).map(m => m.config.module_id).sort(),
      pipeline: data.default_pipeline || [],
      synapseCounts: (data.modules || []).map(m => `${m.config.module_id}:${m.synapses_count}`)
    });
    if (fp !== _registryFingerprint) {
      if (_registryFingerprint !== '') {
        // Registry changed — reload relevant panels silently
        moeModules = data.modules || [];
        defaultPipeline = data.default_pipeline || [];
        
        // Rebuild global dropdown preserving current selection
        const globalSel = document.getElementById('moduleSelectGlobal');
        const currentVal = globalSel.value;
        globalSel.innerHTML = '';
        const routeOpt = document.createElement('option');
        routeOpt.value = 'auto-route';
        routeOpt.textContent = '🧠 Dynamic Router (Auto-Route)';
        routeOpt.selected = (currentVal === 'auto-route');
        globalSel.appendChild(routeOpt);
        moeModules.forEach(m => {
          const opt = document.createElement('option');
          opt.value = m.config.module_id;
          opt.textContent = `${m.config.name} (${m.config.module_id})`;
          if (m.config.module_id === currentVal) opt.selected = true;
          globalSel.appendChild(opt);
        });

        renderPipelineFlow();
        renderModulesList();
        if (activeTab === 'vault')  loadVault();
        if (activeTab === 'stats')  refreshStats();
        pushToast('Registry updated — UI reloaded.');
      }
      _registryFingerprint = fp;
    }
  } catch(_) {}
}
setInterval(_pollRegistryChanges, 3000);
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return HTMLResponse(content=HTML_TEMPLATE, status_code=200)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)