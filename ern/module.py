import os
import uuid
import time
import json
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict, Any
import threading
import copy
from concurrent.futures import ThreadPoolExecutor

from ern.config import SAVE_DIR, _resolve_device
from ern.deltas import DeltaOp, TensorDelta, TensorDeltaStack

class ERNModule:
    # Shared single-thread executor to run background serialization operations sequentially
    _executor = ThreadPoolExecutor(max_workers=1)

    def __init__(self, config: Dict[str, Any], model_name='all-MiniLM-L6-v2', device: str = 'auto', embedder: Optional[SentenceTransformer] = None):
        self._save_timer = None
        self._save_lock = threading.Lock()

        self.module_id = config["module_id"]
        self.name = config["name"]
        self.description = config.get("description", "")
        self.frozen = config.get("frozen", False)
        self.mcp_enabled = config.get("mcp_enabled", True)
        
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
        combined = f"{tags} {text}"
        vec = self._encode(combined)

        # ── Duplicate Detection & Incremental Fusion ──────────────────────────
        if self.memory_bank.size(0) > 0:
            similarities = F.cosine_similarity(vec, self.memory_bank)
            max_sim, max_idx = torch.max(similarities, dim=0)
            print(f"[ERN][{self.name}] Duplicate check: max_sim={max_sim.item():.4f} against index {max_idx.item()}")
            if max_sim.item() > 0.75:
                existing_mem_id = self.labels[max_idx.item()]
                print(f"[ERN][{self.name}] Duplicate detected (sim={max_sim.item():.3f}). Fusing with existing synapse {existing_mem_id}.")

                # 1. Hebbian reinforcement: Boost its energies
                old_e = self.energies[max_idx].item()
                old_st = self.short_term_energies[max_idx].item()
                new_e = min(old_e + 0.6 + 0.2 * old_st, 5.0)
                new_st = min(old_st + 1.0, 3.0)

                self.energies[max_idx] = new_e
                self.short_term_energies[max_idx] = new_st

                # 2. Incremental Fusion: Merge tags and metadata
                existing_entry = self.vault[existing_mem_id]
                old_tags = [t.strip() for t in existing_entry.get("tags", "").split(",") if t.strip()]
                new_tags = [t.strip() for t in tags.split(",") if t.strip()]
                
                merged_tags = []
                seen_tags = set()
                for t in old_tags + new_tags:
                    t_lower = t.lower()
                    if t_lower not in seen_tags:
                        seen_tags.add(t_lower)
                        merged_tags.append(t)

                existing_entry["tags"] = ", ".join(merged_tags)
                existing_entry["timestamp"] = self._now()
                if image_url and not existing_entry.get("image_url"):
                    existing_entry["image_url"] = image_url

                # Overwrite text/vector if the new statement is longer/more detailed
                if len(text) >= len(existing_entry.get("text", "")):
                    existing_entry["text"] = text
                    self.memory_bank[max_idx] = vec.squeeze(0)
                    if memory_type in ("instruction", "question") or existing_entry.get("memory_type") == "fact":
                        existing_entry["memory_type"] = memory_type

                # 3. Deltas transaction logging
                self.deltas.push(TensorDelta(
                    op            = DeltaOp.BOOST,
                    timestamp     = self._now(),
                    delta_id      = str(uuid.uuid4()),
                    prev_size     = self.memory_bank.size(0),
                    next_size     = self.memory_bank.size(0),
                    boost_indices = [max_idx.item()],
                    boost_amounts = [new_e - old_e],
                    memory_id     = existing_mem_id,
                ))

                self._save_state()
                return existing_mem_id

        # ── Form New Synapse ──────────────────────────────────────────────────
        memory_id = str(uuid.uuid4())
        self.vault[memory_id] = {
            "text": text,
            "tags": tags,
            "timestamp": self._now(),
            "memory_type": memory_type
        }
        if image_url:
            self.vault[memory_id]["image_url"] = image_url

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

    def retrieve(self, query_text: str, top_k: int = 5, threshold: float = 0.15, decay: bool = True, tags_filter: Optional[List[str]] = None, memory_type_filter: Optional[str] = None, dry_run: bool = False):
        if self.memory_bank.size(0) == 0:
            return []

        q_vec       = self._encode(query_text)
        similarities = F.cosine_similarity(q_vec, self.memory_bank)
        resonance    = similarities * (1.0 + torch.log1p(self.energies + self.short_term_energies))

        # Apply structured constraints via PyTorch masking
        if (tags_filter and len(tags_filter) > 0) or memory_type_filter:
            mask = torch.ones((self.memory_bank.size(0),), dtype=torch.bool, device=self.device)
            for idx, mem_id in enumerate(self.labels):
                entry = self.vault.get(mem_id)
                if not entry:
                    mask[idx] = False
                    continue

                if memory_type_filter:
                    if entry.get("memory_type") != memory_type_filter:
                        mask[idx] = False
                        continue

                if tags_filter and len(tags_filter) > 0:
                    entry_tags = [t.strip().lower() for t in entry.get("tags", "").split(",") if t.strip()]
                    match = True
                    for filter_tag in tags_filter:
                        if filter_tag.lower() not in entry_tags:
                            match = False
                            break
                    if not match:
                        mask[idx] = False
                        continue

            resonance = resonance * mask.float()

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

                if not dry_run:
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
                        "energy"   : round(new_e if not dry_run else old_e, 3),
                        "stp_energy": round(new_st if not dry_run else old_st, 3),
                        "image_url": self.vault[mem_id].get("image_url"),
                        "timestamp": self.vault[mem_id].get("timestamp", 0.0)
                    })

        if boost_indices and not dry_run:
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

    def boost_node(self, memory_id: str, delta_e: float, recall_chain_id: str) -> bool:
        """
        Surgical single-node Hebbian LTP boost for use during Deep Recall hops.
        Tagged with recall_chain_id so rollback_chain() can precisely reverse it via LTD.
        Thread-safe: acquires _save_lock before mutating live tensors.
        Returns True if the node was found and boosted, False otherwise.
        """
        with self._save_lock:
            idx = self.labels.index(memory_id) if memory_id in self.labels else -1
            if idx < 0:
                return False
            old_e = self.energies[idx].item()
            new_e = min(old_e + delta_e, 5.0)
            actual_delta = new_e - old_e
            self.energies[idx] = new_e
        self.deltas.push(TensorDelta(
            op              = DeltaOp.DEEP_RECALL_BOOST,
            timestamp       = self._now(),
            delta_id        = str(uuid.uuid4()),
            prev_size       = self.memory_bank.size(0),
            next_size       = self.memory_bank.size(0),
            boost_indices   = [idx],
            boost_amounts   = [actual_delta],
            memory_id       = memory_id,
            recall_chain_id = recall_chain_id,
        ))
        self._save_state()  # Fix #5: persist boost immediately
        print(f"[ERN][{self.name}] Deep Recall hop boost: node={memory_id[:8]} Δe=+{actual_delta:.3f} chain={recall_chain_id[:8]}")
        return True

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
            self._save_state(debounce=False)
            print(f"[ERN][{self.name}] Synapse {memory_id} forgotten. Network size: {self.memory_bank.size(0)} nodes.")
            return True
        return False

    def _save_state(self, debounce: bool = True):
        if not debounce:
            self.flush_save()
            return

        with self._save_lock:
            # Debounce: Cancel any existing pending save timer
            if self._save_timer is not None:
                self._save_timer.cancel()

            # Schedule a new background save in 2 seconds
            self._save_timer = threading.Timer(2.0, self._execute_async_save)
            self._save_timer.daemon = True
            self._save_timer.start()

    def _execute_async_save(self):
        # 1. Take fast, sub-millisecond thread-safe copies of data states on the main thread
        with self._save_lock:
            state_snapshot = {
                'memory_bank': self.memory_bank.cpu().clone(),
                'energies': self.energies.cpu().clone(),
                'short_term_energies': self.short_term_energies.cpu().clone(),
                'labels': list(self.labels),
                'vault': copy.deepcopy(self.vault)
            }
            deltas_snapshot = list(self.deltas.stack)
            self._save_timer = None

        # 2. Dispatch to the single-thread background executor to execute serialization
        self._executor.submit(self._write_snapshot_to_disk, state_snapshot, deltas_snapshot)

    def _write_snapshot_to_disk(self, state_snapshot, deltas_snapshot):
        try:
            os.makedirs(self.module_dir, exist_ok=True)
            try:
                os.chmod(self.module_dir, 0o777)
            except Exception:
                pass

            torch.save(state_snapshot, self.state_path)
            torch.save(deltas_snapshot, self.delta_path)

            try:
                os.chmod(self.state_path, 0o666)
                os.chmod(self.delta_path, 0o666)
            except Exception:
                pass
            print(f"[ASYNC PERSISTENCE][{self.name}] ✓ Consolidated state successfully written to disk in background thread.")
        except Exception as e:
            print(f"[ASYNC PERSISTENCE][{self.name}] ERROR: Failed to write state to disk in background thread: {e}")

    def flush_save(self):
        """Immediately writes the current live tensor states to disk, canceling any pending timers."""
        with self._save_lock:
            if self._save_timer is not None:
                self._save_timer.cancel()
                self._save_timer = None
            
            state_snapshot = {
                'memory_bank': self.memory_bank.cpu().clone(),
                'energies': self.energies.cpu().clone(),
                'short_term_energies': self.short_term_energies.cpu().clone(),
                'labels': list(self.labels),
                'vault': copy.deepcopy(self.vault)
            }
            deltas_snapshot = list(self.deltas.stack)

        self._write_snapshot_to_disk(state_snapshot, deltas_snapshot)

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
        try:
            os.chmod(self.modules_dir, 0o777)
        except Exception:
            pass
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
        try:
            with open(self.registry_path, "w") as f:
                json.dump(registry, f, indent=2)
            try:
                os.chmod(self.registry_path, 0o666)
            except Exception:
                pass
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
