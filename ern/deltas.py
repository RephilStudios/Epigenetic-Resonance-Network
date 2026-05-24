import os
import uuid
import torch
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

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
