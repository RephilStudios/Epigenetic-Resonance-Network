"""
ern/deep_recall.py
==================
Iterative Deep Recall — Multi-Hop Query Expansion Engine.

Algorithm (per hop):
1. Retrieve top-k candidate nodes with dry_run=True (no energy mutations).
2. Ask the Ollama judge to select the single highest-quality node index.
3. Apply a targeted Hebbian LTP boost ONLY to the chosen node via boost_node(),
   tagged with a recall_chain_id for transactional LTD reversal (Feature 2).
4. Expand query: "{original_query} | {chosen_node_text}"
5. Repeat n_hops times, accumulating unique results.

The returned DeepRecallResult carries the recall_chain_id so the caller can:
  - Commit the chain (no-op: energies stay boosted permanently).
  - Trigger LTD rollback via module.deltas.rollback_chain(module, chain_id)
    if the downstream code execution fails (Pain Signal).
"""

import re
import uuid
import json
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from ern.config import OLLAMA_URL, JUDGE_MODEL
from ern.module import ERNModule


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DeepRecallResult:
    """Immutable record of one deep recall session."""
    recall_chain_id : str
    original_query  : str
    final_memories  : List[Dict[str, Any]]           # merged unique results from all hops
    hop_log         : List[Dict[str, Any]]            # per-hop detail for agentic_steps UI
    boosted_nodes   : List[Dict[str, Any]]            # [{module_id, memory_id, delta_e}]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _call_judge(
    original_query: str,
    candidates: List[Dict[str, Any]],
    model: str = JUDGE_MODEL,
) -> int:
    """
    Ask the Ollama judge to select the best candidate index (0-based).
    Returns the parsed integer index, or 0 on any failure.
    """
    if not candidates:
        return 0

    candidate_block = "\n".join(
        f"[{i}] {c['text'][:200]}" for i, c in enumerate(candidates)
    )
    prompt = (
        f"Original query: \"{original_query}\"\n\n"
        "Candidate memory nodes:\n"
        f"{candidate_block}\n\n"
        "Which candidate (by index number) is most relevant and highest quality "
        "for expanding the original query? "
        "Reply with ONLY a single integer (0 to "
        f"{len(candidates) - 1}), nothing else."
    )
    try:
        res = requests.post(
            OLLAMA_URL,
            json={
                "model"  : model,
                "messages": [{"role": "user", "content": prompt}],
                "stream" : False,
                "options": {"temperature": 0.0, "num_predict": 4},
            },
            timeout=30,
        )
        raw = res.json().get("message", {}).get("content", "").strip()
        # Extract first integer found in response
        match = re.search(r"\d+", raw)
        if match:
            idx = int(match.group())
            return max(0, min(idx, len(candidates) - 1))
    except Exception as e:
        print(f"[DEEP RECALL] Judge call failed ({e}). Defaulting to index 0.")
    return 0


def _collect_unique(
    accumulator: Dict[str, Dict[str, Any]],
    results: List[Dict[str, Any]],
    module_id: str,
    module_name: str,
) -> None:
    """
    Merge new retrieval results into the accumulator dict, keyed by
    '{module_id}:{memory_id}' to guarantee cross-module deduplication.
    """
    for r in results:
        key = f"{module_id}:{r['memory_id']}"
        if key not in accumulator:
            r_copy = dict(r)
            r_copy["module_id"]   = module_id
            r_copy["module_name"] = module_name
            accumulator[key] = r_copy


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def deep_recall(
    query          : str,
    modules        : List[ERNModule],
    module_ids     : List[str],
    n_hops         : int   = 3,
    top_k_candidates: int  = 10,
    hop_ltp_boost  : float = 0.5,
    threshold      : float = 0.10,
    judge_model    : str   = JUDGE_MODEL,
) -> DeepRecallResult:
    """
    Run the iterative multi-hop deep recall loop.

    Parameters
    ----------
    query           : The original user query string.
    modules         : List of ERNModule instances in the active pipeline.
    module_ids      : Corresponding module ID strings (same order as modules).
    n_hops          : Number of recall hops (default 3).
    top_k_candidates: Candidate pool size per hop (default 10).
    hop_ltp_boost   : LTP energy delta applied to the judge-selected node per hop.
    threshold       : Minimum resonance score to include a result.
    judge_model     : Ollama model used by the hop selector judge.

    Returns
    -------
    DeepRecallResult containing the merged memory set, hop log, and chain ID.
    """
    chain_id    = str(uuid.uuid4())
    current_q   = query
    seen        : Dict[str, Dict[str, Any]] = {}   # deduplication accumulator
    hop_log     : List[Dict[str, Any]]      = []
    boosted     : List[Dict[str, Any]]      = []

    print(f"\n[DEEP RECALL] === Starting {n_hops}-hop recall | chain={chain_id[:8]} ===")
    print(f"[DEEP RECALL] Original query: \"{query}\"")

    for hop in range(n_hops):
        print(f"\n[DEEP RECALL] --- Hop {hop + 1}/{n_hops} | query=\"{current_q[:80]}...\"")

        # ── Step 1: Dry-run retrieval across all pipeline modules ──────────
        hop_candidates: List[Tuple[ERNModule, str, Dict[str, Any]]] = []
        for mod, mid in zip(modules, module_ids):
            results = mod.retrieve(
                current_q,
                top_k=top_k_candidates,
                threshold=threshold,
                decay=False,
                dry_run=True,           # No energy mutations during candidate scoring
            )
            for r in results:
                hop_candidates.append((mod, mid, r))
            # Accumulate ALL candidates (not just chosen) into the final set
            _collect_unique(seen, results, mid, mod.name)

        if not hop_candidates:
            print(f"[DEEP RECALL] Hop {hop + 1}: No candidates above threshold. Stopping early.")
            hop_log.append({
                "hop": hop + 1,
                "query": current_q,
                "candidates_found": 0,
                "chosen_node": None,
                "status": "no_candidates",
            })
            break

        # Sort by resonance descending; keep top top_k_candidates across all modules
        hop_candidates.sort(key=lambda x: x[2]["resonance"], reverse=True)
        hop_candidates = hop_candidates[:top_k_candidates]

        # ── Step 2: Judge selects the single best candidate ────────────────
        candidate_dicts = [c[2] for c in hop_candidates]
        chosen_idx      = _call_judge(query, candidate_dicts, judge_model)
        chosen_mod, chosen_mid, chosen_node = hop_candidates[chosen_idx]

        print(
            f"[DEEP RECALL] Hop {hop + 1}: Judge selected index {chosen_idx} "
            f"→ node={chosen_node['memory_id'][:8]} "
            f"resonance={chosen_node['resonance']:.3f} "
            f"text=\"{chosen_node['text'][:60]}...\""
        )

        # ── Step 3: Targeted Hebbian LTP boost on chosen node ─────────────
        boosted_ok = chosen_mod.boost_node(
            memory_id       = chosen_node["memory_id"],
            delta_e         = hop_ltp_boost,
            recall_chain_id = chain_id,
        )
        if boosted_ok:
            boosted.append({
                "module_id" : chosen_mid,
                "memory_id" : chosen_node["memory_id"],
                "delta_e"   : hop_ltp_boost,
                "hop"       : hop + 1,
            })

        # ── Step 4: Expand query for next hop ─────────────────────────────
        expanded = f"{query} | {chosen_node['text']}"
        hop_log.append({
            "hop"             : hop + 1,
            "query"           : current_q,
            "candidates_found": len(hop_candidates),
            "chosen_index"    : chosen_idx,
            "chosen_node"     : {
                "memory_id"  : chosen_node["memory_id"],
                "module_id"  : chosen_mid,
                "resonance"  : chosen_node["resonance"],
                "text_snippet": chosen_node["text"][:120],
            },
            "expanded_query"  : expanded[:200],
            "ltp_boost_applied": boosted_ok,
            "status"          : "ok",
        })
        current_q = expanded

    # ── Finalise: sort merged results by resonance ─────────────────────────
    final_memories = sorted(seen.values(), key=lambda x: x["resonance"], reverse=True)

    print(
        f"\n[DEEP RECALL] === Complete | chain={chain_id[:8]} | "
        f"hops={len(hop_log)} | unique_nodes={len(final_memories)} | "
        f"boosted={len(boosted)} ==="
    )

    return DeepRecallResult(
        recall_chain_id = chain_id,
        original_query  = query,
        final_memories  = final_memories,
        hop_log         = hop_log,
        boosted_nodes   = boosted,
    )
