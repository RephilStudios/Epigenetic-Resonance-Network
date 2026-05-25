import datetime
from typing import List, Optional, Dict, Any

from ern.module import manager

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
        mem_id = mod.encode_hebbian(text=stripped, tags=tags, memory_type=memory_type)
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
