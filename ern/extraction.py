import os
import re
import io
import base64
import requests
from PIL import Image
from pypdf import PdfReader
from typing import Optional

from ern.config import OLLAMA_URL, JUDGE_MODEL, VISION_MODEL, DEFAULT_MODEL
from ern.module import manager
from ern.helpers import _classify_message

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
                text=user_message,
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
                    text=user_message,
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
            raise Exception(f"Ollama returned HTTP status {res.status_code}: {res.text}")

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
