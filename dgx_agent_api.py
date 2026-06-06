import os
import sys
# Ensure python finds the local package in all runtime environments
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uuid
import time
import asyncio
import datetime
import requests
import uvicorn
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, BackgroundTasks, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# Import classes into entrypoint's namespace to satisfy unpickling legacy data
from ern.deltas import TensorDelta, DeltaOp

# Import ern package components
from ern.config import SAVE_DIR, OLLAMA_URL, DEFAULT_MODEL, JUDGE_MODEL, VISION_MODEL
from ern.models import (
    Message, ChatRequest, ChatResponse, MemoryStoreRequest,
    ModuleConfig, ModulePatchRequest, BuilderRequest,
    DeepRecallRequest, DeepRecallResponse,
    CustomPrompt, CustomPromptPatchRequest,
)
from ern.module import manager
from ern.helpers import (
    _classify_message, _direct_save_message, _format_age,
    format_temporal_memory_block
)
from ern.routing import route_query_to_modules, agentic_search_planner
from ern.deep_recall import deep_recall
from ern.extraction import (
    run_memory_judge, route_fact_to_module, extract_text_from_pdf, run_pdf_extractor_chunk,
    process_pdf_background, process_image_background
)
from ern.dashboard import HTML_TEMPLATE

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
try:
    os.chmod("ern_state", 0o777)
    os.chmod("ern_state/uploads", 0o777)
except Exception:
    pass
app.mount("/static", StaticFiles(directory="ern_state"), name="static")

@app.post("/api/chat", response_model=ChatResponse)
def process_chat(req: ChatRequest, background_tasks: BackgroundTasks):
    user_message = req.message
    
    # 1. Resolve pipeline modules
    modules_list = manager.list_modules()
    active_pipeline = manager.registry.get("default_pipeline", ["default-memory"])
    
    if req.auto_route:
        pipeline_ids = route_query_to_modules(user_message, modules_list)
        # Only include default-memory as a fallback for retrieval if it is actually selected in the default pipeline
        if "default-memory" in active_pipeline and "default-memory" not in pipeline_ids:
            pipeline_ids.append("default-memory")
    elif req.pipeline:
        pipeline_ids = [pid for pid in req.pipeline if pid in manager.registry["modules"]]
    else:
        pipeline_ids = [pid for pid in active_pipeline if pid in manager.registry["modules"]]
    
    # Safety: never operate with a completely empty pipeline. If empty, fall back to selected active pipeline modules.
    if not pipeline_ids:
        pipeline_ids = [pid for pid in active_pipeline if pid in manager.registry["modules"]]
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
            q_mems = mod.retrieve(
                q, top_k=5, threshold=req.focus_threshold, decay=False,
                tags_filter=req.tags_filter, memory_type_filter=req.memory_type_filter
            )
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

    # Passive ingestion — classify type, MoE-route, then save
    # Questions, facts, and instructions each carry different initial energy levels
    # so the network naturally differentiates signal quality without extra logic.
    msg_tags, _ = _classify_message(user_message)
    target_extraction_id = route_fact_to_module(user_message, msg_tags, fallback_module_id="default-memory")

    # Synchronous raw save — type-tagged, guaranteed every turn
    _direct_save_message(user_message, target_extraction_id)

    # LLM-based extraction in background — each extracted fact gets its own MoE routing
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
        try:
            os.chmod(archive_path, 0o666)
        except Exception:
            pass
            
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
            
    if req.mcp_enabled is not None:
        config["mcp_enabled"] = req.mcp_enabled
        if module_id in manager.active_modules:
            manager.active_modules[module_id].mcp_enabled = req.mcp_enabled
            
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


# ── Custom Prompt Registry ────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    """Convert a human name into a valid slash-command slug."""
    import re
    return re.sub(r'[^a-z0-9_]+', '_', name.lower().strip()).strip('_')

@app.get("/api/prompts")
def list_prompts_endpoint():
    prompts = manager.registry.get("custom_prompts", {})
    return {"prompts": list(prompts.values())}

@app.post("/api/prompts")
def create_prompt_endpoint(req: CustomPrompt):
    prompts = manager.registry.setdefault("custom_prompts", {})
    slug = _slugify(req.name)
    if slug in prompts:
        return {"error": f"Prompt '{slug}' already exists.", "status": "Failed"}
    entry = {"name": slug, "description": req.description, "text": req.text}
    prompts[slug] = entry
    manager.registry["custom_prompts"] = prompts
    manager._save_registry(manager.registry)
    return {"status": "Created", "prompt": entry}

@app.patch("/api/prompts/{name}")
def patch_prompt_endpoint(name: str, req: CustomPromptPatchRequest):
    prompts = manager.registry.get("custom_prompts", {})
    if name not in prompts:
        return {"error": f"Prompt '{name}' not found.", "status": "Failed"}
    if req.description is not None:
        prompts[name]["description"] = req.description
    if req.text is not None:
        prompts[name]["text"] = req.text
    manager.registry["custom_prompts"] = prompts
    manager._save_registry(manager.registry)
    return {"status": "Updated", "prompt": prompts[name]}

@app.delete("/api/prompts/{name}")
def delete_prompt_endpoint(name: str):
    prompts = manager.registry.get("custom_prompts", {})
    if name not in prompts:
        return {"error": f"Prompt '{name}' not found.", "status": "Failed"}
    deleted = prompts.pop(name)
    manager.registry["custom_prompts"] = prompts
    manager._save_registry(manager.registry)
    return {"status": "Deleted", "prompt": deleted}


@app.get("/api/memories")
def get_all_memories(q: Optional[str] = None, module_id: str = "default-memory", tags_filter: Optional[str] = None, memory_type_filter: Optional[str] = None):
    # When auto-route is selected, only aggregate modules in the active pipeline
    if module_id == "auto-route":
        module_ids = manager.registry.get("default_pipeline", ["default-memory"])
        if not module_ids:
            module_ids = ["default-memory"]
    else:
        module_ids = [module_id]

    results = []
    parsed_tags = [t.strip().lower() for t in tags_filter.split(",") if t.strip()] if tags_filter else []

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

            if memory_type_filter and mtype != memory_type_filter:
                continue

            if parsed_tags:
                entry_tags = [t.strip().lower() for t in data["tags"].split(",") if t.strip()]
                match = True
                for t in parsed_tags:
                    if t not in entry_tags:
                        match = False
                        break
                if not match:
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


# ---------------------------------------------------------------------------
# Deep Recall endpoints (Feature 1 + 2)
# ---------------------------------------------------------------------------

@app.post("/api/recall/deep", response_model=DeepRecallResponse)
async def deep_recall_endpoint(req: DeepRecallRequest):
    """
    Iterative multi-hop deep recall with Hebbian LTP boosting.
    Returns a recall_chain_id that can later be committed or rolled back via LTD.
    """
    # Resolve pipeline
    active_pipeline = manager.registry.get("default_pipeline", ["default-memory"])
    if req.pipeline:
        pipeline_ids = [pid for pid in req.pipeline if pid in manager.registry["modules"]]
    else:
        pipeline_ids = [pid for pid in active_pipeline if pid in manager.registry["modules"]]
    if not pipeline_ids:
        pipeline_ids = ["default-memory"]

    modules    = [m for pid in pipeline_ids if (m := manager.get_module(pid)) is not None]
    module_ids = [pid for pid in pipeline_ids if manager.get_module(pid) is not None]

    if not modules:
        return DeepRecallResponse(
            recall_chain_id="",
            original_query=req.query,
            final_memories=[],
            hop_log=[{"status": "error", "detail": "No valid modules resolved."}],
            boosted_nodes=[],
        )

    # Run the hop loop in a thread executor to avoid blocking uvicorn's event loop
    loop   = asyncio.get_running_loop()  # Fix #6: deprecated get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: deep_recall(
            query            = req.query,
            modules          = modules,
            module_ids       = module_ids,
            n_hops           = req.n_hops,
            top_k_candidates = req.top_k_candidates,
            hop_ltp_boost    = req.hop_ltp_boost,
        ),
    )

    return DeepRecallResponse(
        recall_chain_id = result.recall_chain_id,
        original_query  = result.original_query,
        final_memories  = result.final_memories,
        hop_log         = result.hop_log,
        boosted_nodes   = result.boosted_nodes,
    )


@app.post("/api/recall/rollback/{chain_id}")
def trigger_ltd_rollback(chain_id: str, module_id: Optional[str] = None):
    """
    LTD Pain Signal — atomically reverses all DEEP_RECALL_BOOST deltas tagged
    with chain_id across ALL active modules (deep recall spans multiple experts).
    Node vectors and vault entries are NEVER deleted; only the hallucinated
    energy pathway is severed.
    """
    from ern.deltas import DeltaOp

    # Fix #2: search all active modules, not just one
    search_modules = (
        [manager.get_module(module_id)] if module_id and manager.get_module(module_id)
        else list(manager.active_modules.values())
    )

    total_undone = 0
    modules_affected = []
    for mod in search_modules:
        chain_deltas = [
            d for d in mod.deltas.stack
            if d.op == DeltaOp.DEEP_RECALL_BOOST
            and getattr(d, "recall_chain_id", None) == chain_id
        ]
        if chain_deltas:
            undone = mod.deltas.rollback_chain(mod, chain_id)
            total_undone += undone
            modules_affected.append(mod.module_id)

    if total_undone == 0:
        return {
            "error"   : f"Chain '{chain_id}' not found in any active module's delta stack "
                        "(already rolled back or never committed).",
            "status"  : "not_found",
        }

    return {
        "status"          : "LTD rollback complete. Hallucinated neural bridge severed.",
        "chain_id"        : chain_id,
        "deltas_reversed" : total_undone,
        "modules_affected": modules_affected,
    }


@app.post("/api/recall/commit/{chain_id}")
def commit_recall_chain(chain_id: str, module_id: Optional[str] = None):
    """
    Permanently commit a recall chain — re-tags DEEP_RECALL_BOOST deltas as
    standard BOOST ops across ALL active modules so they are no longer eligible
    for chain rollback. Call this when the downstream code execution succeeds.
    """
    from ern.deltas import DeltaOp

    # Fix #3: commit across all active modules, not just one
    search_modules = (
        [manager.get_module(module_id)] if module_id and manager.get_module(module_id)
        else list(manager.active_modules.values())
    )

    total_committed = 0
    modules_affected = []
    for mod in search_modules:
        committed = 0
        with mod.deltas._lock:
            for d in mod.deltas.stack:
                if d.op == DeltaOp.DEEP_RECALL_BOOST and getattr(d, "recall_chain_id", None) == chain_id:
                    d.op              = DeltaOp.BOOST
                    d.recall_chain_id = None
                    committed        += 1
        if committed > 0:
            mod._save_state(debounce=False)
            total_committed += committed
            modules_affected.append(mod.module_id)

    if total_committed == 0:
        return {
            "error"   : f"Chain '{chain_id}' not found in any active module's delta stack "
                        "(already committed, rolled back, or never created).",
            "status"  : "not_found",
        }

    return {
        "status"          : "Chain committed. LTP boosts are now permanent.",
        "chain_id"        : chain_id,
        "committed"       : total_committed,
        "modules_affected": modules_affected,
    }


@app.get("/api/memory/query")
def semantic_query_memories(q: str, module_id: str = "default-memory", top_k: int = 5):
    """
    Fix #10: Proper semantic vector search via mod.retrieve() — used by the MCP
    query_ern_memory tool. Unlike /api/memories which does substring matching,
    this performs cosine similarity + resonance scoring in the embedding space.
    """
    mod = manager.get_module(module_id)
    if not mod:
        return {"error": f"Module '{module_id}' not found.", "memories": []}
    results = mod.retrieve(q, top_k=top_k, decay=False)
    for r in results:
        r["module_id"] = module_id
        r["module_name"] = mod.name
    return {"memories": results, "module_id": module_id, "query": q}


@app.get("/api/models")
def get_models():
    try:
        base   = OLLAMA_URL.rsplit("/api/", 1)[0]
        res    = requests.get(f"{base}/api/tags", timeout=10)
        models = [m["name"] for m in res.json().get("models", [])]
        return {"models": sorted(models)}
    except Exception as e:
        return {"models": [DEFAULT_MODEL], "error": str(e)}


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return HTMLResponse(content=HTML_TEMPLATE, status_code=200)


@app.on_event("shutdown")
def shutdown_event():
    print("[SYSTEM] FastAPI shutting down — flushing all active module debounced memory writes...")
    for mod in manager.active_modules.values():
        mod.flush_save()
    print("[SYSTEM] Memory flush complete.")

import atexit
@atexit.register
def exit_handler():
    # Fallback sync flush to completely guarantee no memory leaks on system exit
    for mod in manager.active_modules.values():
        try:
            mod.flush_save()
        except Exception:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)