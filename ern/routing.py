import re
import requests
import torch
import torch.nn.functional as F
from typing import List, Dict, Any

from ern.config import OLLAMA_URL, DEFAULT_MODEL, JUDGE_MODEL
from ern.module import manager

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
