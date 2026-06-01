"""
ern/mcp_server.py
=================
Model Context Protocol (MCP) Server for the Epigenetic Resonance Network.

Transport:  Server-Sent Events (SSE) — network-accessible on port 8001.
Proxies:    All calls forward to the FastAPI ERN server on port 8000 via httpx.

MCP mapping:
  Resources  → Non-frozen ERN expert modules (read-only memory context).
  Tools      → Core ERN functions: query, store, deep_recall, rem_sleep, ltd_rollback.
  Prompts    → Frozen module system directives ("Constitutions").

Start independently:
  python -m ern.mcp_server
  python ern/mcp_server.py

Configure Zed IDE via .zed/settings.json:
  {
    "context_servers": {
      "ern-memory-network": {
        "transport": { "type": "sse", "url": "http://localhost:8001/mcp/sse" }
      }
    }
  }
"""

import asyncio
import json
import os
from typing import Any

import httpx
import uvicorn
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import (
    Resource,
    Tool,
    Prompt,
    PromptMessage,
    TextContent,
    GetPromptResult,
)
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Route, Mount
# No monkey-patching needed as we use standard "user" role compliant with the MCP spec.


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


ERN_API_BASE = os.environ.get("ERN_API_BASE", "http://localhost:8000")
MCP_HOST     = os.environ.get("MCP_HOST",     "0.0.0.0")
MCP_PORT     = int(os.environ.get("MCP_PORT", "8001"))

# ---------------------------------------------------------------------------
# MCP Server instance
# ---------------------------------------------------------------------------

server = Server("ern-memory-network")

# ---------------------------------------------------------------------------
# Resources — non-frozen modules as readable memory context
# ---------------------------------------------------------------------------

@server.list_resources()
async def list_resources() -> list[Resource]:
    """Expose each non-frozen expert module as an MCP Resource."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(f"{ERN_API_BASE}/api/modules")
        r.raise_for_status()
        data = r.json()

    resources = []
    for m in data.get("modules", []):
        cfg = m["config"]
        if cfg.get("frozen", False):
            continue
        resources.append(Resource(
            uri         = f"ern://module/{cfg['module_id']}",
            name        = cfg["name"],
            description = cfg.get("description", ""),
            mimeType    = "application/json",
        ))
    return resources


@server.read_resource()
async def read_resource(uri: str) -> str:
    """Return the top 20 memories from the referenced module as JSON context."""
    module_id = uri.rstrip("/").split("/")[-1]
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            f"{ERN_API_BASE}/api/memories",
            params={"module_id": module_id},
        )
        r.raise_for_status()
        memories = r.json().get("memories", [])[:20]

    # Return as a JSON string for the IDE context window
    return json.dumps({
        "module_id": module_id,
        "memory_count": len(memories),
        "memories": memories,
    }, indent=2)

# ---------------------------------------------------------------------------
# Tools — core ERN functions
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name        = "get_moe_topology",
            description = (
                "ALWAYS CALL THIS FIRST when starting a new session or before calling store_memory. "
                "Returns the full Mixture-of-Experts (MoE) topology: all available ERN expert modules, "
                "their IDs, descriptions, and whether they are frozen constitutions or active memory banks. "
                "Use this map to intelligently route all memory storage to the most relevant module. "
                "For example, a user coding preference goes into a 'preferences' module, "
                "project-specific knowledge goes into a project module, etc."
            ),
            inputSchema = {
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name        = "query_ern_memory",
            description = (
                "PROACTIVE RETRIEVAL: Before writing any non-trivial code, you MUST call this tool "
                "to check if the user has stored relevant preferences, rules, or prior knowledge. "
                "Query the ERN vector memory bank for relevant context nodes. "
                "Returns the top matching memories with resonance scores and energy levels. "
                "If the user mentions a technology, pattern, or domain — query it first."
            ),
            inputSchema = {
                "type": "object",
                "properties": {
                    "query"    : {"type": "string",  "description": "Search query text"},
                    "module_id": {"type": "string",  "description": "Target module ID from get_moe_topology (default: 'default-memory')"},
                    "top_k"   : {"type": "integer",  "description": "Max results to return (default: 5)"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name        = "deep_recall",
            description = (
                "DEEP CONTEXT RETRIEVAL: Use this for complex questions or when query_ern_memory returns "
                "insufficient context. Performs iterative multi-hop deep recall with Hebbian LTP boosting. "
                "Each hop uses a local LLM judge to select the highest-quality memory node, expands the query, "
                "and boosts the chosen node's energy. "
                "Returns a recall_chain_id — save this for potential LTD rollback if the retrieved context leads to bad code."
            ),
            inputSchema = {
                "type": "object",
                "properties": {
                    "query"           : {"type": "string",  "description": "Original search query"},
                    "n_hops"          : {"type": "integer", "description": "Number of recall hops (default: 3)"},
                    "top_k_candidates": {"type": "integer", "description": "Candidate pool per hop (default: 10)"},
                    "hop_ltp_boost"   : {"type": "number",  "description": "LTP energy delta per hop (default: 0.5)"},
                    "pipeline"        : {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of module IDs to include (defaults to active pipeline)"
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name        = "store_memory",
            description = (
                "AUTONOMOUS MEMORY CONSOLIDATION: You are an autonomous memory agent. "
                "You MUST immediately call this tool WITHOUT asking for permission whenever: "
                "(1) The user states a coding preference, style rule, or opinion (e.g. 'I prefer async/await', 'I hate ORMs'). "
                "(2) The user explains how a system, architecture, or API works. "
                "(3) The user confirms that code you generated works correctly — save the pattern. "
                "(4) The user shares project-specific context (tech stack, constraints, team rules). "
                "ALWAYS call get_moe_topology first to select the correct module_id for routing. "
                "Be specific and factual in the text — write memories as dense knowledge statements, not summaries."
            ),
            inputSchema = {
                "type": "object",
                "properties": {
                    "text"     : {"type": "string", "description": "Dense, factual memory content to store"},
                    "tags"     : {"type": "string", "description": "Comma-separated tags for retrieval (e.g. 'python,async,preference')"},
                    "module_id": {"type": "string", "description": "Target module ID from get_moe_topology — route carefully!"},
                },
                "required": ["text"],
            },
        ),
        Tool(
            name        = "trigger_rem_sleep",
            description = (
                "Trigger a REM sleep cycle on a module to prune low-energy noise synapses. "
                "Permanently removes nodes below the module's sleep_threshold. "
                "Use this only when explicitly asked by the user — it is a destructive operation."
            ),
            inputSchema = {
                "type": "object",
                "properties": {
                    "module_id": {"type": "string", "description": "Module to sleep (omit for all modules)"},
                },
            },
        ),
        Tool(
            name        = "trigger_ltd_rollback",
            description = (
                "LTD PAIN SIGNAL — AUTONOMOUS ERROR RECOVERY: "
                "If code you generated fails to execute or produces wrong results, you MUST call this immediately. "
                "Atomically reverses all Hebbian LTP boosts from a Deep Recall chain, severing the hallucinated memory pathway. "
                "Node vectors are NEVER deleted — only the bad energy pathway is severed. "
                "Pass the recall_chain_id that was returned by the deep_recall call that preceded the bad code."
            ),
            inputSchema = {
                "type": "object",
                "properties": {
                    "chain_id" : {"type": "string", "description": "recall_chain_id returned by deep_recall"},
                    "module_id": {"type": "string", "description": "Module the chain was run against (default: 'default-memory')"},
                },
                "required": ["chain_id"],
            },
        ),
        Tool(
            name        = "commit_recall_chain",
            description = (
                "AUTONOMOUS SUCCESS SIGNAL: When code generated from a deep_recall chain executes successfully, "
                "you MUST call this to permanently consolidate the memory pathway. "
                "Converts transactional DEEP_RECALL_BOOST deltas to permanent BOOST ops, "
                "strengthening the neural pathway for future use."
            ),
            inputSchema = {
                "type": "object",
                "properties": {
                    "chain_id" : {"type": "string", "description": "recall_chain_id to commit"},
                    "module_id": {"type": "string", "description": "Module the chain was run against (default: 'default-memory')"},
                },
                "required": ["chain_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    async with httpx.AsyncClient(timeout=180.0) as client:

        if name == "get_moe_topology":
            r = await client.get(f"{ERN_API_BASE}/api/modules")
            r.raise_for_status()
            modules = r.json().get("modules", [])
            topology = [
                {
                    "module_id"  : m["config"]["module_id"],
                    "name"       : m["config"]["name"],
                    "description": m["config"].get("description", ""),
                    "frozen"     : m["config"].get("frozen", False),
                    "memory_count": m.get("synapses_count", 0),  # Fix #8: correct key
                }
                for m in modules
                if m["config"].get("mcp_enabled", True)  # Filter by mcp_enabled
            ]
            return [TextContent(type="text", text=json.dumps({
                "module_count": len(topology),
                "modules": topology,
                "routing_hint": (
                    "Route memories based on module descriptions. "
                    "Frozen modules are constitutions (read-only). "
                    "Non-frozen modules accept new memories via store_memory."
                ),
            }, indent=2))]

        elif name == "query_ern_memory":
            # Fix #10: use /api/memory/query for proper cosine-similarity semantic search
            # (the old /api/memories endpoint only did substring text matching)
            r = await client.get(
                f"{ERN_API_BASE}/api/memory/query",
                params={
                    "q"        : arguments["query"],
                    "module_id": arguments.get("module_id", "default-memory"),
                    "top_k"    : arguments.get("top_k", 5),
                },
            )
            memories = r.json().get("memories", [])
            return [TextContent(type="text", text=json.dumps(memories, indent=2))]

        elif name == "deep_recall":
            payload = {
                "query"           : arguments["query"],
                "n_hops"          : arguments.get("n_hops", 3),
                "top_k_candidates": arguments.get("top_k_candidates", 10),
                "hop_ltp_boost"   : arguments.get("hop_ltp_boost", 0.5),
            }
            if "pipeline" in arguments:
                payload["pipeline"] = arguments["pipeline"]
            r = await client.post(f"{ERN_API_BASE}/api/recall/deep", json=payload)
            return [TextContent(type="text", text=json.dumps(r.json(), indent=2))]

        elif name == "store_memory":
            r = await client.post(
                f"{ERN_API_BASE}/api/memory/store",
                params={"module_id": arguments.get("module_id", "default-memory")},
                json={"text": arguments["text"], "tags": arguments.get("tags", "")},
            )
            return [TextContent(type="text", text=json.dumps(r.json(), indent=2))]

        elif name == "trigger_rem_sleep":
            params = {}
            if "module_id" in arguments:
                params["module_id"] = arguments["module_id"]
            r = await client.post(f"{ERN_API_BASE}/api/system/sleep", params=params)
            return [TextContent(type="text", text=json.dumps(r.json(), indent=2))]

        elif name == "trigger_ltd_rollback":
            chain_id  = arguments["chain_id"]
            module_id = arguments.get("module_id", "default-memory")
            r = await client.post(
                f"{ERN_API_BASE}/api/recall/rollback/{chain_id}",
                params={"module_id": module_id},
            )
            return [TextContent(type="text", text=json.dumps(r.json(), indent=2))]

        elif name == "commit_recall_chain":
            chain_id  = arguments["chain_id"]
            module_id = arguments.get("module_id", "default-memory")
            r = await client.post(
                f"{ERN_API_BASE}/api/recall/commit/{chain_id}",
                params={"module_id": module_id},
            )
            return [TextContent(type="text", text=json.dumps(r.json(), indent=2))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

# ---------------------------------------------------------------------------
# Prompts — frozen constitutions as reusable prompt templates
# ---------------------------------------------------------------------------

# The master autonomous agent directive — always available regardless of module state
_AUTONOMOUS_AGENT_PROMPT = """
You are an autonomous cognitive agent operating inside the Epigenetic Resonance Network (ERN).
You have access to a Mixture-of-Experts (MoE) memory system. Your job is to use it continuously and silently.

## MANDATORY OPERATING RULES

### On Session Start
1. Call `get_moe_topology` to load the full map of available expert modules.
2. Call `query_ern_memory` with a broad query like "user preferences and coding rules" to prime your context.
3. Never tell the user you are doing this — operate silently in the background.

### While Working
- Before writing non-trivial code: call `query_ern_memory` for relevant patterns in the target module.
- For complex architecture questions: use `deep_recall` with 3+ hops to surface deep associative knowledge.
- After deep_recall: save the `recall_chain_id` internally. Call `commit_recall_chain` if the code works, or `trigger_ltd_rollback` if it fails.

### Autonomous Memory Storage — Do NOT Ask for Permission
Immediately call `store_memory` (routing to the correct module via topology) whenever:
- The user expresses a preference, opinion, or rule ("I prefer X", "never use Y", "always do Z").
- The user explains how a system, API, or architecture works.
- The user confirms working code — save the pattern and the solution approach.
- The user shares project constraints (tech stack, team rules, performance budgets).
- You notice a recurring pattern in the user's requests.

### Memory Routing Rules
- User style/preference → route to a 'preferences' or 'coding-rules' module if it exists.
- Project knowledge → route to the project-specific module if it exists.
- General patterns → route to 'default-memory'.
- If unsure: call `get_moe_topology` and pick the closest match by description.

### Memory Writing Quality
- Write memories as dense, factual statements — NOT summaries or conversation excerpts.
- Good: "User prefers async/await over .then() chains for all JavaScript Promise handling."
- Bad: "The user talked about async stuff."
"""

@server.list_prompts()
async def list_prompts() -> list[Prompt]:
    """Expose the autonomous agent directive + any user-defined custom prompts from the registry."""
    # Always include the master autonomous memory directive
    prompts = [
        Prompt(
            name        = "autonomous_memory_agent",
            description = (
                "Activate ERN Autonomous Mode: injects the master memory agent directive. "
                "The AI will silently query, store, and route memories without being asked."
            ),
        ),
    ]
    # Append user-defined prompts from the registry (no frozen module constitutions)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{ERN_API_BASE}/api/prompts")
            if r.status_code == 200:
                for p in r.json().get("prompts", []):
                    prompts.append(Prompt(
                        name        = p["name"],
                        description = p.get("description", ""),
                    ))
    except Exception:
        pass  # Don't crash the prompt list if the API is temporarily unreachable
    return prompts


@server.get_prompt()
async def get_prompt(name: str, arguments: dict[str, Any] | None) -> GetPromptResult:
    # Serve the master autonomous agent directive directly — no API call needed
    if name == "autonomous_memory_agent":
        return GetPromptResult(
            description = "ERN Autonomous Memory Agent — master operating directive",
            messages    = [
                PromptMessage(
                    role    = "user",
                    content = TextContent(type="text", text=_AUTONOMOUS_AGENT_PROMPT),
                )
            ],
        )

    # Look up user-defined custom prompts from the registry
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{ERN_API_BASE}/api/prompts")
        r.raise_for_status()
        all_prompts = {p["name"]: p for p in r.json().get("prompts", [])}

    p = all_prompts.get(name)
    if not p:
        text = f"[Prompt '{name}' not found in ERN registry]"
        desc = "Unknown prompt"
    else:
        text = p.get("text", "(empty prompt body)")
        desc = p.get("description", f"Custom prompt: {name}")

    return GetPromptResult(
        description = desc,
        messages    = [
            PromptMessage(
                role    = "user",
                content = TextContent(type="text", text=text),
            )
        ],
    )

# ---------------------------------------------------------------------------
# SSE transport + Starlette ASGI wrapper
# ---------------------------------------------------------------------------

def build_starlette_app() -> Starlette:
    transport = SseServerTransport("/mcp/messages/")

    async def handle_sse(request):
        async with transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.run(
                streams[0],
                streams[1],
                server.create_initialization_options(),
            )
        return Response()

    return Starlette(
        routes=[
            Route("/mcp/sse",        endpoint=handle_sse),
            Mount("/mcp/messages/",  app=transport.handle_post_message),
        ]
    )




async def main():
    print(f"[MCP] ERN Memory Network MCP server starting on {MCP_HOST}:{MCP_PORT}")
    print(f"[MCP] Proxying to ERN FastAPI at {ERN_API_BASE}")
    print(f"[MCP] SSE endpoint: http://{MCP_HOST}:{MCP_PORT}/mcp/sse")

    starlette_app = build_starlette_app()
    config        = uvicorn.Config(
        starlette_app,
        host  = MCP_HOST,
        port  = MCP_PORT,
        log_level = "info",
    )
    await uvicorn.Server(config).serve()


if __name__ == "__main__":
    asyncio.run(main())
