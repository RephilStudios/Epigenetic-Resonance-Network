# 🧠 ERN // Epigenetic Resonance Network

The **Epigenetic Resonance Network (ERN)** is a state-of-the-art, high-performance, dynamic tensor-based memory architecture designed to transform passive LLM context interfaces into an **autonomous agentic long-term memory cortex**. 

Integrated directly with developer tools (like the **Zed IDE**) via the Model Context Protocol (MCP), ERN enables AI assistants to autonomously query, boost, prune, and manage knowledge, preferences, and coding styles in the background without explicit user intervention.

---

## 🌟 Architecture & Key Features

### 1. Mixture-of-Experts (MoE) Memory Topology
Rather than dumping all historical interactions into a single flat vector index, ERN organizes knowledge into a modular **Mixture-of-Experts (MoE)** network:
*   **Specialized Expertise Modules**: Create isolated memory spaces (e.g., `coding-rules`, `user-preferences`, `project-architecture`).
*   **Constitutions**: Each module can be loaded with custom system directives that are injected into the agent's prompt whenever the expert is actively engaged.
*   **Dynamic Auto-Routing**: A routing engine automatically analyzes incoming queries to engage the most relevant experts, avoiding context bloat and maximizing token efficiency.

### 2. Transactional Recall & Hebbian LTP/LTD Rollbacks
To safeguard against hallucinated memory structures and incorrect neural pathways, ERN implements a transactional vector control system:
*   **Hebbian LTP (Long-Term Potentiation) Boosts**: Retrieving a memory boosts its dynamic "energy level" in real-time, cementing active knowledge in the cache.
*   **Deep Multi-Hop Recall**: Resolves complex questions by traversing associative memory hops across multiple expert modules sequentially.
*   **LTD (Long-Term Depression) Pain Signal**: If code generated from a deep recall hop fails to compile or execute, the agent triggers an **LTD Rollback**. This atomically rolls back the transactional delta stack and severs the bad memory bridge without deleting actual underlying vector data.
*   **Pathway Consolidation**: When the generated code runs successfully, the agent calls a **Commit** signal, consolidating the Hebbian boosts permanently.

### 3. Sleek Web HUD Telemetry Control Center
An integrated, beautifully designed real-time dashboard offers high-fidelity telemetry:
*   **VRAM Telemetry & Canvas Sparklines**: Watch real-time energy surges, resonance coefficients, and neural activity.
*   **MCP Toggle Badges**: Toggle modules as **`MCP`** (exposed) or **`NO MCP`** (hidden) with a single click. Hiding a module immediately strips it from Zed's active topology and prompt registry.
*   ** REM Sleep Cycles**: Manually initiate or automate simulated REM sleep to run synaptic pruning, decaying overall energy and purging faint, outdated memories to prevent memory bloat.

---

## 🚀 Telemetry Control Console

![Dashboard Preview](file:///home/reid/.gemini/antigravity/brain/bcfda9c9-e177-44a0-851a-7881bbce7de2/media__1779744189982.png)
*Real-time MoE topology manager with glows, VRAM sparklines, delta stacks, and interactive MCP toggle badges.*

---

## 📋 Prerequisites

1.  **[Docker](https://docs.docker.com/get-docker/)** & **[Docker Compose](https://docs.docker.com/compose/install/)**
2.  **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)** (Optional, for DGX/GPU hardware acceleration)
3.  **[Ollama](https://ollama.com/)** running locally on the host machine.

### Ollama Model Prep
Ensure the default memory classification and judge model is pulled and running on your host machine:
```bash
ollama run qwen2.5-coder:7b
```

---

## 🚀 Quick Start

The entire multi-container service (FastAPI ERN core server + SSE MCP bridge) runs seamlessly via Docker Compose:

1.  **Start the services**:
    ```bash
    docker compose up --build -d
    ```
2.  **Access the Dashboard**: Open your browser and navigate to **[http://localhost:8000](http://localhost:8000)**.
3.  **Explore the MoE Control Center**: Switch to the `🧠 MOE ARCHITECT` tab to design modules, customize system directives, toggle active pipelines, or toggle MCP visibility.

---

## 🔌 Core API Endpoints

*   `GET /`: Interactive web telemetry console.
*   `POST /api/chat`: Chat console endpoint with vector retrieval.
*   `GET /api/memory/query`: Proper vector-space cosine similarity search.
*   `POST /api/memory/store`: Store custom memories.
*   `POST /api/recall/rollback/{chain_id}`: Roll back hallucinated boosts across all experts (LTD rollback).
*   `POST /api/recall/commit/{chain_id}`: Commit a successful recall chain permanently (consolidate boosts).
*   `POST /api/system/sleep`: Run REM sleep pruning.

---

## 🔌 Zed IDE Integration (Settings)

To integrate ERN's autonomous memory matrix with your local Zed editor, establish a remote bridge to the SSE server.

Add the following to your local `.zed/settings.json` (or global `settings.json`):

```json
"context_servers": {
    "ern-memory-network": {
      "enabled": true,
      "command": "npx",
      "args": [
        "-y",
        "mcp-remote",
        "http://100.106.252.1:8001/mcp/sse",
        "--allow-http",
      ],
    },
  },
```
*(Replace the IP address with your DGX host Tailscale or LAN IP address if hosting remotely).*

---

## 🛠️ MCP Toolset Reference

Once connected, your IDE Agent will autonomously leverage the following tools to manage its memory state:

| Tool | Action | Agentic Role |
| :--- | :--- | :--- |
| `get_moe_topology` | Fetches active modules and descriptions. | Discover available experts and routing pathways. |
| `query_ern_memory` | Performs cosine similarity search against an expert. | Load style rules, project parameters, or preferences. |
| `deep_recall` | Performs multi-hop associative recall across pipeline. | Retrieve complex background context and trigger LTP boosts. |
| `store_memory` | Writes a new dense fact vector into an expert. | Automatically save user preferences or proven code solutions. |
| `trigger_ltd_rollback` | Reverses all transaction boosts for a specific chain. | Recover context integrity if generated code fails. |
| `commit_recall_chain` | Consolidates all transaction boosts in a chain. | Strengthen valid memory pathways after code success. |
| `trigger_rem_sleep` | Triggers REM sleep pruning on a module. | Keep the memory stack clean and performant. |
