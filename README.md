# DGX Unified Agent API

The **DGX Unified Agent API** is a high-performance, agentic LLM wrapper equipped with a custom **PyTorch Epigenetic Memory System (ERN)**. Built with FastAPI and backed by Ollama, this project provides a stateful AI agent capable of forming memories, retrieving them via dense embeddings, and pruning weak connections through simulated REM sleep cycles.

## 🌟 Features

* **PyTorch Epigenetic Resonance Network (ERN)**: A persistent, dynamic tensor-based memory bank running on GPU/CPU.
* **Autonomous Memory Extraction**: A background "Memory Judge" asynchronously extracts factual information from user interactions without blocking the main chat response.
* **Simulated REM Sleep**: Periodically scrubs weak memories (synaptic pruning) to prevent unbounded memory growth.
* **Ollama Integration**: Seamlessly interfaces with local LLMs (defaults to `qwen2.5-coder:7b`) for privacy-preserving, high-speed inference.
* **Web-Based Dashboard**: Includes a sleek, integrated chat UI for testing interactions and observing real-time memory recall (VRAM Recall).
* **Dockerized & GPU Ready**: Leverages NVIDIA's PyTorch container base for optimal DGX/GPU hardware acceleration.

## 📋 Prerequisites

Before running the project, ensure you have the following installed:

1. **[Docker](https://docs.docker.com/get-docker/)** & **[Docker Compose](https://docs.docker.com/compose/install/)**
2. **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)** (for GPU acceleration within Docker)
3. **[Ollama](https://ollama.com/)** running locally on your host machine.

### Ollama Setup

The agent expects an Ollama instance to be running on your local machine. By default, it expects the `qwen2.5-coder:7b` model to be available. 

Run the following command on your host machine to pull the model:
```bash
ollama run qwen2.5-coder:7b
```
*(You can exit the Ollama prompt once it has finished downloading).*

## 🚀 Getting Started

The easiest way to run the DGX Agent API is via `docker-compose`. 

1. **Clone/Navigate** to the project directory.
2. **Start the application**:
   ```bash
   docker-compose up --build
   ```

> **Note:** The `docker-compose.yml` uses `network_mode: host` to allow the container to easily communicate with your host's Ollama instance at `http://localhost:11434`.

3. **Access the Web Dashboard**: Open your browser and navigate to [http://localhost:8000](http://localhost:8000).

## 🧠 Architecture Overview

1. **FastAPI Backend (`dgx_agent_api.py`)**: Handles API routing, WebSocket/HTTP connections, and serves the UI.
2. **Dense Epigenetic Engine**: Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to encode text into dense vectors. Stores these embeddings and their "energy" levels in PyTorch tensors (`ern_state.pt`).
3. **Hebbian Learning & Pruning**: Retrieving memories boosts their "energy" (Hebbian boost). Periodically, a sleep cycle decays overall energy and purges memories that fall below a threshold.
4. **Dual Agent System**: 
   * **Main Model**: Processes the chat history, recalled memories, and user message to generate a response.
   * **Judge Model**: Runs in the background to analyze the user's input and extract concrete facts to store in the ERN.

## 🔌 API Endpoints

* `GET /`: Serves the HTML testing dashboard.
* `POST /api/chat`: Main chat endpoint. Expects a JSON payload with `message`, `model`, `history`, and `focus_threshold`.
* `POST /api/memory/store`: Manually store a memory with explicit text and tags.
* `POST /api/system/sleep`: Manually trigger a REM sleep cycle to prune weak memories.
* `GET /api/models`: Proxies available models from your local Ollama instance.

## 💾 State Persistence

The ERN memory state is saved periodically to `./ern_state/ern_state.pt`. This directory is mounted as a volume in `docker-compose`, meaning your agent's memories will persist across container restarts.
