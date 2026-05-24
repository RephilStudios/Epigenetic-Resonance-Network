import os
import torch

OLLAMA_URL    = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
DEFAULT_MODEL = "qwen2.5-coder:7b"
JUDGE_MODEL   = "qwen2.5-coder:7b"
VISION_MODEL  = os.environ.get("VISION_MODEL", "llama3.2-vision")
SAVE_DIR      = "./ern_state"

def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            t = torch.zeros(1, device='cuda')
            del t
            return torch.device('cuda')
        except Exception as e:
            print(f"[HARDWARE] CUDA detected but unusable ({e}). Falling back to CPU.")
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
