from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from ern.config import DEFAULT_MODEL

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    model: str = DEFAULT_MODEL
    history: List[Message] = []
    focus_threshold: float = 0.15
    agentic_search: bool = True
    temporal_discovery: bool = True
    pipeline: Optional[List[str]] = None
    auto_route: bool = False
    active_expert: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    context_used: str
    memories: List[Dict[str, Any]] = []
    agentic_steps: List[Dict[str, Any]] = []

class MemoryStoreRequest(BaseModel):
    text: str
    tags: str = ""
    memory_type: Optional[str] = None

class ModuleConfig(BaseModel):
    module_id: str
    name: str
    description: str = ""
    frozen: bool = False
    ltp_decay_rate: float = 0.95
    stp_decay_rate: float = 0.80
    sleep_threshold: float = 0.10
    focus_threshold: float = 0.15
    system_directive: str = ""

class ModulePatchRequest(BaseModel):
    frozen: Optional[bool] = None
    ltp_decay_rate: Optional[float] = None
    stp_decay_rate: Optional[float] = None

class BuilderRequest(BaseModel):
    message: str
    history: List[Message] = []
