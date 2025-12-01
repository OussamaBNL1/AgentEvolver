from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid


class Triplet(BaseModel):
    """Triplet data structure (history, action, observation)"""
    triplet_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    env_id: str = ""  # Environment ID
    session_id: Optional[str] = None  # Session ID
    history: str  # History (as string)
    action: str         # Action
    observation: str    # Observation
    reward: float = 0.0  # Reward
    done: bool = False   # Done flag
    timestamp: datetime = Field(default_factory=datetime.now)
    exploration_memory: Optional[str] = None  # Exploration memory (optional)
    original_action: Optional[str] = None  # Original action (optional)

    def to_dict(self) -> dict:
        """Convert to dict"""
        if hasattr(self, 'model_dump'):
            return self.model_dump()
        else:
            return self.dict()


class Task(BaseModel):
    """Task abstracted from triplets"""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    env_id: str = ""  # Environment ID
    session_id: Optional[str] = None  # Session ID
    description: str    # Task description
    query: str         # Task query
    source_triplets: List[str] = []  # Source triplet IDs
    confidence: float = 1.0  # Abstraction confidence
    timestamp: datetime = Field(default_factory=datetime.now)
    gt: Optional[str] = None  # Ground truth (optional)

    def to_dict(self) -> dict:
        """Convert to dict"""
        if hasattr(self, 'model_dump'):
            return self.model_dump()
        else:
            return self.dict()


class Session(BaseModel):
    """Session data for a full three-stage execution"""
    session_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3])
    triplets: List[Triplet] = []
    tasks: List[Task] = []
    config: Dict[str, Any] = {}
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed

    def to_dict(self) -> dict:
        """Convert to dict"""
        if hasattr(self, 'model_dump'):
            return self.model_dump()
        else:
            return self.dict()
