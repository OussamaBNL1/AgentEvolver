from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List


class BaseEnvironment(ABC):
    """Base environment interface"""
    
    def __init__(self, max_steps: int = 30):
        self.max_steps = max_steps
        self.current_step = 0
        self.done = False
    
    @abstractmethod
    def reset(self) -> Tuple[str, Dict[str, Any]]:
        """Reset environment"""
        pass
    
    @abstractmethod
    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute one step"""
        pass
    
    @abstractmethod
    def get_available_actions(self) -> List[str]:
        """Get available actions"""
        pass
    
    @abstractmethod
    def close(self):
        """Close environment"""
        pass
    
    def is_done(self) -> bool:
        """Check termination"""
        return self.done or self.current_step >= self.max_steps
