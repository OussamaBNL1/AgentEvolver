from typing import Tuple, Dict, Any, List
from .base_env import BaseEnvironment
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SciWorldEnvironment(BaseEnvironment):
    """SciWorld environment wrapper implemented based on the provided code"""
    
    def __init__(self, jar_path: str, task_name: str = "task-1-boil-water", 
                 var_num: int = 0, simplification: str = "easy", max_steps: int = 30):
        super().__init__(max_steps)
        self.jar_path = jar_path
        self.task_name = task_name
        self.var_num = var_num
        self.simplification = simplification
        self.env = None
        self._init_environment()
    
    def _init_environment(self):
        """Initialize SciWorld environment (following the example code)"""
        try:
            from scienceworld import ScienceWorldEnv
            
            self.env = ScienceWorldEnv("", self.jar_path, envStepLimit=100)
            self.env.load(self.task_name, self.var_num, self.simplification, generateGoldPath=True)
            logger.info(f"SciWorld environment initialized: {self.task_name}")
            
        except ImportError as e:
            error_message = (
                f"Error importing ScienceWorldEnv: {str(e)}. "
                "Please make sure you have installed the sciworld package successfully, "
                "following the instructions in https://github.com/allenai/ScienceWorld"
            )
            logger.error(error_message)
            raise ImportError(error_message)
    
    def get_available_actions(self) -> List[str]:
        """Get available SciWorld actions"""
        return [
            "open OBJ",
            "close OBJ", 
            "activate OBJ",
            "deactivate OBJ",
            "connect OBJ to OBJ",
            "disconnect OBJ",
            "use OBJ [on OBJ]",
            "look around",
            "examine OBJ",
            "look at OBJ",
            "read OBJ",
            "move OBJ to OBJ",
            "pick up OBJ",
            "pour OBJ into OBJ",
            "mix OBJ",
            "teleport to LOC",
            "focus on OBJ",
            "wait",
            "wait1"
        ]
    
    def get_task_description(self) -> str:
        """Get task description"""
        if self.env:
            return self.env.get_task_description()
        return "No task description available"
    
    def get_gold_action_sequence(self) -> List[str]:
        """Get gold action sequence"""
        if self.env:
            return self.env.get_gold_action_sequence()
        return []
    
    def reset(self) -> Tuple[str, Dict[str, Any]]:
        """Reset environment"""
        if not self.env:
            self._init_environment()
        
        self.current_step = 0
        self.done = False
        
        observation, info = self.env.reset()
        
        # Add task description to observation
        task_desc = self.get_task_description()
        observation = f"Task Description: {task_desc}\n{observation}"
        
        return observation, info
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute one step"""
        if not self.env:
            raise RuntimeError("Environment not initialized")
        
        self.current_step += 1
        
        try:
            observation, reward, done, info = self.env.step(action)
            self.done = done or self.current_step >= self.max_steps
            
            return observation, reward, self.done, info
            
        except Exception as e:
            logger.error(f"Error in environment step: {e}")
            error_obs = f"Error executing action '{action}': {str(e)}"
            return error_obs, 0.0, True, {"error": str(e)}
    
    def close(self):
        """Close environment"""
        if self.env:
            try:
                self.env.close()
                logger.info("SciWorld environment closed")
            except Exception as e:
                logger.error(f"Error closing SciWorld environment: {e}")
            finally:
                self.env = None


def create_sciworld_environment(task_config: Dict[str, Any]) -> SciWorldEnvironment:
    """Create a SciWorld environment instance (following the structure provided)"""
    var_num = task_config.get("var_num", 0)
    task_name = task_config.get("task_name", "task-1-boil-water")
    jar_path = task_config.get("jar_path", "./environments/sciworld.jar")
    simplification = task_config.get("simplification", "easy")
    max_steps = task_config.get("max_steps", 30)
    
    return SciWorldEnvironment(
        jar_path=jar_path,
        task_name=task_name,
        var_num=var_num,
        simplification=simplification,
        max_steps=max_steps
    )
