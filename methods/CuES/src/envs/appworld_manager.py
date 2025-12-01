"""AppWorld Environment Manager"""
import os
import sys
from typing import List, Dict, Any, Tuple

# Optional EnvService import
try:
    from EnvService.env_sandbox.env_client import EnvClient  # type: ignore
except Exception:
    EnvClient = None  # type: ignore

from ..agents.trajectory_evaluator import TrajectoryEvaluator
from ..utils.logger import get_logger
import random

logger = get_logger(__name__)


class AppWorldEnvironmentManager:
    """AppWorld environment manager wrapping EnvClient"""
    
    def __init__(self, server_url="http://localhost:8000", env_type="appworld", client=None):
        if EnvClient is None:
            raise ImportError(
                "EnvService is not available. Please add EnvService to PYTHONPATH or run from the repository root where EnvService is present."
            )
        self.client = EnvClient(server_url)  # type: ignore
        self.env_type = env_type
        self.instance_id = None
        self.current_task_id = None
        self.env_id = None
        self.current_observation = ""
        if client:
            self.evaluator = TrajectoryEvaluator(client)
        else:
            self.evaluator = None
        
    def get_available_tasks(self) -> List[str]:
        """Get available task list"""
        return self.client.get_env_profile(self.env_type)
    
    def reset(self, task_id=None, env_id=None, stage="stage1") -> Tuple[str, Dict[str, Any]]:
        """Reset environment"""
        try:
            if self.instance_id:
                self.close()
                
            if env_id is None:
                tasks = self.get_available_tasks()
                if not tasks:
                    raise RuntimeError("No tasks available")
                env_id = random.choice(tasks)
                # env_id = tasks[0]  # default to the first task

            result = self.client.create_instance(self.env_type, env_id)
            
            instance_id = result.get("info", {}).get("instance_id")
            if not instance_id:
                raise RuntimeError("Failed to get instance_id from create_instance response")
                
            self.instance_id = instance_id
            self.current_task_id = env_id
            self.env_id = env_id

            if stage == "stage1":
                current_observation = [
                str(result.get("state", [{}])[0].get("content", "")),
                ]
            elif stage == "stage2":
                current_observation = [
                str(result.get("state", [{}])[0].get("content", "")),
                ]
            elif stage == "stage3":
                current_observation = [
                str(result.get("state", [{}])[0].get("content", ""))
                ]
            
            self.current_observation = "\n".join(current_observation)

            logger.info(f"Created AppWorld instance: {self.instance_id}, task: {task_id}")

            return self.current_observation, {"task_id": task_id, "instance_id": self.instance_id}
            
        except Exception as e:
            logger.error(f"Reset AppWorld environment failed: {e}")
            raise
    
    def step(self, action: str, task_description: str = None, query: str = None) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute action"""
        if not self.instance_id:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        try:
            if action == "<finish>":
                done = True
            else:
                done = False
            result_str = self.client.step(self.instance_id, {"content": action})
            
            observation = result_str['state'][0]['content']
            if observation == '':
                observation = "No valid action. "
            reward = float(result_str['reward'])
            info = {"action": action, "result": result_str}
            
            self.current_observation = observation

            if self.evaluator and task_description and query:
                done = self.evaluator.evaluate_step_completion(observation = observation, task_description = task_description, query = query)

            return observation, reward, done, info
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            raise
    
    def evaluate(self, sparse: bool = False) -> float:
        """Evaluate current state"""
        if not self.instance_id:
            return 0.0
        
        try:
            score = self.client.evaluate(self.instance_id, sparse)
            return score
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return 0.0
    
    def close(self):
        """Close environment"""
        if self.instance_id:
            try:
                self.client.release_instance(self.instance_id)
                logger.info(f"Released AppWorld instance: {self.instance_id}")
            except Exception as e:
                logger.warning(f"Failed to release instance: {e}")
            finally:
                self.instance_id = None
                self.current_task_id = None
