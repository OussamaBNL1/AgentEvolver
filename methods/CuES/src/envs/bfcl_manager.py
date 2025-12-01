"""BFCL Environment Manager"""
import os
import sys
from typing import List, Dict, Any, Tuple
import json
import random

# Optional import: EnvService client is only required when running BFCL env
try:
    from EnvService.env_sandbox.env_client import EnvClient  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    EnvClient = None  # type: ignore

from ..agents.trajectory_evaluator import TrajectoryEvaluator
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BFCLEnvironmentManager:
    """BFCL environment manager wrapping EnvClient"""
    
    def __init__(self, server_url="http://localhost:8000", env_type="bfcl", client=None):
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
        self.current_tools = []
        self.tool_call_history = []
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
                env_id = random.choice(tasks[:200])
                # env_id = tasks[0]

            result = self.client.create_instance(env_type=self.env_type, task_id=env_id)
            
            instance_id = result.get("info", {}).get("instance_id")
            if not instance_id:
                raise RuntimeError("Failed to get instance_id from create_instance response")
                
            self.instance_id = instance_id
            self.current_task_id = env_id
            self.env_id = env_id
            
            # BFCL specific initialization
            if stage == "stage1":
                current_observation = [
                str(result.get("state", [{}])[0].get("content", ""))
                ]
            elif stage == "stage2":
                current_observation = [
                "",       
                str(result.get("state", [{}])[0].get("content", "")),
                ]
            elif stage == "stage3":
                current_observation = [
                str(result.get("state", [{}])[0].get("content", ""))
                ]
            self.current_observation = "\n".join(current_observation)
            
            self.current_tools = result.get("info", {}).get("tools", [])
            self.tool_call_history = []
            
            logger.info(f"Created BFCL instance: {self.instance_id}, task: {task_id}")
            
            return self.current_observation, {
                "task_id": task_id, 
                "instance_id": self.instance_id,
                "tools": self.current_tools
            }
            
        except Exception as e:
            logger.error(f"Reset BFCL environment failed: {e}")
            raise
    
    def step(self, action: str, task_description: str = None, query: str = None) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute action (tool call or message)"""
        if not self.instance_id:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # try:
        done = False
        
        # Handle different action formats
        if isinstance(action, str):
            if "<finish>" in action:
                done = True
                # Send finish action
                action_dict = {"role": "assistant", "content": action}
                result_str = "<finish>"
                observation = {"content": result_str}
                reward = 0.0
                info = {
                    "action": action_dict, 
                    "result": result_str,
                    "tool_call_history": self.tool_call_history,
                    "tools": self.current_tools,
                    "is_terminated": done
                }
                return observation, reward, done, info
            elif action.startswith("{") and "tool_calls" in action:
                # Parse JSON action with tool_calls
                try:
                    action_dict = json.loads(action)
                    if "tool_calls" in action_dict:
                        self._record_tool_calls(action_dict["tool_calls"])
                except json.JSONDecodeError:
                    action_dict = {"role": "assistant", "content": action}
            else:
                # Plain text action
                action_dict = {"role": "assistant", "content": action}
        elif isinstance(action, dict):
            action_dict = action
            if "tool_calls" in action_dict:
                self._record_tool_calls(action_dict["tool_calls"])
        else:
            action_dict = {"role": "assistant", "content": str(action)}
        
        result_str = self.client.step(self.instance_id, action_dict)
        
        # Extract information from response
        try:
            role = result_str.get('state', {})[0].get('role', "tool")
            if role == "user":
                observation = "Connection timeout or No valid action."
            elif role == "tool":
                observation = result_str.get('state', {})[0].get('content', [{}])
            else:
                observation = 'Connection timeout or No valid action.'
                print(f"result_str: {result_str}")
        except Exception as e:
            print(f"Failed to extract observation: {e}")
            print(f"Result string: {result_str}")
            observation = ""
        observation = {"content": str(observation)}
        reward = float(result_str.get('reward', 0.0))
        done = result_str.get('is_terminated', done)
        
        info = {
            "action": action_dict, 
            "result": result_str,
            "tool_call_history": self.tool_call_history,
            "tools": self.current_tools,
            "is_terminated": done
        }
        
        self.current_observation = observation

        # Use evaluator to check completion if available
        if self.evaluator and task_description and query:
            evaluator_done = self.evaluator.evaluate_step_completion(
                observation=observation, 
                task_description=task_description, 
                query=query
            )
            done = done or evaluator_done

        return observation, reward, done, info
            
        # except Exception as e:
        #     logger.error(f"Tool call execution failed: {e}")
        #     raise
    
    def _record_tool_calls(self, tool_calls: List[Dict[str, Any]]):
        """Record tool calls in history"""
        for tool_call in tool_calls:
            self.tool_call_history.append({
                "id": tool_call.get("id", ""),
                "name": tool_call.get("name", ""),
                "arguments": tool_call.get("arguments", "{}"),
                "type": tool_call.get("type", "tool")
            })
    
    def create_tool_call(self, name: str, arguments: Dict[str, Any], call_id: str = None, index: int = 0) -> Dict[str, Any]:
        """Create a tool call dictionary"""
        if call_id is None:
            call_id = f"{name}_{len(self.tool_call_history)}_{index}"
        
        return {
            "id": call_id,
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
            "type": "tool",
            "index": index
        }
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get available tools for current task"""
        return self.current_tools
    
    def validate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """Validate if tool call is correct"""
        for tool in self.current_tools:
            if tool.get("function", {}).get("name") == tool_name:
                # Basic validation - check required parameters
                parameters = tool.get("function", {}).get("parameters", {})
                required_params = parameters.get("required", [])
                
                for param in required_params:
                    if param not in arguments:
                        logger.warning(f"Missing required parameter '{param}' for tool '{tool_name}'")
                        return False
                return True
        
        logger.warning(f"Tool '{tool_name}' not found in available tools")
        return False
    
    def get_tools_info(self) -> Dict[str, Any]:
        """Get tools information"""
        if not self.instance_id:
            return {}
        
        try:
            tools_info = self.client.get_tools_info(self.instance_id)
            return tools_info
        except Exception as e:
            logger.warning(f"Failed to get tools info: {e}")
            return {}
    
    def evaluate(self, sparse: bool = False) -> float:
        """Evaluate current state"""
        if not self.instance_id:
            return 0.0
        
        try:
            score = self.client.evaluate(self.instance_id, params={"sparse": sparse})
            return score
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return 0.0
    
    def get_execution_result(self) -> Dict[str, Any]:
        """Get tool execution result"""
        return {
            "tool_calls": self.tool_call_history,
            "observation": self.current_observation,
            "task_id": self.current_task_id,
            "tools": self.current_tools
        }
    
    def close(self):
        """Close environment"""
        if self.instance_id:
            try:
                self.client.release_instance(self.instance_id)
                logger.info(f"Released BFCL instance: {self.instance_id}")
            except Exception as e:
                logger.warning(f"Failed to release instance: {e}")
            finally:
                self.instance_id = None
                self.current_task_id = None
                self.env_id = None
                self.current_tools = []
                self.tool_call_history = []


# Example usage
def main():
    """BFCL Environment Manager usage example"""
    manager = BFCLEnvironmentManager()
    
    try:
        # Get available tasks
        tasks = manager.get_available_tasks()
        print(f"Available tasks: {tasks[:3]}...")  # Show the first 3 tasks
        
        # Reset environment
        if tasks:
            observation, info = manager.reset(env_id=tasks[0])
            print(f"Initial observation: {observation}")
            print(f"Available tools: {len(info.get('tools', []))}")
            
            # Create tool call
            tool_call = manager.create_tool_call(
                name="cd",
                arguments={"folder": "document"},
                index=0
            )
            
            # Execute tool call
            action = {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call]
            }
            
            obs, reward, done, step_info = manager.step(action)
            print(f"Step result - Reward: {reward}, Done: {done}")
            print(f"New observation: {obs}")
            
            # Evaluate
            score = manager.evaluate(sparse=True)
            print(f"Evaluation score: {score}")
            
    except Exception as e:
        logger.error(f"Example failed: {e}")
    finally:
        manager.close()


if __name__ == "__main__":
    main()