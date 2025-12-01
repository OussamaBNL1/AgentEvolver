"""
Stage 3: Trajectory Generation
Generate execution trajectories from abstracted tasks using LLM agents
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import sys
import os

# Add EnvService to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../EnvService'))

from ..agents.llm_agent import LLMAgent
from ..agents.trajectory_evaluator import TrajectoryEvaluator
from ..data.storage import DataStorage
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrajectoryStep:
    """Single step in trajectory"""
    observation: str
    action: str
    reward: float
    done: bool
    step_number: int
    history: str = ""  # Optional history of the step, useful for debugging or analysis
    response: str = ""  # LLM response if applicable


@dataclass
class Trajectory:
    """Complete trajectory data"""
    task_id: str
    env_id: str
    description: str
    query: str
    ground_truth: str
    steps: List[TrajectoryStep]
    final_reward: float
    success: bool
    reason: str  # Reason for success or failure, can be an LLM message
    total_steps: int
    strategy_used: str  # "simple" or "reflection"
    response: str = ""
    


class Stage3TrajectoryGeneration:
    """Stage 3: Generate trajectories from tasks"""

    def __init__(self, client, env_config: Dict[str, Any], **kwargs):
        self.client = client
        self.env_config = env_config
        self.max_steps = kwargs.get('max_steps_per_trajectory', kwargs.get('max_steps', 20))

        # Determine env_type from config (envservice.env_type preferred, fallback to environment.type)
        env_type = env_config.get('envservice', {}).get('env_type') or env_config.get('type', 'webshop')
        self.env_type = env_type

            # Initialize other components
        self.llm_agent = LLMAgent(client, env_type=env_type)
        self.evaluator = TrajectoryEvaluator(client, env_type=env_type)
        self.storage = DataStorage(kwargs.get('data_dir', './data'))

        # Ensure output directories exist
        self.output_dir = Path("data/trajectories")
        self.trajectories_dir = self.output_dir
        self.failed_dir = self.output_dir / "failed_tasks"

        for dir_path in [self.output_dir, self.trajectories_dir, self.failed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def run(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run trajectory generation"""
        logger.info("Starting Stage 3: Trajectory Generation")
        logger.info(f"Processing {len(tasks)} tasks from stage 2")
        
        # Generate trajectories
        results = {
            "successful_trajectories": [],
            "failed_tasks": [],
            "statistics": {
                "total_tasks": len(tasks),
                "successful": 0,
                "failed": 0,
                "strategy1_success": 0,
                "strategy2_success": 0
            }
        }
        
        for i, task in enumerate(tasks):
            task_id = task.get('task_id', f'task_{i}')
            env_id = task.get('env_id', 'unknown_env')
            logger.info(f"Processing task {i+1}/{len(tasks)}: {task_id}")
            
            # Validate task structure
            if not self._validate_task(task):
                logger.error(f"Invalid task structure for {task_id}")
                results["failed_tasks"].append(task)
                results["statistics"]["failed"] += 1
                self._save_failed_task(task, "invalid_task_structure")
                continue
            
            try:
                trajectory = self._generate_single_trajectory(task, env_id)
                
                if trajectory and trajectory.success:
                    results["successful_trajectories"].append(trajectory)
                    results["statistics"]["successful"] += 1
                    
                    if trajectory.strategy_used == "simple":
                        results["statistics"]["strategy1_success"] += 1
                    else:
                        results["statistics"]["strategy2_success"] += 1
                        
                    self._save_trajectory(trajectory, failed=False)
                    logger.info(f"Task {task_id} completed successfully with {trajectory.strategy_used} strategy")
                elif trajectory:
                    results["failed_tasks"].append(task)
                    results["statistics"]["failed"] += 1
                    self._save_trajectory(trajectory, failed=True)
                    logger.warning(f"Task {task_id} failed to complete")
                    
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, stopping trajectory generation")
                break
            except Exception as e:
                logger.error(f"Error processing task {task_id}: {e}", exc_info=True)
                results["failed_tasks"].append(task)
                results["statistics"]["failed"] += 1
                self._save_failed_task(task, f"exception: {str(e)}")
                
                # If too many consecutive failures, consider stopping
                if results["statistics"]["failed"] > 5 and results["statistics"]["successful"] == 0:
                    logger.error("Too many consecutive failures, stopping execution")
                    break
        
        # Save statistics
        self._save_statistics(results["statistics"])
        
        success_rate = results['statistics']['successful'] / results['statistics']['total_tasks'] if results['statistics']['total_tasks'] > 0 else 0
        logger.info(f"Stage 3 completed: {results['statistics']['successful']}/{results['statistics']['total_tasks']} successful ({success_rate:.2%})")
        
        return results
    
    def _generate_single_trajectory(self, task: Dict[str, Any], env_id:str) -> Optional[Trajectory]:
        """Generate trajectory for a single task"""
        task_id = task.get('task_id', 'unknown')
        description = task.get('description', '')
        query = task.get('query', '')
        ground_truth = task.get('gt', task.get('ground_truth', ''))
        
        # Strategy 1: Simple execution
        trajectory = self._execute_strategy1(task_id, description, query, ground_truth, env_id)
            
        return trajectory
    
    def _create_environment(self):
        """Create environment via EnvService (following Stage 1 pattern)"""
        try:
            envservice_config = self.env_config.get('envservice', {})
            server_url = envservice_config.get('server_url', 'http://localhost:8080')
            env_type = envservice_config.get('env_type', 'appworld')
            
            # Create environment manager
            if env_type == 'appworld':
                from ..envs.appworld_manager import AppWorldEnvironmentManager
                env = AppWorldEnvironmentManager(server_url, env_type)
            elif env_type == 'bfcl':
                from ..envs.bfcl_manager import BFCLEnvironmentManager
                env = BFCLEnvironmentManager(server_url, env_type)
            elif env_type == 'webshop':
                from ..envs.webshop_manager import WebshopEnvironmentManager
                env = WebshopEnvironmentManager(server_url, env_type)
            
            # Test connection
            tasks = env.get_available_tasks()
            if not tasks:
                logger.error("No available AppWorld tasks")
                return None
            
            logger.info(f"Available tasks: {len(tasks)}")
            return env
            
        except Exception as e:
            logger.error(f"Failed to create AppWorld environment: {e}")
            logger.error("Please ensure EnvService is running: python -m env.env_service")
            return None
    
    def _execute_strategy1(self, task_id: str, description: str, query: str, gt: str, env_id: str) -> Optional[Trajectory]:
        """Execute strategy 1: Simple execution"""
        # Create environment (same as Stage 1)
        env = self._create_environment()
        if not env:
            return None

        steps = []
        try:
            # Reset environment (same as Stage 1)
            observation, info = env.reset(env_id=env_id, stage="stage3")
            if env_id != env.env_id:
                print("Environment ID mismatch")
            # AppWorld observation contains task description - ensure string (same as Stage 1)
            initial_obs = str(observation)
            # Find the last occurrence of "\nTask:" and remove everything after it
            last_task_index = initial_obs.rfind('\nTask:')
            if last_task_index != -1:
                initial_obs = initial_obs[:last_task_index-1]
            # initial_obs = str(observation)
            
            history = [initial_obs]
            
            for step_num in range(self.max_steps):
                try:
                    # Get action from LLM
                    current_obs = initial_obs if step_num == 0 else observation
                    action, response = self.llm_agent.get_next_action(
                        env_discription=initial_obs,
                        task_description=description,
                        query=query,
                        history=[f"Action: {s.action} -> Observation: {s.observation}"for s in steps],
                        ground_truth=gt
                    )
                    
                    if not action:
                        logger.warning(f"No action generated at step {step_num}")
                        break
                    
                    # Execute action (same as Stage 1)
                    if action == "<finish>" or "<finish>" in action:
                        observation, reward, done, info = "<finish>", 0.0, True, {}
                    else:
                        observation, reward, done, info = env.step(action, task_description=description, query=query,)
                    
                    # Ensure observation text string (same as Stage 1)
                    obs_text = str(observation) if observation else ""
                    
                    # Record step
                    step = TrajectoryStep(
                        history=str(current_obs),
                        observation=obs_text,
                        action=str(action),
                        reward=float(reward) if reward is not None else 0.0,
                        done=bool(done),
                        step_number=step_num,
                        response=str(response) if response else ""
                    )
                    steps.append(step)
                    
                    # Update history (same as Stage 1)
                    history.append(f"Action: {action} -> Observation: {obs_text}")
                    if len(history) > 10:  # Keep a reasonable history length
                        history = history[-10:]
                    
                    logger.info(f"Step {step_num}: {action} -> reward: {reward:.2f}")
                    
                    if done:
                        logger.info(f"Environment finished in {step_num + 1} steps")
                        break
                        
                except Exception as step_error:
                    logger.error(f"Action execution failed at step {step_num}: {step_error}")
                    break
            
            if not steps:
                logger.error(f"No valid steps generated for task {task_id}")
                return None
            
            # Evaluate success using LLM
            try:
                # Use last reward as final reward
                final_reward = steps[-1].reward if steps else 0.0
            except Exception as eval_error:
                logger.warning(f"Environment evaluation failed for task {task_id}: {eval_error}")
                final_reward = 0.0
            
            try:
                success, response = self.evaluator.evaluate_trajectory_success(
                    steps=steps,
                    task_description=description,
                    query=query,
                    ground_truth=gt,
                    final_observation=obs_text
                )
            except Exception as eval_error:
                logger.warning(f"LLM evaluation failed for task {task_id}: {eval_error}")
                success = False
                response = ""
            
            trajectory = Trajectory(
                env_id=env_id,
                task_id=task_id,
                description=description,
                query=query,
                ground_truth=gt,
                steps=steps,
                final_reward=final_reward,
                success=success,
                reason=response,
                total_steps=len(steps),
                strategy_used="simple"
            )
            
            return trajectory
            
        except Exception as e:
            logger.error(f"Strategy 1 execution failed for task {task_id}: {e}")
            return None
        finally:
            # Always close environment (same as Stage 1)
            env.close()
    
    def _validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate task structure"""
        required_fields = ['task_id', 'description', 'query']
        
        for field in required_fields:
            if field not in task or not task[field]:
                logger.error(f"Missing or empty required field: {field}")
                return False
        
        # Check if task_id is valid
        task_id = task['task_id']
        if not isinstance(task_id, str) or len(task_id.strip()) == 0:
            logger.error(f"Invalid task_id: {task_id}")
            return False
        
        return True
    
    def _save_failed_task(self, task: Dict[str, Any], reason: str):
        """Save failed task"""
        try:
            task_id = task.get('task_id', 'unknown')
            filename = f"failed_{task_id}.json"
            filepath = self.failed_dir / filename
            
            from datetime import datetime
            import copy
            
            # Create a deep copy of the task to avoid modifying the original object
            task_copy = copy.deepcopy(task)
            
            # Handle potential datetime objects in the task
            def convert_datetime(obj):
                if isinstance(obj, dict):
                    return {k: convert_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime(item) for item in obj]
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                else:
                    return obj
            
            # Convert all datetime objects in the task
            task_copy = convert_datetime(task_copy)
            
            failed_data = {
                "task": task_copy,
                "failure_reason": reason,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(failed_data, f, ensure_ascii=False, indent=2)
                    
            logger.debug(f"Saved failed task: {filename}")
        except Exception as e:
            logger.error(f"Failed to save failed task {task.get('task_id', 'unknown')}: {e}")

    def _save_trajectory(self, trajectory: Trajectory, failed: bool = False):
        """Save successful trajectory"""
        try:
            filename = f"trajectory_{trajectory.task_id}.json"
            if failed:
                filepath = self.failed_dir / filename
            else:
                filepath = self.trajectories_dir / filename
            
            # Convert trajectory to dict, handling dataclass serialization
            trajectory_dict = asdict(trajectory)
            
            # Convert steps to messages format
            messages = [{
                        "role": "user",
                        "content": trajectory.steps[0].history if trajectory.steps else ""
            }]
            for step in trajectory.steps:
                # Add assistant message (action)
                if step.action:
                    messages.append({
                        "role": "assistant", 
                        "content": step.response
                    })

                # Add user message (observation)
                # print(f"Step {step.step_number}: {step}")
                if step.observation:
                    messages.append({
                        "role": "user",
                        "content": step.observation
                    })
                
            # Replace steps with messages while preserving other fields
            trajectory_dict["messages"] = messages
            # Keep original steps for reference if needed
            # trajectory_dict["original_steps"] = trajectory_dict["steps"]
            del trajectory_dict["steps"]
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(trajectory_dict, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Saved trajectory: {filename}")
        except Exception as e:
            logger.error(f"Failed to save trajectory: {e}")
    
    def _save_statistics(self, statistics: Dict[str, Any]):
        """Save execution statistics"""
        try:
            stats_file = self.output_dir / "statistics.json"
            
            from datetime import datetime
            statistics["timestamp"] = datetime.now().isoformat()
            statistics["success_rate"] = statistics["successful"] / statistics["total_tasks"] if statistics["total_tasks"] > 0 else 0
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(statistics, f, ensure_ascii=False, indent=2)
                
            logger.debug("Saved execution statistics")
        except Exception as e:
            logger.error(f"Failed to save statistics: {e}")
