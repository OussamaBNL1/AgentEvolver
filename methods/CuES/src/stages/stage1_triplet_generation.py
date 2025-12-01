"""
Stage 1: Environment interaction to generate triplets
The agent interacts with the AppWorld environment to produce (history, action, observation) triplets.
"""
import uuid
from typing import List, Dict, Any, Optional

# from ..envs.appworld_manager import AppWorldEnvironmentManager
# from ..envs.bfcl_manager import BFCLEnvironmentManager
from ..core.api_client import DashScopeClient, PromptManager
from ..core.memory_manager import MemoryManager
from ..data.models import Triplet
from ..prompts.explorer_random import get_agent_interaction_prompt, parse_action_from_response
from ..prompts.exploration_requirements import format_exploration_requirement
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Stage1TripletGeneration:
    """Stage 1: Triplet generator"""
    
    def __init__(self, client: DashScopeClient, env_config: Dict[str, Any], max_steps: int = 20, storage=None, session_id: Optional[str] = None):
        self.client = client
        self.env_config = env_config
        self.max_steps = max_steps
        self.storage = storage
        # Initialize Memory Manager
        if storage:
            self.memory_manager = MemoryManager(storage, client)
        else:
            self.memory_manager = None
        if session_id:
            self.session_id = session_id
        else:
            self.session_id = None
        self.exploration_requirement = None
    
    def set_exploration_requirement(self, requirement: Optional[str] = None, concepts: Optional[list] = None):
        """Set exploration requirement"""
        if requirement:
            # Format user requirement
            self.exploration_requirement = format_exploration_requirement(self.client, requirement)
        else:
            self.exploration_requirement = None
            
        if concepts:
            import random
            random.shuffle(concepts)
            chosen_concepts = concepts[:12] if len(concepts) > 12 else concepts
            conc = "\nYour exploration should start from " + str(",".join(chosen_concepts)) + ".\n"
            if self.exploration_requirement:
                self.exploration_requirement += conc
            else:
                self.exploration_requirement = conc

        
    def run(self, rollout_num: int = 3, requirement: Optional[str] = None, concepts: Optional[list] = None) -> List[Triplet]:
        """Run Stage 1 to generate triplet data"""
        # Set exploration requirement
        self.set_exploration_requirement(requirement, concepts)
        
        all_triplets = []
        
        for rollout_idx in range(rollout_num):
            logger.info(f"Starting rollout {rollout_idx + 1} of environment interaction")
            triplets = self._single_rollout()
            all_triplets.extend(triplets)
            logger.info(f"Rollout {rollout_idx + 1} finished, generated {len(triplets)} triplets")
        
        logger.info(f"Stage 1 completed, generated {len(all_triplets)} triplets in total")
        return all_triplets
    
    def _single_rollout(self) -> List[Triplet]:
        """Single rollout to interact with environment and generate triplets"""
        # Create environment
        env = self._create_environment()
        if not env:
            return []

        triplets = []
        exploration_memory = None
        env_id = None
        
        # Reset environment
        observation, info = env.reset()
        
        # Ensure observation is string
        initial_obs = str(observation)
        
        # Get environment ID
        env_id = str(env.env_id)
        
        # Load environment exploration memory
        if self.memory_manager and env_id:
            exploration_memory = self.memory_manager.get_env_exploration_memory(env_id)
            logger.info(f"Loaded exploration memory for env {env_id}")
        
        history = [initial_obs]  # do not keep initialization info

        if env_id:
            import os
            log_dir = "./data/env_init"
            os.makedirs(log_dir, exist_ok=True)
            log_file_path = os.path.join(log_dir, f"{env_id}.txt")
            
            if os.path.exists(log_file_path):
                pass
            else:
                # Create new file and write initial_obs
                with open(log_file_path, 'w', encoding='utf-8') as f:
                    f.write(initial_obs)
                logger.info(f"Saved initial_obs to new file: {log_file_path}")
        
        for step in range(self.max_steps):
            # Get agent action (with exploration memory)
            action, response = self._get_agent_action(initial_obs, history, exploration_memory)
            if not action:
                logger.warning(f"Step {step}: no valid action produced")
                break
            
            # Execute action
            observation, reward, done, info = env.step(action)
            
            # Ensure observation string
            obs_str = str(observation) if observation else ""
            
            # Create triplet
            save_history = history[1:]  # do not save initialization info
            triplet = Triplet(
                env_id=env_id,
                history="\n".join(save_history[-3:]),  # keep last 3 history entries
                action=str(action),
                observation=obs_str,
                reward=float(reward) if reward is not None else 0.0,
                done=bool(done),
                exploration_memory=str(exploration_memory) if exploration_memory else None,
                original_action=str(response) if response else None
            )
            triplets.append(triplet)
            
            # Persist triplet and update index
            if self.storage:
                if self.session_id:
                    triplet.session_id = self.session_id
                self.storage.save_triplet(triplet)
            
            # Update history
            history.append(f"Action: {action}")
            history.append(f"Observation: {obs_str}")
            
            if done:
                logger.info(f"Environment finished at step {step}")
                break

        env.close()
        
        logger.info(f"Finished rollout, generated {len(triplets)} triplets")
        # After persisting triplets, refresh and persist exploration memory for this env
        try:
            if self.memory_manager and env_id:
                # Invalidate cache so new triplets are considered
                self.memory_manager.invalidate_env_cache(env_id)
                # Recompute and persist memory summary to data_dir/memories/
                _ = self.memory_manager.get_env_exploration_memory(env_id)
        except Exception as e:
            logger.warning(f"Post-rollout memory update failed for env {env_id}: {e}")
        return triplets
    
    def _create_environment(self):
        """Create environment"""
        # try:
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
            
        # except Exception as e:
        #     logger.error(f"Failed to create AppWorld environment: {e}")
        #     logger.error("Please ensure EnvService is running: python -m env.env_service")
        #     return None
    
    def _get_agent_action(self, initial_obs: str, history: List[str], exploration_memory: str = None) -> str:
        """Get agent action (including exploration memory and requirement)"""
        try:
            # Determine environment type
            env_type = self.env_config.get('envservice', {}).get('env_type') or self.env_config.get('type', 'appworld')
            system_prompt, user_prompt = get_agent_interaction_prompt(
                initial_obs, 
                history, 
                exploration_memory,
                self.exploration_requirement,  # Add exploration requirement
                env_type=env_type,
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.client.chat_with_retry(messages, max_retries=2)
            if not response:
                return "look around", None  # Default action
            
            action = parse_action_from_response(response)
            return action if action else "look around", response
            
        except Exception as e:
            logger.error(f"Failed to get agent action: {e}")
            return "look around", None
