"""
LLM Agent for trajectory generation
"""
from typing import List, Optional
from ..prompts.trajectory_generation import TrajectoryPrompts
from ..prompts.explorer_random import parse_action_from_response
from ..utils.logger import get_logger
import re

logger = get_logger(__name__)


class LLMAgent:
    """LLM-based agent for environment interaction"""
    
    def __init__(self, client, env_type: str = "webshop"):
        self.client = client
        self.env_type = env_type
        self.prompts = TrajectoryPrompts()
    
    def get_next_action(self, env_discription: str, task_description: str, 
                       query: str, history: List[str], ground_truth: str) -> Optional[str]:
        """Get next action using simple strategy"""
        try:
            prompt = self.prompts.simple_action_prompt(
                env_discription=env_discription,
                task_description=task_description,
                query=query,
                action_history=history,
                ground_truth=ground_truth,
                env_type=self.env_type,
            )
            
            response = self.client.chat_with_retry(prompt)
            action = parse_action_from_response(response)
            
            logger.debug(f"Generated action: {action}")
            return action, response
            
        except Exception as e:
            logger.error(f"Failed to get next action: {e}")
            return None, None
    
    # def get_next_action_with_reflection(self, observation: str, task_description: str,
    #                                   query: str, history: List[str], 
    #                                   previous_observations: List[str], ground_truth: str) -> Optional[str]:
    #     """Get next action using reflection strategy"""
    #     try:
    #         prompt = self.prompts.reflection_action_prompt(
    #             observation=observation,
    #             task_description=task_description,
    #             query=query,
    #             action_history=history,
    #             observation_history=previous_observations,
    #             ground_truth=ground_truth
    #         )
            
    #         response = self.client.chat_with_retry(prompt)
    #         action = parse_action_from_response(response)
            
    #         logger.debug(f"Generated reflection action: {action}")
    #         return action
            
    #     except Exception as e:
    #         logger.error(f"Failed to get reflection action: {e}")
    #         return None
    
    def _extract_action(self, response: str) -> Optional[str]:
        """Extract action from LLM response"""
        if not response:
            return None

        if '<finish>' in response:
            return '<finish>'

        try:
            # Parse action within <action> tags
            start_tag = "<action>"
            end_tag = "</action>"
            start_idx = response.find(start_tag)
            end_idx = response.find(end_tag)
            
            if start_idx != -1 and end_idx != -1:
                action = response[start_idx + len(start_tag):end_idx].strip()
                return action
            else:
                # If no tags found, try the first meaningful line
                lines = response.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('**'):
                        return line
        except Exception as e:
            logger.error(f"Error parsing action from response: {e}")
            return None
            
        return None

