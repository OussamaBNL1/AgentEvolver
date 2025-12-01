"""
Trajectory Evaluator using LLM
"""
from typing import List
from ..prompts.trajectory_evaluation import EvaluationPrompts
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TrajectoryEvaluator:
    """Evaluate trajectory success using LLM"""
    
    def __init__(self, client, env_type: str = "webshop"):
        self.client = client
        self.prompts = EvaluationPrompts()
        self.env_type = env_type
    
    def evaluate_trajectory_success(self, steps: List, task_description: str, query: str,
                                  ground_truth: str, final_observation: str) -> bool:
        """Evaluate if trajectory completed the task successfully"""
        try:
            # Create trajectory summary
            trajectory_summary = self._create_trajectory_summary(steps)
            
            # Generate evaluation prompt
            prompt = self.prompts.success_evaluation_prompt(
                task_description=task_description,
                query=query,
                ground_truth=ground_truth,
                trajectory_summary=trajectory_summary,
                final_observation=final_observation,
                env_type=self.env_type,
            )
            
            # Get LLM evaluation
            response = self.client.chat_with_retry(prompt)
            
            # Parse evaluation result
            success = self._parse_evaluation_response(response)
            
            logger.debug(f"Trajectory evaluation result: {success}")
            return success, response
            
        except Exception as e:
            logger.error(f"Failed to evaluate trajectory: {e}")
            return False, ""
    
    def _create_trajectory_summary(self, steps: List) -> str:
        """Create summary of trajectory steps"""
        summary_lines = []
        
        for i, step in enumerate(steps):
            summary_lines.append(f"Step {i+1}: {step.action}")
            if step.observation:
                # Keep observation short
                obs_short = step.observation[:200] + "..." if len(step.observation) > 200 else step.observation
                summary_lines.append(f"  Result: {obs_short}")
        
        return "\n".join(summary_lines)
    
    def _parse_evaluation_response(self, response: str) -> bool:
        """Parse LLM evaluation response"""
        if not response:
            return False
        
        response_lower = response.lower()
        
        # Look for explicit success/failure indicators
        if 'success: true' in response_lower or 'successful: true' in response_lower or 'success: **true**' in response_lower:
            return True
        else:
            return False
    
    def evaluate_step_completion(self, observation: str, task_description: str, query: str = None) -> bool:
        """
        Evaluate if a single step's observation indicates task completion
        
        Args:
            observation: The observation from a single step
            task_description: Description of the task
            query: Optional query context
        
        Returns:
            bool: True if task appears to be completed, False otherwise
        """
        try:
            # Generate step completion evaluation prompt
            prompt = self.prompts.step_completion_prompt(
                task_description=task_description,
                query=query or "",
                observation=observation,
                env_type=self.env_type,
            )
            
            # Get LLM evaluation
            response = self.client.chat_with_retry(prompt)
            
            # Parse evaluation result
            is_completed = self._parse_completion_response(response)
            
            logger.debug(f"Step completion evaluation result: {is_completed}")
            logger.debug(f"Observation: {observation[:100]}...")
            
            return is_completed
            
        except Exception as e:
            logger.error(f"Failed to evaluate step completion: {e}")
            return False

    def _parse_completion_response(self, response: str) -> bool:
        """
        Parse LLM response for step completion evaluation
        """
        if not response:
            return False
        
        response_lower = response.lower()
        
        # Look for explicit completion indicators
        if 'completed: true' in response_lower or 'task_completed: true' in response_lower:
            return True
        elif 'completed: false' in response_lower or 'task_completed: false' in response_lower:
            return False
        
        # Look for completion keywords
        completion_keywords = [
            'completed', 'finished', 'done', 'accomplished', 
            'task complete', 'successfully completed', 'goal achieved'
        ]
        
        incomplete_keywords = [
            'not completed', 'incomplete', 'in progress', 'continue',
            'not finished', 'partial', 'ongoing', 'not done'
        ]
        
        completion_count = sum(1 for keyword in completion_keywords if keyword in response_lower)
        incomplete_count = sum(1 for keyword in incomplete_keywords if keyword in response_lower)
        
        # Default to incomplete if uncertain
        return completion_count > incomplete_count and completion_count > 0
