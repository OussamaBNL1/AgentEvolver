"""
Memory management module
Manage environment exploration history, avoid repeated exploration, support Stage 1 and Stage 2
"""
import json
from typing import List, Dict, Any, Optional
from ..data.models import Triplet, Task
from ..data.storage import DataStorage
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MemoryManager:
    """Environment memory manager"""
    
    def __init__(self, storage: DataStorage, client):
        self.storage = storage
        self.client = client
        self.memory_cache = {}  # Cache of generated memory
        self.processed_triplet_counts = {}  # Track processed triplet count per environment
        # Try loading from storage
        # stored_memory = self.storage.load_memory(env_id, "exploration")
        # if stored_memory is not None:
        #     self.memory_cache[env_id] = stored_memory
    
    def get_env_exploration_memory(self, env_id: str) -> str:
        """Get environment exploration memory summary"""
        # Get historical triplets for this environment
        historical_triplets = self.storage.get_triplets_by_env_id(env_id)
        
        if not historical_triplets:
            memory = ""
            self.memory_cache[env_id] = memory
            self.processed_triplet_counts[env_id] = 0
            return memory
        
        current_count = len(historical_triplets)
        
        # Check whether cache needs updating
        if (env_id in self.memory_cache and 
            env_id in self.processed_triplet_counts and 
            current_count <= self.processed_triplet_counts[env_id]):
            # No new triplets, return cache directly
            return self.memory_cache[env_id]
        
        # There are new triplets; compute which are new
        previous_count = self.processed_triplet_counts.get(env_id, 0)
        new_triplets = historical_triplets[previous_count:] if previous_count > 0 else historical_triplets
        
        # Generate or update memory summary
        if env_id not in self.memory_cache or previous_count == 0:
            # First-time full summary
            memory = self._generate_exploration_summary(historical_triplets)
        else:
            # Incremental update
            previous_memory = self.memory_cache[env_id]
            memory = self._update_exploration_summary(previous_memory, new_triplets)
        
        # Update cache
        self.memory_cache[env_id] = memory
        self.processed_triplet_counts[env_id] = current_count

        # Persist updated memory
        self.storage.save_memory(env_id, "exploration", memory)
        return memory
    
    def get_env_task_memory(self, env_id: str) -> str:
        """Get environment task abstraction memory summary (for Stage 2)"""
        # Get historical tasks for this environment
        historical_tasks = self.storage.get_tasks_by_env_id(env_id)
        
        if not historical_tasks:
            return "No previous tasks have been abstracted for this environment."
        
        # Generate task memory summary
        return self._generate_task_summary(historical_tasks)
    
    def _generate_exploration_summary(self, triplets: List[Triplet]) -> str:
        """Generate exploration memory summary using LLM"""
        try:
            # Extract key info
            actions_and_results = []
            for triplet in triplets[-50:]:  # Limit to recent 50 triplets to avoid too many tokens
                action_result = {
                    'action': triplet.action,
                    'observation': triplet.observation[:200] if triplet.observation else '',  # Truncate long observations
                    'reward': triplet.reward,
                    'success': triplet.done
                }
                actions_and_results.append(action_result)
            
            # Build LLM prompt
            system_prompt = """
You are an exploration memory summarizer. Your task is to convert a user's natural language description of past exploration activity into a structured and concise memory entry for an AI agent.

The memory should:
1. Summarize what was explored
2. Describe key actions or strategies taken during exploration
3. Note outcomes, consequences, or patterns discovered
4. Mention anything the agent deliberately avoided or excluded
5. Be factual and grounded in what was observed (no assumptions)

Format the memory as a clear paragraph that reflects past exploration behavior and its consequences. Avoid speculation and keep it grounded in observed results.

Keep the language concise and use past tense. This memory will be stored and referenced by other AI agents to guide future behavior.
"""
            
            user_prompt = f"""
Here are the previous exploration actions and results in this environment:

{self._format_actions_for_llm(actions_and_results)}

Please Keep the summary under 500 words and convert this into a structured exploration memory that summarizes what was done, what was discovered, and any constraints or patterns observed. This will help future agents make better decisions based on prior experience.
"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.client.chat_with_retry(messages, max_retries=2, max_tokens=8192)
            if response:
                return response.strip()
            else:
                return self._generate_fallback_summary(actions_and_results)
                
        except Exception as e:
            logger.warning(f"Failed to generate LLM exploration summary: {e}")
            return self._generate_fallback_summary(actions_and_results)
    
    def _update_exploration_summary(self, previous_memory: str, new_triplets: List[Triplet]) -> str:
        """Incrementally update exploration memory summary"""
        try:
            # Extract key info from new triplets
            new_actions = []
            for triplet in new_triplets[-50:]:  # Only process the most recent 50 new triplets
                action_result = {
                    'action': triplet.action,
                    'observation': triplet.observation[:150] if triplet.observation else '',
                    'reward': triplet.reward,
                    'success': triplet.done
                }
                new_actions.append(action_result)
            
            # Build LLM prompt
            system_prompt = """
You are an exploration memory manager. You need to update an existing exploration memory with new exploration actions and results.

The existing memory summarizes what's been explored so far. The new actions represent additional exploration.

Update the memory to:
1. Include important new findings or patterns
2. Remove outdated or less relevant information if necessary
3. Maintain focus on guiding future exploration efficiently
4. Keep the summary under 500 words
"""
            
            user_prompt = f"""
EXISTING EXPLORATION MEMORY:
{previous_memory}

NEW EXPLORATION ACTIONS:
{self._format_actions_for_llm(new_actions)}

Please update the exploration memory to incorporate these new actions and results. Focus on maintaining a comprehensive but concise guide for future exploration that avoids repetition.
"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.client.chat_with_retry(messages, max_retries=2, max_tokens=8192)
            if response:
                return response.strip()
            else:
                # If update fails, append a simple addendum to the previous memory
                return self._append_fallback_update(previous_memory, new_actions)
                
        except Exception as e:
            logger.warning(f"Failed to update exploration summary: {e}")
            return self._append_fallback_update(previous_memory, new_actions)
    
    def _append_fallback_update(self, previous_memory: str, new_actions: List[Dict]) -> str:
        """Append a simple addendum when LLM update fails"""
        if not new_actions:
            return previous_memory
            
        successful_actions = [item['action'] for item in new_actions if item['reward'] > 0]
        actions_tried = [item['action'] for item in new_actions]
        
        update = "\n\nRecent update: "
        update += f"Tried {len(actions_tried)} new actions. "
        
        if successful_actions:
            update += f"Successful actions include: {', '.join(set(successful_actions[:3]))}. "
        else:
            update += "No new successful actions. "
            
        update += f"Recent attempts: {', '.join(set(actions_tried[-5:]))}."
        
        # Ensure total length does not exceed a reasonable limit
        if len(previous_memory + update) > 2000:
            # Truncate old memory to make space
            return previous_memory[:1700] + "...(truncated)..." + update
        else:
            return previous_memory + update
    
    def _generate_task_summary(self, tasks: List[Task]) -> str:
        """Generate task memory summary"""
        if not tasks:
            return "No previous tasks abstracted."
        
        task_descriptions = [task.description for task in tasks[-10:]]
        task_queries = [task.query for task in tasks[-10:]]
        
        summary = f"Previously abstracted {len(tasks)} tasks. "
        summary += f"Task types: {', '.join(task_descriptions)}"
        
        return summary
    
    def _format_actions_for_llm(self, actions_and_results: List[Dict]) -> str:
        """Format action results for LLM processing"""
        formatted = []
        for i, item in enumerate(actions_and_results):
            formatted.append(f"{i+1}. Action: {item['action']}")
            formatted.append(f"   Result: {item['observation']}")
            # formatted.append(f"   Reward: {item['reward']}, Success: {item['success']}")
            formatted.append("")
        
        return "\n".join(formatted)
    
    def _generate_fallback_summary(self, actions_and_results: List[Dict]) -> str:
        """Generate fallback summary (when LLM call fails)"""
        if not actions_and_results:
            return "No previous exploration history available."
        
        actions = [item['action'] for item in actions_and_results]
        successful_actions = [item['action'] for item in actions_and_results if item['reward'] > 0]
        
        summary = f"Previous exploration attempted {len(actions)} actions. "
        if successful_actions:
            summary += f"Successful actions include: {', '.join(set(successful_actions[:5]))}. "
        
        summary += f"Recently tried actions: {', '.join(actions[-5:])}."
        return summary
    
    def clear_cache(self):
        """Clear cache"""
        self.memory_cache.clear()
        self.processed_triplet_counts.clear()
    
    def invalidate_env_cache(self, env_id: str):
        """Invalidate cache for a specific environment"""
        if env_id in self.memory_cache:
            del self.memory_cache[env_id]
        if env_id in self.processed_triplet_counts:
            del self.processed_triplet_counts[env_id]
