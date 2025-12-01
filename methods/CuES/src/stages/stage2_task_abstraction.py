"""
Stage 2: Abstract tasks from triplets
Derive concrete task goals and queries from the agent interaction triplet sequence
"""
from typing import List, Dict, Any, Optional
from ..core.api_client import DashScopeClient
from ..data.models import Triplet, Task
from ..prompts.judge_task_extract import get_task_extraction_prompt, parse_tasks_from_response
from ..utils.logger import get_logger
from ..core.memory_manager import MemoryManager
logger = get_logger(__name__)


class Stage2TaskAbstraction:
    """Stage 2: Task abstraction component"""
    
    def __init__(self, client: DashScopeClient, env_config: Dict[str, Any], min_confidence: float = 0.5, storage=None, session_id: Optional[str] = None):
        self.client = client
        self.min_confidence = min_confidence
        self.storage = storage
        self.session_id = session_id if session_id else None
        self.env_config = env_config
        # Initialize Memory Manager
        if storage:
            self.memory_manager = MemoryManager(storage, client)
        else:
            self.memory_manager = None
    
    def run(self, triplets: List[Triplet], batch_size: int = 10) -> List[Task]:
        """Run Stage 2 to abstract tasks from triplets"""
        if not triplets:
            logger.warning("No triplet data for task abstraction")
            return []
        
        logger.info(f"Starting task abstraction, processing {len(triplets)} triplets")
        
        # Group triplets by env_id
        env_groups = {}
        for triplet in triplets:
            env_id = triplet.env_id
            if env_id not in env_groups:
                env_groups[env_id] = []
            env_groups[env_id].append(triplet)  # Always append
        
        
        # Batch process triplets, ensuring same env_id triplets are in the same batch
        all_tasks = []
        batch_num = 1
        
        for env_id, env_triplets in env_groups.items():
            # Split triplets of the same env_id into batches
            for i in range(0, len(env_triplets), batch_size):
                batch_triplets = env_triplets[i:i + batch_size]
                batch_tasks = self._extract_tasks_from_batch(batch_triplets, batch_num, env_id)
                batch_tasks = self._filter_and_deduplicate_tasks(batch_tasks)
                # Save tasks
                # for task in batch_tasks:
                #     task.session_id = self.session_id
                #     self.storage.save_task(task)
                all_tasks.extend(batch_tasks)
                batch_num += 1
        
        # Dedup and filter
        # filtered_tasks = self._filter_and_deduplicate_tasks(all_tasks)
        
        logger.info(f"Stage 2 completed, abstracted {len(all_tasks)} tasks")
        return all_tasks
    
    def _create_environment(self):
        """Create environment"""
        try:
            envservice_config = self.env_config.get('envservice', {})
            # Align default with config.yaml (8080) and allow override via config
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
            
            # Optionally test connection
            # tasks = env.get_available_tasks()
            # if not tasks:
            #     logger.error("No available AppWorld tasks")
            #     return None
            
            # logger.info(f"Available tasks: {len(tasks)}")
            return env
            
        except Exception as e:
            logger.error(f"Failed to create AppWorld environment: {e}")
            logger.error("Please ensure EnvService is running: python -m env.env_service")
            return None
    
    def _extract_tasks_from_batch(self, triplets: List[Triplet], batch_num: int, env_id: Optional[str]) -> List[Task]:
        """Abstract tasks from a batch of triplets"""
        logger.info(f"Processing batch {batch_num} with {len(triplets)} triplets")
        
        try:
            # Convert triplets to dicts for prompt
            triplet_dicts = []
            for triplet in triplets:
                triplet_dicts.append({
                    'history': triplet.history,
                    'action': triplet.action,
                    'observation': triplet.observation,
                    'reward': triplet.reward
                })

            # Get environment description
            if env_id:
                # Read environment description from log file
                log_file_path = f"./data/env_init/{env_id}.txt"
                try:
                    with open(log_file_path, 'r', encoding='utf-8') as f:
                        env_discription = f.read().strip()
                    logger.info(f"Read environment description from {log_file_path}")
                except FileNotFoundError:
                    print(f"Environment description file not found: {log_file_path}")
                    env_discription = ""
                except Exception as e:
                    print(f"Failed to read environment description: {e}")
                    env_discription = ""
            else:
                print(f"Failed to read environment description")
                env = self._create_environment()
                observation, info = env.reset(env_id=env_id, stage="stage2")
                env_discription = str(observation['content'])
                env.close()

            # Get task abstraction memory
            if self.memory_manager and env_id:
                exploration_memory = self.memory_manager.get_env_task_memory(env_id)
                logger.info(f"Get task memory for environment {env_id}")
            else:
                exploration_memory = None

            # Build prompts for task abstraction
            # Determine environment type for prompt formatting
            env_type_for_prompt = self.env_config.get('envservice', {}).get('env_type') or self.env_config.get('type', 'webshop')
            system_prompt, user_prompt = get_task_extraction_prompt(triplet_dicts, exploration_memory, env_discription, env_type=env_type_for_prompt)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call LLM API
            response = self.client.chat_with_retry(messages, max_retries=3)
            if not response:
                return []
            
            # Parse tasks
            task_infos = parse_tasks_from_response(response)
            
            # Create Task objects
            tasks = []
            for i, task_info in enumerate(task_infos):
                if task_info.get('confidence', 0) >= self.min_confidence:
                    task = Task(
                        env_id=triplets[0].env_id,
                        description=task_info['description'],
                        query=task_info['query'],
                        confidence=task_info['confidence'],
                        gt=task_info['gt'],
                        source_triplets=[triplet.triplet_id for triplet in triplets]  # Record source triplet IDs
                    )
                    tasks.append(task)
            
            # Save tasks
            for task in tasks:
                task.session_id = self.session_id
                self.storage.save_task(task)

            logger.info(f"Batch {batch_num} produced {len(tasks)} valid tasks")
            return tasks
            
        except Exception as e:
            logger.error(f"Task abstraction failed for batch {batch_num}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _filter_and_deduplicate_tasks(self, tasks: List[Task]) -> List[Task]:
        """Filter and deduplicate tasks"""
        if not tasks:
            return []
        
        # Sort by confidence descending
        tasks.sort(key=lambda x: x.confidence, reverse=True)
        
        # Simple deduplication: based on query text similarity
        unique_tasks = []
        seen_queries = set()
        
        for i, task in enumerate(tasks):

            # Normalize query for dedup comparison
            normalized_query = task.query.lower().strip()
            
            # Check if similar query exists
            is_duplicate = False
            for seen_query in seen_queries:
                if self._queries_are_similar(normalized_query, seen_query):
                    is_duplicate = True
                    break
            
            if task.gt == "":  # Filter out if ground truth is missing
                is_duplicate = True
            
            if not is_duplicate:
                unique_tasks.append(task)
                seen_queries.add(normalized_query)
        return unique_tasks
    
    def _queries_are_similar(self, query1: str, query2: str, threshold: float = 0.8) -> bool:
        """Simple query similarity check"""
        # Simple similarity metric based on lexical overlap
        words1 = set(query1.split())
        words2 = set(query2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union) if union else 0
        return similarity >= threshold
    
    def get_task_statistics(self, tasks: List[Task]) -> Dict[str, Any]:
        """Get task statistics"""
        if not tasks:
            return {}
        
        confidences = [task.confidence for task in tasks]
        
        return {
            'total_tasks': len(tasks),
            'avg_confidence': sum(confidences) / len(confidences),
            'max_confidence': max(confidences),
            'min_confidence': min(confidences),
            'high_confidence_tasks': len([t for t in tasks if t.confidence >= 0.8]),
            'medium_confidence_tasks': len([t for t in tasks if 0.5 <= t.confidence < 0.8]),
            'low_confidence_tasks': len([t for t in tasks if t.confidence < 0.5])
        }