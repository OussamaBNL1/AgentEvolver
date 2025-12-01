import json
import jsonlines
import os
from typing import List, Dict, Set, Optional
from datetime import datetime
from pathlib import Path
from .models import Triplet, Task, Session
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataStorage:
    """Lightweight data storage manager"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / "triplets").mkdir(exist_ok=True)
        (self.base_dir / "tasks").mkdir(exist_ok=True)
        (self.base_dir / "sessions").mkdir(exist_ok=True)
        (self.base_dir / "memories").mkdir(exist_ok=True)  # Memory storage directory
        # env_id indices
        self.env_triplet_index: Dict[str, Set[Path]] = {}
        self.env_task_index: Dict[str, Set[Path]] = {}
        self._build_indices()
    
    def _build_indices(self):
        """Build indices from env_id to file paths"""
        # Triplet index
        triplets_dir = self.base_dir / "triplets"
        if triplets_dir.exists():
            for file_path in triplets_dir.glob("*.jsonl"):
                self._index_triplet_file(file_path)
        
        # Task index
        tasks_dir = self.base_dir / "tasks"
        if tasks_dir.exists():
            for file_path in tasks_dir.glob("*.json"):
                self._index_task_file(file_path)
            # Also check jsonl files
            for file_path in tasks_dir.glob("*.jsonl"):
                self._index_task_jsonl_file(file_path)
    
    def _index_triplet_file(self, file_path: Path):
        """Index a triplet file"""
        try:
            with jsonlines.open(file_path, mode='r') as reader:
                for obj in reader:
                    env_id = obj.get('env_id')
                    if env_id:
                        if env_id not in self.env_triplet_index:
                            self.env_triplet_index[env_id] = set()
                        self.env_triplet_index[env_id].add(file_path)
        except Exception as e:
            logger.error(f"Error indexing triplet file {file_path}: {e}")
    
    def _index_task_file(self, file_path: Path):
        """Index a task JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for task_data in data:
                        env_id = task_data.get('env_id')
                        if env_id:
                            if env_id not in self.env_task_index:
                                self.env_task_index[env_id] = set()
                            self.env_task_index[env_id].add(file_path)
                elif isinstance(data, dict):
                    env_id = data.get('env_id')
                    if env_id:
                        if env_id not in self.env_task_index:
                            self.env_task_index[env_id] = set()
                        self.env_task_index[env_id].add(file_path)
        except Exception as e:
            logger.error(f"Error indexing task file {file_path}: {e}")
    
    def _index_task_jsonl_file(self, file_path: Path):
        """Index a task JSONL file"""
        try:
            with jsonlines.open(file_path, mode='r') as reader:
                for obj in reader:
                    env_id = obj.get('env_id')
                    if env_id:
                        if env_id not in self.env_task_index:
                            self.env_task_index[env_id] = set()
                        self.env_task_index[env_id].add(file_path)
        except Exception as e:
            logger.error(f"Error indexing task jsonl file {file_path}: {e}")
    
    def get_triplets_by_env_id(self, env_id: str) -> List[Triplet]:
        """Get triplets by environment ID"""
        triplets = []
        
        # Rebuild indices if empty
        if not self.env_triplet_index:
            self._build_indices()
        
        # Use index to find related files
        file_paths = self.env_triplet_index.get(env_id, set())
        for file_path in file_paths:
            try:
                with jsonlines.open(file_path, mode='r') as reader:
                    for obj in reader:
                        if obj.get('env_id') == env_id:
                            # Handle datetime field
                            if 'timestamp' in obj and isinstance(obj['timestamp'], str):
                                obj['timestamp'] = datetime.fromisoformat(obj['timestamp'])
                            triplets.append(Triplet(**obj))
            except Exception as e:
                logger.error(f"Error reading triplet file {file_path}: {e}")
        
        return triplets
    
    def get_tasks_by_env_id(self, env_id: str) -> List[Task]:
        """Get tasks by environment ID"""
        tasks = []
        
        # Rebuild indices if empty
        if not self.env_task_index:
            self._build_indices()
        
        # Use index to find related files
        file_paths = self.env_task_index.get(env_id, set())
        for file_path in file_paths:
            try:
                if file_path.suffix == '.json':
                    # Handle JSON file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for task_data in data:
                                if task_data.get('env_id') == env_id:
                                    # Handle datetime field
                                    if 'timestamp' in task_data and isinstance(task_data['timestamp'], str):
                                        task_data['timestamp'] = datetime.fromisoformat(task_data['timestamp'])
                                    tasks.append(Task(**task_data))
                        elif data.get('env_id') == env_id:
                            # Handle datetime field
                            if 'timestamp' in data and isinstance(data['timestamp'], str):
                                data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                            tasks.append(Task(**data))
                elif file_path.suffix == '.jsonl':
                    # Handle JSONL file
                    with jsonlines.open(file_path, mode='r') as reader:
                        for obj in reader:
                            if obj.get('env_id') == env_id:
                                # Handle datetime field
                                if 'timestamp' in obj and isinstance(obj['timestamp'], str):
                                    obj['timestamp'] = datetime.fromisoformat(obj['timestamp'])
                                tasks.append(Task(**obj))
            except Exception as e:
                logger.error(f"Error reading task file {file_path}: {e}")
        
        return tasks
    
    def _update_triplet_index(self, triplet: Triplet, file_path: Path):
        """Update triplet index"""
        env_id = triplet.env_id
        if env_id:
            if env_id not in self.env_triplet_index:
                self.env_triplet_index[env_id] = set()
            self.env_triplet_index[env_id].add(file_path)
    
    def _update_task_index(self, task: Task, file_path: Path):
        """Update task index"""
        env_id = task.env_id
        if env_id:
            if env_id not in self.env_task_index:
                self.env_task_index[env_id] = set()
            self.env_task_index[env_id].add(file_path)

    # def save_triplets(self, triplets: List[Triplet], session_id: str):
    #     """Save triplets to jsonl"""
    #     filename = f"triplets_{session_id}.jsonl"
    #     filepath = self.base_dir / "triplets" / filename
    #     
    #     with jsonlines.open(filepath, mode='w') as writer:
    #         for triplet in triplets:
    #             # Dict conversion and datetime serialization - Pydantic v1/v2 compatible
    #             if hasattr(triplet, 'model_dump'):
    #                 triplet_dict = triplet.model_dump()
    #             else:
    #                 triplet_dict = triplet.dict()
    #             # Convert datetime to string
    #             if 'timestamp' in triplet_dict:
    #                 triplet_dict['timestamp'] = str(triplet_dict['timestamp'])
    #             writer.write(triplet_dict)
    #     
    #     # Update index
    #     for triplet in triplets:
    #         self._update_triplet_index(triplet, filepath)

    def load_triplets(self, session_id: str) -> List[Triplet]:
        """Load triplets from jsonl"""
        filename = f"triplets_{session_id}.jsonl"
        filepath = self.base_dir / "triplets" / filename
        
        triplets = []
        if filepath.exists():
            with jsonlines.open(filepath, mode='r') as reader:
                for obj in reader:
                    # Handle datetime field
                    if 'timestamp' in obj and isinstance(obj['timestamp'], str):
                        obj['timestamp'] = datetime.fromisoformat(obj['timestamp'])
                    triplets.append(Triplet(**obj))
        return triplets

    def save_tasks(self, tasks: List[Task], session_id: str):
        """Save tasks to JSON file"""
        filename = f"tasks_{session_id}.json"
        filepath = self.base_dir / "tasks" / filename
        
        tasks_data = []
        for task in tasks:
            # Pydantic v1/v2 compatibility
            if hasattr(task, 'model_dump'):
                task_dict = task.model_dump()
            else:
                task_dict = task.dict()
            # Convert datetime to string
            if 'timestamp' in task_dict:
                task_dict['timestamp'] = str(task_dict['timestamp'])
            tasks_data.append(task_dict)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(tasks_data, f, indent=2, ensure_ascii=False)
        
        # Update index
        for task in tasks:
            self._update_task_index(task, filepath)

    def load_tasks(self, session_id: str) -> List[Task]:
        """Load tasks from JSON file"""
        filename = f"tasks_{session_id}.json"
        filepath = self.base_dir / "tasks" / filename
        
        tasks = []
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                tasks_data = json.load(f)
                for task_data in tasks_data:
                    # Handle datetime field
                    if 'timestamp' in task_data and isinstance(task_data['timestamp'], str):
                        task_data['timestamp'] = datetime.fromisoformat(task_data['timestamp'])
                    tasks.append(Task(**task_data))
        return tasks

    def save_session(self, session: Session):
        """Save a complete session"""
        filename = f"session_{session.session_id}.json"
        filepath = self.base_dir / "sessions" / filename
        
        # Pydantic v1/v2 compatibility
        if hasattr(session, 'model_dump'):
            session_dict = session.model_dump()
        else:
            session_dict = session.dict()
        # Handle datetime fields
        for key in ['start_time', 'end_time']:
            if key in session_dict and session_dict[key] is not None:
                session_dict[key] = str(session_dict[key])
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_dict, f, indent=2, ensure_ascii=False)
        
        return session.session_id

    def load_session(self, session_id: str) -> Session:
        """Load a session"""
        filename = f"session_{session_id}.json"
        filepath = self.base_dir / "sessions" / filename
        
        with open(filepath, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
            # Handle datetime fields
            for key in ['start_time', 'end_time']:
                if key in session_data and session_data[key] is not None and isinstance(session_data[key], str):
                    session_data[key] = datetime.fromisoformat(session_data[key])
            return Session(**session_data)

    def save_triplet(self, triplet: Triplet):
        """Save a single triplet"""
        session_id = triplet.session_id or "default"
        filename = f"triplets_{session_id}.jsonl"
        filepath = self.base_dir / "triplets" / filename
        
        # Dict conversion and datetime serialization - Pydantic v1/v2 compatible
        if hasattr(triplet, 'model_dump'):
            triplet_dict = triplet.model_dump()
        else:
            triplet_dict = triplet.dict()
        if 'timestamp' in triplet_dict:
            triplet_dict['timestamp'] = str(triplet_dict['timestamp'])
        
        # Append mode write
        with jsonlines.open(filepath, mode='a') as writer:
            writer.write(triplet_dict)
        
        # Update index
        self._update_triplet_index(triplet, filepath)

    def save_memory(self, env_id: str, memory_type: str, content: str):
        """Save environment memory summary
        
        Args:
            env_id: Environment ID
            memory_type: memory namespace, e.g., 'exploration' or 'task'
            content: memory content
        """
        # Ensure memory directory exists
        memory_dir = self.base_dir / "memories"
        memory_dir.mkdir(exist_ok=True)
        
        # Save memory file
        filename = f"memory_{env_id}_{memory_type}.txt"
        filepath = memory_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return filepath

    def load_memory(self, env_id: str, memory_type: str) -> Optional[str]:
        """Load environment memory summary
        
        Args:
            env_id: Environment ID
            memory_type: memory namespace, e.g., 'exploration' or 'task'
            
        Returns:
            memory content if present, otherwise None
        """
        filename = f"memory_{env_id}_{memory_type}.txt"
        filepath = self.base_dir / "memories" / filename
        
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        
        return None

    def save_task(self, task: Task):
        """Save a single task"""
        session_id = task.session_id or "default"
        filename = f"tasks_{session_id}.jsonl"
        filepath = self.base_dir / "tasks" / filename
        
        # Dict conversion and datetime serialization - Pydantic v1/v2 compatible
        if hasattr(task, 'model_dump'):
            task_dict = task.model_dump()
        else:
            task_dict = task.dict()
        if 'timestamp' in task_dict:
            task_dict['timestamp'] = str(task_dict['timestamp'])
        
        # Append mode write
        with jsonlines.open(filepath, mode='a') as writer:
            writer.write(task_dict)
        
        # Update index
        self._update_task_index(task, filepath)
