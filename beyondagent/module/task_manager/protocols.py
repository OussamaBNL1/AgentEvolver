import abc
from typing import Any, Protocol

from beyondagent.schema.task import Task, TaskObjective


class LlmClient(Protocol):
    def chat(self, messages: list[dict[str, str]], sampling_params: dict[str, Any]) -> str:
        ...


class TaskObjectiveRetrieval(abc.ABC):
    """支持任务相关任务 objective 检索，用于避免重复探索"""
    
    @abc.abstractmethod
    def retrieve_objectives(self, task: Task) -> list[TaskObjective]:
        ...
    
    @abc.abstractmethod
    def add_objective(self, objective: TaskObjective):
        ...