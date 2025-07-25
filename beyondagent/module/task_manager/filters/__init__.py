import abc
from typing import Sequence

from beyondagent.schema.task import TaskObjective


class TaskPostFilter(abc.ABC):
    @abc.abstractmethod
    def filter(self, tasks: Sequence[TaskObjective]) -> list[TaskObjective]:
        pass
    

from .filters import NaiveTaskPostFilter

__all__ = [
    "NaiveTaskPostFilter"
]