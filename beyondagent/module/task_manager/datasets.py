from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from typing import (
    Iterable,
    Sequence,
)

from loguru import logger
from torch.utils.data import IterableDataset,Dataset
from beyondagent.module.task_manager.adapter import OnflyRlDataset, to_rl_dataset
from beyondagent.module.task_manager.data_mixture import MixtureStrategy, OriginalOnlyStrategy

from beyondagent.module.task_manager.task_manager import RewardProps, TaskManager
from beyondagent.schema.task import Task, TaskObjective

class FullDataset(Dataset):
    """FullDataset with MixtureStrategy support and auto-refresh after one DataLoader epoch"""
    
    def __init__(self, 
                 manager: TaskManager, 
                 tasks: Sequence[TaskObjective],
                 mixture_strategy: MixtureStrategy,
                 reward_config:RewardProps,
                 *, 
                 tokenizer, 
                 config, 
                 processor):
        self._manager = manager
        self._tasks = list(tasks)
        assert all([x.task.evaluator==reward_config["original_grader"] for x in tasks]), "task evaluator must be set as the config"
        self._mixture_strategy = mixture_strategy
        self._reward_config=reward_config
        self._tokenizer = tokenizer
        self._config = config
        self._processor = processor
        self._objectives = []
        self._dataset = None
        self._synthetic_objectives = []
        
        # 标记是否需要在下一轮迭代开始前刷新
        self._refresh_after_epoch = False

    def _rebuild_dataset(self):
        """使用混合策略重新生成 dataset"""
        self._objectives = self._mixture_strategy.mix_data(self._synthetic_objectives, self._tasks)
        self._dataset = to_rl_dataset(self._objectives, self._tokenizer, self._config, self._processor)
        logger.info(f"Auto-refreshed dataset: #objectives={len(self._objectives)}, #rlhf={len(self._dataset)}")

    def set_mixture_strategy(self, strategy: MixtureStrategy):
        self._mixture_strategy = strategy
        logger.info(f"mixture strategy updated to: {type(strategy).__name__}")
    
    def save_to_file(self, filepath: str):
        with open(filepath, "w") as f:
            f.writelines([ob.json() + "\n" for ob in self._synthetic_objectives])
        logger.info(f"Saved {len(self._objectives)} objectives to {filepath}")
    
    def load_from_file(self, filepath: str):
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                self._synthetic_objectives = []
                for line in filter(lambda x: x.strip() != "", f.readlines()):
                    tmp=TaskObjective.parse_raw(line)
                    # patch old data
                    if tmp.ground_truth is None:
                        tmp.ground_truth = json.loads(line)['ground_truth']
                    self._synthetic_objectives.append(tmp)
        else:
            logger.warning(f"failed to load objectives from {filepath}, file not found.")
            self._synthetic_objectives = []
        
        for item in self._synthetic_objectives:
            assert item.ground_truth is not None
        
        logger.info("patching grader config to all synthetic data")
        for item in self._synthetic_objectives:
            item.task.evaluator=self._reward_config["synthetic_grader"]
        
        self._rebuild_dataset()
    
    def reload(self):
        self._synthetic_objectives = self._manager.generate_task([x.task for x in self._tasks], show_progress=True)
        logger.info("patching grader config to all synthetic data")
        for item in self._synthetic_objectives:
            item.task.evaluator=self._reward_config["synthetic_grader"]
        self._rebuild_dataset()
    
    def get_statistics(self) -> dict:
        if not self._objectives:
            return {
                "total": 0, 
                "synthetic": 0, 
                "original": 0,
                "synthetic_ratio": 0.0,
                "strategy_info": str(self._mixture_strategy)
            }
        
        synthetic_count = sum(1 for obj in self._objectives if obj.task.evaluator != "env")
        original_count = len(self._objectives) - synthetic_count
        
        return {
            "total": len(self._objectives),
            "synthetic": synthetic_count,
            "original": original_count,
            "synthetic_ratio": synthetic_count / len(self._objectives) if len(self._objectives) > 0 else 0,
            "strategy_info": str(self._mixture_strategy)
        }
    
    def __getitem__(self, index):
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded. Call reload() or load_from_file() first.")
        return self._dataset[index]
    
    def __len__(self):
        if self._dataset is None:
            return 0
        return len(self._dataset)

    def __iter__(self):
        """在 DataLoader 完成一轮迭代后自动刷新"""
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded. Call reload() or load_from_file() first.")
        
        # 如果上一轮标记了刷新，则先重建
        if self._refresh_after_epoch:
            self._rebuild_dataset()
            self._refresh_after_epoch = False

        for i in range(len(self._dataset)):
            yield self._dataset[i]
        
        # 一轮迭代结束后，标记下次迭代前刷新
        self._refresh_after_epoch = True


# wrapper for data auto-reloading
class AutoReloadDataset(IterableDataset):
    """AytoReloadDataset
    
    the number of workers of DataLoader must be 1.
    """
    def __init__(self,manager:TaskManager, tasks:Iterable[Task], bs: int, mix_origins:bool=False, *, tokenizer, config, processor):
        self._manager=manager
        self._tasks=tasks
        self._bs = bs
        self._mix_origins=mix_origins
        assert self._mix_origins==False, "mix_origins is not supported yet"
        self._tokenizer = tokenizer
        self._config=config
        self._processor = processor
        
        self._dataset = OnflyRlDataset(release_used_dataset=True)
    
    def reload(self):
        delta = []
        for task in self._tasks:
            delta.append(task)
            if len(delta) == self._bs:
                break

        ls = self._manager.generate_task(delta)
        while len(ls) < self._bs * self._manager._n:
            logger.debug("failed to generate enough tasks, retrying")
            ls = self._manager.generate_task(delta)

        self._dataset.append_dataset(to_rl_dataset(ls, self._tokenizer, self._config,self._processor))
        return self._dataset.num_rest_data

    def __iter__(self):
        return self

    def __next__(self):
        if self._dataset.num_rest_data == 0:
            logger.debug("no data left")
            if self.reload() == 0:
                logger.debug("no task left, stop reloading and iteration")
                raise StopIteration
        return next(self._dataset)