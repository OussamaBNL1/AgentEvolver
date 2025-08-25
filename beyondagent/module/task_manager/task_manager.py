from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import functools
import json
import os
import pickle
import random
import threading
import time
from typing import (
    Callable,
    Iterable,
    NotRequired,
    Optional,
    Sequence,
    TypedDict,
    Unpack,
)

import hydra
from loguru import logger
from omegaconf import DictConfig
import requests
from torch.utils.data import IterableDataset,Dataset
from tqdm import tqdm
from beyondagent.client.env_client import EnvClient
from beyondagent.client.llm_client import DashScopeClient
from beyondagent.module.agent_flow.agent_flow import AgentFlow
from beyondagent.module.agent_flow.base_agent_flow import BaseAgentFlow
from beyondagent.module.task_manager import adapter
from beyondagent.module.task_manager.adapter import OnflyRlDataset, to_rl_dataset
from beyondagent.module.task_manager.data_mixture import MixtureStrategy, OriginalOnlyStrategy
from beyondagent.module.task_manager.filters.llm_filter import LlmFilter
from beyondagent.module.task_manager.strategies import TaskExploreStrategy
from beyondagent.module.task_manager.explorer import EnvWorkerWithPrompt
from beyondagent.module.task_manager.filters.filters import NaiveTaskPostFilter, TaskPostFilter

from beyondagent.module.task_manager.base import LlmClient, TaskObjectiveRetrieval
from beyondagent.module.task_manager.strategies.deduplication import LlmDedupSamplingExploreStrategy
from beyondagent.module.task_manager.strategies.random import LlmRandomSamplingExploreStrategy
from beyondagent.module.task_manager.datasets import FullDataset,AutoReloadDataset
from beyondagent.schema.task import Task, TaskObjective
from beyondagent.schema.trajectory import Trajectory
from verl.utils.dataset.rl_dataset import RLHFDataset

class TaskManagerProps(TypedDict):
    num_explore_threads: int
    n: int # 重复探索的控制必须放在这里，task manager 要规划 task 执行顺序，避免在同时探索相同任务导致潜在的 query 重复

class RewardProps(TypedDict):
    original_grader:str
    synthetic_grader:str

class TaskManager(object):

    def __init__(
        self,
        config: DictConfig,
        exploration_strategy: str,
        exploration_strategy_args,
        llm_client: LlmClient,
        old_retrival: TaskObjectiveRetrieval,
        mixture_strategy: MixtureStrategy,
        reward_config: RewardProps,
        tokenizer,
        env_service_url: str,
        **kwargs: Unpack[TaskManagerProps],
    ):
        self._config = config
        self._exploration_strategy=get_exploration_strategy(exploration_strategy,exploration_strategy_args,tokenizer=tokenizer,config=config)
        self._llm_client = llm_client
        self._old_retrival = old_retrival
        self._mixture_strategy = mixture_strategy
        self._reward_config=reward_config
        self._env_service_url = env_service_url
        self._tokenizer = tokenizer  # cc: 这玩意似乎不该在这
        self._num_exploration_threads = kwargs["num_explore_threads"] or 10
        self._n = kwargs["n"]

        self._realtime_filters: list[TaskPostFilter] = [NaiveTaskPostFilter()]
        self._post_filter: list[TaskPostFilter] = [LlmFilter(env_service_url,llm_client,self._num_exploration_threads,tokenizer=tokenizer,config=config)]
        
        self._tasks: list[Task]=[]
        self._exploration_strategy._inject_deps(self._old_retrival,self._llm_client,DashScopeClient(model_name='qwq-plus',max_tokens=8192))
    
    @property
    def seed_tasks(self):
        return self._tasks
    
    def load_tasks(self,tasks:Sequence[Task]):
        self._tasks.extend(tasks)
        assert all([x.query is None for x in self._tasks]), "query of seed task must be empty"
        logger.info(f"loaded tasks, #tasks={len(self._tasks)}")
        
    def load_tasks_from_dataset(self, dataset: RLHFDataset,*, env_type:str):
        self._tasks.extend(adapter.convert_to_tasks(dataset,env_type=env_type,grader=self._reward_config["original_grader"]))
        assert all([x.query is None for x in self._tasks]), "query of seed task must be empty"
        logger.info(f"loaded tasks from dataset, #tasks={len(self._tasks)}")
    
    def load_tasks_from_environment(self, env: EnvClient, *, env_type: str, split: str, params: Optional[dict] = None):
        try:
            response = env.get_env_profile(env_type, split, params)
            self._tasks.extend([Task(task_id=str(x),env_type=env_type,evaluator=self._reward_config["original_grader"]) for x in response])
            assert all([x.query is None for x in self._tasks]), "query of seed task must be empty"
            logger.info(f"loaded tasks from environment, #tasks={len(self._tasks)}")
        except requests.exceptions.RequestException as e:
            logger.error(f"failed to load tasks from environment: {e}")
            return 0
        return len(response)

    def register_filter(self, filter: TaskPostFilter):
        self._realtime_filters.append(filter)

    def get_onthefly_dataset(self, bs: int, tokenizer, config,processor):
        """
        Get dataset on the fly.

        Args:
            tasks: Iterable[Task]
            bs: int. 该 batch size 决定一次读取的 task 数量。每次生成的 dataset 大小为 bs * self._n。
            tokenizer: transformers.tokenization_utils.PreTrainedTokenizer
            config: DictConfig. Only for RLHFDataset.
        """
        # autoreloaddataset 没适配 mixture
        raise NotImplementedError("get_onthefly_dataset is not implemented")
        # return AutoReloadDataset(self,iter(self._tasks),bs,self._mix_original_tasks,tokenizer=tokenizer,config=config,processor=processor)
    
    def get_or_load_full_dataset(self,filepath:Optional[str],*,config,tokenizer,processor)->"FullDataset":
        """Get the full dataset, or load from file.
        """
        seed_tasks=[TaskObjective(task=task,confidence=1.0,reward=None) for task in self._tasks]
        dataset=FullDataset(self,seed_tasks,self._mixture_strategy,self._reward_config,tokenizer=tokenizer,config=config,processor=processor)
        
        if filepath is not None and os.path.exists(filepath):
            logger.info(f"loading full dataset from {filepath}")
            dataset.load_from_file(filepath)
        else:
            dataset.reload()
            if filepath is not None:
                dataset.save_to_file(filepath)
        
        return dataset
    
    def get_original_dataset(self,*,tokenizer,config,processor)->"FullDataset":
        """Get the original dataset.
        """
        seed_tasks=[TaskObjective(task=task,confidence=1.0,reward=None) for task in self._tasks]
        dataset = FullDataset(self,seed_tasks,OriginalOnlyStrategy(),self._reward_config,tokenizer=tokenizer,config=config,processor=processor)
        dataset.load_from_file('[unknown]')
        return dataset
    
    
    def generate_task(self, tasks: Sequence[Task],*,show_progress=False) -> list[TaskObjective]:
        task_q = list(copy.copy(tasks)) * self._n
        res = []
        
        # 每次最多探索所有不同任务，或者最大线程个任务，防止同批次中生成相同任务
        parallel_num = min(self._num_exploration_threads, len(tasks))
        with ThreadPoolExecutor(max_workers=self._num_exploration_threads) as pool:
            for i in tqdm(range(0, len(task_q), parallel_num),desc="generating tasks", disable=not show_progress):
                futures = [
                    pool.submit(self._exlore_and_summarize, task, data_id, rollout_id)
                    for task, data_id, rollout_id in zip(
                        task_q[i : i + parallel_num],
                        ["unknown"] * parallel_num,
                        ["unknown"] * parallel_num,
                    )
                ]
                task_objectives = sum([future.result() for future in futures], [])
                res.extend(task_objectives)
                # realtime filter
                res = functools.reduce(lambda x, f: f.filter(x), self._realtime_filters, res)
                self._old_retrival.reset()
                for i in res:
                    self._old_retrival.add_objective(i)
                    
        res = functools.reduce(lambda x, f: f.filter(x), self._realtime_filters, res)
        # post filter
        logger.info("running post filter on generated tasks")
        cnt_before_filter=len(res)
        res = functools.reduce(lambda x, f: f.filter(x), self._post_filter, res)
        cnt_after_filter=len(res)
        logger.info(f"finish post filter: #before={cnt_before_filter}, #after={cnt_after_filter}")
        random.shuffle(res) # shuffle

        return res

    
    def _exlore_and_summarize(self,task:Task,data_id:str,rollout_id:str)->list[TaskObjective]:
        trajectories=self._step_explore(task,data_id,rollout_id)
        task_objectives=sum([self._step_summarize(task,trajectory) for trajectory in trajectories],[])
        return task_objectives


    def _step_explore(self, task: Task, data_id: str, rollout_id: str)->list[Trajectory]:
        """
        Step 1: explore the environment to find out possible actions and their results.
        """
        return self._exploration_strategy.explore(task,data_id,rollout_id)


    def _step_summarize(
        self, task: Task, trajectory: Trajectory
    ) -> list[TaskObjective]:
        """
        Step 2: summarize the results of the exploration to generate the TASK (query and gt).

        Args:
            task: Task
            trajectories: Trajectory.
        """
        return self._exploration_strategy.summarize(task, trajectory)


def get_exploration_strategy(name:str, strategy_args, *, tokenizer, config)->TaskExploreStrategy:
    """Get exploration strategy by name."""
    logger.info(f"loading exploration strategy {name}")
    if name=="random":
        return LlmRandomSamplingExploreStrategy(tokenizer=tokenizer,config=config,**strategy_args)
    elif name == "deduplication":
        return LlmDedupSamplingExploreStrategy(tokenizer=tokenizer,config=config,**strategy_args)
    else:
        raise NotImplementedError(f"exploration strategy {name} not implemented")


