import json
import time
from typing import Any, Callable, Optional, cast

from loguru import logger
from omegaconf import DictConfig

from beyondagent.client.env_client import EnvClient
from beyondagent.module.agent_flow.reward_calculator import RewardCalculator
from beyondagent.module.task_manager.base import LlmClient, LlmRawClient
from beyondagent.schema.trajectory import Trajectory
from beyondagent.utils.utils import convert_tool_to_user_message, clip_state_content_correctly

class AgentFlow:
    def __init__(self,
                 llm: LlmRawClient,
                 env_client:EnvClient,
                 instance_id:str,
                 reward_caculator:Optional[RewardCalculator],
                 max_steps:int,
                 max_model_len:int,
                 max_response_len:int,
                 max_env_len:int,
                 sampling_params:dict,
                 enable_request_id:bool,
                 *,
                 tokenizer: Any):
        self._llm=llm
        self._env_client=env_client
        self._instance_id=instance_id
        self._reward_calculator = reward_caculator
        if self._reward_calculator is not None:
            logger.info(f"reward_calculator={self._reward_calculator}")
        
        self._max_steps = max_steps
        self._max_model_len = max_model_len
        self._max_response_len=max_response_len
        self._max_env_len = max_env_len
        self._sampling_params=sampling_params
        self._enable_request_id=enable_request_id
        
        self._tokenizer = tokenizer
    
    
    def chat(self, trajectory:Trajectory)->Trajectory:
        t_start = time.time()
        # callback llm server, messages.size=1
        llm_output: dict = {}
        try:
            # TODO handle request id
            llm_output = self._llm.chat(trajectory.steps,sampling_params=self._sampling_params)
        except Exception as e:
            logger.exception(f"call llm_chat_fn error with {e}")

        time_cost = round(time.time() - t_start, 4)
        new_request_id: str = llm_output.pop("request_id", "")

        info_dict = {
            "act_step": self._act_step,
            "llm_output": llm_output,
            "new_request_id": new_request_id,
            "request_id": self._request_id,
            "time_cost": time_cost,
        }
        logger.info(f"info_dict={json.dumps(info_dict)}")

        self._request_id = new_request_id
        trajectory.steps.append(llm_output)
        
        return trajectory
    
    
    def act(self,trajectory:Trajectory)->Trajectory:
        try:
            env_output = self._env_client.step(self._instance_id, trajectory.steps[-1])
            env_messages: list[dict] = env_output["state"]
        except Exception as e:
            logger.exception(f"call env.step error with {e}")
            raise e
        
        # useless: for tool role
        assert len(env_messages)>0, "env returns empty messages"
        for env_message in env_messages:
            if env_message["role"] == "tool":
                env_message = cast(dict, convert_tool_to_user_message(env_message, format="qwen"))
            
            state_content: str = env_message["content"]
            
            env_message["content"] = clip_state_content_correctly(
                self._tokenizer, 
                state_content,
                self._max_env_len
            )
            

            trajectory.steps.append(sanitize_env_state(env_message))
        trajectory.is_terminated = env_output["is_terminated"]
        
        return trajectory

    
    def act_flow(self, trajectory:Trajectory):
        while True:
            yield self.chat(trajectory)
            yield self.act(trajectory)
        

    def execute(self, trajectory: Trajectory) -> Trajectory:
        self._request_id: str = ""
        self._act_step=0
        
        generator=self.act_flow(trajectory)
        for act_step in range(self._max_steps):
            self._act_step=act_step
            # TODO add /no_think in llm client
            prompt_text = self._tokenizer.apply_chat_template(trajectory.steps, 
                                                                tokenize=False,
                                                                add_generation_prompt=True)
            current_token_len = len(self._tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"][0])

            # yunpeng 0623: to prevent add an imend token to an uncompleted seq, 
            # because the message-type output will be applied chat_template.
            if trajectory.steps[-1]['role'] != 'assistant' and current_token_len + self._max_response_len > self._max_model_len:
                logger.warning(f"exceed max model len={self._max_model_len}, stopping this flow...")
                break
            
            trajectory=next(generator)

            if trajectory.is_terminated:
                break
        
        if self._reward_calculator is not None:
            score = self._reward_calculator.calculate_reward(trajectory, self._env_client)
        else:
            score = self._env_client.evaluate(self._instance_id, params={"sparse": False})
        trajectory.reward.outcome = score
        trajectory.reward.description = "Outcome 1 = success, 0 = failure."

        if trajectory.steps[-1]["role"] == "user":
            trajectory.steps = trajectory.steps[:-1]

        return trajectory


def sanitize_env_state(state: dict):
    """
    sanitize env state
    """
    # remove empty tool_calls
    if "tool_calls" in state and not state["tool_calls"]:
        state.pop("tool_calls")
    
    return state
