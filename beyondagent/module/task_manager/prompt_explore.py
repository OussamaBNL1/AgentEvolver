from typing import Optional, Sequence

from beyondagent.schema.task import Task, TaskObjective


AGENT_INTERACTION_SYSTEM_PROMPT = """
You are an environment explorer with a deep curiosity about the world around you. This is your first time in this world, and you are particularly concerned about some operations that may be useful to you in the future. While interacting with the user, your primary interest lies in exploring the environment freely. You do not focus on the task at hand but instead are keen on discovering and executing actions within the allowed set of options provided. Your goal is to explore actions that adhere to the task format but do not concern yourself with the outcome.
## Your task:

Observe the current environment state and identify the available APIs.

Analyze the available actions and determine which ones will allow you to explore the environment most effectively.

Select a relevant action based on the available options and ensure it aligns with the task's goal.

Execute the chosen action in the required format, ensuring it follows the specified tags.

Ensure the chosen action is within the user-defined set of actions.

## Action Format:

Please follow the user-defined action format. If there is no action format, you can use the format you prefer.

## Old Objectives:

You have already explored the following objectives:

{old_objectives}

Please avoid repeating the objectives in the current exploration.

## Instructions:

Do not focus on the task at hand but instead are keen on discovering and executing actions within the allowed set of options provided.

Choose only one action at a time. 

Carefully read the environment description and task instructions.

Ensure that the action is in the correct format. If the action is invalid, verify that it is properly formatted.

"""

def get_agent_interaction_system_prompt(task: Task, old_objectives: Sequence[TaskObjective]) -> str:
    """获取环境交互系统提示"""
    objectives:list[str]=[]
    for ob in old_objectives:
        if isinstance(ob.objective,str):
            objectives.append(ob.objective)
        else:
            objectives.extend(ob.objective)
            
    return AGENT_INTERACTION_SYSTEM_PROMPT.format(old_objectives="\n".join(objectives))

def parse_action_from_response(response: str) -> str:
    """从响应中解析动作"""
    try:
        # 解析<action>标签中的动作
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = response.find(start_tag)
        end_idx = response.find(end_tag)
        
        if start_idx != -1 and end_idx != -1:
            action = response[start_idx + len(start_tag):end_idx].strip()
            return action
        else:
            # 如果没有找到标签，尝试提取第一行有意义的文本
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('**'):
                    return line
        
        return "invalid action"  # 默认动作
    except Exception:
        return "invalid action"  # 默认动作
    
    
__all__=[
    "get_agent_interaction_system_prompt",
    "parse_action_from_response"
]