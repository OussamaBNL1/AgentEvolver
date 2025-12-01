"""
Prompts for trajectory generation
"""


def _build_action_format(env_type: str) -> str:
    env = (env_type or "").lower()
    if env == "bfcl":
        return (
            "<action>\n"
            "<tool_call>\n"
            "{\n"
            "    \"id\": \"unique_call_id\",\n"
            "    \"name\": \"tool_function_name\",\n"
            "    \"arguments\": \"{\\\"param1\\\": \\\"value1\\\"}\",\n"
            "    \"type\": \"tool\",\n"
            "    \"index\": 0\n"
            "}\n"
            "</tool_call>\n"
            "</action>\n"
        )
    if env == "appworld":
        return (
            "Enclose the action within <action>```python\n \n```</action> tags. Such as <action>```python\nprint('hello appworld!!')\n```</action>. Every response must contain a valid action enclosed in these tags.\n"
        )
    # default: webshop
    return (
        "<action>\n"
        "\\boxed{search[machine wash men's dress shirts]}\n"
        "\\boxed{click[machine wash men's dress shirts]}\n"
        "</action>\n"
    )

class TrajectoryPrompts:
    """Prompt templates for trajectory generation"""
    
    def simple_action_prompt(self, env_discription: str, task_description: str,
                           query: str, action_history: list, ground_truth: str, env_type: str = "webshop") -> list:
        """Simple strategy action prompt"""
        
        history_text = "\n".join([f"- {action}" for action in action_history[-3:]])  # Last 3 actions
        action_format = _build_action_format(env_type)
        
        messages = [
            {
                "role": "user",
                "content": f"""You are an AI assistant helping to complete tasks in an interactive environment.

Task Description: {task_description}
Query: {query}
Tips: {ground_truth}
Environment information: {env_discription}
Previous Actions:
{history_text if history_text else "None"}

## Instructions:

Choose only one action at a time.

Tips are not always correct and may skip some steps.

Carefully read the environment description and task instructions.

Ensure that the action is in the correct format.

If the action is invalid, verify that it is properly formatted.

Always include a valid action and action tags in your reply.

Please Observe the current environment state and identify the available APIs. 

Analyze the available actions and provide the next action to take. 

The output format of the tool call should contain <tool_call> and </tool_call> and should not contain ```json and ``` blocks. Follow the format:

# Action Format (must follow):
{action_format}

Every response must contain a valid action enclosed in these tags.

First enter your reason, then enter your action. 

Focus on completing the task step by step. Be specific and clear in your action.

Don't cheat such as directly output tips without any reason. 

Every step should be reasonable and logical.

When you think all the task is complete and do not need any action, you should output <action><finish></action>."""

            }
        ]
# To use any tool, respond with a JSON object containing, and finally return the action in the following format:
# """
# """

##bfcl
# <action>
# <tool_call>
# {
#     \"id\": \"unique_call_id\",
#     \"name\": \"tool_function_name\",
#     \"arguments\": \"{\\\"param1\\\": \\\"value1\\\"}\",
#     \"type\": \"tool\",
#     \"index\": 0
# }
# </tool_call>
# </action>

##webshop
# <action>
# \\boxed{click[something]} 
# \\boxed{search[something]}
# </action>

##appworld
# Enclose the action within <action>```python\n \n```</action> tags. Such as <action>```python\nprint('hello appworld!!')\n```</action>. Every response must contain a valid action enclosed in these tags.

        return messages

#     def reflection_action_prompt(self, observation: str, task_description: str,
#                                query: str, action_history: list, 
#                                observation_history: list, ground_truth: str) -> list:
#         """Reflection strategy action prompt"""
        
#         # Create history summary
#         history_summary = []
#         start_idx = max(0, len(action_history) - 3)  # Last 3 steps
        
#         for i in range(start_idx, len(action_history)):
#             action = action_history[i]
#             obs = observation_history[i] if i < len(observation_history) else "No observation"
#             step_num = i + 1
            
#             history_summary.append(f"Step {step_num}: {action}")
#             obs_short = obs[:100] + "..." if len(obs) > 100 else obs
#             history_summary.append(f"  Result: {obs_short}")
        
#         history_text = "\n".join(history_summary)
        
#         messages = [
#             {
#                 "role": "user", 
#                 "content": f"""You are an AI assistant helping to complete tasks in an interactive environment.
# You should reflect on previous actions and their results to plan better next steps.

# Task Description: {task_description}
# Query: {query}
# Tips: {ground_truth}

# Current Observation: {observation}

# Recent Action History:
# {history_text if history_text else "None"}

# Please reflect on the previous actions and their results. Consider what worked, what didn't work, and what you should try next.
# Then provide the next best action to take.

# Thought: [Your reflection on previous actions and current situation]
# Action: [The next action to take]"""
#             }
#         ]
        
#         return messages
