"""
Stage 1: Agent-environment interaction to generate triplet data
Dynamically builds the Action Format section according to env_type: webshop | bfcl | appworld
"""

def _build_action_format(env_type: str) -> str:
    env = (env_type or "").lower()
    if env == "bfcl":
        return (
            "To use any tool, respond with a JSON object and finally return the action in the following format:\n\n"
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
            "Enclose the action within <action>```python\\n \\n+```</action> tags.\n"
            "For example: <action>```python\\nprint('hello appworld!!')\\n```</action>.\n"
            "Every response must contain a valid action enclosed in these tags.\n"
        )
    # default: webshop
    return (
        "To use any tool, respond with an action in the following format:\n\n"
        "<action>\n"
        "\\boxed{search[machine wash men's dress shirts]}\n"
        "\\boxed{click[machine wash men's dress shirts]}\n"
        "</action>\n"
    )


def build_agent_interaction_system_prompt(env_type: str) -> str:
    return f"""
You are an environment explorer with a deep curiosity about the world around you. This is your first time in this world, and you are particularly concerned about some operations that may be useful to you in the future. While interacting with the user, your primary interest lies in exploring the environment freely. You focus on discovering and executing actions within the allowed set of options provided. Your goal is to explore actions that adhere to the task format but do not concern yourself with the outcome.

## Your task:

Observe the current environment state and identify the available Actions.

Analyze the available actions and determine which ones will allow you to explore the environment most effectively.

Select a relevant action based on the available options and ensure it aligns with the task's goal.

Execute the chosen action in the required format, ensuring it follows the specified tags.

Ensure the chosen action is within the user-defined set of actions.

# Action Format:

{_build_action_format(env_type)}

## Instructions:

Choose only one action at a time.

Carefully read the environment description and task instructions.

Ensure that the action is in the correct format.

Do not use undefined actions. The actions used must be defined in the given action set.

If the action is invalid, verify that it is properly formatted.

Always include a valid action and action tags in your reply.

First enter your reason, then enter your action.
"""

##bfcl
# To use any tool, respond with a JSON object containing, and finally return the action in the following format:

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

##appworld
# Enclose the action within <action>```python\n \n```</action> tags. Such as <action>```python\nprint('hello appworld!!')\n```</action>. Every response must contain a valid action enclosed in these tags.

##webshop
# To use any tool, respond with a JSON object containing, and finally return the action in the following format:
# <action>
# \\boxed{search[machine wash men's dress shirts]}
# \\boxed{click[machine wash men's dress shirts]}
# </action>

def parse_action_from_response(response: str) -> str:
    """Parse action from LLM response"""
    try:
        # Try to extract action from <action>...</action> tags
        import re
        action_match = re.search(r'<action>(.*?)</action>', response, re.DOTALL)
        if action_match:
            action = action_match.group(1).strip()  # Clean to reduce transfer size
            return action.strip()
        else:
            # If no <action> tag found, return the whole response
            return response.strip()
        
    except Exception as e:
        # On failure, return a default action
        return "look around"

def get_agent_interaction_prompt(
    initial_obs: str,
    history: list = None,
    exploration_memory: str = None,
    exploration_requirement: str = "You are completely free to explore the environment. You should prioritize trying new and untried actions. ",
    env_type: str = "webshop",
) -> tuple:
    """Build the full prompt for agent interaction, including exploration memory and user requirements"""
    
    # Build exploration requirement context
    requirement_text = ""
    if exploration_requirement:
        requirement_text = f"""
## Exploration Requirement:
{exploration_requirement}

Your exploration should prioritize this requirement while still maintaining curiosity about the environment.
"""
    
    if initial_obs and initial_obs!= "":
        initial_obs = f"""
## Environment Description:
{initial_obs}

"""
        
    
    # Build memory context
    memory_text = ""
    if exploration_memory and exploration_memory!= "":
        # Ensure non-empty memory
        memory_text = f"""
## Environment Exploration Memory:
{exploration_memory}

Based on this memory, try to explore new areas and avoid repeating actions that have already been thoroughly tested, especially those that led to poor outcomes.
"""
    
    # Build recent history context
    history_text = ""
    history = history[1:] if history else []  # Skip the first one
    if history:
        history_text = "\n".join([f"- {h}" for h in history[-5:]])  # Keep last 5
        history_text = f"\n## Recent History:\n{history_text}\n"
        
    user_prompt = f"""
{requirement_text}
{memory_text}
{history_text}
{initial_obs}

Please select an appropriate action based on the current environment state, exploration memory, historical information, and any specified exploration requirements. Focus on exploring new areas or actions that haven't been thoroughly tested yet.
"""
    
    system_prompt = build_agent_interaction_system_prompt(env_type)
    return system_prompt, user_prompt

