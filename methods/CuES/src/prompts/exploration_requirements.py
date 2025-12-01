"""
Handle user exploration requirements.
Convert natural language requirements into structured exploration directives.
"""
from typing import Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)


def format_exploration_requirement(client, user_requirement: Optional[str] = None) -> str:
    """Format a user requirement into a standard exploration directive"""
    if not user_requirement:
        # If there is no user requirement, return None to indicate comprehensive exploration
        return None
    
    # Use LLM to convert natural language into a structured exploration directive
    system_prompt = """
You are an exploration requirement formatter. Your task is to convert a user's natural language exploration requirement into a clear, structured exploration directive for an AI agent.

The directive should:
1. Be specific about what to explore
2. Include constraints on what not to explore (if specified)
3. Suggest strategies for achieving the exploration goal
4. Be concise and actionable
5. Focus only on exploration, not on completing specific tasks

Format the directive as a clear, concise paragraph that will guide the exploration agent.
"""

    user_prompt = f"""
The user has provided the following exploration requirement:

"{user_requirement}"

Please convert this into a clear, structured exploration directive for an AI agent that will be exploring an application environment.
"""

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = client.chat_with_retry(messages, max_retries=2)
        if not response:
            logger.warning("Failed to format exploration requirement, using default")
            return f"Focus exploration on: {user_requirement}"
        
        formatted_requirement = response.strip()
        logger.info(f"Formatted user requirement: {formatted_requirement}")
        return formatted_requirement
        
    except Exception as e:
        logger.error(f"Error formatting exploration requirement: {e}")
        # Return simplified format on error
        return f"Focus exploration on: {user_requirement}"
