import os
import time
from typing import List, Dict, Any, Optional
import requests
import json
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DashScopeClient:
    """Aliyun DashScope API client"""
    
    def __init__(self, api_key: str = None, model_name: str = "qwen-plus", 
                 temperature: float = 0.7, max_tokens: int = 2048):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Please set DASHSCOPE_API_KEY environment variable or pass it directly.")
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Make a chat completion request"""
        url = f"{self.base_url}/chat/completions"
        # Handle max_tokens parameter
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        
        # Merge parameters
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=params, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
            else:
                logger.error(f"Unexpected response format: {result}")
                return ""
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return ""
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response: {e}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error in API call: {e}")
            return ""
    
    def chat_with_retry(self, messages: List[Dict[str, str]], max_retries: int = 3, 
                       retry_delay: float = 1.0, **kwargs) -> str:
        """Chat completion with retry mechanism"""
        for attempt in range(max_retries):
            try:
                result = self.chat_completion(messages, **kwargs)
                if result:  # Valid response
                    return result
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
            if attempt < max_retries - 1:  # Not the last attempt
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
        
        logger.error(f"All {max_retries} attempts failed")
        return ""


class PromptManager:
    """Prompt manager for formatting different prompt types"""
    
    @staticmethod
    def format_system_message(content: str) -> Dict[str, str]:
        """Format a system message"""
        return {"role": "system", "content": content}
    
    @staticmethod
    def format_user_message(content: str) -> Dict[str, str]:
        """Format a user message"""
        return {"role": "user", "content": content}
    
    @staticmethod
    def format_assistant_message(content: str) -> Dict[str, str]:
        """Format an assistant message"""
        return {"role": "assistant", "content": content}
    
    @staticmethod
    def build_conversation(system_prompt: str, user_inputs: List[str], 
                          assistant_responses: List[str] = None) -> List[Dict[str, str]]:
        """Build a full conversation"""
        messages = [PromptManager.format_system_message(system_prompt)]
        
        assistant_responses = assistant_responses or []
        
        for i, user_input in enumerate(user_inputs):
            messages.append(PromptManager.format_user_message(user_input))
            if i < len(assistant_responses):
                messages.append(PromptManager.format_assistant_message(assistant_responses[i]))
        
        return messages
