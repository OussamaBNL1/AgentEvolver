import yaml
import os
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Handle environment variables
    if 'api' in config and 'api_key' in config['api']:
        if not config['api']['api_key'] or config['api']['api_key'] == 'your-api-key-here':
            config['api']['api_key'] = os.getenv('DASHSCOPE_API_KEY')
    
    return config


def ensure_dir(path: str):
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)


def generate_session_id() -> str:
    """Generate a session ID"""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")
