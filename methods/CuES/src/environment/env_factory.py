from typing import Dict, Any, Union
from ..utils.logger import get_logger

logger = get_logger(__name__)


def create_environment(env_config: Dict[str, Any]) -> Any:
    """Create an environment instance (inspired by SciWorld code you provided)"""
    env_type = env_config.get("type", "sciworld").lower()
    
    if env_type == "sciworld":
        return create_sciworld_environment(env_config.get("sciworld", {}))
    elif env_type == "textworld":
        return create_textworld_environment(env_config.get("textworld", {}))
    else:
        raise ValueError(f"Unsupported environment type: {env_type}")


def create_sciworld_environment(task_config: Dict[str, Any]):
    """Create a SciWorld environment instance (based on your code)"""
    try:
        from scienceworld import ScienceWorldEnv
        
        var_num = task_config.get("var_num", 0)
        task_name = task_config.get("task_name", "task-1-boil-water")
        jar_path = task_config.get("jar_path", "./environments/sciworld.jar")
        simplification = task_config.get("simplification", "easy")
        env_step_limit = task_config.get("env_step_limit", 100)
        
        logger.info(f"Creating SciWorld environment: {task_name}, var_num: {var_num}")
        
        # Create environment following SciWorld example
        env = ScienceWorldEnv("", jar_path, envStepLimit=env_step_limit)
        env.load(task_name, var_num, simplification, generateGoldPath=True)
        
        return env
        
    except ImportError as e:
        error_message = (
            f"Error importing ScienceWorldEnv: {str(e)}. "
            "Please make sure you have installed the sciworld package successfully, "
            "following the instructions in https://github.com/allenai/ScienceWorld"
        )
        logger.error(error_message)
        raise ImportError(error_message)


def create_textworld_environment(task_config: Dict[str, Any]):
    """Create a TextWorld environment instance"""
    try:
        import textworld
        
        game_file = task_config.get("game_file")
        
        if game_file:
            # Load specific game file
            env = textworld.start(game_file)
        else:
            # Create a default simple environment
            env = textworld.start()
            
        logger.info("TextWorld environment created")
        return env
        
    except ImportError as e:
        error_message = (
            f"Error importing TextWorld: {str(e)}. "
            "Please install TextWorld following the instructions at "
            "https://github.com/microsoft/TextWorld"
        )
        logger.error(error_message)
        raise ImportError(error_message)
