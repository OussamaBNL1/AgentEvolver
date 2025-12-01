from .base_env import BaseEnvironment

# Direct import of real environments, no simulators by default
try:
    from scienceworld import ScienceWorldEnv
    SCIWORLD_AVAILABLE = True
except ImportError:
    SCIWORLD_AVAILABLE = False
    ScienceWorldEnv = None

try:
    import textworld
    TEXTWORLD_AVAILABLE = True
except ImportError:
    TEXTWORLD_AVAILABLE = False
    textworld = None

__all__ = ['BaseEnvironment', 'ScienceWorldEnv', 'textworld', 
           'SCIWORLD_AVAILABLE', 'TEXTWORLD_AVAILABLE']
