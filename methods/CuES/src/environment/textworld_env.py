import random
from typing import Tuple, Dict, Any, List
from .base_env import BaseEnvironment
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TextWorldEnvironment(BaseEnvironment):
    """TextWorld environment simulator"""
    
    def __init__(self, max_steps: int = 30, **kwargs):
        super().__init__(max_steps)
        self.initial_observation = ""
        self.current_location = "middle of a room"
        self.inventory = []
        self.locations = {
            "cabinet 1": {"items": ["cloth 1", "soapbar 1", "soapbottle 1"]},
            "cabinet 2": {"items": ["towel 1", "shampoo 1"]},
            "cabinet 3": {"items": ["toothbrush 1", "toothpaste 1"]},
            "cabinet 4": {"items": ["perfume 1", "comb 1"]},
            "countertop 1": {"items": ["mirror 1", "soap dispenser 1"]},
            "garbagecan 1": {"items": []},
            "handtowelholder 1": {"items": ["handtowel 1"]},
            "handtowelholder 2": {"items": ["handtowel 2"]},
            "sinkbasin 1": {"items": []},
            "sinkbasin 2": {"items": []},
            "toilet 1": {"items": []},
            "toiletpaperhanger 1": {"items": ["toiletpaper 1"]},
            "towelholder 1": {"items": ["towel 2"]}
        }
        self.visited_locations = set()
        
    def get_available_actions(self) -> List[str]:
        """Get available actions"""
        return [
            "look",
            "inventory", 
            "go to (receptacle)",
            "open (receptacle)",
            "close (receptacle)", 
            "take (object) from (receptacle)",
            "move (object) to (receptacle)",
            "examine (something)",
            "use (object)",
            "heat (object) with (receptacle)",
            "clean (object) with (receptacle)",
            "cool (object) with (receptacle)",
            "slice (object) with (object)"
        ]
    
    def reset(self) -> Tuple[str, Dict[str, Any]]:
        """Reset environment"""
        self.current_step = 0
        self.done = False
        self.current_location = "middle of a room"
        self.inventory = []
        self.visited_locations = set()
        
        # Reset item locations
        self.locations = {
            "cabinet 1": {"items": ["cloth 1", "soapbar 1", "soapbottle 1"]},
            "cabinet 2": {"items": ["towel 1", "shampoo 1"]},
            "cabinet 3": {"items": ["toothbrush 1", "toothpaste 1"]},
            "cabinet 4": {"items": ["perfume 1", "comb 1"]},
            "countertop 1": {"items": ["mirror 1", "soap dispenser 1"]},
            "garbagecan 1": {"items": []},
            "handtowelholder 1": {"items": ["handtowel 1"]},
            "handtowelholder 2": {"items": ["handtowel 2"]},
            "sinkbasin 1": {"items": []},
            "sinkbasin 2": {"items": []},
            "toilet 1": {"items": []},
            "toiletpaperhanger 1": {"items": ["toiletpaper 1"]},
            "towelholder 1": {"items": ["towel 2"]}
        }
        
        observation = (
            "You are in the middle of a room. Looking quickly around you, you see "
            "a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, "
            "a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, "
            "a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1."
        )
        self.initial_observation = observation
        
        return observation, {"step": self.current_step}
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute one step"""
        self.current_step += 1
        reward = 0.0
        
        observation = self._process_action(action)
        
        # Simple reward: visiting a new location yields reward
        if "go to" in action.lower():
            location = self._extract_location(action)
            if location and location not in self.visited_locations:
                self.visited_locations.add(location)
                reward = 1.0
        
        # Check termination
        if self.current_step >= self.max_steps:
            self.done = True
        
        info = {
            "step": self.current_step,
            "current_location": self.current_location,
            "inventory": self.inventory.copy(),
            "visited_locations": list(self.visited_locations)
        }
        
        return observation, reward, self.done, info
    
    def _process_action(self, action: str) -> str:
        """Handle the action and return the observation"""
        action = action.strip().lower()
        
        if action == "look":
            return self._look_around()
        elif action == "inventory":
            return self._check_inventory()
        elif action.startswith("go to"):
            return self._go_to_location(action)
        elif action.startswith("examine"):
            return self._examine_object(action)
        elif action.startswith("take"):
            return self._take_object(action)
        elif action.startswith("open"):
            return self._open_receptacle(action)
        elif action.startswith("close"):
            return self._close_receptacle(action)
        else:
            return f"I don't understand the action: {action}"
    
    def _look_around(self) -> str:
        """Look around"""
        if self.current_location == "middle of a room":
            return self.initial_observation
        else:
            location_info = self.locations.get(self.current_location, {})
            items = location_info.get("items", [])
            if items:
                items_str = ", ".join(items)
                return f"You are at {self.current_location}. On the {self.current_location}, you see {items_str}."
            else:
                return f"You are at {self.current_location}. The {self.current_location} is empty."
    
    def _check_inventory(self) -> str:
        """Check inventory"""
        if not self.inventory:
            return "Your inventory is empty."
        else:
            items_str = ", ".join(self.inventory)
            return f"You are carrying: {items_str}."
    
    def _go_to_location(self, action: str) -> str:
        """Move to the specified location"""
        location = self._extract_location(action)
        if location and location in self.locations:
            self.current_location = location
            self.visited_locations.add(location)
            location_info = self.locations[location]
            items = location_info.get("items", [])
            if items:
                items_str = ", ".join(items)
                return f"You arrive at {location}. On the {location}, you see {items_str}."
            else:
                return f"You arrive at {location}. The {location} is empty."
        else:
            return f"You can't go to {location}. It doesn't exist or is not accessible."
    
    def _examine_object(self, action: str) -> str:
        """Examine an object"""
        # Simplified: return basic info
        object_name = action.replace("examine", "").strip()
        return f"You examine the {object_name}. It looks normal."
    
    def _take_object(self, action: str) -> str:
        """Take an object"""
        # Parse "take X from Y"
        parts = action.split(" from ")
        if len(parts) == 2:
            object_name = parts[0].replace("take", "").strip()
            location = parts[1].strip()
            
            if location in self.locations:
                items = self.locations[location]["items"]
                if object_name in items:
                    items.remove(object_name)
                    self.inventory.append(object_name)
                    return f"You take the {object_name} from {location}."
                else:
                    return f"There is no {object_name} on {location}."
            else:
                return f"You can't find {location}."
        else:
            return "Please specify what to take and from where (e.g., 'take cloth 1 from cabinet 1')."
    
    def _open_receptacle(self, action: str) -> str:
        """Open a receptacle"""
        receptacle = action.replace("open", "").strip()
        return f"You open the {receptacle}."
    
    def _close_receptacle(self, action: str) -> str:
        """Close a receptacle"""
        receptacle = action.replace("close", "").strip()
        return f"You close the {receptacle}."
    
    def _extract_location(self, action: str) -> str:
        """Extract location name from the action"""
        # Simplified: search through known locations
        for location in self.locations.keys():
            if location in action:
                return location
        return ""
    
    def close(self):
        """Close environment"""
        logger.info("TextWorld environment closed")


# TextWorld random action generator
class TextWorldRandomActionGenerator:
    """TextWorld random action generator"""
    
    def __init__(self, env: TextWorldEnvironment):
        self.env = env
        self.basic_actions = ["look", "inventory"]
        self.locations = list(env.locations.keys())
    
    def generate_random_action(self) -> str:
        """Generate a random action"""
        action_type = random.choice([
            "basic", "go_to", "examine", "take", "open", "close"
        ])
        
        if action_type == "basic":
            return random.choice(self.basic_actions)
        elif action_type == "go_to":
            location = random.choice(self.locations)
            return f"go to {location}"
        elif action_type == "examine":
            location = random.choice(self.locations)
            return f"examine {location}"
        elif action_type == "open":
            location = random.choice(self.locations)
            return f"open {location}"
        elif action_type == "close":
            location = random.choice(self.locations)
            return f"close {location}"
        elif action_type == "take":
            # Randomly choose a location with items
            available_locations = [loc for loc, info in self.env.locations.items() 
                                 if info.get("items")]
            if available_locations:
                location = random.choice(available_locations)
                items = self.env.locations[location]["items"]
                if items:
                    item = random.choice(items)
                    return f"take {item} from {location}"
        
        # Default action
        return "look"
