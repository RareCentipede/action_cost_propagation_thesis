from enum import Enum
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class Pos:
    name: str
    pos: Tuple[float, float, float]
    clear: bool = True
    occupied_by: Optional['Object'] = None

@dataclass
class Object:
    name: str
    at: Pos
    at_top: bool = True
    on: Optional['Object'] = None
    below: Optional['Object'] = None

@dataclass
class Robot:
    name: str
    at: Pos
    holding: Optional[Object] = None
    gripper_empty: bool = True

