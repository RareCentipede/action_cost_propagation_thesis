from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Tuple, List, Optional, NewType, Dict, Union, Callable, Type
from abc import abstractmethod

Poses = NewType('Poses', Dict[str, 'Pose'])
Objects = NewType('Objects', Dict[str, 'Object'])
State = NewType('State', Dict[str, Any])
Condition = Callable[..., bool]

@dataclass
class Thing:
    name: str

@dataclass
class Pose(Thing):
    name: str
    pos: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    clear: bool = True
    occupied_by: Optional['Object'] = None

@dataclass
class Object(Thing):
    name: str
    at: Pose
    at_top: bool = True
    on: Optional['Object'] = None
    below: Optional['Object'] = None

@dataclass
class Robot(Thing):
    name: str
    at: Pose
    holding: Optional[Object] = None
    gripper_empty: bool = True


move_conditions = [lambda robot, start_pose, target_pose: robot.at == start_pose,
                   lambda robot, start_pose, target_pose: robot.at != target_pose]
move_effects = [lambda robot, start_pose, target_pose: setattr(robot, 'at', target_pose)]

def action(parameters: Dict[str, Thing], conditions: List[Condition], effects: List[Callable]) -> bool:
    if not all(cond(*parameters.values()) for cond in conditions):
        return False

    for effect in effects:
        effect(*parameters.values())

    return True

@dataclass
class Domain:
    things: Dict[str, List[Thing]]
    states: List[State] = field(default_factory=list)
    actions: Dict[str, Tuple[Dict[str, Thing], List[Condition], List[Callable]]] = field(default_factory=dict)

p1 = Pose(name="p1", pos=(0, 0, 0))
p2 = Pose(name="p2", pos=(0, 0, 0))
p3 = Pose(name="p3", pos=(1, 1, 0))
p4 = Pose(name="p4", pos=(1, 1, 0))
p5 = Pose(name="p5", pos=(2, 2, 0))

robot = Robot(name="robot1", at=p1)
block_1 = Object(name="block1", at=p2)
block_2 = Object(name="block2", at=p3)
block_3 = Object(name="block3", at=p4)

things = {
    'robot': [robot],
    'poses': [p1, p2, p3, p4, p5],
    'objects': [block_1, block_2, block_3],
}

domain = Domain(things=things, actions={'move': (things, move_conditions, move_effects)})

# Build Domain Transition Graphs (DTGs)

p_dict = {pose.name: pose for pose in things['poses']}
poses = Enum('PosesEnum', p_dict)

@dataclass
class TestRobot:
    at: poses

    @property
    def at_vals(self):
        return list(poses)

r = TestRobot(at=poses['p1'])
print(r.at.name)

for p in r.at_vals:
    print(p.value)