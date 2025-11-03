import numpy as np

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Sequence, Tuple, List, Optional, NewType, Dict, Union, Callable, Type, ClassVar, cast
from abc import abstractmethod

from eas.EAS import (Thing, State, SimpleCondition, Domain, Node,
                     is_action_applicable, apply_action,
                     ComputedCondition, Condition)

@dataclass(eq=False)
class NonePose(Thing):
    name: str = "NonePose"

@dataclass(eq=False)
class NoneObj(Thing):
    name: str = "NoneObject"

@dataclass(eq=False)
class Pose(Thing):
    pos: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    clear: bool = True
    occupied_by: 'Object | NoneObj' = field(default_factory=NoneObj)
    variables = ("clear", "occupied_by")

@dataclass(eq=False)
class Object(Thing):
    at: Pose | NonePose
    at_top: bool = True
    on: 'Object | NoneObj' = field(default_factory=NoneObj)
    below: 'Object | NoneObj' = field(default_factory=NoneObj)
    variables = ("at", "at_top", "on", "below")

@dataclass(eq=False)
class Robot(Thing):
    at: Pose
    holding: 'Object | NoneObj' = field(default_factory=NoneObj)
    gripper_empty: bool = True
    variables = ("at", "holding", "gripper_empty")

move_parameters = {'robot': Robot,
                   'start_pose': Pose,
                   'target_pose': Pose}
move_conditions = [SimpleCondition(('robot', 'at', 'start_pose'))]
move_effects = [SimpleCondition(('robot', 'at', 'target_pose'))]

pick_parameters = {'robot': Robot,
                   'object': Object,
                   'object_pose': Pose}
pick_conditions = [SimpleCondition(('robot', 'at', 'object_pose')),
                   SimpleCondition(('robot', 'gripper_empty', True)),
                   SimpleCondition(('object', 'at', 'object_pose')),
                   SimpleCondition(('object', 'at_top', True))]
pick_effects = [SimpleCondition(('robot', 'holding', 'object')),
                SimpleCondition(('robot', 'gripper_empty', False)),
                SimpleCondition(('object', 'at', NonePose())),
                SimpleCondition(('object_pose', 'occupied_by', NoneObj()))]

place_parameters = {'robot': Robot,
                    'object': Object,
                    'target_pose': Pose}
place_conditions = [SimpleCondition(('robot', 'at', 'target_pose')),
                    SimpleCondition(('robot', 'holding', 'object')),
                    SimpleCondition(('target_pose', 'clear', True))]
place_effects = [SimpleCondition(('robot', 'holding', NoneObj())),
                 SimpleCondition(('robot', 'gripper_empty', True)),
                 SimpleCondition(('object', 'at', 'target_pose')),
                 SimpleCondition(('target_pose', 'occupied_by', 'object')),
                 SimpleCondition(('target_pose', 'clear', False))]

block_domain = Domain(things={}, states=[], goal_state=State({}), actions={'move': (move_parameters, move_conditions, move_effects),
                                                                           'pick': (pick_parameters, pick_conditions, pick_effects),
                                                                           'place': (place_parameters, place_conditions, place_effects)})
