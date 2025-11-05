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

def create_domatin_transition_graph(domain: Domain) -> Dict[str, Node]:
    robot_dtg = {}
    block_dtg = {}

    robots = domain.things.get(Robot, [])
    if robots:
        robot = robots[0]

    for pose in domain.things.get(Pose, []):
        node_name = f"{robot.name}_at_{pose.name}"
        robot_dtg[node_name] = Node(name=node_name, values=(robot, pose))

        for block in domain.things.get(Object, []):
            node_name = f"{block.name}_at_{pose.name}"
            block_dtg[node_name] = Node(name=node_name, values=(block, pose))

            none_node_name = f"{block.name}_at_None"
            if block_dtg.get(none_node_name, None) is None:
                block_dtg[none_node_name] = Node(name=none_node_name, values=(block, NonePose()))

    robot_nodes = list(robot_dtg.values())
    while robot_nodes:
        node = robot_nodes.pop(0)

        for other_node in robot_nodes:
            edge = ('move', other_node)

            node.edges.append(edge)
            other_node.edges.append(('move', node))

    block_nodes = list(block_dtg.values())
    while block_nodes:
        node = block_nodes.pop(0)
        thing, value = node.values

        for other_node in block_nodes:
            other_thing, other_value = other_node.values

            if thing.name != other_thing.name:
                continue

            if isinstance(value, NonePose):
                node.edges.append(('place', other_node))
                other_node.edges.append(('pick', node))
            elif isinstance(other_value, NonePose):
                other_node.edges.append(('place', node))
                node.edges.append(('pick', other_node))

    dtg = {**robot_dtg, **block_dtg}
    return dtg
