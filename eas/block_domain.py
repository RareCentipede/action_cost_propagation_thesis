import numpy as np

from dataclasses import dataclass, field
from typing import Tuple, List, Dict, cast

from eas.EAS import Thing, State, SimpleCondition, Domain, Node, Condition

@dataclass(eq=False)
class Ground(Thing):
    name: str = "GND"

@dataclass(eq=False)
class Pose(Thing):
    pos: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    clear: bool = True
    occupied_by: 'Object | None' = None
    on: 'Pose | Ground' = Ground()
    below: 'Pose | None' = None
    variables = ("clear", "occupied_by", "on", "below", "supported")
    _supported: bool = False

    @property
    def supported(self) -> bool:
        if self.on == 'GND':
            return True

        pose_below = cast(Pose, self.on) # Poses are never on a NonePose. Either ground or another valid pose.
        below_obj = pose_below.occupied_by

        if below_obj is not None:
            return True

        return False

    @supported.setter
    def supported(self, value: bool) -> None:
        self._supported = self.supported

@dataclass(eq=False)
class Object(Thing):
    at: Pose | None
    at_top: bool = True
    on: 'Object | None | Ground' = None
    below: 'Object | None' = None
    variables = ("at", "at_top", "on", "below")

@dataclass(eq=False)
class Robot(Thing):
    at: Pose
    holding: 'Object | None' = None
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
                SimpleCondition(('object', 'at', None)),
                SimpleCondition(('object_pose', 'occupied_by', None)),
                SimpleCondition(('object_pose', 'clear', True)),
                SimpleCondition(('object.on', 'at_top', True)),
                SimpleCondition(('object.on', 'below', None)),
                SimpleCondition(('object', 'on', None)),]
place_parameters = {'robot': Robot,
                    'object': Object,
                    'target_pose': Pose}
place_conditions = [SimpleCondition(('robot', 'at', 'target_pose')),
                    SimpleCondition(('robot', 'holding', 'object')),
                    SimpleCondition(('target_pose', 'clear', True)),
                    SimpleCondition(('target_pose', 'supported', True))]
place_effects = [SimpleCondition(('robot', 'holding', None)),
                 SimpleCondition(('robot', 'gripper_empty', True)),
                 SimpleCondition(('object', 'at', 'target_pose')),
                 SimpleCondition(('object', 'on', 'target_pose.occupied_by')),
                 SimpleCondition(('target_pose', 'occupied_by', 'object')),
                 SimpleCondition(('target_pose', 'clear', False)),
                 SimpleCondition(('target_pose.on.occupied_by', 'at_top', False)),
                 SimpleCondition(('target_pose.on.occupied_by', 'below', 'object'))]

move_conditions = cast(List[Condition], move_conditions)
move_effects = cast(List[Condition], move_effects)
pick_conditions = cast(List[Condition], pick_conditions)
pick_effects = cast(List[Condition], pick_effects)
place_conditions = cast(List[Condition], place_conditions)
place_effects = cast(List[Condition], place_effects)

domain = Domain(things={}, states=[], goal_state=State({}), actions={'move': (move_parameters, move_conditions, move_effects),
                                                                     'pick': (pick_parameters, pick_conditions, pick_effects),
                                                                     'place': (place_parameters, place_conditions, place_effects)})

def create_domain_transition_graph(domain: Domain) -> Dict[str, Node]:
    robot_dtg, block_dtg = create_nodes(domain)

    robot_nodes = list(robot_dtg.values())
    block_nodes = list(block_dtg.values())

    connect_robot_nodes(robot_nodes)
    connect_block_nodes(block_nodes)

    dtg = {**robot_dtg, **block_dtg}
    return dtg

def create_nodes(domain: Domain) -> Tuple[Dict[str, Node], Dict[str, Node]]:
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
            block_dtg[node_name] = Node(name=node_name, values=(robot, block, pose))
            none_node_name = f"{block.name}_at_None"
            if block_dtg.get(none_node_name, None) is None:
                block_dtg[none_node_name] = Node(name=none_node_name, values=(robot, block, None))

    return robot_dtg, block_dtg

def connect_robot_nodes(robot_nodes: List[Node]) -> None:
    while robot_nodes:
        node = robot_nodes.pop(0)

        for other_node in robot_nodes:
            edge = ('move', other_node)

            node.edges.append(edge)
            other_node.edges.append(('move', node))

def connect_block_nodes(block_nodes: List[Node]) -> None:
    while block_nodes:
        node = block_nodes.pop(0)
        if len(node.values) == 3:
            _, thing, value = node.values
        else:
            thing, value = node.values

        for other_node in block_nodes:
            if len(other_node.values) == 3:
                _, other_thing, other_value = other_node.values
            else:
                other_thing, other_value = other_node.values

            if thing.name != other_thing.name:
                continue

            if value is None:
                node.edges.append(('place', other_node))
                other_node.edges.append(('pick', node))
            elif other_value is None:
                other_node.edges.append(('place', node))
                node.edges.append(('pick', other_node))

def create_goal_nodes(domain: Domain, dtg: Dict[str, Node]) -> Dict[str, Node]:
    goal_nodes = {}
    for var, val in domain.goal_state.items():
        dtg_key = f"{var}_{val}"
        goal_node = dtg.get(dtg_key, None)

        if goal_node:
            goal_nodes[dtg_key] = goal_node

    return goal_nodes