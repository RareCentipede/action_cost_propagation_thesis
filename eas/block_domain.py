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

p1 = Pose(name="p1", pos=(0, 0, 0))
p2 = Pose(name="p2", pos=(0, 0, 0))
p3 = Pose(name="p3", pos=(1, 1, 0))
p4 = Pose(name="p4", pos=(1, 1, 0))
p5 = Pose(name="p5", pos=(2, 2, 0))
p6 = Pose(name="p6", pos=(2, 2, 0))
p7 = Pose(name="p7", pos=(3, 3, 0))
p8 = Pose(name="p8", pos=(3, 3, 0))

robot = Robot(name="robot1", at=p1)
block_1 = Object(name="block1", at=p2)
block_2 = Object(name="block2", at=p3)
block_3 = Object(name="block3", at=p4)

things_init = {
    Robot: [robot],
    Pose: [p1, p2, p3, p4, p5],
    Object: [block_1, block_2, block_3],
}

init_state = State({})
for thing_list in things_init.values():
    for thing in thing_list:
        init_state.update(thing.state)
    
domain = Domain(things=things_init, states=[init_state], actions={'move': (move_parameters, move_conditions, move_effects),
                                                                  'pick': (pick_parameters, pick_conditions, pick_effects),
                                                                  'place': (place_parameters, place_conditions, place_effects)})

# Build Domain Transition Graphs (DTGs)
dtg = {}

# Different approach:
# First get all the variables (predicates)
# Then for each variable, find all possible values it can take and build nodes
# Finally, connect nodes based on possible actions

# TODO: Need to take care of the cases where block is on/below itself
for thing_type, things in domain.things.items():
    for thing in things:
        for attribute in thing.variables:
            if attribute != 'at' and attribute != 'holding':
                continue

            vars = []
            var_type = type(getattr(thing, attribute))

            if var_type is NonePose:
                var_type = Pose
                vars = [cast(Thing, NonePose())]
            elif var_type is NoneObj:
                var_type = Object
                vars = [cast(Thing, NoneObj())]
            elif var_type is Pose:
                if thing_type is not Robot:
                    vars = [cast(Thing, NonePose())]
            elif var_type is Object:
                vars = [cast(Thing, NoneObj())]

            if var_type is bool:
                vars = [True, False]
            else:
                vars.extend(domain.things.get(var_type, []))

            for var in vars:
                var_name = getattr(var, 'name', var)
                node_name = f"{thing.name}_{attribute}_{var_name}"
                dtg[node_name] = Node(name=node_name, type=attribute, values=(thing, var))
                # Maybe add applicable actions to the nodes here based on the values

                if type(thing) is Robot and attribute == 'at':
                    # Move action only applies to robot's 'at' attribute
                    dtg[node_name].applicable_actions.append('move')

                else:
                    dtg[node_name].applicable_actions = ['pick', 'place']

# Get only nodes involving robot, move actions are only for robots
robot_nodes = [node for node in dtg.values() if type(node.values[0]) is Robot]

new_dtg = {}

# Get all values for corresponding parameters in move action
# move edges
move_nodes = [node for node in dtg.values() if 'move' in node.applicable_actions]
for node in move_nodes:
    other_nodes = [n for n in move_nodes if n != node]
    for other in other_nodes:
        edge = ('move', other)

        if edge not in node.edges:
            node.edges.append(edge)

pick_place_nodes = [node for node in dtg.values() if 'move' not in node.applicable_actions]
node_types = set(node.type for node in pick_place_nodes)

for node_type in node_types:
    selected_nodes = [node for node in pick_place_nodes if node.type == node_type]

    match node_type:
        case 'at':
            none_nodes = [n for n in selected_nodes if n.values[1] in (NoneObj(), NonePose())]
            other_nodes = [n for n in selected_nodes if n not in none_nodes and n.values[1] not in (NoneObj(), NonePose())]

            for none_node in none_nodes:
                same_obj_nodes = [n for n in other_nodes if n.values[0].name == none_node.values[0].name]
                none_node.edges = [('place', other) for other in same_obj_nodes]

                for other in same_obj_nodes:
                    other.edges.append(('pick', none_node))

        case 'gripper_empty':
            true_node = [n for n in selected_nodes if n.values[1] is True][0]
            false_node = [n for n in selected_nodes if n.values[1] is False][0]

            true_node.edges = [('pick', false_node)]
            false_node.edges = [('place', true_node)]

        case 'holding':
            none_node = [n for n in selected_nodes if n.values[1] in (NoneObj(), NonePose())][0]
            other_nodes = [n for n in selected_nodes if n != none_node]

            none_node.edges = [('pick', other) for other in other_nodes]
            for other in other_nodes:
                other.edges.append(('place', none_node))


# for node in dtg.values():
#     print(node)
goal_state = State({
    'block1_at': p6,
    'block2_at': p7,
    'block3_at': p8
})

goal_reached = False

while not goal_reached:
    # Check if goal state is reached
    goal_conds = []
    for goal_var, goal_val in goal_state.items():
        current_val = domain.current_state.get(goal_var, None)
        goal_conds.append(current_val == goal_val)

    if all(goal_conds):
        goal_reached = True
        print("Goal reached!")
        break

    current_state_nodes = []
    for var, val in domain.current_state.items():
        node_name = f"{var}_{val}"
        current_state_nodes.append(dtg.get(node_name, None))

    # Here you can implement the logic to choose and apply actions based on the DTG
    visited_node = []
    for node in current_state_nodes:
        if node and node not in visited_node:
            visited_node.append(node)

            for edge in node.edges:
                action_name, target_node = edge

                action = domain.actions.get(action_name, None)
                if action is None:
                    continue

                action_conds = action[1]
                action_params = action[0]

                param_names = list(action_params.keys())
                params = {param_names[0]: node.values[0],
                          param_names[1]: node.values[1],
                          param_names[2]: target_node.values[1]}

                # print(node)

                # if is_action_applicable(params, list(action_conds)):
                    # apply_action(params, action_conds, domain.actions.get(action_name, (None, [], []))[2])
                    # print(f"From Node: {node.name} --[{action_name}]--> To Node: {target_node.name}")

                # if action_params is not None and action_conds is not None:
                #     if is_action_applicable(params, list(action_conds)):
                #         apply_action(params, action_conds, domain.actions.get(action_name, (None, [], []))[2])
                #         print(f"From Node: {node.name} --[{action_name}]--> To Node: {target_node.name}")