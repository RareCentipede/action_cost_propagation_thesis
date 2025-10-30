import numpy as np

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Sequence, Tuple, List, Optional, NewType, Dict, Union, Callable, Type, ClassVar, cast
from abc import abstractmethod

Poses = NewType('Poses', Dict[str, 'Pose'])
Objects = NewType('Objects', Dict[str, 'Object'])
State = NewType('State', Dict[str, Any])
SimpleCondition = NewType('SimpleCondition', Tuple[str, str, Any])
ComputedCondition = Callable[..., bool]
Condition = Union[SimpleCondition, ComputedCondition]

@dataclass(eq=False)
class Thing:
    name: str
    # variables describe which attributes compose the state for this Thing.
    # They are class-level and should not be part of the dataclass init/fields.
    variables: ClassVar[Tuple[str, ...]] = ()
    @property
    def state(self) -> State:
        state = State({})

        for var in self.variables:
            var_val = getattr(self, var, None)
            val = getattr(var_val, 'name', var_val)
            state[f"{self.name}_{var}"] = val

        return state

    def __str__(self):
        return self.state.__str__()

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, Thing):
            return False
        return self.name == other.name

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

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
move_conditions = [SimpleCondition(('robot', 'at', 'start_pose')),
                   SimpleCondition(('target_pose', 'occupied_by', '~robot'))]
move_effects = [SimpleCondition(('robot', 'at', 'target_pose')),
                SimpleCondition(('start_pose', 'occupied_by', '~robot'))]

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

def apply_action(parameters: Dict[str, Thing], conditions: List[SimpleCondition], effects: List[SimpleCondition]) -> bool:
    for cond in conditions:
        if isinstance(cond, Tuple):
            parent_name, attr, value = cond
            parent = parameters[parent_name]
            if isinstance(value, str):
                value = parameters[value]

            if not (getattr(parent, attr) == value):
                if not (isinstance(value, str) and value.startswith('~')):
                    return False

    for effect in effects:
        if isinstance(effect, Tuple):
            parent_name, attr, value = effect
            parent = parameters[parent_name]

            if isinstance(value, str):
                value = parameters[value]

            setattr(parent, attr, value)

    return True

@dataclass
class Domain:
    things: Dict[Type[Thing], List[Thing]]
    states: List[State] = field(default_factory=list)
    actions: Dict[str, Tuple[Dict[str, Thing],
                             Sequence[SimpleCondition | ComputedCondition],
                             Sequence[SimpleCondition | ComputedCondition]]] = field(default_factory=dict)

    @property
    def current_state(self) -> State:
        return self.states[-1] if self.states else State({})

p1 = Pose(name="p1", pos=(0, 0, 0))
p2 = Pose(name="p2", pos=(0, 0, 0))
p3 = Pose(name="p3", pos=(1, 1, 0))
p4 = Pose(name="p4", pos=(1, 1, 0))
p5 = Pose(name="p5", pos=(2, 2, 0))

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
    
domain = Domain(things=things_init, states=[init_state], actions={'move': (move_parameters, move_conditions, move_effects)})

# Build Domain Transition Graphs (DTGs)
dtg = {}

@dataclass
class Node:
    name: str
    type: str
    values: Tuple[Any, ...]
    applicable_actions: List[str] = field(default_factory=list)
    edges: List[Tuple[str, 'Node']] = field(default_factory=list)

    def __str__(self):
        node_name = f"Node: {self.name}, "
        values = f"values: {[f'{value.name}: {type(value).__name__}' if hasattr(value, 'name') else value for value in self.values]}, "
        applicable_actions = f"applicable_actions: {self.applicable_actions}, "
        edges = f"edges: {[(edge[0], edge[1].name if hasattr(edge[1], 'name') else edge[1]) for edge in self.edges]}"
        return node_name + values + applicable_actions + edges

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

                # print(dtg[node_name])

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
# for node in pick_place_nodes:
#     node_type = node.type
#     other_nodes = [n for n in pick_place_nodes if n != node]

#     match node_type:
#         case 'holding':
#             other_nodes = [n for n in other_nodes if n.type == 'holding']
#             node.edges = [('pick', other) for other in other_nodes]

#             for other in other_nodes:
#                 other.edges.append(('place', node))

    # print(node)
print(node_types)
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


for node in dtg.values():
    print(node)