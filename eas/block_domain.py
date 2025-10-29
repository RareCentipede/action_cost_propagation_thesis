import numpy as np

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Tuple, List, Optional, NewType, Dict, Union, Callable, Type, ClassVar
from abc import abstractmethod

Poses = NewType('Poses', Dict[str, 'Pose'])
Objects = NewType('Objects', Dict[str, 'Object'])
State = NewType('State', Dict[str, Any])
Condition = Callable[..., bool]

@dataclass
class Thing:
    name: str
    # variables describe which attributes compose the state for this Thing.
    # They are class-level and should not be part of the dataclass init/fields.
    variables: ClassVar[Tuple[str, ...]]

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

@dataclass
class Pose(Thing):
    pos: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    clear: bool = True
    occupied_by: Optional['Object'] = None
    variables: ClassVar[Tuple[str, ...]] = ("clear", "occupied_by")

@dataclass
class Object(Thing):
    at: Pose
    at_top: bool = True
    on: Optional['Object'] = None
    below: Optional['Object'] = None
    variables: ClassVar[Tuple[str, ...]] = ("at", "at_top", "on", "below")

@dataclass
class Robot(Thing):
    at: Pose
    holding: Optional[Object] = None
    gripper_empty: bool = True
    variables: ClassVar[Tuple[str, ...]] = ("at", "holding", "gripper_empty")

move_parameters = {'robot': Robot,
                   'start_pose': Pose,
                   'target_pose': Pose}
move_conditions = [lambda robot, start_pose, target_pose: robot.at == start_pose,
                   lambda robot, start_pose, target_pose: robot.at != target_pose]
move_effects = [lambda robot, start_pose, target_pose: setattr(robot, 'at', target_pose)]

pick_parameters = {'robot': Robot,
                   'object': Object,
                   'object_pose': Pose}
pick_conditions = [lambda robot, object, object_pose: robot.at == object_pose,
                   lambda robot, object, object_pose: robot.gripper_empty,
                   lambda robot, object, object_pose: object.at == object_pose,
                   lambda robot, object, object_pose: object.at_top]
pick_effects = [lambda robot, object, object_pose: setattr(robot, 'holding', object),
                lambda robot, object, object_pose: setattr(robot, 'gripper_empty', False),
                lambda robot, object, object_pose: setattr(object, 'at', None),
                lambda robot, object, object_pose: setattr(object_pose, 'occupied_by', None)]

place_parameters = {'robot': Robot,
                    'object': Object,
                    'target_pose': Pose}
place_conditions = [lambda robot, object, target_pose: robot.at == target_pose,
                    lambda robot, object, target_pose: robot.holding == object,
                    lambda robot, object, target_pose: target_pose.clear]
place_effects = [lambda robot, object, target_pose: setattr(robot, 'holding', None),
                 lambda robot, object, target_pose: setattr(robot, 'gripper_empty', True),
                 lambda robot, object, target_pose: setattr(object, 'at', target_pose),
                 lambda robot, object, target_pose: setattr(target_pose, 'occupied_by', object),
                 lambda robot, object, target_pose: setattr(target_pose, 'clear', False)]

def apply_action(parameters: Dict[str, Thing], conditions: List[Condition], effects: List[Callable]) -> bool:
    if not all(cond(*parameters.values()) for cond in conditions):
        return False

    for effect in effects:
        effect(*parameters.values())

    return True

@dataclass
class Domain:
    things: Dict[Type[Thing], List[Thing]]
    states: List[State] = field(default_factory=list)
    actions: Dict[str, Tuple[Dict[str, Thing], List[Condition], List[Callable]]] = field(default_factory=dict)

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

# print(init_state)

domain = Domain(things=things_init, states=[init_state], actions={'move': (move_parameters, move_conditions, move_effects)})

# Build Domain Transition Graphs (DTGs)
dtg = {}
# robot_vars = list(list(robot.state.values())[0].keys())
robot_vars = []
for attr in list(robot.__annotations__.keys())[1:]:
    robot_vars.append(f"robot1_{attr}")

@dataclass
class Node:
    name: str
    values: Tuple[Any, ...]
    edges: List[Tuple[str, Any]] = field(default_factory=list)

    def __str__(self):
        node_name = f"Node: {self.name}, "
        values = f"values: {[value.name if hasattr(value, 'name') else value for value in self.values]}, "
        edges = f"edges: {[(edge[0], edge[1].name if hasattr(edge[1], 'name') else edge[1]) for edge in self.edges]}"
        return node_name + values + edges

# Different approach:
# First get all the variables (predicates)
# Then for each variable, find all possible values it can take and build nodes
# Finally, connect nodes based on possible actions

for thing_type in domain.things.values():
    for thing in thing_type:
        for variable in thing.variables:
            var_type = type(getattr(thing, variable))
            print(thing.name, var_type)
            vars = domain.things.get(var_type, [])
            for var in vars:
                node_name = f"{thing.name}_{variable}_{var.name if hasattr(var, 'name') else var}"
                dtg[node_name] = Node(name=node_name, values=(thing, var))

                print(dtg[node_name])

# Get only nodes involving robot, move actions are only for robots
robot_nodes = [node for node in dtg.values() if type(node.values[0]) is Robot]

for node in robot_nodes:
    other_nodes = [n for n in robot_nodes if n != node]
    for other in other_nodes:
        edge = ('move', other)
        if edge not in node.edges:
            node.edges.append(edge)

    print(node)