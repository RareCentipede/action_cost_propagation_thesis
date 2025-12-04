from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Tuple, List, NewType, Dict, Union, Callable, Type, ClassVar, cast
from copy import deepcopy
    
State = NewType('State', Dict[str, Any]) # {object_name}_{variable_name}: value
Action = NewType('Action', Tuple[str, List[Any]]) # (action_name, [param1, param2, ...])
ConditionType = Enum('ConditionType', 'SIMPLE COMPUTED')
StateStatus = Enum('StateStatus', 'ALIVE DEAD GOAL')

@dataclass
class Effect:
    name: str
    src_name: str
    var_name: str
    target_value: Any

@dataclass
class Condition:
    name: str
    cond_tp: ConditionType
    src_name: str
    var_name: str
    target_value: Any


@dataclass(eq=False)
class LinkedState:
    state_id: int
    state: State
    type_: StateStatus = StateStatus.ALIVE
    parent: 'Tuple[Action, LinkedState] | None' = None # Parent state and the action connecting them. Only the root node has no parent
    branches_to_explore: List[Tuple['Node', str, 'Node']] = field(default_factory=list)  # home node, action name, target node
    edges: List[Tuple[str, 'LinkedState']] = field(default_factory=list) # action name, linked state

    def __hash__(self):
        return hash(self.state.__str__())

    def __eq__(self, other):
        if not isinstance(other, LinkedState):
            return False
        return self.state == other.state

    def __str__(self):
        return f"State {self.state_id} --> {[e[1].state_id for e in self.edges]}"

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

@dataclass
class Domain:
    things: Dict[Type[Thing], List[Thing]]
    states: List[State]
    goal_state: State
    actions: Dict[str, Tuple[Dict[str, Thing],
                             List[Condition],
                             List[Condition]]] = field(default_factory=dict)
    name_things: Dict[str, Thing] = field(default_factory=dict)

    @property
    def current_state(self) -> State:
        return self.states[-1] if self.states else State({})

    @property
    def goal_reached(self) -> bool:
        for goal_key, goal_value in self.goal_state.items():
            current_value = self.current_state.get(goal_key, None)
            if current_value != goal_value:
                return False

        return True

    def update_state(self, new_state: State) -> None:
        self.states.append(new_state)

        for name, value_name in new_state.items():
            parent_name, variable_name = tuple(name.split('_', 1))
            thing = self.name_things[parent_name]
            value = self.name_things.get(value_name, value_name)

            if thing and hasattr(thing, variable_name):
                if variable_name == 'supported':
                    continue
                setattr(thing, variable_name, value)

    def reset_state(self) -> None:
        """
            Remove the last state and update the state values according to the new last state.
        """
        self.states.pop(-1)
        self.update_state(self.current_state)
        self.states.pop(-1)

@dataclass
class Node:
    name: str
    values: Tuple[Any, ...]
    edges: List[Tuple[str, 'Node']] = field(default_factory=list)

    def __str__(self):
        node_name = f"Node: {self.name}, "
        values = f"values: {[f'{value.name}: {type(value).__name__}' if hasattr(value, 'name') else value for value in self.values]}, "
        edges = f"edges: {[(edge[0], edge[1].name if hasattr(edge[1], 'name') else edge[1]) for edge in self.edges]}"
        return node_name + values + edges

def is_action_applicable(conditions: List[Condition], parameters: Dict[str, Thing], verbose: bool = False) -> bool:
    for cond in conditions:
        parent_name, variable_name, target_name = cond.src_name, cond.var_name, cond.target_value
        param = parameters.get(parent_name)

        if type(target_name) is str:
            target = parameters.get(target_name)
        else:
            target = target_name

        if not param:
            raise ValueError(f"Parameter {parent_name} not found in parameters")

        current_val = getattr(param, variable_name)

        if current_val != target:
            if verbose:
                print(f"\nCondition {cond.name} failed: {cond.src_name}_{cond.var_name} is {current_val}, expected {target}\n")
            return False

    return True

def apply_action(state: State, conditions: List[Condition], parameters: Dict[str, Thing], effects: List[Effect]) -> State:
    new_state = deepcopy(state)

    action_applicable = is_action_applicable(conditions, parameters)
    if not action_applicable:
        print("Action not applicable!")
        return State({})

    for effect in effects:
        parent_name, variable_name, target_name = effect.src_name, effect.var_name, effect.target_value

        # In case of nested attributes like 'target_pose.on.occupied_by'
        if '.' not in parent_name:
            parent = parameters.get(parent_name)
        else:
            attrs = parent_name.split('.')
            ancestor = parameters.get(attrs[0])
            parent = ancestor

            for attr in attrs[1:]:
                parent = getattr(parent, attr, None)
                if parent and parent.name == 'GND':
                    break

        if parent and parent.name == 'GND':
            continue

        if not parent:
            raise ValueError(f"Parent {parent_name} for {ancestor.name if ancestor else 'unknown'} not found in parameters when applying effect {effect.name}")

        state_key = f"{parent.name}_{variable_name}"

        if type(target_name) is str:
            if '.' not in target_name:
                target = parameters.get(target_name)
                target = target.name if target else None
            else:
                attrs = target_name.split('.')
                target = parameters.get(attrs[0])

                for attr in attrs[1:]:
                    target = getattr(target, attr, None)
                    if target:
                        target = target.name
                        if target == 'GND':
                            break
        else:
            target = target_name

        new_state.update({state_key: target})

    return new_state

def parse_action_params(action_name: str, node: Node, target: Node) -> Dict[str, Thing]:
    action_params = {}
    match action_name:
        case 'move':
            action_params = {
                'robot': node.values[0],
                'start_pose': node.values[1],
                'target_pose': target.values[1]
            }

        case 'pick':
            action_params = {
                'robot': node.values[0],
                'object': node.values[1],
                'object_pose': node.values[2]
            }

        case 'place':
            action_params = {
                'robot': node.values[0],
                'object': node.values[1],
                'target_pose': target.values[-1]
            }

        case _:
            raise ValueError(f"Unknown action: {action_name}")

    return action_params

def query_nodes(dtg: Dict[str, Node], state: State) -> List[Node]:
    nodes = []
    for var, val in state.items():
        dtg_key = f"{var}_{val}"
        node = dtg.get(dtg_key, None)
        if node:
            nodes.append(node)
    return nodes

def query_current_nodes(dtg: Dict[str, Node], current_state: State, goal_nodes: Dict[str, Node]) -> List[Node]:
    current_nodes = []
    for var, val in current_state.items():
        dtg_key = f"{var}_{val}"
        current_node = dtg.get(dtg_key, None)
        if current_node and current_node not in goal_nodes.values():
            current_nodes.append(current_node)
    return current_nodes