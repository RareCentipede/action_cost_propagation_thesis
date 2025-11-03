from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Sequence, Tuple, List, Optional, NewType, Dict, Union, Callable, Type, ClassVar, cast
from abc import abstractmethod
from copy import deepcopy

State = NewType('State', Dict[str, Any]) # {object_name}_{variable_name}: value
SimpleCondition = NewType('SimpleCondition', Tuple[str, str, Any]) # (object_name, variable_name, value)
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

def is_action_applicable(state: State, conditions: List[SimpleCondition]) -> bool:
    for cond in conditions:
        parent_name, variable_name, value = cond
        state_key = f"{parent_name}_{variable_name}"
        current_val = state.get(state_key, None)

        if current_val != value:
            return False

    return True

def apply_action(state: State, conditions: List[SimpleCondition], effects: List[SimpleCondition]) -> State | None:
    new_state = deepcopy(state)

    action_applicable = is_action_applicable(state, conditions)
    if not action_applicable:
        return None

    for effect in effects:
        parent_name, variable_name, value = effect
        state_key = f"{parent_name}_{variable_name}"
        new_state[state_key] = value

    return new_state

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

    def update_state(self, new_state: State):
        self.states.append(new_state)

        for name, value in new_state.items():
            parent_name, variable_name = name.split('_')

            match parent_name[:-1]:
                case 'robot':
                    thing_type = Robot
                case 'object':
                    thing_type = Object
                case 'pose':
                    thing_type = Pose
                case _:             

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

# TODO: Plan tasks and timeline again
# TODO: Think about experiments and expected results, types of graphs
# TODO: What simulations are needed