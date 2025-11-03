from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Sequence, Tuple, List, Optional, NewType, Dict, Union, Callable, Type, ClassVar, cast
from abc import abstractmethod

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

def is_action_applicable(parameters: Dict[str, Thing], conditions: List[SimpleCondition]) -> bool:
    for cond in conditions:
        if isinstance(cond, Tuple):
            parent_name, attr, value = cond
            parent = parameters[parent_name]
            if isinstance(value, str):
                value = parameters[value]
                current_val = getattr(parent, attr)
                print(parent, attr, value, current_val)

            if not (current_val == value):
                return False

    return True

def apply_action(parameters: Dict[str, Thing], conditions: List[SimpleCondition], effects: List[SimpleCondition]) -> bool:
    if not is_action_applicable(parameters, conditions):
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