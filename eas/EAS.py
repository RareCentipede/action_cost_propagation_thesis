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

def is_action_applicable(conditions: List[SimpleCondition], parameters: Dict[str, Thing]) -> bool:
    for cond in conditions:
        parent_name, variable_name, target_name = cond
        param = parameters.get(parent_name)
        target = parameters.get(target_name)

        current_val = getattr(param, variable_name, None) if param else None
        if current_val != target:
            return False

    return True

def apply_action(state: State, conditions: List[SimpleCondition], parameters: Dict[str, Thing], effects: List[SimpleCondition]) -> State:
    new_state = deepcopy(state)

    action_applicable = is_action_applicable(conditions, parameters)
    if not action_applicable:
        return State({})

    for effect in effects:
        parent_name, variable_name, target_name = effect
        parent = parameters.get(parent_name)

        if not parent:
            return State({})

        state_key = f"{parent.name}_{variable_name}"
        target = parameters.get(target_name)

        if not target:
            return State({})

        new_state.update({state_key: target.name})

    return new_state

@dataclass
class Domain:
    things: Dict[Type[Thing] | str, List[Thing] | Thing]
    states: List[State]
    goal_state: State
    actions: Dict[str, Tuple[Dict[str, Thing],
                             List[SimpleCondition] | List[ComputedCondition],
                             List[SimpleCondition] | List[ComputedCondition]]] = field(default_factory=dict)

    def map_name_to_things(self):
        things_copy = deepcopy(self.things)
        for things in things_copy.values():
            if isinstance(things, list):
                for thing in things:
                    self.things[thing.name] = thing

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

    def update_state(self, new_state: State):
        self.states.append(new_state)

        for name, value in new_state.items():
            parent_name, variable_name = tuple(name.split('_', 1))
            thing = self.things[parent_name]
            if thing and hasattr(thing, variable_name):
                setattr(thing, variable_name, value)

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