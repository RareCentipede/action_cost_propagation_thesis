from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

POSE_VAR_DOMAIN = ("p1", "p2", "p3")
BLOCK__VAR_DOMAIN = ("b1", "b2", "b3")
AT_TOP_VAR_DOMAIN = (True, False)
VarDomiain = {'pos': POSE_VAR_DOMAIN, 'block': BLOCK__VAR_DOMAIN, 'at_top': AT_TOP_VAR_DOMAIN}

@dataclass
class Variable:
    """
        A typed state variable with a finite domain.
        Holds the current value and validates on assignment.
    """
    domain: Tuple[Any, ...]
    _value: Optional[Any] = field(default=None, init=False, repr=False)

    @property
    def value(self) -> Optional[Any]:
        return self._value

    @value.setter
    def value(self, v: Optional[Any]) -> None:
        if v is not None and v not in self.domain:
            raise ValueError(f"{v!r} not in domain {self.domain}")
        self._value = v

    def __call__(self, index: int) -> Optional[Any]:
        """
            Enumerate valid values by index.
        """
        return self.domain[index] if index < len(self.domain) else None

    def __str__(self) -> Any:
        if not self._value:
            return f"∅  ∈ {self.domain}"
        return f"{self._value!r} ∈ {self.domain}"

@dataclass
class Entity:
    name: str

    @property
    def state(self) -> Dict[str, Any]:
        attrs = vars(self)

        state_dict = {}
        for attr_name, attr_value in attrs.items():
            if isinstance(attr_value, Variable):
                state_dict[f"{self.name}_{attr_name}"] = attr_value.value

        return state_dict

    def __str__(self) -> str:
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.state.items())})"

@dataclass
class Object(Entity):
    at: Variable = field(default_factory=lambda: Variable(domain=POSE_VAR_DOMAIN))
    at_top: Variable = field(default_factory=lambda: Variable(domain=AT_TOP_VAR_DOMAIN))

    def __post_init__(self):
        self.at_top.value = True

@dataclass
class Pose(Entity):
    occupied_by: Variable = field(default_factory=lambda: Variable(domain=BLOCK__VAR_DOMAIN))

obj = Object(name="block1")
print(obj.at)  # Output: p1
print(obj.at(1))  # Output: p2
print(obj.at(2))  # Output: p3
obj.at.value = "p2"
print(obj.at.value)  # Output: p2
print(obj.at)
print(obj)

p1 = Pose(name="p1")
p2 = Pose(name="p2")