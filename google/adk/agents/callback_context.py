from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class State:
    data: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def update(self, other: Dict[str, Any]) -> None:
        self.data.update(other)

    def setdefault(self, key: str, default: Any = None) -> Any:
        return self.data.setdefault(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.data)


@dataclass
class CallbackContext:
    agent_name: str
    state: State


ReadonlyContext = CallbackContext  # simple alias for now
