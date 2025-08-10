from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Session:
    id: str
    user_id: str


class InMemorySessionService:
    def __init__(self):
        self._sessions = {}
        self._counter = 0

    async def create_session(self, app_name: str, user_id: str) -> _Session:
        self._counter += 1
        s = _Session(id=str(self._counter), user_id=user_id)
        self._sessions[s.id] = s
        return s
