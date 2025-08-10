from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Optional

from google.genai import types
from google.adk.agents.llm_agent import SequentialAgent, Agent


@dataclass
class _Session:
    id: str
    user_id: str


class _InMemorySessionService:
    def __init__(self):
        self._sessions = {}
        self._counter = 0

    async def create_session(self, app_name: str, user_id: str) -> _Session:
        self._counter += 1
        s = _Session(id=str(self._counter), user_id=user_id)
        self._sessions[s.id] = s
        return s


class InMemoryRunner:
    def __init__(self, agent: Agent | SequentialAgent, app_name: str):
        self.agent = agent
        self.app_name = app_name
        self.session_service = _InMemorySessionService()

    async def run_async(
        self,
        user_id: str,
        session_id: str,
        new_message: types.Content,
    ) -> AsyncGenerator[types.Content, None]:
        state = self._ensure_state()
        # Place message into state if desired
        state["last_user_message"] = new_message
        # Execute the agent synchronously for now
        self.agent.run(state)
        # Retrieve a reply from state
        reply_text = state.get("last_model_text", "")
        yield types.Content(parts=[types.Part(text=reply_text)], role="model")

    def _ensure_state(self):
        from google.adk.agents.callback_context import State
        if not hasattr(self, "_state"):
            self._state = State()
        return self._state


Runner = InMemoryRunner
