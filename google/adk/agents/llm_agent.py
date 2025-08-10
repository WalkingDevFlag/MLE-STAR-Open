from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Union

from google.genai import types
from .callback_context import CallbackContext, State
from machine_learning_engineering.shared_libraries.llm_provider import get_llm


class LlmResponse:
    def __init__(self, content: Any | None = None):
        self.content = content


class LlmRequest:
    def __init__(self):
        pass


# Protocols for type hints
class BeforeModelCallback(Protocol):
    def __call__(self, callback_context: CallbackContext, llm_request: LlmRequest, /) -> Optional[LlmResponse]:
        ...


class AfterModelCallback(Protocol):
    def __call__(self, callback_context: CallbackContext, llm_response: LlmResponse, /) -> Optional[LlmResponse]:
        ...


InstructionProvider = Union[str, Callable[[CallbackContext], str]]


@dataclass
class Agent:
    model: Optional[str]
    name: str
    description: str = ""
    instruction: InstructionProvider = ""
    global_instruction: Optional[str] = None
    before_agent_callback: Optional[Callable[[CallbackContext], Any]] = None
    before_model_callback: Optional[BeforeModelCallback] = None
    after_model_callback: Optional[AfterModelCallback] = None
    after_agent_callback: Optional[Callable[[CallbackContext], Any]] = None
    generate_content_config: Optional[types.GenerateContentConfig] = None
    include_contents: str = "all"
    tools: Optional[List[Callable[..., Any]]] = None
    # Accept a list of child agents (any agent-like object with run(state))
    sub_agents: Optional[List[Any]] = None

    def run(self, state: State) -> None:
        ctx = CallbackContext(agent_name=self.name, state=state)
        # Before-agent (context-only) callback
        if self.before_agent_callback is not None:
            self.before_agent_callback(ctx)
        # If this Agent is a container, just run sub_agents and exit
        if self.sub_agents:
            for a in self.sub_agents:
                a.run(state)
            if self.after_agent_callback is not None:
                self.after_agent_callback(ctx)
            return
        req = LlmRequest()
        # Run before-model callback
        if self.before_model_callback is not None:
            maybe_resp = self.before_model_callback(ctx, req)
            if isinstance(maybe_resp, LlmResponse):
                return
        # Build instruction
        instruction = self.instruction(ctx) if callable(self.instruction) else str(self.instruction)
        # Call LLM
        llm = get_llm()
        # Map config to temperature if available
        temp = 0.2
        if self.generate_content_config and self.generate_content_config.temperature is not None:
            temp = float(self.generate_content_config.temperature)
        msg = []
        if self.global_instruction:
            msg.append({"role": "system", "content": self.global_instruction})
        msg.append({"role": "user", "content": instruction})
        text = llm.chat(msg, temperature=temp)
        # Wrap in a simple response content structure similar to ADK
        class _Part:
            def __init__(self, text: str):
                self.text = text
        class _Content:
            def __init__(self, text: str):
                self.parts = [_Part(text)]
        resp = LlmResponse(content=_Content(text))
        # Store last model text for simple runner integration
        state["last_model_text"] = text
        if self.after_model_callback is not None:
            self.after_model_callback(ctx, resp)
        # After-agent (context-only) callback
        if self.after_agent_callback is not None:
            self.after_agent_callback(ctx)


@dataclass
class SequentialAgent:
    name: str
    sub_agents: List[Agent] = field(default_factory=list)
    description: str = ""
    before_agent_callback: Optional[Callable[[CallbackContext], Any]] = None
    after_agent_callback: Optional[Callable[[CallbackContext], Any]] = None

    def run(self, state: State) -> None:
        ctx = CallbackContext(agent_name=self.name, state=state)
        if self.before_agent_callback:
            self.before_agent_callback(ctx)
        for a in self.sub_agents:
            a.run(state)
        if self.after_agent_callback:
            self.after_agent_callback(ctx)


@dataclass
class ParallelAgent:
    name: str
    sub_agents: List[Agent] = field(default_factory=list)
    description: str = ""
    before_agent_callback: Optional[Callable[[CallbackContext], Any]] = None

    def run(self, state: State) -> None:
        # Simple sequential fallback for parallel
        ctx = CallbackContext(agent_name=self.name, state=state)
        if self.before_agent_callback:
            self.before_agent_callback(ctx)
        for a in self.sub_agents:
            a.run(state)


@dataclass
class LoopAgent:
    name: str
    sub_agents: List[Union[Agent, SequentialAgent]] = field(default_factory=list)
    description: str = ""
    max_iterations: int = 1
    before_agent_callback: Optional[Callable[[CallbackContext], Any]] = None
    after_agent_callback: Optional[Callable[[CallbackContext], Any]] = None

    def run(self, state: State) -> None:
        ctx = CallbackContext(agent_name=self.name, state=state)
        for _ in range(self.max_iterations):
            if self.before_agent_callback:
                self.before_agent_callback(ctx)
            for a in self.sub_agents:
                a.run(state)
        if self.after_agent_callback:
            self.after_agent_callback(ctx)
