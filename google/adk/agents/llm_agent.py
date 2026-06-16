from enum import Enum


class AgentRole(str, Enum):
    SYSTEM = "system"
    USER = "user"


class AgentStateKey(str, Enum):
    LAST_MODEL_TEXT = "last_model_text"


class Agent:
    # ... existing code ...

    def run(self):
        # Example usage of the constants instead of magic strings
        messages = [
            {"role": AgentRole.SYSTEM.value, "content": "System message content"},
            {"role": AgentRole.USER.value, "content": "User message content"},
        ]

        # Access state keys via constants
        last_text = self.state.get(AgentStateKey.LAST_MODEL_TEXT.value, "")

        # ... rest of the run method ...

    # ... rest of the Agent class ...
