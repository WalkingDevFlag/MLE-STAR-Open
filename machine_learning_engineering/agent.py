"""Demonstration of Machine Learning Engineering Agent using Agent Development Kit"""

import os
import json
from typing import Optional
from google.genai import types
from google.adk.agents import callback_context as callback_context_module

from google.adk import agents
from machine_learning_engineering.sub_agents.initialization import agent as initialization_agent_module
from machine_learning_engineering.sub_agents.refinement import agent as refinement_agent_module
from machine_learning_engineering.sub_agents.ensemble import agent as ensemble_agent_module
from machine_learning_engineering.sub_agents.submission import agent as submission_agent_module

from machine_learning_engineering import prompt


def save_state(
    callback_context: callback_context_module.CallbackContext
) -> Optional[types.Content]:
    """Prints the current state of the callback context."""
    workspace_dir = callback_context.state.get("workspace_dir", "")
    task_name = callback_context.state.get("task_name", "")
    run_cwd = os.path.join(workspace_dir, task_name)
    with open(os.path.join(run_cwd, "final_state.json"), "w") as f:
        json.dump(callback_context.state.to_dict(), f, indent=2)
    return None


def get_mle_pipeline_agent():
    """Creates and returns the MLE pipeline agent to avoid circular imports."""
    # Import agent creation functions and call them to get the agents
    initialization_agent = initialization_agent_module.get_initialization_agents()["initialization_agent"]
    refinement_agent = refinement_agent_module.get_refinement_agent()
    ensemble_agent = ensemble_agent_module.get_ensemble_agent()
    submission_agent = submission_agent_module.get_submission_agent()
    
    mle_pipeline_agent = agents.SequentialAgent(
        name="mle_pipeline_agent",
        sub_agents=[
            initialization_agent,
            refinement_agent,
            ensemble_agent,
            submission_agent,
        ],
        description="Executes a sequence of sub-agents for solving the MLE task.",
        after_agent_callback=save_state,
    )
    return mle_pipeline_agent


def get_root_agent():
    """Creates and returns the root agent to avoid circular imports."""
    mle_pipeline_agent = get_mle_pipeline_agent()
    
    # For ADK tools compatibility, the root agent must be named `root_agent`
    root_agent = agents.Agent(
        model=os.getenv("ROOT_AGENT_MODEL"),
        name="mle_frontdoor_agent",
        instruction=prompt.FRONTDOOR_INSTRUCTION,
        global_instruction=prompt.SYSTEM_INSTRUCTION,
        sub_agents=[mle_pipeline_agent],
        generate_content_config=types.GenerateContentConfig(temperature=0.01),
    )
    return root_agent


# For ADK tools compatibility, the root agent must be named `root_agent`
# Use lazy initialization to avoid circular imports
_root_agent = None

def _get_root_agent_lazy():
    global _root_agent
    if _root_agent is None:
        _root_agent = get_root_agent()
    return _root_agent

# Create a property-like access to root_agent that creates it on first access
class _RootAgentProxy:
    def __getattr__(self, name):
        return getattr(_get_root_agent_lazy(), name)
    
    def run(self, *args, **kwargs):
        return _get_root_agent_lazy().run(*args, **kwargs)

root_agent = _RootAgentProxy()
