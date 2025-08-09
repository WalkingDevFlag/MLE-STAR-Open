"""Basic evalualtion for Machine Learning Engineering Agent"""

import pathlib

import dotenv
import pytest
from google.adk.evaluation.agent_evaluator import AgentEvaluator

from machine_learning_engineering.shared_libraries import config


pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(scope="session", autouse=True)
def load_env():
    dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_full_interaction(monkeypatch):
    """Test the agent's full ability on a task."""
    monkeypatch.setattr(config.CONFIG, "exec_timeout", 30)
    await AgentEvaluator.evaluate(
        "machine_learning_engineering",
        str(pathlib.Path(__file__).parent / "./full.test.json"),
        num_runs=1,
    )
