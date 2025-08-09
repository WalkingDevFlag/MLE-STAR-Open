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
async def test_basic_interaction():
    """Test the agent's basic ability on a few examples."""
    await AgentEvaluator.evaluate(
        "machine_learning_engineering",
        str(pathlib.Path(__file__).parent / "./simple.test.json"),
        num_runs=1,
    )
