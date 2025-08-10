from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure repo root is on sys.path so package imports work when running from scripts/
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from machine_learning_engineering.shared_libraries import config as cfg
from machine_learning_engineering.agent import root_agent
from google.adk.agents.callback_context import State


def main():
    load_dotenv(override=True)
    ap = argparse.ArgumentParser(description="Run the MLE pipeline on a task directory.")
    ap.add_argument("--task-name", required=True, help="Task folder name under machine_learning_engineering/tasks")
    ap.add_argument("--workspace", default=None, help="Optional workspace dir (default from config)")
    args = ap.parse_args()

    # Update config
    if args.workspace:
        cfg.CONFIG.workspace_dir = args.workspace
    cfg.CONFIG.task_name = args.task_name

    # Ensure directories
    Path(cfg.CONFIG.workspace_dir, cfg.CONFIG.task_name).mkdir(parents=True, exist_ok=True)

    # Run the root agent
    state = State()
    root_agent.run(state)


if __name__ == "__main__":
    main()
