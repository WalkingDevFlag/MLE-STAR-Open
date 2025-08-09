"""Ensemble agent for Machine Learning Engineering."""

from typing import Optional
import os
import shutil
import numpy as np

from google.adk import agents
from google.adk.agents import callback_context as callback_context_module
from google.adk.models import llm_response as llm_response_module
from google.adk.models import llm_request as llm_request_module
from google.genai import types

from machine_learning_engineering.sub_agents.ensemble import prompt
from machine_learning_engineering.shared_libraries import debug_util
from machine_learning_engineering.shared_libraries import common_util
from machine_learning_engineering.shared_libraries import config


def update_ensemble_loop_states(
    callback_context: callback_context_module.CallbackContext
) -> Optional[types.Content]:
    """Updates ensemble loop states."""
    callback_context.state["ensemble_iter"] += 1
    return None


def init_ensemble_loop_states(
    callback_context: callback_context_module.CallbackContext
) -> Optional[types.Content]:
    """Initializes ensemble loop states."""
    callback_context.state["ensemble_iter"] = 0
    return None


def get_init_ensemble_plan(
    callback_context: callback_context_module.CallbackContext,
    llm_response: llm_response_module.LlmResponse,
) -> Optional[llm_response_module.LlmResponse]:
    """Gets the initial plan to ensemble solutions."""
    response_text = common_util.get_text_from_response(llm_response)
    callback_context.state["ensemble_plans"] = [response_text]
    return None


def check_ensemble_plan_implement_finish(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest,
) -> Optional[llm_response_module.LlmResponse]:
    """Checks if the ensemble plan implement is finished."""
    ensemble_iter = callback_context.state.get("ensemble_iter", 0)
    result_dict = callback_context.state.get(
        f"ensemble_code_exec_result_{ensemble_iter}", {}
    )
    callback_context.state[
        f"ensemble_plan_implement_skip_data_leakage_check_{ensemble_iter}"
    ] = True
    if result_dict:
        return llm_response_module.LlmResponse()
    callback_context.state[
        f"ensemble_plan_implement_skip_data_leakage_check_{ensemble_iter}"
    ] = False
    return None


def get_refined_ensemble_plan(
    callback_context: callback_context_module.CallbackContext,
    llm_response: llm_response_module.LlmResponse,
) -> Optional[llm_response_module.LlmResponse]:
    """Gets the refined ensemble plan from the response."""
    response_text = common_util.get_text_from_response(llm_response)
    callback_context.state["ensemble_plans"].append(response_text)
    return None


def get_init_ensemble_plan_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the initial ensemble plan agent instruction."""
    num_solutions = context.state.get("num_solutions", 2)
    outer_loop_round = context.state.get("outer_loop_round", 2)
    python_solutions = []
    for task_id in range(1, num_solutions + 1):
        code = context.state.get(
            f"train_code_{outer_loop_round}_{task_id}", ""
        )
        formatted_str = f"# Python Solution {task_id}\n```python\n{code}\n```\n"
        python_solutions.append(formatted_str)
    instruction = prompt.INIT_ENSEMBLE_PLAN_INSTR.format(
        num_solutions=num_solutions,
        python_solutions="\n".join(python_solutions),
    )
    return instruction


def get_ensemble_plan_refinement_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets ensemble plan refinement instruction."""
    num_solutions = context.state.get("num_solutions", 2)
    outer_loop_round = context.state.get("outer_loop_round", 2)
    num_top_plans = context.state.get("num_top_plans", 3)
    lower = context.state.get("lower", True)
    prev_plans = context.state.get("ensemble_plans", [])
    prev_scores = []
    for k in range(len(prev_plans)):
        exec_result = context.state.get(
            f"ensemble_code_exec_result_{k}", {}
        )
        prev_scores.append(exec_result["score"])
    sorted_idx = np.argsort(prev_scores)[::-1]
    if lower:
        sorted_idx = sorted_idx[-num_top_plans:]
        criteria = "lower"
    else:
        sorted_idx = sorted_idx[:num_top_plans]
        criteria = "higher"
    prev_plans_and_scores = ""
    for k in sorted_idx:
        prev_plans_and_scores += f"## Plan: {prev_plans[k]}\n"
        prev_plans_and_scores += f"## Score: {prev_scores[k]:.5f}\n\n"
    python_solutions = []
    for task_id in range(1, num_solutions + 1):
        code = context.state.get(
            f"train_code_{outer_loop_round}_{task_id}", ""
        )
        formatted_str = f"# Python Solution {task_id}\n```python\n{code}\n```\n"
        python_solutions.append(formatted_str)
    return prompt.ENSEMBLE_PLAN_REFINE_INSTR.format(
        num_solutions=num_solutions,
        python_solutions="\n".join(python_solutions),
        prev_plans_and_scores=prev_plans_and_scores,
        criteria=criteria,
    )


def get_ensemble_plan_implement_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the ensemble plan implement agent instruction."""
    num_solutions = context.state.get("num_solutions", 2)
    outer_loop_round = context.state.get("outer_loop_round", 2)
    python_solutions = []
    for task_id in range(1, num_solutions + 1):
        code = context.state.get(
            f"train_code_{outer_loop_round}_{task_id}", ""
        )
        formatted_str = f"# Python Solution {task_id}\n```python\n{code}\n```\n"
        python_solutions.append(formatted_str)
    prev_plans = context.state.get(f"ensemble_plans", [""])
    return prompt.ENSEMBLE_PLAN_IMPLEMENT_INSTR.format(
        num_solutions=num_solutions,
        python_solutions="\n".join(python_solutions),
        plan=prev_plans[-1],
    )


def create_workspace(
    callback_context: callback_context_module.CallbackContext
) -> Optional[types.Content]:
    """Creates workspace."""
    data_dir = callback_context.state.get("data_dir", "")
    workspace_dir = callback_context.state.get("workspace_dir", "")
    task_name = callback_context.state.get("task_name", "")
    run_cwd = os.path.join(workspace_dir, task_name, "ensemble")
    if os.path.exists(run_cwd):
      shutil.rmtree(run_cwd)
    # make required directories
    os.makedirs(os.path.join(workspace_dir, task_name, "ensemble"), exist_ok=True)
    os.makedirs(os.path.join(workspace_dir, task_name, "ensemble", "input"), exist_ok=True)
    os.makedirs(os.path.join(workspace_dir, task_name, "ensemble", "final"), exist_ok=True)
    # copy files to input directory
    files = os.listdir(os.path.join(data_dir, task_name))
    for file in files:
        if os.path.isdir(os.path.join(data_dir, task_name, file)):
            shutil.copytree(
                os.path.join(data_dir, task_name, file),
                os.path.join(workspace_dir, task_name, "ensemble", "input", file),
            )
        else:
            if "answer" not in file:
                common_util.copy_file(
                    os.path.join(data_dir, task_name, file),
                    os.path.join(workspace_dir, task_name, "ensemble", "input"),
                )
    return None


init_ensemble_plan_agent = agents.Agent(
    model=config.CONFIG.agent_model,
    name="init_ensemble_plan_agent",
    description="Generate an initial plan to ensemble solutions.",
    instruction=get_init_ensemble_plan_agent_instruction,
    before_agent_callback=init_ensemble_loop_states,
    after_model_callback=get_init_ensemble_plan,
    generate_content_config=types.GenerateContentConfig(
        temperature=1.0,
    ),
    include_contents="none",
)
init_ensemble_plan_implement_agent = debug_util.get_run_and_debug_agent(
    prefix="ensemble_plan_implement_initial",
    suffix="",
    agent_description="Implement the initial plan to ensemble solutions.",
    instruction_func=get_ensemble_plan_implement_agent_instruction,
    before_model_callback=check_ensemble_plan_implement_finish,
)
ensemble_plan_refine_agent = agents.Agent(
    model=config.CONFIG.agent_model,
    name="ensemble_plan_refine_agent",
    description="Refine the ensemble plan.",
    instruction=get_ensemble_plan_refinement_instruction,
    after_model_callback=get_refined_ensemble_plan,
    generate_content_config=types.GenerateContentConfig(
        temperature=1.0,
    ),
    include_contents="none",
)
ensemble_plan_implement_agent = debug_util.get_run_and_debug_agent(
    prefix="ensemble_plan_implement",
    suffix="",
    agent_description="Implement the plan to ensemble solutions.",
    instruction_func=get_ensemble_plan_implement_agent_instruction,
    before_model_callback=check_ensemble_plan_implement_finish,
)
ensemble_plan_refine_and_implement_agent = agents.SequentialAgent(
    name="ensemble_plan_refine_and_implement_agent",
    description="Refine the ensemble plan and then implement it.",
    sub_agents=[
        ensemble_plan_refine_agent,
        ensemble_plan_implement_agent,
    ],
    after_agent_callback=update_ensemble_loop_states,
)
ensemble_plan_refine_and_implement_loop_agent = agents.LoopAgent(
    name="ensemble_plan_refine_and_implement_loop_agent",
    description="Iteratively refine the ensemble plan and implement it.",
    sub_agents=[ensemble_plan_refine_and_implement_agent],
    before_agent_callback=update_ensemble_loop_states,
    max_iterations=config.CONFIG.ensemble_loop_round,
)
ensemble_agent = agents.SequentialAgent(
    name="ensemble_agent",
    description="Ensemble multiple solutions.",
    sub_agents=[
        init_ensemble_plan_agent,
        init_ensemble_plan_implement_agent,
        ensemble_plan_refine_and_implement_loop_agent,
    ],
    before_agent_callback=create_workspace,
    after_agent_callback=None,
)
