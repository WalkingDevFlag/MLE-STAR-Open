"""Submission agent for Machine Learning Engineering."""

from typing import Optional

from google.adk.agents import callback_context as callback_context_module
from google.adk.models import llm_response as llm_response_module
from google.adk.models import llm_request as llm_request_module

from machine_learning_engineering.sub_agents.submission import prompt
from machine_learning_engineering.shared_libraries import debug_util


def check_submission_finish(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest,
) -> Optional[llm_response_module.LlmResponse]:
    """Checks if adding codes for submission is finished."""
    result_dict = callback_context.state.get(
        "submission_code_exec_result", {}
    )
    callback_context.state[
        "submission_skip_data_leakage_check"
    ] = True
    if result_dict:
        return llm_response_module.LlmResponse()
    callback_context.state[
        "submission_skip_data_leakage_check"
    ] = False
    return None


def get_submission_and_debug_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the submission agent instruction."""
    num_solutions = context.state.get("num_solutions", 2)
    outer_loop_round = context.state.get("outer_loop_round", 2)
    ensemble_loop_round = context.state.get("ensemble_loop_round", 2)
    task_description = context.state.get("task_description", "")
    lower = context.state.get("lower", True)
    final_solution = ""
    best_score = None
    for task_id in range(1, num_solutions + 1):
        curr_code = context.state.get(
            f"train_code_{outer_loop_round}_{task_id}", ""
        )
        curr_exec_result = context.state.get(
            f"train_code_exec_result_{outer_loop_round}_{task_id}", ""
        )
        curr_score = curr_exec_result["score"]
        if (best_score is None) or (lower and curr_score < best_score) or (not lower and curr_score > best_score):
            final_solution = curr_code
            best_score = curr_score
    for ensemble_iter in range(ensemble_loop_round + 1):
        curr_code = context.state.get(
            f"ensemble_code_{ensemble_iter}", {}
        )
        curr_exec_result = context.state.get(
            f"ensemble_code_exec_result_{ensemble_iter}", {}
        )
        curr_score = curr_exec_result["score"]
        if (best_score is None) or (lower and curr_score < best_score) or (not lower and curr_score > best_score):
            final_solution = curr_code
            best_score = curr_score
    return prompt.ADD_TEST_FINAL_INSTR.format(
        task_description=task_description,
        code=final_solution,
    )


submission_agent = debug_util.get_run_and_debug_agent(
    prefix="submission",
    suffix="",
    agent_description="Add codes for creating a submission file.",
    instruction_func=get_submission_and_debug_agent_instruction,
    before_model_callback=check_submission_finish,
)
