# --- Stubs for missing referenced functions ---
def get_task_summary(*args, **kwargs):
    pass

def update_merger_states(*args, **kwargs):
    pass

def create_workspace(*args, **kwargs):
    pass

def prepare_task(*args, **kwargs):
    pass
# --- Callback stubs for agent creation ---
def check_model_finish(*args, **kwargs):
    pass
# --- Callback stubs for agent creation ---
def check_model_eval_finish(*args, **kwargs):
    pass

def rank_candidate_solutions(*args, **kwargs):
    pass

def check_merger_finish(*args, **kwargs):
    pass

def select_best_solution(*args, **kwargs):
    pass

def skip_data_use_check(*args, **kwargs):
    pass
from google.adk import agents
"""Initialization agent for Machine Learning Engineering."""

from typing import Optional
import dataclasses
import os
import shutil
import time
import ast


from google.adk.agents import callback_context as callback_context_module
from google.adk.models import llm_response as llm_response_module
from google.adk.models import llm_request as llm_request_module
from google.genai import types
from google.adk.tools.google_search_tool import google_search

from machine_learning_engineering.sub_agents.initialization import prompt
from machine_learning_engineering.shared_libraries import debug_util
from machine_learning_engineering.shared_libraries import common_util
from machine_learning_engineering.shared_libraries import config


def get_model_candidates(
    callback_context: callback_context_module.CallbackContext,
    llm_response: llm_response_module.LlmResponse,
) -> Optional[llm_response_module.LlmResponse]:
    task_summarization_agent = agents.Agent(
        model=config.CONFIG.agent_model,
        name="task_summarization_agent",
        description="Summarize the task description.",
        instruction=prompt.SUMMARIZATION_AGENT_INSTR,
        after_model_callback=get_task_summary,
        generate_content_config=types.GenerateContentConfig(
            temperature=0.0,
        ),
        include_contents="none",
    )
    init_parallel_sub_agents = []
    for k in range(config.CONFIG.num_solutions):
        model_retriever_agent = agents.Agent(
            model=config.CONFIG.agent_model,
            name=f"model_retriever_agent_{k+1}",
            description="Retrieve effective models for solving a given task.",
            instruction=get_model_retriever_agent_instruction,
            tools=[google_search],
            before_model_callback=check_model_finish,
            after_model_callback=get_model_candidates,
            generate_content_config=types.GenerateContentConfig(
                temperature=1.0,
            ),
            include_contents="none",
        )
        model_retriever_loop_agent = agents.LoopAgent(
            name=f"model_retriever_loop_agent_{k+1}",
            description="Retrieve effective models until it succeeds.",
            sub_agents=[model_retriever_agent],
            max_iterations=config.CONFIG.max_retry,
        )
        init_solution_gen_sub_agents = [
            model_retriever_loop_agent,
        ]
        for l in range(config.CONFIG.num_model_candidates):
            model_eval_and_debug_loop_agent = debug_util.get_run_and_debug_agent(
                prefix="model_eval",
                suffix=f"{k+1}_{l+1}",
                agent_description="Generate a code using the given model",
                instruction_func=get_model_eval_agent_instruction,
                before_model_callback=check_model_eval_finish,
            )
            init_solution_gen_sub_agents.append(model_eval_and_debug_loop_agent)
        rank_agent = agents.SequentialAgent(
            name=f"rank_agent_{k+1}",
            description="Rank the solutions based on the scores.",
            before_agent_callback=rank_candidate_solutions,
        )
        init_solution_gen_sub_agents.append(rank_agent)
        for l in range(1, config.CONFIG.num_model_candidates):
            merge_and_debug_loop_agent = debug_util.get_run_and_debug_agent(
                prefix="merger",
                suffix=f"{k+1}_{l}",
                agent_description="Integrate two solutions into a single solution",
                instruction_func=get_merger_agent_instruction,
                before_model_callback=check_merger_finish,
            )
            merger_states_update_agent = agents.SequentialAgent(
                name=f"merger_states_update_agent_{k+1}_{l}",
                description="Updates the states after merging.",
                before_agent_callback=update_merger_states,
            )
            init_solution_gen_sub_agents.extend(
                [
                    merge_and_debug_loop_agent,
                    merger_states_update_agent,
                ]
            )
        selection_agent = agents.SequentialAgent(
            name=f"selection_agent_{k+1}",
            description="Select the best solution.",
            before_agent_callback=select_best_solution,
        )
        init_solution_gen_sub_agents.append(selection_agent)
        if config.CONFIG.use_data_usage_checker:
            check_data_use_and_debug_loop_agent = debug_util.get_run_and_debug_agent(
                prefix="check_data_use",
                suffix=f"{k+1}",
                agent_description="Check if all the provided information is used",
                instruction_func=get_check_data_use_instruction,
                before_model_callback=skip_data_use_check,
            )
            init_solution_gen_sub_agents.append(check_data_use_and_debug_loop_agent)
        init_solution_gen_agent = agents.SequentialAgent(
            name=f"init_solution_gen_agent_{k+1}",
            description="Generate an initial solutions for the given task.",
            sub_agents=init_solution_gen_sub_agents,
            before_agent_callback=create_workspace,
        )
        init_parallel_sub_agents.append(init_solution_gen_agent)
    init_parallel_agent = agents.ParallelAgent(
        name="init_parallel_agent",
        description="Generate multiple initial solutions for the given task in parallel.",
        sub_agents=init_parallel_sub_agents,
    )
    initialization_agent = agents.SequentialAgent(
        name="initialization_agent",
        description="Initialize the states and generate initial solutions.",
        sub_agents=[
            task_summarization_agent,
            init_parallel_agent,
        ],
        before_agent_callback=prepare_task,
    )
    return {
        "task_summarization_agent": task_summarization_agent,
        "init_parallel_agent": init_parallel_agent,
        "initialization_agent": initialization_agent
    }
    # make required directories
    os.makedirs(os.path.join(workspace_dir, task_name, task_id), exist_ok=True)
    os.makedirs(os.path.join(workspace_dir, task_name, task_id, "input"), exist_ok=True)
    os.makedirs(os.path.join(workspace_dir, task_name, task_id, "model_candidates"), exist_ok=True)
    # copy files to input directory
    files = os.listdir(os.path.join(data_dir, task_name))
    for file in files:
        if os.path.isdir(os.path.join(data_dir, task_name, file)):
            shutil.copytree(
                os.path.join(data_dir, task_name, file),
                os.path.join(workspace_dir, task_name, task_id, "input", file),
            )
        else:
            if "answer" not in file:
                common_util.copy_file(
                    os.path.join(data_dir, task_name, file),
                    os.path.join(workspace_dir, task_name, task_id, "input"),
                )
    return None


def get_model_eval_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the model evaluation agent instruction."""
    task_description = context.state.get("task_description", "")
    model_id = context.agent_name.split("_")[-1]
    task_id = context.agent_name.split("_")[-2]
    model_description = context.state.get(
        f"init_{task_id}_model_{model_id}",
        {},
    ).get("model_description", "")
    return prompt.MODEL_EVAL_INSTR.format(
        task_description=task_description,
        model_description=model_description,
    )


def get_model_retriever_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the model retriever agent instruction."""
    task_summary = context.state.get("task_summary", "")
    num_model_candidates = context.state.get("num_model_candidates", 2)
    return prompt.MODEL_RETRIEVAL_INSTR.format(
        task_summary=task_summary,
        num_model_candidates=num_model_candidates,
    )


def get_merger_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the integrate agent instruction."""
    reference_idx = int(context.agent_name.split("_")[-1])
    task_id = context.agent_name.split("_")[-2]
    performance_results = context.state.get(f"performance_results_{task_id}", [])
    base_solution = context.state.get(f"base_solution_{task_id}", "")
    if reference_idx < len(performance_results):
        reference_solution = performance_results[reference_idx][1].replace(
            "```python", ""
        ).replace("```", "")
    else:
        reference_solution = ""
    return prompt.CODE_INTEGRATION_INSTR.format(
        base_code=base_solution,
        reference_code=reference_solution,
    )


def get_check_data_use_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the check data use agent instruction."""
    task_id = context.agent_name.split("_")[-1]
    task_description = context.state.get("task_description", "")
    code = context.state.get(f"train_code_0_{task_id}", "")
    return prompt.CHECK_DATA_USE_INSTR.format(
        code=code,
        task_description=task_description,
    )



# All agent object creation is now inside this function to avoid circular imports
def get_initialization_agents():
    task_summarization_agent = agents.Agent(
        model=config.CONFIG.agent_model,
        name="task_summarization_agent",
        description="Summarize the task description.",
        instruction=prompt.SUMMARIZATION_AGENT_INSTR,
        after_model_callback=get_task_summary,
        generate_content_config=types.GenerateContentConfig(
            temperature=0.0,
        ),
        include_contents="none",
    )
    init_parallel_sub_agents = []
    for k in range(config.CONFIG.num_solutions):
        model_retriever_agent = agents.Agent(
            model=config.CONFIG.agent_model,
            name=f"model_retriever_agent_{k+1}",
            description="Retrieve effective models for solving a given task.",
            instruction=get_model_retriever_agent_instruction,
            tools=[google_search],
            before_model_callback=check_model_finish,
            after_model_callback=get_model_candidates,
            generate_content_config=types.GenerateContentConfig(
                temperature=1.0,
            ),
            include_contents="none",
        )
        model_retriever_loop_agent = agents.LoopAgent(
            name=f"model_retriever_loop_agent_{k+1}",
            description="Retrieve effective models until it succeeds.",
            sub_agents=[model_retriever_agent],
            max_iterations=config.CONFIG.max_retry,
        )
        init_solution_gen_sub_agents = [
            model_retriever_loop_agent,
        ]
        for l in range(config.CONFIG.num_model_candidates):
            model_eval_and_debug_loop_agent = debug_util.get_run_and_debug_agent(
                prefix="model_eval",
                suffix=f"{k+1}_{l+1}",
                agent_description="Generate a code using the given model",
                instruction_func=get_model_eval_agent_instruction,
                before_model_callback=check_model_eval_finish,
            )
            init_solution_gen_sub_agents.append(model_eval_and_debug_loop_agent)
        rank_agent = agents.SequentialAgent(
            name=f"rank_agent_{k+1}",
            description="Rank the solutions based on the scores.",
            before_agent_callback=rank_candidate_solutions,
        )
        init_solution_gen_sub_agents.append(rank_agent)
        for l in range(1, config.CONFIG.num_model_candidates):
            merge_and_debug_loop_agent = debug_util.get_run_and_debug_agent(
                prefix="merger",
                suffix=f"{k+1}_{l}",
                agent_description="Integrate two solutions into a single solution",
                instruction_func=get_merger_agent_instruction,
                before_model_callback=check_merger_finish,
            )
            merger_states_update_agent = agents.SequentialAgent(
                name=f"merger_states_update_agent_{k+1}_{l}",
                description="Updates the states after merging.",
                before_agent_callback=update_merger_states,
            )
            init_solution_gen_sub_agents.extend(
                [
                    merge_and_debug_loop_agent,
                    merger_states_update_agent,
                ]
            )
        selection_agent = agents.SequentialAgent(
            name=f"selection_agent_{k+1}",
            description="Select the best solution.",
            before_agent_callback=select_best_solution,
        )
        init_solution_gen_sub_agents.append(selection_agent)
        if config.CONFIG.use_data_usage_checker:
            check_data_use_and_debug_loop_agent = debug_util.get_run_and_debug_agent(
                prefix="check_data_use",
                suffix=f"{k+1}",
                agent_description="Check if all the provided information is used",
                instruction_func=get_check_data_use_instruction,
                before_model_callback=skip_data_use_check,
            )
            init_solution_gen_sub_agents.append(check_data_use_and_debug_loop_agent)
        init_solution_gen_agent = agents.SequentialAgent(
            name=f"init_solution_gen_agent_{k+1}",
            description="Generate an initial solutions for the given task.",
            sub_agents=init_solution_gen_sub_agents,
            before_agent_callback=create_workspace,
        )
        init_parallel_sub_agents.append(init_solution_gen_agent)
    init_parallel_agent = agents.ParallelAgent(
        name="init_parallel_agent",
        description="Generate multiple initial solutions for the given task in parallel.",
        sub_agents=init_parallel_sub_agents,
    )
    initialization_agent = agents.SequentialAgent(
        name="initialization_agent",
        description="Initialize the states and generate initial solutions.",
        sub_agents=[
            task_summarization_agent,
            init_parallel_agent,
        ],
        before_agent_callback=prepare_task,
    )
    return {
        "task_summarization_agent": task_summarization_agent,
        "init_parallel_agent": init_parallel_agent,
        "initialization_agent": initialization_agent
    }
