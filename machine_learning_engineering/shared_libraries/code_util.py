"""Code related utility functions."""

from typing import Any
import subprocess
import os
import time

from google.adk.agents import callback_context as callback_context_module


class Result:
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def run_python_code(
    code_text: str,
    run_cwd: str,
    py_filepath: str,
    exec_timeout: int,
) -> dict[str, Any]:
    start_time = time.time()
    output_filepath = os.path.join(run_cwd, py_filepath)
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(code_text)
    try:
        result = subprocess.run(
            ["python", py_filepath],
            cwd=run_cwd,
            capture_output=True,
            text=True,
            timeout=exec_timeout,
        )
    except Exception as e:
        result = Result(returncode=1, stdout="", stderr=str(e))
    end_time = time.time()
    execution_time = end_time - start_time
    result_dict = {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "execution_time": execution_time,
    }
    return result_dict


def extract_performance_from_text(text: str) -> float | None:
    """Extracts the final validation performance score from the text."""
    lines = text.splitlines()
    performance_value = None
    for line in lines:
        if "Final Validation Performance" in line:
            try:
                parts = line.split(":")
                # score_str = line.split("Final Validation Performance:")[-1].strip()
                score_str = parts[-1].strip()
                performance_value = float(score_str)
            except ValueError:
                pass
    return performance_value


def get_name_with_prefix_and_suffix(
    base_name: str,
    prefix: str = "",
    suffix: str = "",
) -> str:
    """Gets the name with the specified prefix and suffix."""
    new_name = base_name
    if prefix:
        new_name = prefix + "_" + new_name
    if suffix:
        new_name = new_name + "_" + suffix
    return new_name


def get_updated_suffix(
    callback_context: callback_context_module.CallbackContext,
) -> str:
    """Gets the suffix string."""
    agent_name = callback_context.agent_name
    if agent_name.startswith("model_eval"):
        model_id = agent_name.split("_")[-1]
        task_id = agent_name.split("_")[-2]
        suffix = f"{task_id}_{model_id}"
    elif agent_name.startswith("merger"):
        reference_idx = agent_name.split("_")[-1]
        task_id = agent_name.split("_")[-2]
        suffix = f"{task_id}_{reference_idx}"
    elif agent_name.startswith("check_data_use"):
        task_id = agent_name.split("_")[-1]
        suffix = f"{task_id}"
    elif agent_name.startswith("ablation"):
        task_id = agent_name.split("_")[-1]
        step = callback_context.state.get(f"refine_step_{task_id}", 0)
        suffix = f"{step}_{task_id}"
    elif agent_name.startswith("plan_implement"):
        task_id = callback_context.agent_name.split("_")[-1]
        step = callback_context.state.get(f"refine_step_{task_id}", 0)
        inner_iter = callback_context.state.get(f"inner_iter_{task_id}", 0)
        suffix = f"{inner_iter}_{step}_{task_id}"
    elif agent_name.startswith("ensemble_plan_implement"):
        ensemble_iter = callback_context.state.get("ensemble_iter", 0)
        suffix = f"{ensemble_iter}"
    elif agent_name.startswith("submission"):
        suffix = ""
    else:
        raise ValueError(f"Unexpected agent name: {agent_name}.")
    return suffix


def get_code_state_key(
    agent_name: str,
    suffix: str,
) -> str:
    """Gets the state key for the code."""
    if agent_name.startswith("model_eval"):
        key = f"init_code_{suffix}"
    elif agent_name.startswith("merger"):
        key = f"merger_code_{suffix}"
    elif agent_name.startswith("check_data_use"):
        key = f"train_code_0_{suffix}"
    elif agent_name.startswith("ablation"):
        key = f"ablation_code_{suffix}"
    elif agent_name.startswith("plan_implement"):
        key = f"train_code_improve_{suffix}"
    elif agent_name.startswith("ensemble_plan_implement"):
        key = f"ensemble_code_{suffix}"
    elif agent_name.startswith("submission"):
        key = "submission_code"
    else:
        raise ValueError(f"Unexpected agent name: {agent_name}.")
    return key


def get_code_execution_result_state_key(
    agent_name: str,
    suffix: str,
) -> str:
    """Gets the state key for the code execution result."""
    if agent_name.startswith("model_eval"):
        key = f"init_code_exec_result_{suffix}"
    elif agent_name.startswith("merger"):
        key = f"merger_code_exec_result_{suffix}"
    elif agent_name.startswith("check_data_use"):
        key = f"train_code_exec_result_0_{suffix}"
    elif agent_name.startswith("ablation"):
        key = f"ablation_code_exec_result_{suffix}"
    elif agent_name.startswith("plan_implement"):
        key = f"train_code_improve_exec_result_{suffix}"
    elif agent_name.startswith("ensemble_plan_implement"):
        key = f"ensemble_code_exec_result_{suffix}"
    elif agent_name.startswith("submission"):
        key = "submission_code_exec_result"
    else:
        raise ValueError(f"Unexpected agent name: {agent_name}.")
    return key


def get_run_code_condition(
    agent_name: str,
    raw_code: str,
) -> bool:
    """Gets the condition for running the code."""
    if agent_name.startswith("ensemble_plan_implement"):
        if "debug_agent" not in agent_name:
            return True
        if "Final Validation Performance" in raw_code and "exit()" not in raw_code:
            return True
    elif agent_name.startswith("ablation"):
        if "debug_agent" not in agent_name:
            return True
        if "exit()" not in raw_code:
            return True
    elif agent_name.startswith("submission"):
        if "debug_agent" not in agent_name and "exit()" not in raw_code and "submission.csv" in raw_code:
            return True
        if "debug_agent" in agent_name and "exit()" not in raw_code:
            return True
    elif "Final Validation Performance" in raw_code and "exit()" not in raw_code:
        return True
    return False


def evaluate_code(
    callback_context: callback_context_module.CallbackContext,
) -> None:
    """Evaluates the given code."""
    lower = callback_context.state.get("lower", True)
    exec_timeout = callback_context.state.get("exec_timeout", 1800)
    agent_name = callback_context.agent_name
    suffix = get_updated_suffix(callback_context=callback_context)
    code_state_key = get_code_state_key(
        agent_name=agent_name,
        suffix=suffix,
    )
    raw_code = callback_context.state.get(code_state_key, "")
    if agent_name.startswith("model_eval"):
        model_id = agent_name.split("_")[-1]
        task_id = agent_name.split("_")[-2]
        py_filepath = f"init_code_{model_id}.py"
    elif agent_name.startswith("merger"):
        reference_idx = agent_name.split("_")[-1]
        task_id = agent_name.split("_")[-2]
        py_filepath = f"train0_{reference_idx}.py"
    elif agent_name.startswith("check_data_use"):
        task_id = agent_name.split("_")[-1]
        py_filepath = "train0.py"
    elif agent_name.startswith("ablation"):
        task_id = agent_name.split("_")[-1]
        step = callback_context.state.get(f"refine_step_{task_id}", 0)
        py_filepath = f"ablation_{step}.py"
    elif agent_name.startswith("plan_implement"):
        task_id = agent_name.split("_")[-1]
        step = callback_context.state.get(f"refine_step_{task_id}", 0)
        inner_iter = callback_context.state.get(f"inner_iter_{task_id}", 0)
        py_filepath = f"train{step}_improve{inner_iter}.py"
    elif agent_name.startswith("ensemble_plan_implement"):
        task_id = "ensemble"
        py_filepath = f"ensemble{suffix}.py"
    elif agent_name.startswith("submission"):
        task_id = "ensemble"
        py_filepath = "final_solution.py"
    else:
        raise ValueError(f"Unexpected agent name: {agent_name}.")
    if get_run_code_condition(
        agent_name=agent_name,
        raw_code=raw_code,
    ):
        workspace_dir = callback_context.state.get("workspace_dir", "")
        task_name = callback_context.state.get("task_name", "")
        run_cwd = os.path.join(workspace_dir, task_name, task_id)
        result_dict = run_python_code(
            code_text=raw_code,
            run_cwd=run_cwd,
            py_filepath=py_filepath,
            exec_timeout=exec_timeout,
        )
        if agent_name.startswith("ablation"):
            if result_dict["returncode"] == 0:
                ablation_result = result_dict.get("stdout", "None")
            else:
                ablation_result = "None"
            result_dict["ablation_result"] = ablation_result
        else:
            if result_dict.get("returncode", 1) == 0:
                try:
                    score = extract_performance_from_text(result_dict.get("stdout", ""))
                    score = float(score)
                except:
                    score = 1e9 if lower else 0
            else:
                score = 1e9 if lower else 0
            result_dict["score"] = score
    else:
        result_dict = {}
    code_execution_result_state_key = get_code_execution_result_state_key(
        agent_name=agent_name,
        suffix=suffix,
    )
    callback_context.state[code_execution_result_state_key] = result_dict
    return None
