"""Utility functions for leakage check agent."""

from typing import Optional
import json
import functools

from google.adk import agents
from google.adk.agents import callback_context as callback_context_module
from google.adk.models import llm_response as llm_response_module
from google.adk.models import llm_request as llm_request_module
from google.genai import types

from machine_learning_engineering.shared_libraries import data_leakage_prompt
from machine_learning_engineering.shared_libraries import code_util
from machine_learning_engineering.shared_libraries import common_util
from machine_learning_engineering.shared_libraries import config


def get_check_leakage_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the check leakage agent instruction."""
    agent_name = context.agent_name
    suffix = code_util.get_updated_suffix(callback_context=context)
    code_state_key = code_util.get_code_state_key(
        agent_name=agent_name,
        suffix=suffix,
    )
    code = context.state.get(code_state_key, "")
    return data_leakage_prompt.CHECK_LEAKAGE_INSTR.format(
        code=code,
    )


def get_refine_leakage_agent_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the refine leakage agent instruction."""
    agent_name = context.agent_name
    suffix = code_util.get_updated_suffix(callback_context=context)
    code_state_key = code_util.get_code_state_key(
        agent_name=agent_name,
        suffix=suffix,
    )
    code = context.state.get(code_state_key, "")
    return data_leakage_prompt.LEAKAGE_REFINE_INSTR.format(
        code=code,
    )


def parse_leakage_status(text: str) -> tuple[str, str]:
    """Parses the leakage status from the text."""
    start_idx, end_idx = text.find("["), text.rfind("]")+1
    text = text[start_idx:end_idx]
    result = json.loads(text)[0]
    leakage_status = result["leakage_status"]
    code_block = result["code_block"].replace(f"```python", "").replace("```", "")
    return leakage_status, code_block


def update_extract_status(
    callback_context: callback_context_module.CallbackContext,
    llm_response: llm_response_module.LlmResponse,
    prefix: str,
) -> Optional[llm_response_module.LlmResponse]:
    """Updates the status of extraction."""
    response_text = common_util.get_text_from_response(llm_response)
    agent_name = callback_context.agent_name
    suffix = code_util.get_updated_suffix(callback_context=callback_context)
    code_state_key = code_util.get_code_state_key(
        agent_name=agent_name,
        suffix=suffix,
    )
    code = callback_context.state.get(code_state_key, "")
    if "No Data Leakage" in response_text:
        leakage_status = "No Data Leakage"
    try:
        leakage_status, code_block = parse_leakage_status(response_text)
        if leakage_status == "No Data Leakage":
            extract_status = True
        else:
            extract_status = code_block in code
    except:
        code_block = ""
        extract_status = False
    extract_status_key = code_util.get_name_with_prefix_and_suffix(
        base_name="extract_status",
        prefix=prefix,
        suffix=suffix,
    )
    leakage_block_key = code_util.get_name_with_prefix_and_suffix(
        base_name="leakage_block",
        prefix=prefix,
        suffix=suffix,
    )
    leakage_status_key = code_util.get_name_with_prefix_and_suffix(
        base_name="leakage_status",
        prefix=prefix,
        suffix=suffix,
    )
    callback_context.state[extract_status_key] = extract_status
    callback_context.state[leakage_block_key] = code_block
    callback_context.state[leakage_status_key] = leakage_status
    return None


def check_extract_status(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest,
    prefix: str,
) -> Optional[llm_response_module.LlmResponse]:
    """Checks the status of extraction."""
    suffix = code_util.get_updated_suffix(callback_context=callback_context)
    extract_status_key = code_util.get_name_with_prefix_and_suffix(
        base_name="extract_status",
        prefix=prefix,
        suffix=suffix,
    )
    skip_data_leakage_check_key = code_util.get_name_with_prefix_and_suffix(
        base_name="skip_data_leakage_check",
        prefix=prefix,
        suffix=suffix,
    )
    extract_status = callback_context.state.get(extract_status_key, False)
    skip_flag = callback_context.state.get(skip_data_leakage_check_key, False)
    if skip_flag or extract_status:
        return llm_response_module.LlmResponse()
    return None


def replace_leakage_code(
    callback_context: callback_context_module.CallbackContext,
    llm_response: llm_response_module.LlmResponse,
    prefix: str,
) -> Optional[llm_response_module.LlmResponse]:
    """Replace the code block that has the data leakage issue."""
    response_text = common_util.get_text_from_response(llm_response)
    refined_code_block = response_text.replace("```python", "").replace("```", "")
    agent_name = callback_context.agent_name
    suffix = code_util.get_updated_suffix(callback_context=callback_context)
    leakage_block_key = code_util.get_name_with_prefix_and_suffix(
        base_name="leakage_block",
        prefix=prefix,
        suffix=suffix,
    )
    code_block = callback_context.state.get(leakage_block_key, "")
    code_state_key = code_util.get_code_state_key(
        agent_name=agent_name,
        suffix=suffix,
    )
    code = callback_context.state.get(code_state_key, "")
    refined_code = code.replace(code_block, refined_code_block)
    callback_context.state[code_state_key] = refined_code
    code_util.evaluate_code(callback_context=callback_context)
    return None


def check_data_leakage(
    callback_context: callback_context_module.CallbackContext,
    llm_request: llm_request_module.LlmRequest,
    prefix: str,
) -> Optional[llm_response_module.LlmResponse]:
    """Checks if the code has the data leakage issue."""
    suffix = code_util.get_updated_suffix(callback_context=callback_context)
    leakage_status_key = code_util.get_name_with_prefix_and_suffix(
        base_name="leakage_status",
        prefix=prefix,
        suffix=suffix,
    )
    skip_data_leakage_check_key = code_util.get_name_with_prefix_and_suffix(
        base_name="skip_data_leakage_check",
        prefix=prefix,
        suffix=suffix,
    )
    leakage_status = callback_context.state.get(leakage_status_key, "")
    skip_flag = callback_context.state.get(skip_data_leakage_check_key, False)
    if skip_flag or ("Yes Data Leakage" not in leakage_status):
        return llm_response_module.LlmResponse()
    return None


def get_data_leakage_checker_agent(
    prefix: str,
    suffix: str,
) -> agents.SequentialAgent:
    """Gets the data leakage checker agent."""
    check_leakage_agent = agents.Agent(
        model=config.CONFIG.agent_model,
        name=code_util.get_name_with_prefix_and_suffix(
            base_name="check_leakage_agent",
            prefix=prefix,
            suffix=suffix,
        ),
        description="Check if the code has the data leakage issue.",
        instruction=get_check_leakage_agent_instruction,
        before_model_callback=functools.partial(
            check_extract_status,
            prefix=prefix,
        ),
        after_model_callback=functools.partial(
            update_extract_status,
            prefix=prefix,
        ),
        generate_content_config=types.GenerateContentConfig(
            temperature=0.0,
        ),
        include_contents="none",
    )
    check_leakage_loop_agent = agents.LoopAgent(
        name=code_util.get_name_with_prefix_and_suffix(
            base_name="check_leakage_loop_agent",
            prefix=prefix,
            suffix=suffix,
        ),
        description="Check if the code has the data leakage issue until extraction succeeds.",
        sub_agents=[
            check_leakage_agent,
        ],
        max_iterations=config.CONFIG.max_retry,
    )
    refine_leakage_agent = agents.Agent(
        model=config.CONFIG.agent_model,
        name=code_util.get_name_with_prefix_and_suffix(
            base_name="refine_leakage_agent",
            prefix=prefix,
            suffix=suffix,
        ),
        description="Refine the code to address the data leakage issue.",
        instruction=get_refine_leakage_agent_instruction,
        before_model_callback=functools.partial(
            check_data_leakage,
            prefix=prefix,
        ),
        after_model_callback=replace_leakage_code,
        generate_content_config=types.GenerateContentConfig(
            temperature=0.0,
        ),
        include_contents="none",
    )
    data_leakage_checker_agent = agents.SequentialAgent(
        name=code_util.get_name_with_prefix_and_suffix(
            base_name="data_leakage_checker_agent",
            prefix=prefix,
            suffix=suffix,
        ),
        description="Check and refine the code to address the data leakage issue.",
        sub_agents=[
            check_leakage_loop_agent,
            refine_leakage_agent,
        ],
    )
    return data_leakage_checker_agent
