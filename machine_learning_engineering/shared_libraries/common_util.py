"""Common utility functions."""

import random
import torch
import os
import shutil
import numpy as np

from google.adk.models import llm_response


def get_text_from_response(
    response: llm_response.LlmResponse,
) -> str:
  """Extracts text from response."""
  final_text = ""
  if response.content and response.content.parts:
    num_parts = len(response.content.parts)
    for i in range(num_parts):
        if hasattr(response.content.parts[i], "text"):
            final_text += response.content.parts[i].text
  return final_text


def set_random_seed(seed: int) -> None:
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def copy_file(source_file_path: str, destination_dir: str) -> None:
    """Copies a file to the specified directory."""
    if not os.path.isdir(destination_dir):
        os.makedirs(destination_dir, exist_ok=True)
    shutil.copy2(source_file_path, destination_dir)
