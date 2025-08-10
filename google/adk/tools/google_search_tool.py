# Free replacement for Google Search tool using DuckDuckGo.
from __future__ import annotations

from typing import List, Dict
from machine_learning_engineering.shared_libraries.web_search import web_search as _web_search


def google_search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    return _web_search(query, max_results=num_results)
def google_search(*args, **kwargs):
    # No-op placeholder to keep interface compatible.
    return None
