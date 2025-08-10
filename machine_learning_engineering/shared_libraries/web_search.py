from __future__ import annotations

from typing import List, Dict
from duckduckgo_search import DDGS


def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search the web via DuckDuckGo and return a list of results.

    Each result contains: title, href, and body snippet.
    """
    results: List[Dict[str, str]] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "href": r.get("href", ""),
                "body": r.get("body", ""),
            })
    return results
