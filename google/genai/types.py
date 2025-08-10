"""Local shim to satisfy imports of google.genai.types.

This avoids pulling Google SDKs while keeping type names referenced in code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class GenerateContentConfig:
    temperature: Optional[float] = None


@dataclass
class Part:
    text: Optional[str] = None


@dataclass
class Content:
    parts: List[Part]
    role: Optional[str] = None
