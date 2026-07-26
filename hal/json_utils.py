"""Tolerant parsing of JSON / Python-list output produced by an LLM.

LLMs regularly wrap JSON in markdown fences, prepend prose, or emit a Python
literal instead of JSON.  These helpers recover the payload without ever
raising on malformed model output -- callers get ``None`` (or a default) and can
decide how to react.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional

_FENCE_RE = re.compile(r"```(?:json|python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def strip_code_fences(text: str) -> str:
    if not text:
        return ""
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _find_balanced(text: str, open_ch: str, close_ch: str) -> Optional[str]:
    """Return the first balanced ``open_ch``...``close_ch`` block, ignoring
    delimiters that appear inside string literals."""
    start = text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    in_string = False
    quote = ""
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                in_string = False
            continue
        if ch in ('"', "'"):
            in_string = True
            quote = ch
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _loads(candidate: str) -> Optional[Any]:
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(candidate)
        except Exception:
            continue
    # Last resort: drop trailing commas, which JSON rejects.
    cleaned = re.sub(r",\s*([\]}])", r"\1", candidate)
    if cleaned != candidate:
        try:
            return json.loads(cleaned)
        except Exception:
            return None
    return None


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from ``text``; ``None`` if there is none."""
    body = strip_code_fences(text or "")
    for candidate in (body, _find_balanced(body, "{", "}")):
        if not candidate:
            continue
        value = _loads(candidate)
        if isinstance(value, dict):
            return value
    return None


def parse_json_array(text: str) -> Optional[List[Any]]:
    """Extract the first JSON/Python array from ``text``."""
    body = strip_code_fences(text or "")
    for candidate in (body, _find_balanced(body, "[", "]")):
        if not candidate:
            continue
        value = _loads(candidate)
        if isinstance(value, list):
            return value
    return None


def parse_json_any(text: str) -> Optional[Any]:
    obj = parse_json_object(text)
    if obj is not None:
        return obj
    return parse_json_array(text)


def as_str_list(value: Any) -> List[str]:
    """Coerce a parsed value into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        value = list(value.values())
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    for item in value:
        if isinstance(item, str):
            item = item.strip()
            if item:
                out.append(item)
        elif isinstance(item, dict):
            for key in ("event", "event_name", "name", "title", "candidate"):
                if isinstance(item.get(key), str) and item[key].strip():
                    out.append(item[key].strip())
                    break
    return out


def get_str(data: Dict[str, Any], *keys: str, default: str = "") -> str:
    """First non-empty string value among ``keys``."""
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
    return default


def get_float(data: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = data.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            match = re.search(r"-?\d+(?:\.\d+)?", value)
            if match:
                try:
                    return float(match.group())
                except ValueError:
                    pass
    return default


def get_list(data: Dict[str, Any], *keys: str) -> List[str]:
    for key in keys:
        if key in data:
            items = as_str_list(data[key])
            if items:
                return items
    return []
