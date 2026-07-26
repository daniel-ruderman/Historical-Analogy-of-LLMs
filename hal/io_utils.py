"""jsonl helpers and access to the datasets shipped with the original repo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from .config import DATASET_DIR

# Dataset files of the original repository (schemas verified by inspection):
#   popular_analogy.jsonl : event_name, event_intro, target_event   (20 rows)
#   general_analogy.jsonl : event_name, event_intro, event_type     (160 rows)
#   event_pool.jsonl      : url, history_event_text, history_time_text,
#                           history_intro_text                      (658 rows)
#   similarity_embeddings-example.jsonl : url, embeddings (1536-d, OpenAI)
DATASETS = {
    "popular": DATASET_DIR / "popular_analogy.jsonl",
    "general": DATASET_DIR / "general_analogy.jsonl",
    "event_pool": DATASET_DIR / "event_pool.jsonl",
    "openai_embeddings": DATASET_DIR / "similarity_embeddings-example.jsonl",
}


def configure_stdout() -> None:
    """Make printing dataset text safe on consoles with a non-UTF-8 codepage.

    The datasets contain Arabic, Chinese and accented text; a Windows console
    running e.g. cp1255 raises ``UnicodeEncodeError`` on ``print``.
    """
    import sys

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def read_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a .jsonl file (same helper as the original repository)."""
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_jsonl(file_path: Union[str, Path], rows: Iterable[Dict[str, Any]],
                mode: str = "w") -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(file_path: Union[str, Path], row: Dict[str, Any]) -> None:
    write_jsonl(file_path, [row], mode="a+")


def load_dataset(name_or_path: str) -> List[Dict[str, Any]]:
    """Load ``popular``/``general``/``event_pool`` or an explicit .jsonl path."""
    if name_or_path in DATASETS:
        return read_jsonl(DATASETS[name_or_path])
    return read_jsonl(name_or_path)


def load_event_pool(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """The 658-event Google Arts & Culture pool used by the retrieval methods."""
    pool = read_jsonl(DATASETS["event_pool"])
    if limit is not None:
        pool = pool[:limit]
    return pool


def event_text(event: Dict[str, Any]) -> str:
    """``"name: description"`` for either dataset schema."""
    name = event.get("event_name") or event.get("history_event_text", "")
    intro = event.get("event_intro") or event.get("history_intro_text", "")
    return f"{name}: {intro}"
