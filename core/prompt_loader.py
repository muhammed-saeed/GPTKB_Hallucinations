# core/prompt_loader.py
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict

def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.exists():
        return p
    here = Path(__file__).resolve().parents[1]  # project root (.. from core/)
    p2 = (here / p).resolve()
    if p2.exists():
        return p2
    p3 = Path.cwd() / p
    if p3.exists():
        return p3
    raise FileNotFoundError(f"Prompt not found. Tried: {p}, {p2}, {p3}")

def load_messages_from_prompt_json(path: str | Path, **vars) -> List[Dict[str, str]]:
    obj = json.loads(_resolve(path).read_text(encoding="utf-8"))
    system = (obj.get("system") or "").format(**vars)
    user   = (obj.get("user") or "").format(**vars)
    return [
        {"role": "system", "content": system.strip()},
        {"role": "user",   "content": user.strip()},
    ]
