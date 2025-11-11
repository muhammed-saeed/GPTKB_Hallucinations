# tracer.py
from __future__ import annotations
import json, time, threading, datetime
from typing import Any, Dict, List, Optional

_jsonl_lock = threading.Lock()

def append_jsonl(path: str, obj: dict):
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with _jsonl_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

def _knob(v):
    try:
        return float(v) if v is not None else None
    except Exception:
        return None

def _now():
    return datetime.datetime.utcnow().isoformat() + "Z"

class TracedLLM:
    """
    Wraps an LLM client (e.g., from llm.factory.make_llm_from_config).
    Logs request/response metadata to a JSONL file so you can verify
    temperature/top_p/top_k/max_tokens, model, provider, durations, etc.
    """
    def __init__(self, llm, *, name: str, trace_path: str, echo: bool = False):
        self._llm = llm
        self._name = name
        self._trace_path = trace_path
        self._echo = echo  # also print a short line to stdout

    # --- helpers to read config fields if present ---
    def _cfg_str(self, attr, default=None):
        try:
            return getattr(self._llm, attr)
        except Exception:
            return default

    def _cfg_num(self, attr):
        return _knob(self._cfg_str(attr, None))

    def _provider(self):
        # we try provider from config; fallback to class/module names
        prov = self._cfg_str("provider", None)
        if prov: return str(prov)
        return f"{self._llm.__class__.__module__}.{self._llm.__class__.__name__}"

    def _model(self):
        return self._cfg_str("model", None)

    def _max_tokens(self):
        return self._cfg_num("max_tokens")

    def _knobs_snapshot(self) -> Dict[str, Any]:
        return {
            "temperature": self._cfg_num("temperature"),
            "top_p": self._cfg_num("top_p"),
            "top_k": self._cfg_num("top_k"),
            "max_tokens": self._max_tokens(),
        }

    def _messages_meta(self, messages) -> Dict[str, Any]:
        try:
            n = len(messages) if isinstance(messages, list) else None
            total_chars = 0
            if isinstance(messages, list):
                for m in messages:
                    c = m.get("content")
                    if isinstance(c, str):
                        total_chars += len(c)
            return {"count": n, "total_chars": total_chars}
        except Exception:
            return {"count": None, "total_chars": None}

    def _batch_meta(self, messages_list) -> Dict[str, Any]:
        try:
            n = len(messages_list) if isinstance(messages_list, list) else None
            counts = []
            chars = 0
            if isinstance(messages_list, list):
                for msgs in messages_list:
                    mm = self._messages_meta(msgs)
                    counts.append(mm["count"])
                    chars += (mm["total_chars"] or 0)
            return {"batches": n, "per_batch_counts": counts[:10], "total_chars": chars}
        except Exception:
            return {"batches": None, "per_batch_counts": None, "total_chars": None}

    def _log(self, payload: dict):
        payload.setdefault("ts", _now())
        payload.setdefault("who", self._name)
        append_jsonl(self._trace_path, payload)
        if self._echo:
            # single-line echo for quick eyes-on
            kind = payload.get("event")
            model = payload.get("model")
            prov = payload.get("provider")
            took = payload.get("took_ms")
            knobs = payload.get("knobs", {})
            print(f"[api-trace] {kind} {prov}:{model} took={took}ms "
                  f"temp={knobs.get('temperature')} top_p={knobs.get('top_p')} top_k={knobs.get('top_k')} max_tokens={knobs.get('max_tokens')}",
                  flush=True)

    # ---------------- public call wrappers ----------------
    def __call__(self, messages: List[dict], **kwargs):
        t0 = time.time()
        req = {
            "event": "request",
            "provider": self._provider(),
            "model": self._model(),
            "api_method": "__call__",
            "knobs": self._knobs_snapshot(),
            "messages_meta": self._messages_meta(messages),
            "kwargs": {
                # we only record presence / types for sensitive fields; avoid dumping prompts
                "json_schema": bool(kwargs.get("json_schema") is not None),
                "timeout": kwargs.get("timeout", None),
            },
        }
        self._log(req)
        try:
            out = self._llm(messages, **kwargs)
            took = int((time.time() - t0) * 1000)
            resp = {
                "event": "response",
                "provider": self._provider(),
                "model": self._model(),
                "api_method": "__call__",
                "took_ms": took,
                # light footprint: record rough size/info, not full content
                "response_meta": _safe_shape(out),
            }
            self._log(resp)
            return out
        except Exception as e:
            took = int((time.time() - t0) * 1000)
            self._log({
                "event": "error",
                "provider": self._provider(),
                "model": self._model(),
                "api_method": "__call__",
                "took_ms": took,
                "error": repr(e),
            })
            raise

    def batch(self, messages_list: List[List[dict]], **kwargs):
        t0 = time.time()
        req = {
            "event": "request",
            "provider": self._provider(),
            "model": self._model(),
            "api_method": "batch",
            "knobs": self._knobs_snapshot(),
            "batch_meta": self._batch_meta(messages_list),
            "kwargs": {
                "json_schema": bool(kwargs.get("json_schema") is not None),
                "timeout": kwargs.get("timeout", None),
            },
        }
        self._log(req)
        try:
            out = self._llm.batch(messages_list, **kwargs)
            took = int((time.time() - t0) * 1000)
            resp = {
                "event": "response",
                "provider": self._provider(),
                "model": self._model(),
                "api_method": "batch",
                "took_ms": took,
                "response_meta": _safe_shape(out),
            }
            self._log(resp)
            return out
        except Exception as e:
            took = int((time.time() - t0) * 1000)
            self._log({
                "event": "error",
                "provider": self._provider(),
                "model": self._model(),
                "api_method": "batch",
                "took_ms": took,
                "error": repr(e),
            })
            raise

def _safe_shape(obj):
    """
    Record a tiny ‘shape’ so you can see what came back without storing payloads.
    """
    try:
        if isinstance(obj, list):
            return {"type": "list", "len": len(obj)}
        if isinstance(obj, dict):
            keys = list(obj.keys())
            return {"type": "dict", "keys": keys[:12], "nkeys": len(keys)}
        if isinstance(obj, str):
            return {"type": "str", "len": len(obj)}
        return {"type": type(obj).__name__}
    except Exception:
        return {"type": "unknown"}
