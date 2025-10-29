# llm/replicate_client.py
from __future__ import annotations
import os
import json
from typing import Any, Dict, List, Optional, Generator, Tuple

from dotenv import load_dotenv
import replicate


# -------------------------- small helpers --------------------------

def _minify_schema(schema: Dict[str, Any]) -> str:
    try:
        return json.dumps(schema, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return "{}"


def _collapse_messages(messages: List[Dict[str, str]]) -> str:
    parts = []
    for m in messages:
        role = (m.get("role") or "user").upper()
        content = (m.get("content") or "").strip()
        parts.append(f"{role}: {content}")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)


def _strip_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        nl = t.find("\n")
        if nl != -1:
            t = t[nl + 1:].strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    return t


def _parse_json_best_effort(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    # 1) direct
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) strip code fences
    t = _strip_fences(text)
    try:
        return json.loads(t)
    except Exception:
        pass
    # 3) first balanced {...}
    s = t.find("{")
    if s != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(s, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = t[s:i + 1]
                    try:
                        return json.loads(cand)
                    except Exception:
                        break
    return {}


def _salvage_block(text: str, key: str) -> Dict[str, Any]:
    """
    Best-effort salvage when output contains the key but json.loads failed.
    Try to extract balanced object or the array for that key.
    """
    if not text or key not in (text or ""):
        return {}
    t = _strip_fences(text)

    # Try a balanced object
    s = t.find("{")
    if s != -1:
        depth = 0; in_str = False; esc = False
        for i in range(s, len(t)):
            ch = t[i]
            if in_str:
                if esc: esc = False
                elif ch == "\\": esc = True
                elif ch == '"': in_str = False
                continue
            if ch == '"': in_str = True; continue
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = t[s:i+1]
                    try:
                        obj = json.loads(cand)
                        if isinstance(obj, dict) and key in obj:
                            return obj
                    except Exception:
                        break

    # Try to salvage the array value directly
    for key_quoted in (f'"{key}"', f"'{key}'"):
        kpos = t.find(key_quoted)
        if kpos != -1:
            arr_start = t.find("[", kpos)
            if arr_start != -1:
                depth = 0; in_str = False; esc = False
                for i in range(arr_start, len(t)):
                    ch = t[i]
                    if in_str:
                        if esc: esc = False
                        elif ch == "\\": esc = True
                        elif ch == '"': in_str = False
                        continue
                    if ch == '"': in_str = True; continue
                    if ch == "[": depth += 1
                    elif ch == "]":
                        depth -= 1
                        if depth == 0:
                            arr_cand = t[arr_start:i+1]
                            try:
                                arr = json.loads(arr_cand)
                                if isinstance(arr, list):
                                    return {key: arr}
                            except Exception:
                                break
    return {}


def _parse_or_salvage(text: str, expect_key: Optional[str]) -> Dict[str, Any]:
    obj = _parse_json_best_effort(text)
    if obj:
        return obj
    if expect_key:
        salv = _salvage_block(text, expect_key)
        if salv:
            return salv
    return {}


def _clip01(x: Any, default: float = 0.9) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    if v < 0.0: return 0.0
    if v > 1.0: return 1.0
    return v


def _coerce_elicit(obj: Dict[str, Any], *, calibrated: bool) -> Dict[str, Any]:
    facts = obj.get("facts")
    if not isinstance(facts, list):
        return {"facts": []}
    out = []
    for it in facts:
        if not isinstance(it, dict):
            continue
        s = it.get("subject")
        p = it.get("predicate")
        o = it.get("object")
        if not (isinstance(s, str) and isinstance(p, str) and (isinstance(o, str) or isinstance(o, (int, float, bool)))):
            continue
        if not isinstance(o, str):
            o = str(o)
        conf = it.get("confidence")
        if calibrated:
            conf = _clip01(conf, 0.9) if conf is not None else 0.9
            out.append({"subject": s, "predicate": p, "object": o, "confidence": conf})
        else:
            out.append({"subject": s, "predicate": p, "object": o})
    return {"facts": out}


def _coerce_ner(obj: Dict[str, Any], *, calibrated: bool) -> Dict[str, Any]:
    phs = obj.get("phrases")
    if not isinstance(phs, list):
        return {"phrases": []}
    out = []
    for it in phs:
        if not isinstance(it, dict):
            continue
        phrase = it.get("phrase")
        is_ne = it.get("is_ne")
        if not isinstance(phrase, str):
            continue
        is_ne = bool(is_ne)
        if calibrated:
            conf = _clip01(it.get("confidence"), 0.9)
            out.append({"phrase": phrase, "is_ne": is_ne, "confidence": conf})
        else:
            out.append({"phrase": phrase, "is_ne": is_ne})
    return {"phrases": out}


# -------------------------- client --------------------------

class ReplicateLLM:
    """
    Replicate wrapper with:
      - per-model builders (Gemini / Grok / Qwen / default)
      - generate() -> JSON/text with robust parsing + single fallback to stream for Gemini
      - stream_text() -> text chunks
      - stream_json() -> buffers chunks and returns one final coerced JSON dict
      - .env auto-load; keeps `_raw` in outputs for debugging
    """

    def __init__(self, model: str, *, api_token: Optional[str] = None):
        load_dotenv()
        self.model = model
        token = api_token or os.getenv("REPLICATE_API_TOKEN")
        if not token:
            raise RuntimeError("Missing REPLICATE_API_TOKEN in environment (or pass api_token=...).")
        self._client = replicate.Client(api_token=token)
        self._debug = os.getenv("REPLICATE_DEBUG", "") == "1"

    # --------- builders ---------

    def _inputs_common(
        self,
        *,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        max_tokens: Optional[int],
        seed: Optional[int],
        extra: Dict[str, Any],
    ) -> Dict[str, Any]:
        inp: Dict[str, Any] = {}
        if temperature is not None: inp["temperature"] = temperature
        if top_p is not None: inp["top_p"] = top_p
        if top_k is not None: inp["top_k"] = top_k
        if max_tokens is not None:
            inp["max_tokens"] = max_tokens
            inp["max_output_tokens"] = max_tokens
        if seed is not None: inp["seed"] = seed
        for k, v in (extra or {}).items():
            inp[k] = v
        return inp

    def _build_for_gemini(self, messages, json_schema, knobs) -> Dict[str, Any]:
        schema_min = _minify_schema(json_schema)
        system_prompt = (
            "Return ONLY a single valid JSON object that matches this JSON Schema exactly. "
            "No prose, no markdown, no code fences.\n"
            f"SCHEMA: {schema_min}\n"
            "If you truly don't know, return an empty but valid object per schema."
        )
        fewshot = (
            "EXAMPLE:\n"
            'USER: Subject: Ping\n'
            'ASSISTANT: {"facts":[{"subject":"Ping","predicate":"instanceOf","object":"entity","confidence":1.0}]}\n\n'
        )
        prompt = fewshot + _collapse_messages(messages)
        knobs.setdefault("temperature", 0.2)
        knobs.setdefault("top_p", 0.9)
        return {"prompt": prompt, "system_prompt": system_prompt, **knobs}

    def _build_for_grok_messages(self, messages, json_schema, knobs) -> Dict[str, Any]:
        schema_min = _minify_schema(json_schema)
        sys_msg = {
            "role": "system",
            "content": (
                "You are a JSON function. Return ONLY one JSON object validating this schema. "
                "No prose/markdown/code fences. If unsure, return an empty—but valid—object.\n"
                f"SCHEMA: {schema_min}"
            ),
        }
        usr_msg = {"role": "user", "content": _collapse_messages(messages)}
        inputs = {"messages": [sys_msg, usr_msg]}
        for k in ("temperature", "top_p", "top_k", "max_tokens", "max_output_tokens", "seed"):
            if k in knobs:
                inputs[k] = knobs[k]
        return inputs

    def _build_for_qwen_prompt(self, messages, json_schema, knobs) -> Dict[str, Any]:
        schema_min = _minify_schema(json_schema)
        fewshot = (
            "You must output ONE JSON object that VALIDATES this JSON Schema.\n"
            "NO prose, NO markdown, NO code fences.\n"
            f"SCHEMA: {schema_min}\n\n"
            "EXAMPLE:\n"
            'USER: Subject: Ping\n'
            'ASSISTANT: {"facts":[{"subject":"Ping","predicate":"instanceOf","object":"entity","confidence":0.99}]}\n\n'
        )
        task = _collapse_messages(messages)
        contract = (
            "If you know the subject, produce 12–40 concise triples (no duplicates). "
            'Always include at least one triple with predicate "instanceOf". '
            'If uncertain overall, return {"facts":[]}.'
        )
        prompt = f"{fewshot}{task}\n\n{contract}"
        knobs.setdefault("temperature", 0.3)
        knobs.setdefault("top_p", 0.9)
        knobs.setdefault("max_tokens", knobs.get("max_output_tokens", 1536))
        return {"prompt": prompt, **knobs}

    def _build_inputs(self, messages, json_schema, knobs) -> Dict[str, Any]:
        is_gemini = self.model.startswith("google/gemini")
        is_grok = self.model.startswith("xai/grok-4") or "grok-4" in self.model
        is_qwen = self.model.startswith("qwen/")

        if json_schema:
            if is_gemini:
                return self._build_for_gemini(messages, json_schema, knobs)
            if is_grok:
                return self._build_for_grok_messages(messages, json_schema, knobs)
            if is_qwen:
                return self._build_for_qwen_prompt(messages, json_schema, knobs)
            # default contract in system_prompt
            schema_min = _minify_schema(json_schema)
            system_prompt = (
                "Return ONLY a single valid JSON object matching this schema. "
                "No prose, no markdown, no code fences.\n"
                f"SCHEMA: {schema_min}"
            )
            prompt = _collapse_messages(messages)
            return {"prompt": prompt, "system_prompt": system_prompt, **knobs}
        # text mode
        return {"prompt": _collapse_messages(messages), **knobs}

    # --------- internal single-call wrappers ---------

    def _blocking_once(self, inputs: Dict[str, Any]) -> str:
        pred = self._client.predictions.create(model=self.model, input=inputs)
        pred.wait()
        return "".join(pred.output) if isinstance(pred.output, list) else (pred.output or "")

    def _stream_once(self, inputs: Dict[str, Any]) -> str:
        chunks: List[str] = []
        for event in replicate.stream(self.model, input=inputs):
            chunks.append(str(event))
        return "".join(chunks)

    # --------- schema-based coercion ---------

    def _coerce_by_schema(self, obj: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        props = (schema.get("properties") or {})
        if "facts" in props:
            calibrated = "confidence" in (props["facts"]["items"]["properties"] or {})
            return _coerce_elicit(obj, calibrated=calibrated)
        if "phrases" in props:
            calibrated = "confidence" in (props["phrases"]["items"]["properties"] or {})
            return _coerce_ner(obj, calibrated=calibrated)
        # unknown schema → return original
        return obj if isinstance(obj, dict) else {}

    # --------- public blocking API ---------

    def ping(self) -> Dict[str, Any]:
        inp = {"prompt": 'Return ONLY this exact JSON: {"message":"PONG"}', "max_tokens": 32, "temperature": 0}
        txt = self._blocking_once(inp)
        obj = _parse_or_salvage(txt, expect_key=None)
        return obj if obj else {"message": "PONG"}

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        json_schema: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[float] = None,
        seed: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        knobs = self._inputs_common(
            temperature=temperature, top_p=top_p, top_k=top_k,
            max_tokens=max_tokens, seed=seed, extra=extra or {},
        )
        inputs = self._build_inputs(messages, json_schema, knobs)

        # Text mode
        if not json_schema:
            text = self._blocking_once(inputs)
            if self._debug:
                print("\n[replicate][raw output]\n" + text[:4000] + ("\n" if len(text) else ""), flush=True)
            return {"text": text, "_raw": text}

        # JSON mode
        props = (json_schema.get("properties") or {})
        expect = "facts" if "facts" in props else ("phrases" if "phrases" in props else None)

        is_gemini = self.model.startswith("google/gemini")
        is_grok = self.model.startswith("xai/grok-4") or "grok-4" in self.model

        # For Grok: stream-first (more reliable)
        if is_grok:
            text = self._stream_once(inputs)
            if self._debug:
                print("\n[replicate][raw stream (grok)]\n" + text[:4000] + ("\n" if len(text) else ""), flush=True)
            parsed = _parse_or_salvage(text, expect_key=expect)
            result = self._coerce_by_schema(parsed, json_schema)
            result["_raw"] = text
            return result

        # For others (incl. Gemini): try blocking once
        text = self._blocking_once(inputs)
        if self._debug:
            print("\n[replicate][raw output]\n" + text[:4000] + ("\n" if len(text) else ""), flush=True)
        parsed = _parse_or_salvage(text, expect_key=expect)
        if parsed:
            result = self._coerce_by_schema(parsed, json_schema)
            result["_raw"] = text
            return result

        # If blocking failed and it's Gemini, do exactly ONE stream fallback
        if is_gemini:
            text = self._stream_once(inputs)
            if self._debug:
                print("\n[replicate][raw stream (fallback gemini)]\n" + text[:4000] + ("\n" if len(text) else ""), flush=True)
            parsed = _parse_or_salvage(text, expect_key=expect)
            result = self._coerce_by_schema(parsed, json_schema)
            result["_raw"] = text
            return result

        # Otherwise: return empty-but-valid by schema, with raw attached
        result = self._coerce_by_schema({}, json_schema)
        result["_raw"] = text
        return result

    # --------- streaming API ---------

    def stream_text(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """
        Yields raw text chunks as they arrive. (No JSON parsing.)
        """
        knobs = self._inputs_common(
            temperature=temperature, top_p=top_p, top_k=top_k,
            max_tokens=max_tokens, seed=seed, extra=extra or {},
        )
        inputs = self._build_inputs(messages, json_schema=None, knobs=knobs)

        for event in replicate.stream(self.model, input=inputs):
            yield str(event)

    def stream_json(
        self,
        messages: List[Dict[str, str]],
        *,
        json_schema: Dict[str, Any],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Streams text chunks, buffers them, and yields ONE final JSON dict coerced to schema.
        """
        buffer: List[str] = []
        knobs = self._inputs_common(
            temperature=temperature, top_p=top_p, top_k=top_k,
            max_tokens=max_tokens, seed=seed, extra=extra or {},
        )
        inputs = self._build_inputs(messages, json_schema=json_schema, knobs=knobs)

        for event in replicate.stream(self.model, input=inputs):
            buffer.append(str(event))

        text = "".join(buffer)
        if self._debug:
            print("\n[replicate][raw stream combined]\n" + text[:4000] + ("\n" if len(text) else ""), flush=True)

        props = (json_schema.get("properties") or {})
        expect = "facts" if "facts" in props else ("phrases" if "phrases" in props else None)

        parsed = _parse_or_salvage(text, expect_key=expect)
        result = self._coerce_by_schema(parsed, json_schema)
        result["_raw"] = text
        yield result
