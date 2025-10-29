# llm/replicate_client.py
from __future__ import annotations
import os
import json
from typing import Any, Dict, List, Optional

import replicate  # pip install replicate


def _minify_schema(schema: Dict[str, Any]) -> str:
    try:
        return json.dumps(schema, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return "{}"


def _collapse_messages(messages: List[Dict[str, str]]) -> str:
    """
    Collapse OpenAI-style chat into a single prompt string.
    We keep it simple so provider quirks aren't triggered.
    """
    parts = []
    for m in messages:
        role = (m.get("role") or "user").upper()
        content = (m.get("content") or "").strip()
        parts.append(f"{role}: {content}")
    # trailing ASSISTANT: helps some models finish cleanly
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)


def _strip_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        nl = t.find("\n")
        if nl != -1:
            t = t[nl + 1 :].strip()
    return t


def _parse_json_best_effort(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    # 1) direct
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) strip fences
    t = _strip_fences(text)
    try:
        return json.loads(t)
    except Exception:
        pass
    # 3) first balanced {...}
    s = t.find("{")
    if s != -1:
        depth = 0
        for i, ch in enumerate(t[s:], start=s):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = t[s : i + 1]
                    try:
                        return json.loads(cand)
                    except Exception:
                        break
    return {}


class ReplicateLLM:
    """
    Replicate text model wrapper with per-model prompt strategy.

    - Gemini (google/gemini-*): strong JSON contract in system_prompt + tiny few-shot.
    - Grok-4 (xai/grok-4): minimal system, contract inside USER prompt, no few-shot.
    - Robust JSON parsing (fences & first-balanced-object).
    """

    def __init__(self, model: str):
        self.model = model
        token = os.getenv("REPLICATE_API_TOKEN")
        if not token:
            raise RuntimeError("Missing REPLICATE_API_TOKEN in environment.")
        replicate.Client(api_token=token)
        self._debug = os.getenv("REPLICATE_DEBUG", "") == "1"

    # ---------- internal builders ----------

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
        if temperature is not None:
            inp["temperature"] = temperature
        if top_p is not None:
            inp["top_p"] = top_p
        if top_k is not None:
            inp["top_k"] = top_k
        if max_tokens is not None:
            inp["max_tokens"] = max_tokens
            inp["max_output_tokens"] = max_tokens  # helps Gemini
        if seed is not None:
            inp["seed"] = seed
        # copy extras except keys we always manage locally
        for k, v in (extra or {}).items():
            if k in {"prefer"}:
                continue
            inp[k] = v
        return inp

    def _build_for_gemini(
        self,
        messages: List[Dict[str, str]],
        json_schema: Dict[str, Any],
        knobs: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Strong schema in system prompt + tiny few-shot
        schema_min = _minify_schema(json_schema)
        system_prompt = (
            "You MUST output a single valid JSON object that matches this JSON Schema exactly. "
            "No prose, no markdown, no code fences. If uncertain, return an empty but valid object.\n"
            f"SCHEMA: {schema_min}"
        )

        fewshot = (
            "EXAMPLE:\n"
            'USER: Return EXACTLY this JSON: {"facts":[{"subject":"Ping","predicate":"says","object":"hello"}]}\n'
            'ASSISTANT: {"facts":[{"subject":"Ping","predicate":"says","object":"hello"}]}\n\n'
        )

        prompt = fewshot + _collapse_messages(messages)

        inputs = {"prompt": prompt, "system_prompt": system_prompt}
        inputs.update(knobs)
        return inputs

    def _build_for_grok(
        self,
        messages: List[Dict[str, str]],
        json_schema: Dict[str, Any],
        knobs: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Grok prefers a very light system prompt and the contract inside USER content.
        schema_min = _minify_schema(json_schema)
        # keep system minimal (or allow user extra to override)
        sys_prompt = knobs.pop("system_prompt", None) or "You are a helpful assistant."
        # Put the contract directly into the user prompt, no few-shot.
        contract = (
            "USER: Return ONLY a single valid JSON object matching this schema. "
            "No prose, no markdown, no code fences.\n"
            f"SCHEMA: {schema_min}\n\n"
        )
        prompt = contract + _collapse_messages(messages)

        inputs = {"prompt": prompt, "system_prompt": sys_prompt}
        inputs.update(knobs)
        return inputs

    # ---------- main ----------

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        json_schema: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        knobs = self._inputs_common(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            seed=seed,
            extra=extra or {},
        )

        is_gemini = self.model.startswith("google/gemini")
        is_grok = self.model.startswith("xai/grok-4") or "grok-4" in self.model

        # Build inputs
        if json_schema:
            if is_gemini:
                inputs = self._build_for_gemini(messages, json_schema, knobs)
            elif is_grok:
                inputs = self._build_for_grok(messages, json_schema, knobs)
            else:
                # default: light contract in system prompt
                schema_min = _minify_schema(json_schema)
                system_prompt = (
                    "Return ONLY a single valid JSON object matching this schema. "
                    "No prose, no markdown, no code fences.\n"
                    f"SCHEMA: {schema_min}"
                )
                prompt = _collapse_messages(messages)
                inputs = {"prompt": prompt, "system_prompt": system_prompt}
                inputs.update(knobs)
        else:
            # no schema: just collapse messages
            prompt = _collapse_messages(messages)
            inputs = {"prompt": prompt}
            inputs.update(knobs)

        # Call Replicate once
        prediction = replicate.predictions.create(model=self.model, input=inputs)
        prediction.wait()
        if prediction.status != "succeeded":
            raise RuntimeError(
                f"Replicate prediction failed: status={prediction.status} details={prediction.error}"
            )

        out = prediction.output
        text = "".join(out) if isinstance(out, list) else (out or "")

        if self._debug:
            print("\n[replicate][raw output]\n" + text[:4000] + ("\n" if len(text) else ""), flush=True)

        if not json_schema:
            return {"text": text}

        obj = _parse_json_best_effort(text)
        if obj:
            return obj

        # Retry once with an alternate shape if model-specific
        # (Sometimes moving the contract helps.)
        try_again = False
        if is_gemini:
            # Second shot: move contract from system to top of user prompt
            schema_min = _minify_schema(json_schema)
            alt_prompt = (
                "USER: Return ONLY a single valid JSON object matching this schema. No prose or fences.\n"
                f"SCHEMA: {schema_min}\n\n" + _collapse_messages(messages)
            )
            alt_inputs = {"prompt": alt_prompt}
            alt_inputs.update(self._inputs_common(
                temperature=temperature, top_p=top_p, top_k=top_k,
                max_tokens=max_tokens, seed=seed, extra=extra or {},
            ))
            try_again = True
            inputs = alt_inputs
        elif is_grok:
            # Second shot: put the contract ONLY in system prompt
            schema_min = _minify_schema(json_schema)
            alt_prompt = _collapse_messages(messages)
            alt_inputs = {
                "prompt": alt_prompt,
                "system_prompt": (
                    "Return ONLY a single valid JSON object that matches this schema. "
                    "No prose, no markdown, no code fences.\n"
                    f"SCHEMA: {schema_min}"
                ),
            }
            alt_inputs.update(self._inputs_common(
                temperature=temperature, top_p=top_p, top_k=top_k,
                max_tokens=max_tokens, seed=seed, extra=extra or {},
            ))
            try_again = True
            inputs = alt_inputs

        if try_again:
            prediction = replicate.predictions.create(model=self.model, input=inputs)
            prediction.wait()
            if prediction.status == "succeeded":
                out = prediction.output
                text = "".join(out) if isinstance(out, list) else (out or "")
                obj = _parse_json_best_effort(text)
                if obj:
                    return obj

        # Final fallback for schema-mode so your pipeline never crashes
        schema_str = json.dumps(json_schema)
        if '"facts"' in schema_str:
            return {"facts": []}
        if '"phrases"' in schema_str:
            return {"phrases": []}
        return {}
