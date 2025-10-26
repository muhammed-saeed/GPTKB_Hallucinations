# llm/deepseek_client.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import json, re, os
from dotenv import load_dotenv
from openai import OpenAI, BadRequestError

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_json_loose(text: str) -> Optional[Dict[str, Any]]:
    # 1) fenced ```json ... ```
    m = _JSON_FENCE_RE.search(text or "")
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # 2) first balanced {...}
    s = (text or "").find("{")
    if s != -1:
        depth = 0
        for i, ch in enumerate(text[s:], start=s):
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[s:i+1])
                    except Exception:
                        break
    # 3) raw json
    try:
        return json.loads(text)
    except Exception:
        return None


def _schema_hint(schema: Optional[Dict[str, Any]]) -> str:
    """Short human hint that mirrors the JSON schema fields used."""
    if not schema:
        return ""
    try:
        js = json.dumps(schema)
        if '"facts"' in js:
            needs_conf = '"confidence"' in js
            if needs_conf:
                return (
                    "\nReturn ONLY valid JSON with shape:\n"
                    '{ "facts": [ { "subject": str, "predicate": str, "object": str, "confidence": number } ] }\n'
                    "No prose. No code fences."
                )
            else:
                return (
                    "\nReturn ONLY valid JSON with shape:\n"
                    '{ "facts": [ { "subject": str, "predicate": str, "object": str } ] }\n'
                    "No prose. No code fences."
                )
        if '"phrases"' in js:
            needs_conf = '"confidence"' in js
            if needs_conf:
                return (
                    "\nReturn ONLY valid JSON with shape:\n"
                    '{ "phrases": [ { "phrase": str, "is_ne": bool, "confidence": number } ] }\n'
                    "No prose. No code fences."
                )
            else:
                return (
                    "\nReturn ONLY valid JSON with shape:\n"
                    '{ "phrases": [ { "phrase": str, "is_ne": bool } ] }\n'
                    "No prose. No code fences."
                )
    except Exception:
        pass
    return "\nReturn ONLY valid JSON. No prose. No code fences."


def _normalize_elicitation_payload(data: dict) -> dict:
    if not isinstance(data, dict):
        return {"_raw": data}
    if "facts" not in data and isinstance(data.get("triples"), list):
        data["facts"] = data["triples"]
    if "facts" not in data and isinstance(data.get("items"), list):
        data["facts"] = data["items"]
    if "facts" not in data:
        data["facts"] = []
    return data


class DeepSeekLLM:
    """
    DeepSeek chat client via OpenAI SDK (base_url=https://api.deepseek.com).
    DeepSeek is OpenAI-compatible but does not support JSON Schema; we:
      1) Try response_format=json_object
      2) If that fails, add strict JSON instruction with a schema hint and parse loosely.
    """
    def __init__(self, model: str, api_key: str, base_url: str = "https://api.deepseek.com"):
        load_dotenv()
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        if os.getenv("DEBUG_PROMPTS"):
            print(f"[deepseek] model={self.model} base_url={base_url}")

    def generate(
        self,
        messages: List[Dict[str, Any]],
        *,
        json_schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.2,
        top_p: float = 1.0,
        top_k: Optional[int] = None,   # ignored
        max_tokens: int = 2048,
        seed: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        base_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        # (1) Try json_object first
        try:
            kwargs = dict(base_kwargs)
            kwargs["response_format"] = {"type": "json_object"}
            out = self.client.chat.completions.create(**kwargs)
            text = (out.choices[0].message.content or "").strip()
            parsed = _parse_json_loose(text)
            if parsed is not None:
                return _normalize_elicitation_payload(parsed)
        except BadRequestError:
            pass
        except Exception:
            pass

        # (2) Forced JSON via instruction + schema hint
        forced = list(messages)
        schema_tip = _schema_hint(json_schema)
        forced.append({
            "role": "user",
            "content": "Return ONLY valid JSON. No prose. No markdown fences." + schema_tip
        })

        kwargs = dict(base_kwargs)
        kwargs["messages"] = forced
        out = self.client.chat.completions.create(**kwargs)
        text = (out.choices[0].message.content or "").strip()
        parsed = _parse_json_loose(text)
        if parsed is not None:
            return _normalize_elicitation_payload(parsed)

        # Fallback keeps pipeline alive without retries
        if json_schema and "facts" in json.dumps(json_schema):
            return {"facts": []}
        if json_schema and "phrases" in json.dumps(json_schema):
            return {"phrases": []}
        return {"_raw": text}
