# llm/openai_client.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import json
from openai import OpenAI


class OpenAIClient:
    """
    Unified OpenAI client that can call either:
      • Chat Completions API (gpt-4o, gpt-4o-mini, etc.)
      • Responses API (gpt-5 family, e.g. gpt-5-nano)

    Usage:
        client = OpenAIClient(
            model="gpt-4o-mini",
            api_key="sk-...",
            base_url=None,               # or custom compatible base
            max_tokens=1024,
            temperature=0.0,
            top_p=1.0,
            use_responses_api=False,     # True for gpt-5 family
            extra_inputs={
                # only used by Responses API:
                # "reasoning": {"effort": "low|medium|high|minimal"},
                # "text": {"verbosity": "low|medium|high"},
            },
        )
        out = client(messages, json_schema=SCHEMA_OR_None)
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        max_tokens: Optional[int] = 1024,
        temperature: Optional[float] = 0.0,
        top_p: Optional[float] = 1.0,
        use_responses_api: bool = False,
        extra_inputs: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.use_responses_api = bool(use_responses_api or (model or "").startswith("gpt-5"))
        self.extra_inputs = extra_inputs or {}

        # Construct OpenAI SDK client
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

    # ----- Public callable -----
    def __call__(self, messages: List[Dict[str, str]], json_schema: Optional[Dict[str, Any]] = None):
        if self.use_responses_api:
            return self._call_responses(messages, json_schema)
        return self._call_chat(messages, json_schema)

    # ----- Internal: Chat Completions API -----
    def _call_chat(self, messages: List[Dict[str, str]], json_schema: Optional[Dict[str, Any]]):
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

        if json_schema:
            # Chat Completions requires schema name
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "schema",
                    "schema": json_schema,
                },
            }

        resp = self.client.chat.completions.create(**kwargs)
        text = (resp.choices[0].message.content or "").strip()

        if json_schema:
            # try to parse JSON; fall back to a dict with _raw
            try:
                return json.loads(text)
            except Exception:
                return {"_raw": text}
        else:
            return {"text": text}

    # --- inside llm/openai_client.py ---

    def _call_responses(self, messages, json_schema):
        """
        Responses API (gpt-5 family). 
        Handles both modern SDKs (with or without response_format) 
        and automatically omits unsupported parameters.
        """
        reasoning = self.extra_inputs.get("reasoning")
        text_opts = self.extra_inputs.get("text")

        # Base kwargs: omit temperature/top_p since GPT-5 disallows them
        base_kwargs = {
            "model": self.model,
            "input": messages,
            "max_output_tokens": self.max_tokens,
        }

        if reasoning:
            base_kwargs["reasoning"] = reasoning
        if text_opts:
            base_kwargs["text"] = text_opts

        # Try to include schema (new SDKs only)
        if json_schema:
            with_schema_kwargs = dict(base_kwargs)
            with_schema_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "schema", "schema": json_schema},
            }
        else:
            with_schema_kwargs = dict(base_kwargs)
            with_schema_kwargs["response_format"] = {"type": "text"}

        try:
            # Newer SDK (supports response_format)
            resp = self.client.responses.create(**with_schema_kwargs)
        except TypeError:
            # Older SDK, retry without response_format
            resp = self.client.responses.create(**base_kwargs)
        except Exception as e:
            # Some versions reject unsupported args; print and retry minimal
            if "Unsupported parameter" in str(e):
                resp = self.client.responses.create(**base_kwargs)
            else:
                raise

        # Extract text from output
        output_text = getattr(resp, "output_text", None)
        if not output_text:
            try:
                parts = []
                for block in getattr(resp, "output", []) or []:
                    for c in getattr(block, "content", []) or []:
                        if getattr(c, "type", "") == "output_text":
                            parts.append(getattr(c, "text", ""))
                output_text = "".join(parts).strip()
            except Exception:
                output_text = ""

        # Return parsed JSON or raw text
        if json_schema:
            try:
                return json.loads(output_text)
            except Exception:
                return {"_raw": output_text}
        else:
            return {"text": output_text}


__all__ = ["OpenAIClient"]
