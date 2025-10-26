# llm/factory.py
from __future__ import annotations
import os
from typing import Any, List, Optional
from dotenv import load_dotenv

from .config import ModelConfig
from .openai_client import OpenAIClient
from .replicate_client import ReplicateLLM
from .deepseek_client import DeepSeekLLM

try:
    from .unsloth_client import UnslothLLM
    _HAS_UNSLOTH = True
except Exception:
    _HAS_UNSLOTH = False

load_dotenv()


def _get_key(env_name: Optional[str], fallbacks: Optional[List[str]] = None) -> Optional[str]:
    if env_name:
        v = os.getenv(env_name)
        if v:
            return v
    if fallbacks:
        for f in fallbacks:
            v = os.getenv(f)
            if v:
                return v
    return None


def _is_gpt5_model(model_name: Optional[str]) -> bool:
    """Heuristic: OpenAI GPT-5 family (Responses API)."""
    if not model_name:
        return False
    return str(model_name).lower().startswith("gpt-5")


def make_llm_from_config(cfg: ModelConfig):
    """
    Returns a callable:
        out = llm(messages, json_schema)
    Out shape:
      - with json_schema: parsed dict matching your schema (never raw string)
      - without schema  : {"text": "..."}
    """
    provider = (cfg.provider or "").lower()

    # -------- OpenAI / compatible (single-call client) --------
    if provider in ("openai", "openai_compatible"):
        key = _get_key(cfg.api_key_env, ["OPENAI_API_KEY"])
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set.")

        # Auto-select Responses API for GPT-5* models (e.g., gpt-5-nano) or when explicitly requested
        use_responses_api = bool(cfg.use_responses_api or _is_gpt5_model(cfg.model))

        # Prefer cfg.base_url; otherwise OPENAI_BASE_URL; default official
        base_url = cfg.base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"

        # NOTE: OpenAIClient internally handles both Chat Completions and Responses API,
        # controlled by use_responses_api flag; it also passes through extra_inputs
        client = OpenAIClient(
            model=cfg.model,
            max_tokens=cfg.max_tokens or 1024,
            temperature=cfg.temperature if cfg.temperature is not None else 0.0,
            top_p=cfg.top_p if cfg.top_p is not None else 1.0,
            api_key=key,
            base_url=base_url,
            extra_inputs=cfg.extra_inputs,   # for GPT-5: e.g. {"reasoning":{"effort":"minimal"}, "text":{"verbosity":"low"}}
            use_responses_api=use_responses_api,
        )

        def _gen(messages, json_schema=None):
            return client(messages, json_schema)

        return _gen

    # -------- DeepSeek (OpenAI-compatible via base_url) --------
    if provider == "deepseek":
        api_key = _get_key(cfg.api_key_env, ["DEEPSEEK_API_KEY"])
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set.")
        client = DeepSeekLLM(
            model=cfg.model,
            api_key=api_key,
            base_url=cfg.base_url or "https://api.deepseek.com",
        )

        def _gen(messages, json_schema=None):
            return client.generate(
                messages,
                json_schema=json_schema,
                temperature=cfg.temperature if cfg.temperature is not None else 0.0,
                top_p=cfg.top_p if cfg.top_p is not None else 1.0,
                max_tokens=cfg.max_tokens or 1024,
                seed=getattr(cfg, "seed", None),
                extra=cfg.extra_inputs,
            )

        return _gen

    # -------- Replicate --------
    # llm/factory.py (Replicate section)
    # -------- Replicate --------
    if provider == "replicate":
        if not os.getenv("REPLICATE_API_TOKEN"):
            raise RuntimeError("REPLICATE_API_TOKEN not set.")
        client = ReplicateLLM(model=cfg.model)

        def _gen(messages, json_schema=None):
            return client.generate(
                messages,
                json_schema=json_schema,
                temperature=cfg.temperature if cfg.temperature is not None else None,
                top_p=cfg.top_p if cfg.top_p is not None else None,
                top_k=cfg.top_k if cfg.top_k is not None else None,
                max_tokens=cfg.max_tokens if cfg.max_tokens is not None else None,
                seed=getattr(cfg, "seed", None),
                extra=cfg.extra_inputs,
            )

        return _gen


    # -------- Local via Unsloth --------
    if provider == "unsloth":
        if not _HAS_UNSLOTH:
            raise RuntimeError("Unsloth backend not available. Install unsloth & deps or remove 'unsloth' models.")
        extra = cfg.extra_inputs or {}
        client = UnslothLLM(
            model_name=cfg.model,
            max_seq_length=int(extra.get("max_seq_length", 2048)),
            dtype=extra.get("dtype"),
            load_in_4bit=bool(extra.get("load_in_4bit", False)),
            device=extra.get("device"),
            trust_remote_code=True,
            extra=extra,
        )

        def _gen(messages, json_schema=None):
            return client.generate(
                messages,
                json_schema=json_schema,
                temperature=cfg.temperature if cfg.temperature is not None else 0.0,
                top_p=cfg.top_p if cfg.top_p is not None else 1.0,
                top_k=cfg.top_k,
                max_tokens=cfg.max_tokens if cfg.max_tokens is not None else 512,
                seed=getattr(cfg, "seed", None),
                extra=cfg.extra_inputs,
            )

        return _gen

    raise ValueError(f"Unknown provider: {cfg.provider!r}")
