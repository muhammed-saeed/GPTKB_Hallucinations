# tools/replicate_stream_check.py
import os, sys, argparse, json, textwrap
from dotenv import load_dotenv

# Ensure project root on sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import replicate  # pip install replicate
from settings import settings

load_dotenv()


def _minify_schema(schema: dict) -> str:
    try:
        return json.dumps(schema, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return "{}"


def _strict_system_preface(schema: dict, model_slug: str) -> str:
    mini = _minify_schema(schema)
    if model_slug.startswith("google/gemini"):
        return (
            "You MUST output a single valid JSON object that matches this JSON Schema exactly. "
            "Do not include any extra keys, comments, or prose. If uncertain, return an empty valid object per schema.\n"
            f"SCHEMA: {mini}"
        )
    else:
        return (
            "Return ONLY a single valid JSON object that matches this schema. No prose, no markdown.\n"
            f"SCHEMA: {mini}"
        )


def _fewshot() -> str:
    return (
        "EXAMPLE:\n"
        'USER: Return EXACTLY this JSON: {"facts":[{"subject":"Ping","predicate":"says","object":"hello"}]}\n'
        'ASSISTANT: {"facts":[{"subject":"Ping","predicate":"says","object":"hello"}]}\n'
    )


def _collapse_messages(messages):
    parts = []
    for m in messages:
        role = (m.get("role") or "user").upper()
        content = (m.get("content") or "").strip()
        parts.append(f"{role}: {content}")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)


def _build_inputs(cfg, *, model_slug: str, messages, json_schema: dict | None, max_tokens: int | None):
    # extras from settings (e.g., dynamic_thinking, prefer)
    extra = dict(cfg.extra_inputs or {})

    # strict preface as system_prompt when schema is present
    system_prompt = None
    if json_schema:
        system_prompt = _strict_system_preface(json_schema, model_slug)

    inputs = {
        "prompt": _collapse_messages(messages),
    }
    if system_prompt:
        inputs["system_prompt"] = system_prompt

    # knobs
    if cfg.temperature is not None:
        inputs["temperature"] = cfg.temperature
    if cfg.top_p is not None:
        inputs["top_p"] = cfg.top_p
    if max_tokens is not None:
        inputs["max_tokens"] = max_tokens
        inputs["max_output_tokens"] = max_tokens

    # copy extras (except fields we set explicitly)
    for k, v in extra.items():
        if k in {"prefer"}:
            continue
        inputs[k] = v

    return inputs


def _try_parse_json(text: str) -> dict:
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        # strip ``` fences and retry
        t = text.strip()
        if t.startswith("```"):
            t = t.strip("`")
            nl = t.find("\n")
            if nl != -1:
                t = t[nl+1:].strip()
        try:
            return json.loads(t)
        except Exception:
            return {}


def stream_once(model_slug: str, inputs: dict, show_events: bool = False) -> str:
    print(f"\n== replicate.stream => {model_slug} ==")
    preview = {k: inputs.get(k) for k in ("system_prompt","temperature","top_p","max_tokens","max_output_tokens") if k in inputs}
    print("Inputs (preview):", json.dumps(preview, ensure_ascii=False))
    print("\n--- STREAM OUTPUT ---")
    try:
        chunks = []
        for event in replicate.stream(model_slug, input=inputs):
            s = str(event)
            chunks.append(s)
            if show_events:
                print(f"[{event.__class__.__name__}] {s}", end="")
            else:
                print(s, end="")
        print("\n---------------------")
        return "".join(chunks)
    except Exception as e:
        print(f"\n[STREAM ERROR] {e}")
        return ""


def main():
    ap = argparse.ArgumentParser(description="Stream-check Replicate models defined in settings.MODELS.")
    ap.add_argument("--model-key", default="gemini-flash",
                    help="Key from settings.MODELS (replicate-backed). e.g., gemini-flash or grok4")
    ap.add_argument("--schema", choices=["facts","phrases","none"], default="facts",
                    help="Which schema to enforce.")
    ap.add_argument("--subject", default="Ali ibn Abi Talib")
    ap.add_argument("--show-events", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    if args.model_key not in settings.MODELS:
        raise SystemExit(f"Unknown --model-key '{args.model_key}'. Available: {', '.join(settings.MODELS.keys())}")

    cfg = settings.MODELS[args.model_key]
    if (cfg.provider or "").lower() != "replicate":
        raise SystemExit(f"--model-key '{args.model_key}' is not provider=replicate (provider={cfg.provider}).")

    model_slug = cfg.model

    if args.schema == "facts":
        schema = {
            "type":"object",
            "properties":{
                "facts":{"type":"array","items":{
                    "type":"object",
                    "properties":{
                        "subject":{"type":"string"},
                        "predicate":{"type":"string"},
                        "object":{"type":"string"}
                    },
                    "required":["subject","predicate","object"]
                }}
            },
            "required":["facts"]
        }
        user = (
            "You are a knowledge base construction expert.\n"
            "Return ONLY JSON per the schema.\n\n"
            f"Subject: {args.subject}\n\n"
            "Produce as many (subject, predicate, object) triples as you can."
        )
    elif args.schema == "phrases":
        schema = {
            "type":"object",
            "properties":{
                "phrases":{"type":"array","items":{
                    "type":"object",
                    "properties":{
                        "phrase":{"type":"string"},
                        "is_ne":{"type":"boolean"}
                    },
                    "required":["phrase","is_ne"]
                }}
            },
            "required":["phrases"]
        }
        user = (
            "Given the list of lines below, return ONLY JSON with tokens to keep as named entities.\n"
            "Lines:\nMIT\nNew York City\nAlan Turing\n"
        )
    else:
        schema = None
        user = "Say hello as JSON with a single key 'text'."

    messages = [
        {"role": "system", "content": "Return JSON only. No prose. No code fences."},
        {"role": "user", "content": _fewshot() + "\n" + user},
    ]

    inputs = _build_inputs(cfg, model_slug=model_slug, messages=messages, json_schema=schema, max_tokens=args.max_tokens)
    raw = stream_once(model_slug, inputs, show_events=args.show_events)

    if not raw:
        print("\n[WARN] Empty output from Replicate stream.")
        return

    parsed = _try_parse_json(raw)
    if parsed:
        print("\n== Parsed JSON ==\n" + json.dumps(parsed, indent=2, ensure_ascii=False))
    else:
        print("\n[WARN] Could not parse JSON. Raw follows:\n")
        print(raw[:4000])


if __name__ == "__main__":
    main()
