# tools/replicate_ping.py
import os, sys, json, argparse
from dotenv import load_dotenv

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import replicate
from settings import settings

load_dotenv()


def _minify_schema(s: dict) -> str:
    try:
        return json.dumps(s, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return "{}"


def _preface(schema: dict, slug: str) -> str:
    mini = _minify_schema(schema)
    if slug.startswith("google/gemini"):
        return (
            "You MUST output a single valid JSON object that matches this JSON Schema exactly. "
            "No prose or code fences. If unsure, return an empty but valid object.\n"
            f"SCHEMA: {mini}"
        )
    return "Return ONLY a valid JSON object per schema. No prose.\n" f"SCHEMA: {mini}"


def _fewshot() -> str:
    return (
        "EXAMPLE:\n"
        'USER: Return EXACTLY this JSON: {"facts":[{"subject":"Ping","predicate":"says","object":"hello"}]}\n'
        'ASSISTANT: {"facts":[{"subject":"Ping","predicate":"says","object":"hello"}]}\n'
    )


def _collapse(messages):
    parts = []
    for m in messages:
        parts.append(f"{m['role'].upper()}: {m['content'].strip()}")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)


def _try_parse(text: str) -> dict:
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
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


def main():
    ap = argparse.ArgumentParser(description="Ping a Replicate model with a strict JSON request.")
    ap.add_argument("--model-key", default="gemini-flash", help="Key from settings.MODELS (replicate-backed).")
    ap.add_argument("--subject", default="Ali ibn Abi Talib")
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    if args.model_key not in settings.MODELS:
        raise SystemExit(f"Unknown --model-key '{args.model_key}'. Available: {', '.join(settings.MODELS.keys())}")
    cfg = settings.MODELS[args.model_key]
    if (cfg.provider or "").lower() != "replicate":
        raise SystemExit(f"--model-key '{args.model_key}' is not provider=replicate (provider={cfg.provider}).")

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

    messages = [
        {"role": "system", "content": "Return JSON only. No prose. No code fences."},
        {"role": "user", "content": _fewshot() + f"\nSubject: {args.subject}\nReturn as many (subject,predicate,object) triples as possible."},
    ]

    inputs = {
        "prompt": _collapse(messages),
        "system_prompt": _preface(schema, cfg.model),
        "temperature": cfg.temperature if cfg.temperature is not None else 0.2,
        "top_p": cfg.top_p if cfg.top_p is not None else 0.9,
        "max_tokens": args.max_tokens,
        "max_output_tokens": args.max_tokens,
    }
    # carry extras
    for k, v in (cfg.extra_inputs or {}).items():
        if k == "prefer":
            continue
        inputs[k] = v

    replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))
    print(f"== replicate_ping ==\nmodel={cfg.model}\n")
    print("Inputs (preview):", json.dumps({k: inputs.get(k) for k in ["system_prompt","temperature","top_p","max_tokens","max_output_tokens"]}, ensure_ascii=False))
    pred = replicate.predictions.create(model=cfg.model, input=inputs)
    pred.wait()

    if pred.status != "succeeded":
        print(f"[ERROR] status={pred.status} details={pred.error}")
        sys.exit(1)

    out = pred.output
    text = "".join(out) if isinstance(out, list) else (out or "")
    print("\n--- RAW OUTPUT (first 1200 chars) ---\n" + (text[:1200] if text else "") + "\n---------------------------\n")

    parsed = _try_parse(text)
    if parsed:
        print("== Parsed JSON ==\n" + json.dumps(parsed, indent=2, ensure_ascii=False))
    else:
        print("[WARN] Could not parse JSON. See RAW OUTPUT above.")


if __name__ == "__main__":
    main()
