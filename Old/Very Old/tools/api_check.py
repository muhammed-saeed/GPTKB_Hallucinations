# tools/api_check.py
import os, sys, json, argparse
from dotenv import load_dotenv

# import factory so we can use all providers through one interface
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from llm.config import ModelConfig
from llm.factory import make_llm_from_config

load_dotenv()

ELICIT_SCHEMA_BASE = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                },
                "required": ["subject", "predicate", "object"],
            },
        }
    },
    "required": ["facts"],
}


def mask_key(key: str) -> str:
    if not key:
        return "None"
    if len(key) < 8:
        return "*" * len(key)
    return key[:3] + "…" + key[-4:]


def main():
    ap = argparse.ArgumentParser(description="Connectivity & JSON echo test for all LLM providers.")
    ap.add_argument("--provider", choices=["openai", "deepseek", "replicate"], required=True)
    ap.add_argument("--model", required=True, help="Model slug (e.g. gpt-4o-mini, xai/grok-4, google/gemini-2.5-flash)")
    ap.add_argument("--effort", choices=["minimal","low","medium","high"], default=None)
    ap.add_argument("--verbosity", choices=["low","medium","high"], default=None)
    ap.add_argument("--subject", default="Ali ibn Abi Talib")
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    print("== api_check ==\n")

    # ---- Environment diagnostics ----
    env_path = os.path.join(os.getcwd(), ".env")
    print(f"ENV loaded from: {env_path}")
    print("Keys (masked):")
    for k in ["OPENAI_API_KEY","DEEPSEEK_API_KEY","REPLICATE_API_TOKEN"]:
        v = os.getenv(k)
        print(f"  {k:<22} = {mask_key(v)}")
    print()

    # ---- Build config dynamically ----
    extra_inputs = {}
    if args.provider == "openai" and ("gpt-5" in args.model or args.effort or args.verbosity):
        extra_inputs["reasoning"] = {"effort": args.effort or "low"}
        extra_inputs["text"] = {"verbosity": args.verbosity or "low"}

    cfg = ModelConfig(
        provider=args.provider,
        model=args.model,
        api_key_env={
            "openai": "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "replicate": "REPLICATE_API_TOKEN",
        }[args.provider],
        max_tokens=args.max_tokens,
        temperature=0.0 if args.provider != "replicate" else 0.2,
        top_p=1.0 if args.provider != "replicate" else 0.9,
        use_responses_api=("gpt-5" in args.model),
        extra_inputs=extra_inputs,
    )

    # ---- Build model instance via factory ----
    llm = make_llm_from_config(cfg)

    # Prompt similar to your elicitation
    preface = (
        "SYSTEM: Return ONLY a single valid JSON object matching the schema. No prose, no code fences, no extra keys.\n"
        'SCHEMA-HINT: { "facts": [ { "subject": str, "predicate": str, "object": str } ] }\n\n'
        "EXAMPLE:\n"
        'USER: Return EXACTLY this JSON: {"facts":[{"subject":"Ping","predicate":"says","object":"hello"}]}\n'
        'ASSISTANT: {"facts":[{"subject":"Ping","predicate":"says","object":"hello"}]}\n\n'
    )
    prompt = preface + f"Subject: {args.subject}\nReturn as many (subject, predicate, object) triples as you can."

    messages = [
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": prompt},
    ]

    print(f"provider={args.provider}\nmodel={args.model}\n")
    print("--- REQUEST PROMPT (first 800 chars) ---\n" + prompt[:800] + "\n----------------------------------------\n")

    try:
        raw = llm(messages, json_schema=ELICIT_SCHEMA_BASE)
    except Exception as e:
        print(f"[API ERROR] {e}")
        sys.exit(1)

    print("--- RAW RESPONSE ---")
    print(raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False))
    print("--------------------\n")

    # Try to show parsed nicely if it looks dict-y
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"[Parse error] {e}")
        print(raw)


if __name__ == "__main__":
    main()
