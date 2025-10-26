#!/usr/bin/env python3
# tools/deepseek_check.py
from __future__ import annotations
import os, json, argparse, textwrap

from dotenv import load_dotenv

# Local imports
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llm.deepseek_client import DeepSeekLLM  # uses your fixed client

# --- simple schemas used by crawler ---
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

NER_SCHEMA_BASE = {
    "type": "object",
    "properties": {
        "phrases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "phrase": {"type": "string"},
                    "is_ne": {"type": "boolean"},
                },
                "required": ["phrase", "is_ne"],
            },
        }
    },
    "required": ["phrases"],
}


def build_elicitation_prompt(subject: str, max_facts_hint: int = 50) -> list[dict]:
    txt = f"""
You are a knowledge base construction expert.

Given a subject entity, return as many facts as possible as a list of (subject, predicate, object) triples.

Rules:
- For very famous subjects, try to reach {max_facts_hint}+ distinct facts.
- Each triple must be concise, factual, and in plain language.
- Use multiple triples rather than long objects with commas.
- Include at least one triple with predicate "instanceOf".
- If you do not know the subject, return an empty list.
- Output only JSON, no markdown or text commentary.

Output format:
{{
  "facts": [
    {{ "subject": "...", "predicate": "...", "object": "..." }}
  ]
}}

Subject: {subject}

Now respond with JSON only.
""".strip()

    return [
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": txt},
    ]


def build_ner_prompt(lines: list[str]) -> list[dict]:
    phrases_block = "\n".join(lines)
    txt = f"""
For each line, decide if it is a named entity you would expand in a knowledge graph.
Return JSON only, shaped as:
{{
  "phrases": [{{"phrase": str, "is_ne": bool}}]
}}

Lines:
{phrases_block}
""".strip()
    return [
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": txt},
    ]


def head(items, n=8):
    return items[:n] if isinstance(items, list) else items


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="DeepSeek sanity checker for elicitation/NER.")
    ap.add_argument("--model", default="deepseek-chat", help="DeepSeek model slug (e.g. deepseek-chat, deepseek-reasoner)")
    ap.add_argument("--mode", choices=["elicitation", "ner"], default="elicitation")
    ap.add_argument("--subject", default="Albert Einstein")
    ap.add_argument("--ner-lines", nargs="*", default=["Alan Turing", "New York City", "MIT", "research", "2024"])
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--show-raw", action="store_true", help="Print raw first 1200 chars returned by model.")
    args = ap.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        print("[ERROR] Set DEEPSEEK_API_KEY in your environment or .env file.")
        raise SystemExit(2)

    print("== deepseek_check ==\n")
    print(f"model={args.model}")
    print(f"temperature={args.temperature} top_p={args.top_p} max_tokens={args.max_tokens}")
    print(f"base_url={base_url}\n")

    client = DeepSeekLLM(model=args.model, api_key=api_key, base_url=base_url)

    if args.mode == "elicitation":
        messages = build_elicitation_prompt(args.subject)
        schema = ELICIT_SCHEMA_BASE
        print(f"Subject: {args.subject}\n")
    else:
        messages = build_ner_prompt(args.ner_lines)
        schema = NER_SCHEMA_BASE
        print(f"NER lines: {args.ner_lines}\n")

    out = client.generate(
        messages,
        json_schema=schema,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # Optionally print raw snippet if client put anything there
    if args.show_raw and isinstance(out, dict) and "_raw" in out:
        print("--- RAW (first 1200 chars) ---")
        print(str(out["_raw"])[:1200])
        print("------------------------------\n")

    # Print parsed/normalized view that crawler will receive
    print("== Parsed JSON ==")
    print(json.dumps(out, ensure_ascii=False, indent=2))

    # Quick validation / summary
    if args.mode == "elicitation":
        facts = out.get("facts") or []
        print(f"\nFacts count: {len(facts)}")
        print("Head:", json.dumps(head(facts, 8), ensure_ascii=False, indent=2))
    else:
        phrases = out.get("phrases") or []
        print(f"\nPhrases count: {len(phrases)}")
        print("Head:", json.dumps(head(phrases, 8), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
