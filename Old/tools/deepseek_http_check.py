#!/usr/bin/env python3
# tools/deepseek_http_check.py
import os, json, argparse, textwrap
import requests
from dotenv import load_dotenv

def parse_json_loose(text: str):
    """Lenient JSON extractor: strip fences, then first balanced {...}."""
    if not isinstance(text, str):
        return None
    t = text.strip()
    # strip ``` or ```json fences if present
    if t.startswith("```"):
        # remove opening fence line
        nl = t.find("\n")
        if nl != -1:
            t = t[nl+1:].strip()
        # remove trailing fence if present
        if t.endswith("```"):
            t = t[:-3].strip()
    # direct parse first
    try:
        return json.loads(t)
    except Exception:
        pass
    # first balanced {...}
    s = t.find("{")
    if s == -1:
        return None
    depth = 0
    for i in range(s, len(t)):
        ch = t[i]
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                cand = t[s:i+1]
                try:
                    return json.loads(cand)
                except Exception:
                    break
    return None

def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="Direct DeepSeek /v1/chat/completions probe.")
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY in .env")

    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Minimal prompt (no schema) – DeepSeek tends to obey when asked plainly.
    user_prompt = textwrap.dedent(f"""
    You are a knowledge base construction expert.
    Given a subject entity, return as many (subject, predicate, object) triples as possible.

    Output JSON ONLY (no markdown), like:
    {{ "facts": [ {{ "subject":"...", "predicate":"...", "object":"..." }} ] }}

    Subject: {args.subject}
    Now respond with JSON only.
    """).strip()

    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": user_prompt},
        ],
        # NOTE: DeepSeek supports response_format=json_object; this often helps.
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "top_p": 1.0,
        "max_tokens": 2048,
    }

    print("\n== deepseek_http_check ==\n")
    print(f"POST {url}")
    print(f"model={args.model} timeout={args.timeout}s\n")

    r = requests.post(url, headers=headers, json=body, timeout=args.timeout)
    print(f"HTTP {r.status_code}\n")

    try:
        data = r.json()
    except Exception:
        print("--- RAW TEXT ---")
        print(r.text[:2000])
        print("----------------")
        raise

    # Show a preview so you can see exactly what the model said
    content = ""
    try:
        content = (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        pass

    if content:
        print("\n--- CONTENT (first 1200 chars) ---")
        print(content[:1200])
        print("----------------------------------\n")
    else:
        print("\n(No message content in response)\n")

    # Robust parse
    obj = parse_json_loose(content) or {}
    print("== Parsed JSON ==")
    print(json.dumps(obj, ensure_ascii=False, indent=2))
    facts = obj.get("facts") if isinstance(obj, dict) else None
    print(f"\nfacts count: {len(facts) if isinstance(facts, list) else 0}")

if __name__ == "__main__":
    main()
