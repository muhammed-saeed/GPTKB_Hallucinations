#!/usr/bin/env python3
# tools/deepseek_sanity.py
import os, json, argparse, textwrap
import requests
from dotenv import load_dotenv

def strip_fences(t: str) -> str:
    if not isinstance(t, str):
        return ""
    s = t.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl+1:].strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    return s

def parse_json_loose(text: str):
    if not isinstance(text, str):
        return None
    t = strip_fences(text)
    # 1) direct
    try:
        return json.loads(t)
    except Exception:
        pass
    # 2) first balanced {...}
    s = t.find("{")
    if s == -1:
        return None
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
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = t[s:i+1]
                    try:
                        return json.loads(cand)
                    except Exception:
                        break
    return None

def build_elicitation_prompt(subject: str) -> str:
    return textwrap.dedent(f"""
    You are a knowledge base construction expert.
    Given a subject entity, return as many (subject, predicate, object) triples as possible.
    Aim for 60+ concise, distinct facts when possible.

    Output JSON ONLY (no markdown), exactly like:
    {{ "facts": [ {{ "subject":"...", "predicate":"...", "object":"..." }} ] }}

    Subject: {subject}
    Now respond with JSON only.
    """).strip()

def call_deepseek(subject: str, model: str, base_url: str, temperature: float, top_p: float, max_tokens: int, timeout: float, debug: bool):
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}",
        "Content-Type": "application/json",
    }
    messages = [
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": build_elicitation_prompt(subject)},
    ]
    body = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    if debug:
        print("== Request ==")
        print("POST", url)
        print("model:", model)
        print("body (truncated):", json.dumps({**body, "messages": "[...]"}, ensure_ascii=False)[:300], "\n")

    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    if debug:
        print("HTTP", r.status_code)
    r.raise_for_status()

    try:
        data = r.json()
        content = (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        content = r.text.strip()

    if debug:
        print("\n--- CONTENT (first 1200) ---")
        print(content[:1200])
        print("----------------------------\n")

    obj = parse_json_loose(content) or {}
    facts = obj.get("facts") if isinstance(obj, dict) else None
    count = len(facts) if isinstance(facts, list) else 0
    return obj, count

def main():
    load_dotenv()  # <-- reads .env automatically
    ap = argparse.ArgumentParser(description="DeepSeek sanity test (elicitation).")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--base-url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--save-json", default=None, help="Optional path to save parsed JSON.")
    args = ap.parse_args()

    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY not found. Put it in your .env and re-run.")

    obj, count = call_deepseek(
        subject=args.subject,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        debug=args.debug,
    )

    print("== Parsed JSON ==")
    print(json.dumps(obj, ensure_ascii=False, indent=2))
    print(f"\nFacts count: {count}")

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        print(f"Saved to: {args.save_json}")

if __name__ == "__main__":
    main()
