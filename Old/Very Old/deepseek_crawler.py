#!/usr/bin/env python3
# deepseek_crawler.py
#
# Minimal DeepSeek-only crawler using HTTP (requests) + response_format=json_object.
# - No SQLite; simple in-memory queue with JSONL/JSON outputs.
# - No regex-heavy parsing; uses a lightweight "strip fences + first balanced {...}" JSON extractor.
# - Expansion heuristic: take object strings of accepted triples (1–6 tokens) as next subjects.

from __future__ import annotations
import os, json, time, argparse, textwrap
from typing import List, Dict, Tuple, Optional, Set
import requests
from dotenv import load_dotenv

# ------------------------- JSON helpers -------------------------

def parse_json_loose(text: str) -> Optional[dict]:
    """Lenient JSON extractor: strip fences, try direct json, then first balanced {...}."""
    if not isinstance(text, str):
        return None
    t = text.strip()
    # strip ``` or ```json fences if present (fast, no regex)
    if t.startswith("```"):
        nl = t.find("\n")
        if nl != -1:
            t = t[nl+1:].strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    # direct parse
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

# ------------------------- I/O helpers -------------------------

def ensure_dir(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d

def append_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ------------------------- Prompt builders -------------------------

def build_elicitation_prompt(subject: str, max_facts_hint: int) -> str:
    # Minimal, proven-effective prompt (same spirit as your http_check).
    return textwrap.dedent(f"""
    You are a knowledge base construction expert.

    Given a subject entity, return as many (subject, predicate, object) triples as possible.
    Aim for {max_facts_hint}+ distinct, concise facts when possible.

    Output JSON ONLY (no markdown), exactly like:
    {{ "facts": [ {{ "subject":"...", "predicate":"...", "object":"..." }} ] }}

    Subject: {subject}
    Now respond with JSON only.
    """).strip()

# ------------------------- DeepSeek call -------------------------

def deepseek_elicit(
    api_key: str,
    base_url: str,
    model: str,
    subject: str,
    *,
    temperature: float = 0.2,
    top_p: float = 1.0,
    max_tokens: int = 2048,
    timeout: float = 60.0,
    max_facts_hint: int = 60,
    debug: bool = False,
) -> Dict[str, List[dict]]:
    """
    Returns {"facts":[...]} or {"facts": []} on failure — never raises for model issues.
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    user_prompt = build_elicitation_prompt(subject, max_facts_hint)

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": user_prompt},
        ],
        # DeepSeek supports this and it often yields clean JSON.
        "response_format": {"type": "json_object"},
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    if debug:
        print("\n--- ELICITATION PROMPT ---\n", user_prompt, "\n--------------------------\n", flush=True)

    try:
        r = requests.post(url, headers=headers, json=body, timeout=timeout)
    except Exception as e:
        if debug:
            print(f"[deepseek] HTTP error: {e}")
        return {"facts": []}

    if r.status_code != 200:
        if debug:
            print(f"[deepseek] HTTP {r.status_code}: {r.text[:400]}")
        return {"facts": []}

    try:
        data = r.json()
        content = (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        # fallback to raw text if JSON parse fails
        content = r.text.strip()

    if debug and content:
        print("[deepseek] content preview:", content[:800], flush=True)

    obj = parse_json_loose(content) or {}
    facts = obj.get("facts")
    if not isinstance(facts, list):
        facts = []
    # keep only valid triples (strings)
    cleaned = []
    for t in facts:
        if not isinstance(t, dict):
            continue
        s, p, o = t.get("subject"), t.get("predicate"), t.get("object")
        if isinstance(s, str) and isinstance(p, str) and isinstance(o, str):
            cleaned.append({"subject": s, "predicate": p, "object": o})
    return {"facts": cleaned}

# ------------------------- Expansion heuristic -------------------------

def extract_next_subjects(facts: List[dict]) -> List[str]:
    """
    Very light heuristic: use object strings (1–6 tokens) as candidate subjects.
    Dedup and keep short phrases to reduce drift.
    """
    seen: Set[str] = set()
    out: List[str] = []
    for t in facts:
        o = t.get("object")
        if not isinstance(o, str):
            continue
        tok = o.strip().split()
        if 1 <= len(tok) <= 6:
            key = " ".join(tok)
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out

# ------------------------- Main crawl loop -------------------------

def crawl_deepseek(
    seed: str,
    out_dir: str,
    *,
    model: str = "deepseek-chat",
    max_depth: int = 2,
    max_subjects: int = 0,         # 0 = unlimited
    ner_batch_size: int = 32,      # ignored: no NER phase here; kept for parity
    temperature: float = 0.2,
    top_p: float = 1.0,
    max_tokens: int = 2048,
    timeout: float = 60.0,
    max_facts_hint: int = 60,
    debug: bool = False,
) -> None:
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY in your .env")

    ensure_dir(out_dir)
    paths = {
        "facts_jsonl": os.path.join(out_dir, "facts.jsonl"),
        "facts_json": os.path.join(out_dir, "facts.json"),
        "queue_jsonl": os.path.join(out_dir, "queue.jsonl"),
        "queue_json": os.path.join(out_dir, "queue.json"),
        "errors_log": os.path.join(out_dir, "errors.log"),
    }

    print(f"[deepseek_crawler] output_dir: {out_dir}")
    # in-memory queue: list of (subject, hop, status)
    queue: List[Tuple[str, int, str]] = []
    queue.append((seed, 0, "pending"))
    append_jsonl(paths["queue_jsonl"], {"subject": seed, "hop": 0})

    visited: Set[str] = set()
    accepted_facts: List[dict] = []

    processed_count = 0
    start = time.time()

    while queue:
        # cap if requested
        if max_subjects and processed_count >= max_subjects:
            if debug:
                print(f"[deepseek_crawler] reached --max-subjects={max_subjects}; stopping.")
            break

        # fetch next pending
        idx = next((i for i, (_, _, st) in enumerate(queue) if st == "pending"), None)
        if idx is None:
            if debug:
                print("[deepseek_crawler] queue drained.")
            break

        subject, hop, _ = queue[idx]
        queue[idx] = (subject, hop, "working")

        if subject in visited:
            queue[idx] = (subject, hop, "done")
            continue

        if hop > max_depth:
            queue[idx] = (subject, hop, "done")
            continue

        if debug:
            print(f"[deepseek_crawler] eliciting '{subject}' (hop={hop})")

        try:
            out = deepseek_elicit(
                api_key, base_url, model, subject,
                temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                timeout=timeout, max_facts_hint=max_facts_hint, debug=debug
            )
            facts = out.get("facts", [])

            # write stream + accumulate
            for t in facts:
                record = dict(t)
                record.update({"hop": hop, "model": model, "strategy": "baseline"})
                append_jsonl(paths["facts_jsonl"], record)
                accepted_facts.append(record)

            # expansion
            next_subjects = extract_next_subjects(facts) if hop < max_depth else []
            for ns in next_subjects:
                if ns not in visited:
                    queue.append((ns, hop + 1, "pending"))
                    append_jsonl(paths["queue_jsonl"], {"subject": ns, "hop": hop + 1})

            queue[idx] = (subject, hop, "done")
            visited.add(subject)
            processed_count += 1

        except KeyboardInterrupt:
            print("\n[deepseek_crawler] Interrupted.")
            break
        except Exception as e:
            # log error and mark pending again to retry later (or ignore)
            with open(paths["errors_log"], "a", encoding="utf-8") as ef:
                ef.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] subject={subject}\n{repr(e)}\n")
            # simple: mark done to avoid loops
            queue[idx] = (subject, hop, "done")

    # snapshots
    save_json(paths["facts_json"], {"accepted": accepted_facts})
    save_json(paths["queue_json"], [
        {"subject": s, "hop": h, "status": st} for (s, h, st) in queue
    ])

    dur = time.time() - start
    print(f"[deepseek_crawler] finished in {dur:.1f}s → outputs in {out_dir}")
    print(f"[deepseek_crawler] facts.jsonl : {paths['facts_jsonl']}")
    print(f"[deepseek_crawler] facts.json  : {paths['facts_json']}")
    print(f"[deepseek_crawler] queue.jsonl : {paths['queue_jsonl']}")
    print(f"[deepseek_crawler] queue.json  : {paths['queue_json']}")
    print(f"[deepseek_crawler] errors.log  : {paths['errors_log']}")

# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser(description="DeepSeek-only HTTP crawler (no DB, minimal regex).")
    ap.add_argument("--seed", required=True)
    ap.add_argument("--output-dir", required=True)

    # Crawl knobs
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--max-subjects", type=int, default=0)  # 0 = unlimited
    ap.add_argument("--max-facts-hint", type=int, default=60)

    # Sampler + HTTP
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=float, default=60.0)

    # Debug
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    crawl_deepseek(
        seed=args.seed,
        out_dir=ensure_dir(args.output_dir),
        model=args.model,
        max_depth=args.max_depth,
        max_subjects=args.max_subjects,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_facts_hint=args.max_facts_hint,
        debug=args.debug,
    )

if __name__ == "__main__":
    main()
