# tools/replicate_probe.py
import os, sys, json
from dotenv import load_dotenv

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llm.replicate_client import ReplicateLLM

load_dotenv()

CANDIDATES = [
    # Gemma
    "google-deepmind/gemma-2-9b-it",
    "google/gemma-2-9b-it",
    # Mistral
    "mistralai/mistral-7b-instruct-v0.2",
    "mistralai/mixtral-8x7b-instruct",
    # Llama 3
    "meta/llama-3-8b-instruct",
    "meta/meta-llama-3.1-8b-instruct",
    # Other
    "tiiuae/falcon-7b-instruct",
]

PROMPT = 'Return EXACTLY this JSON:\n{"facts":[{"subject":"Ping","predicate":"says","object":"ok"}]}'
SCHEMA = {
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

def main():
    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        print("[ERROR] Missing REPLICATE_API_TOKEN in .env", file=sys.stderr)
        sys.exit(2)

    results = []
    for slug in CANDIDATES:
        try:
            llm = ReplicateLLM(slug, api_token=token)
            out = llm.generate(
                [{"role":"user","content":PROMPT}],
                json_schema=SCHEMA,
                temperature=0.0,
                top_p=1.0,
                max_tokens=128,
                prefer="prompt",
            )
            ok = isinstance(out, dict) and isinstance(out.get("facts"), list)
            results.append({"model": slug, "ok": ok, "sample": out if ok else str(out)})
        except Exception as e:
            results.append({"model": slug, "ok": False, "error": str(e)})

    print("\n== replicate_probe ==\n")
    for r in results:
        if r["ok"]:
            print(f"[OK]  {r['model']}")
        else:
            print(f"[BAD] {r['model']}  — {r.get('error','no details')}")

if __name__ == "__main__":
    main()
