# testing_tools/test_replicate_stream.py
from __future__ import annotations
import argparse
import json
from typing import List, Dict

from settings import settings, ELICIT_SCHEMA_CAL, NER_SCHEMA_CAL
from llm.replicate_client import ReplicateLLM

def _elicitation_prompt(seed: str) -> List[Dict[str, str]]:
    user = (
        "You are a knowledge base construction expert.\n"
        f"Subject: {seed}\n\n"
        "Return JSON ONLY with key 'facts' = array of {subject,predicate,object,confidence}.\n"
        "confidence ∈ [0,1]. Include at least one 'instanceOf'. "
        "If unsure, return {\"facts\":[]}."
    )
    return [
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": user},
    ]

def _ner_prompt() -> List[Dict[str, str]]:
    phrases = ["Jon Snow", "Westeros", "King's Landing", "Emilia Clarke"]
    user = (
        "You are an expert in NER. For each input line, return {phrase,is_ne,confidence}.\n"
        "Return JSON only with key 'phrases'.\n\n"
        "Phrases:\n" + "\n".join(phrases)
    )
    return [
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": user},
    ]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", default="Game of Thrones")
    ap.add_argument("--task", choices=["elicitation", "ner"], default="elicitation")
    ap.add_argument("--text", action="store_true")
    args = ap.parse_args()

    cfg = settings.MODELS[args.model]
    if cfg.provider != "replicate":
        raise SystemExit("Only provider=replicate supported here.")
    client = ReplicateLLM(cfg.model)

    if args.task == "elicitation":
        messages = _elicitation_prompt(args.seed)
        schema = ELICIT_SCHEMA_CAL
    else:
        messages = _ner_prompt()
        schema = NER_SCHEMA_CAL

    if args.text:
        print("[stream_text] BEGIN")
        for chunk in client.stream_text(
            messages,
            temperature=cfg.temperature, top_p=cfg.top_p,
            top_k=getattr(cfg, "top_k", None), max_tokens=cfg.max_tokens,
            extra=cfg.extra_inputs or {},
        ):
            print(chunk, end="", flush=True)
        print("\n[stream_text] END")
    else:
        print("[stream_json] BEGIN")
        for final in client.stream_json(
            messages,
            json_schema=schema,
            temperature=cfg.temperature, top_p=cfg.top_p,
            top_k=getattr(cfg, "top_k", None), max_tokens=cfg.max_tokens,
            extra=cfg.extra_inputs or {},
        ):
            if "facts" in final:
                miss = sum(1 for f in final["facts"] if "confidence" not in f)
                print(f"[elicitation] facts={len(final['facts'])} missing_conf={miss}")
                print(json.dumps({"facts": final["facts"][:10]}, ensure_ascii=False, indent=2))
            if "phrases" in final:
                miss = sum(1 for p in final["phrases"] if "confidence" not in p)
                print(f"[ner] phrases={len(final['phrases'])} missing_conf={miss}")
                print(json.dumps({"phrases": final["phrases"][:10]}, ensure_ascii=False, indent=2))
        print("[stream_json] END")

if __name__ == "__main__":
    main()
