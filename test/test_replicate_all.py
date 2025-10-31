# test_replicate_all.py
from __future__ import annotations
import argparse
import json

from settings import settings, ELICIT_SCHEMA_CAL, NER_SCHEMA_CAL
from llm.replicate_client import ReplicateLLM


def _elicitation_prompt(seed: str) -> list[dict]:
    user = (
        "You are a knowledge base construction expert.\n"
        f"Subject: {seed}\n\n"
        "Return JSON ONLY with key 'facts' = array of {subject,predicate,object,confidence}.\n"
        "confidence ∈ [0,1]. Include at least one 'instanceOf'. "
        "If unsure, return {\"facts\":[]}."
    )
    return [{"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": user}]


def _ner_prompt() -> list[dict]:
    phrases = ["Jon Snow", "Westeros", "King's Landing", "Emilia Clarke"]
    block = "\n".join(phrases)
    user = (
        "You are an expert in NER. For each input line, return {phrase,is_ne,confidence}.\n"
        "Return JSON only with key 'phrases'.\n\n"
        f"Phrases:\n{block}"
    )
    return [{"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": user}]


def run_one(key: str, seed: str, verbose: bool, stream: bool):
    if key not in settings.MODELS:
        raise SystemExit(f"Unknown model key: {key}")
    cfg = settings.MODELS[key]
    if cfg.provider != "replicate":
        print(f"[skip] {key}: provider={cfg.provider} (this tester is for Replicate)")
        return

    print(f"\n=== {key} :: {cfg.model} ===")
    client = ReplicateLLM(cfg.model)

    # ping
    pong = client.ping()
    print(f"[ping] ok={'message' in pong} text={json.dumps(pong, ensure_ascii=False)}")

    # elicitation
    el_messages = _elicitation_prompt(seed)
    if stream:
        res = None
        for out in client.stream_json(
            el_messages, json_schema=ELICIT_SCHEMA_CAL,
            temperature=cfg.temperature, top_p=cfg.top_p,
            top_k=getattr(cfg, "top_k", None), max_tokens=cfg.max_tokens,
            extra=cfg.extra_inputs or {}
        ):
            res = out
        facts = (res or {}).get("facts", [])
    else:
        res = client.generate(
            el_messages, json_schema=ELICIT_SCHEMA_CAL,
            temperature=cfg.temperature, top_p=cfg.top_p,
            top_k=getattr(cfg, "top_k", None), max_tokens=cfg.max_tokens,
            extra=cfg.extra_inputs or {}
        )
        facts = res.get("facts", [])

    miss = sum(1 for f in facts if "confidence" not in f)
    sample = facts[:3]
    print(f"[elicitation] facts={len(facts)} missing_conf={miss} sample={json.dumps(sample, ensure_ascii=False)}")
    if verbose:
        print(json.dumps({"facts": facts[:18]}, ensure_ascii=False, indent=2))

    # NER
    ner_messages = _ner_prompt()
    if stream:
        res = None
        for out in client.stream_json(
            ner_messages, json_schema=NER_SCHEMA_CAL,
            temperature=cfg.temperature, top_p=cfg.top_p,
            top_k=getattr(cfg, "top_k", None), max_tokens=cfg.max_tokens,
            extra=cfg.extra_inputs or {}
        ):
            res = out
        phrases = (res or {}).get("phrases", [])
    else:
        res = client.generate(
            ner_messages, json_schema=NER_SCHEMA_CAL,
            temperature=cfg.temperature, top_p=cfg.top_p,
            top_k=getattr(cfg, "top_k", None), max_tokens=cfg.max_tokens,
            extra=cfg.extra_inputs or {}
        )
        phrases = res.get("phrases", [])

    miss = sum(1 for p in phrases if "confidence" not in p)
    sample = phrases[:4]
    print(f"[ner] phrases={len(phrases)} missing_conf={miss} sample={json.dumps(sample, ensure_ascii=False)}")
    if verbose:
        print(json.dumps({"phrases": phrases[:12]}, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="Game of Thrones")
    ap.add_argument("--models", nargs="*", help="space-delimited model keys from settings")
    ap.add_argument("--all", action="store_true", help="run all replicate models in settings")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--stream", action="store_true", help="use streaming JSON path")
    args = ap.parse_args()

    if args.all and args.models:
        raise SystemExit("Use either --all or --models, not both.")
    if args.all:
        keys = [k for k, v in settings.MODELS.items() if v.provider == "replicate"]
    else:
        keys = args.models or ["claude37s"]

    print(f"[info] testing models: {keys}")
    for key in keys:
        try:
            run_one(key, args.seed, args.verbose, args.stream)
        except Exception as e:
            print(f"[{key}] ERROR: {e}")

if __name__ == "__main__":
    main()
