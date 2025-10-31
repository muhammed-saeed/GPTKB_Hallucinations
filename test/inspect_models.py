#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from typing import Dict, List, Tuple

# repo-local imports
from settings import (
    settings,
    ELICIT_SCHEMA_BASE,
    ELICIT_SCHEMA_CAL,
    NER_SCHEMA_BASE,
    NER_SCHEMA_CAL,
)
from prompter_parser import get_prompt_messages
from llm.factory import make_llm_from_config


# --------------------------- Pretty helpers ---------------------------

def _banner(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _show_messages(label: str, messages: List[Dict[str, str]]):
    print(f"\n--- {label} MESSAGES ---")
    for m in messages:
        role = (m.get("role") or "").upper()
        content = (m.get("content") or "").strip()
        preview = content if len(content) <= 1200 else (content[:1200] + "…")
        print(f"{role}: {preview}")
    print("-" * 80)


def _loose_json_try(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        return {}


def _salvage_if_raw(obj):
    """Many providers (e.g. replicate/anthropic) ignore schemas and we pipe their
    raw text in '_raw'. Try to parse it to a dict; otherwise return original."""
    if isinstance(obj, dict) and obj.get("_raw"):
        parsed = _loose_json_try(obj["_raw"])
        if isinstance(parsed, dict) and parsed:
            return parsed
    return obj


def _count_ner_items(d: dict) -> Tuple[int, str]:
    """Support both {'phrases':[...]} and {'entities':[...]}."""
    if not isinstance(d, dict):
        return 0, "not-a-dict"
    if "phrases" in d and isinstance(d["phrases"], list):
        return len(d["phrases"]), "phrases"
    if "entities" in d and isinstance(d["entities"], list):
        return len(d["entities"]), "entities"
    return 0, "none"


# --------------------------- Vars builders ---------------------------

def _build_vars_for_elicitation(domain: str, subject: str, root_subject: str | None) -> Dict[str, str]:
    return {
        "subject_name": subject,
        "root_subject": (root_subject or subject) if domain == "topic" else "",
    }


def _build_vars_for_ner(domain: str, subject: str, root_subject: str | None) -> Dict[str, str]:
    phrases = [
        subject,                      # likely relevant
        "Albert Einstein",            # NE
        "March 14, 1879",             # date (false)
        "https://example.com",        # URL (false)
        "physics",                    # literal/category (false)
        "Mona Lisa",                  # NE
        "Dragonfruit Phoenix Festival",  # nonsense (false)
        "New York City",              # NE
    ]
    return {
        "phrases_block": "\n".join(phrases),
        "root_subject": (root_subject or subject) if domain == "topic" else "",
    }


# --------------------------- CLI + Main ---------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Inspect prompts and (optionally) call models with one test subject."
    )
    ap.add_argument("--subject", default="Ada Lovelace", help="Test subject for elicitation.")
    ap.add_argument("--domain", choices=["general", "topic"], default="general")
    ap.add_argument("--elicitation-strategy", choices=["baseline", "icl", "dont_know", "calibrate"], default="baseline")
    ap.add_argument("--ner-strategy", choices=["baseline", "icl", "dont_know", "calibrate"], default="baseline")
    ap.add_argument("--root-subject", default="", help="Topic anchor when --domain topic.")
    ap.add_argument("--call", action="store_true", help="Actually call the models (uses your API keys).")
    ap.add_argument("--models", default="", help="Comma-separated model keys (as in settings.MODELS). Empty = all.")
    ap.add_argument("--max-models", type=int, default=0, help="Hard limit on number of models to run (0 = all).")
    ap.add_argument("--only", choices=["elicitation", "ner", "both"], default="both")
    ap.add_argument("--reasoning-effort", choices=["minimal","low","medium","high"], default=None)
    ap.add_argument("--verbosity", choices=["low","medium","high"], default=None)
    ap.add_argument("--report-json", default="", help="Optional path to write a JSON report.")
    args = ap.parse_args()

    # ---------- Render elicitation messages ----------
    el_vars = _build_vars_for_elicitation(args.domain, args.subject, args.root_subject or args.subject)
    try:
        el_messages = get_prompt_messages(
            args.elicitation_strategy, "elicitation", domain=args.domain, variables=el_vars
        )
    except Exception as e:
        _banner("ERROR rendering elicitation messages")
        print(e)
        sys.exit(1)

    _banner("Elicitation message preview (system & user)")
    _show_messages("ELICITATION", el_messages)
    assert el_messages and el_messages[0]["role"] == "system" and el_messages[1]["role"] == "user", \
        "Elicitation messages must begin with system then user."

    # ---------- Render NER messages ----------
    ner_vars = _build_vars_for_ner(args.domain, args.subject, args.root_subject or args.subject)
    try:
        ner_messages = get_prompt_messages(
            args.ner_strategy, "ner", domain=args.domain, variables=ner_vars
        )
    except Exception as e:
        _banner("ERROR rendering NER messages")
        print(e)
        sys.exit(1)

    _banner("NER message preview (system & user)")
    _show_messages("NER", ner_messages)
    assert ner_messages and ner_messages[0]["role"] == "system" and ner_messages[1]["role"] == "user", \
        "NER messages must begin with system then user."

    if not args.call:
        _banner("DRY RUN ✓  (No API calls were made)")
        print("Confirmed: both ELICITATION and NER have system+user messages rendered correctly.")
        return

    # ---------- Choose schemas ----------
    el_schema = ELICIT_SCHEMA_CAL if args.elicitation_strategy == "calibrate" else ELICIT_SCHEMA_BASE
    ner_schema = NER_SCHEMA_CAL if args.ner_strategy == "calibrate" else NER_SCHEMA_BASE

    # ---------- Select models ----------
    selected_keys = []
    if args.models.strip():
        wanted = {k.strip() for k in args.models.split(",") if k.strip()}
        for k in settings.MODELS.keys():
            if k in wanted:
                selected_keys.append(k)
    else:
        selected_keys = list(settings.MODELS.keys())

    if args.max_models and args.max_models > 0:
        selected_keys = selected_keys[: args.max_models]

    # ---------- Iterate models ----------
    _banner("Live calls (will use API keys if required)")
    report = {
        "subject": args.subject,
        "domain": args.domain,
        "elicitation_strategy": args.elicitation_strategy,
        "ner_strategy": args.ner_strategy,
        "models": []
    }

    for key in selected_keys:
        cfg = settings.MODELS[key].model_copy(deep=True)

        # Responses API extras (OpenAI GPT-5 family, etc.)
        if getattr(cfg, "use_responses_api", False):
            if cfg.extra_inputs is None:
                cfg.extra_inputs = {}
            cfg.extra_inputs.setdefault("reasoning", {})
            cfg.extra_inputs.setdefault("text", {})
            if args.reasoning_effort:
                cfg.extra_inputs["reasoning"]["effort"] = args.reasoning_effort
            if args.verbosity:
                cfg.extra_inputs["text"]["verbosity"] = args.verbosity

        print("\n" + "-" * 80)
        print(f"[{key}] provider={cfg.provider} model={cfg.model}")
        print("-" * 80)
        try:
            llm = make_llm_from_config(cfg)
        except Exception as e:
            print(f"[{key}] SKIP (client init failed): {e}")
            report["models"].append({
                "key": key, "provider": cfg.provider, "model": cfg.model,
                "init_error": str(e)
            })
            continue

        entry = {
            "key": key,
            "provider": cfg.provider,
            "model": cfg.model,
            "elicitation": {"status": "skipped", "facts": None, "error": None},
            "ner": {"status": "skipped", "items": None, "field": None, "error": None},
        }

        # --- Elicitation call ---
        if args.only in ("elicitation", "both"):
            try:
                print(f"[{key}] Calling ELICITATION …")
                el_out = llm(el_messages, json_schema=el_schema)
                el_out = _salvage_if_raw(el_out)
                if isinstance(el_out, dict) and "facts" in el_out and isinstance(el_out["facts"], list):
                    n_facts = len(el_out["facts"])
                    print(f"[{key}] elicitation: got {n_facts} facts")
                    entry["elicitation"] = {"status": "ok", "facts": n_facts, "error": None}
                else:
                    trunc = (json.dumps(el_out, ensure_ascii=False) if not isinstance(el_out, str) else el_out)[:800]
                    print(f"[{key}] elicitation raw/truncated: {trunc}")
                    entry["elicitation"] = {"status": "no_facts", "facts": 0, "error": None}
            except Exception as e:
                print(f"[{key}] elicitation FAILED: {e}")
                entry["elicitation"] = {"status": "error", "facts": 0, "error": str(e)}

        # --- NER call ---
        if args.only in ("ner", "both"):
            try:
                print(f"[{key}] Calling NER …")
                ner_out = llm(ner_messages, json_schema=ner_schema)
                ner_out = _salvage_if_raw(ner_out)
                n, field = _count_ner_items(ner_out)
                if n > 0:
                    print(f"[{key}] ner: got {n} items ({field})")
                    entry["ner"] = {"status": "ok", "items": n, "field": field, "error": None}
                else:
                    if isinstance(ner_out, dict) and "_raw" in ner_out and ner_out["_raw"]:
                        print(f"[{key}] ner _raw: {ner_out['_raw'][:800]}")
                    else:
                        trunc = (json.dumps(ner_out, ensure_ascii=False) if not isinstance(ner_out, str) else ner_out)[:800]
                        print(f"[{key}] ner raw/truncated: {trunc}")
                    entry["ner"] = {"status": "zero", "items": 0, "field": field, "error": None}
            except Exception as e:
                print(f"[{key}] ner FAILED: {e}")
                entry["ner"] = {"status": "error", "items": 0, "field": None, "error": str(e)}

        report["models"].append(entry)

    # ---------- Summaries ----------
    _banner("Summary")
    bad_elicitation = [m for m in report["models"] if m["elicitation"]["status"] in ("error", "no_facts")]
    bad_ner_err     = [m for m in report["models"] if m["ner"]["status"] == "error"]
    bad_ner_zero    = [m for m in report["models"] if m["ner"]["status"] == "zero"]

    def _names(rows): return ", ".join(f"{r['key']}" for r in rows) or "—"

    print(f"Elicitation errors/no_facts : {len(bad_elicitation)}  -> {_names(bad_elicitation)}")
    print(f"NER errors                  : {len(bad_ner_err)}      -> {_names(bad_ner_err)}")
    print(f"NER zero items              : {len(bad_ner_zero)}     -> {_names(bad_ner_zero)}")

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nWrote JSON report: {args.report_json}")


if __name__ == "__main__":
    main()
