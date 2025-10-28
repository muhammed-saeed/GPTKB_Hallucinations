# crawler_simple.py
from __future__ import annotations
import argparse
import datetime
import json
import os
import sqlite3
import time
import traceback
from typing import Dict, List, Tuple

from dotenv import load_dotenv

from settings import (
    settings,
    ELICIT_SCHEMA_BASE,
    ELICIT_SCHEMA_CAL,
    NER_SCHEMA_BASE,
    NER_SCHEMA_CAL,
)
from prompter_parser import get_prompt_template
from llm.factory import make_llm_from_config
from db_models import (
    open_queue_db,
    open_facts_db,
    enqueue_subjects,
    write_triples_accepted,
    write_triples_sink,
    queue_has_rows,
    reset_working_to_pending,
)

load_dotenv()


# ===================== Prompt rendering =====================

def _dbg(msg: str):
    print(msg, flush=True)

def _should_debug_prompts() -> bool:
    return os.getenv("DEBUG_PROMPTS", "") == "1"

def _render_elicitation(strategy: str, subject_name: str, max_facts_hint: str,
                        domain: str = "general", topic: str | None = None) -> str:
    tpl = get_prompt_template(strategy, "elicitation", domain=domain, topic=topic)
    txt = tpl.render(
        subject_name=subject_name,
        subject=subject_name,       # tolerate templates that use {{ subject }}
        seed=subject_name,          # tolerate older templates that used {{ seed }}
        max_facts_hint=max_facts_hint,
        hint=max_facts_hint,        # tolerate {{ hint }}
    )
    return txt

def _render_ner(strategy: str, lines: List[str],
                domain: str = "general", topic: str | None = None) -> str:
    tpl = get_prompt_template(strategy, "ner", domain=domain, topic=topic)
    txt = tpl.render(
        lines="\n".join(lines),
        phrases="\n".join(lines),        # tolerate {{ phrases }}
        phrases_block="\n".join(lines),  # tolerate {{ phrases_block }}
    )
    return txt


# ===================== IO helpers =====================

def _ensure_output_dir(base_dir: str | None) -> str:
    out = base_dir or os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    return out

def _build_paths(out_dir: str) -> dict:
    return {
        "queue_sqlite": os.path.join(out_dir, "queue.sqlite"),
        "facts_sqlite": os.path.join(out_dir, "facts.sqlite"),
        "queue_jsonl": os.path.join(out_dir, "queue.jsonl"),
        "facts_jsonl": os.path.join(out_dir, "facts.jsonl"),
        "queue_json": os.path.join(out_dir, "queue.json"),
        "facts_json": os.path.join(out_dir, "facts.json"),
        "errors_log": os.path.join(out_dir, "errors.log"),
        "ner_jsonl": os.path.join(out_dir, "ner_decisions.jsonl"),
    }

def _append_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ===================== DB helpers =====================

def _fetch_one_pending(conn: sqlite3.Connection, max_depth: int) -> Tuple[str, int] | None:
    cur = conn.cursor()
    cur.execute("SELECT subject, hop FROM queue WHERE status='pending' AND hop<=? LIMIT 1", (max_depth,))
    row = cur.fetchone()
    if not row:
        return None
    s, h = row
    cur.execute("UPDATE queue SET status='working' WHERE subject=?", (s,))
    conn.commit()
    return s, h

def _mark_done(conn: sqlite3.Connection, subject: str):
    conn.execute("UPDATE queue SET status='done' WHERE subject=?", (subject,))
    conn.commit()

def _counts(conn: sqlite3.Connection, max_depth: int):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(1) FROM queue WHERE status='done' AND hop<=?", (max_depth,))
    done = cur.fetchone()[0]
    cur.execute("SELECT COUNT(1) FROM queue WHERE status='working' AND hop<=?", (max_depth,))
    working = cur.fetchone()[0]
    cur.execute("SELECT COUNT(1) FROM queue WHERE status='pending' AND hop<=?", (max_depth,))
    pending = cur.fetchone()[0]
    return done, working, pending, done + working + pending


# ===================== Normalizers =====================

def _parse_obj(maybe_json) -> dict:
    if isinstance(maybe_json, dict):
        return maybe_json
    if isinstance(maybe_json, str):
        try:
            return json.loads(maybe_json)
        except Exception:
            return {}
    return {}

# --- replace your _normalize_elicitation_output with this version ---
def _normalize_elicitation_output(out) -> Dict[str, list]:
    """
    Accepts dict or JSON str.
    Supports:
      - { "facts": [ ... ] }
      - { "triples": [ ... ] }
      - { "text": "<raw string possibly with ```json fences>" }
      - "<raw JSON string>"
    Returns: { "facts": [ {subject,predicate,object,confidence?}, ... ] }
    """
    def _strip_fences(t: str) -> str:
        t = t.strip()
        if t.startswith("```"):
            # remove ```lang header if present
            t = t.strip("`")
            nl = t.find("\n")
            if nl != -1:
                t = t[nl+1:].strip()
        return t

    # 1) dict straight-through
    if isinstance(out, dict):
        # Replicate wrapper may return {"text": "..."} when not enforcing schema
        if "text" in out and isinstance(out["text"], str):
            raw = _strip_fences(out["text"])
            try:
                obj = json.loads(raw)
            except Exception:
                return {"facts": []}
        else:
            obj = out
    # 2) string -> parse
    elif isinstance(out, str):
        raw = _strip_fences(out)
        try:
            obj = json.loads(raw)
        except Exception:
            return {"facts": []}
    else:
        return {"facts": []}

    # normalize shapes
    facts = obj.get("facts")
    if isinstance(facts, list):
        return {"facts": [t for t in facts if isinstance(t, dict)]}

    triples = obj.get("triples")
    if isinstance(triples, list):
        return {"facts": [t for t in triples if isinstance(t, dict)]}

    return {"facts": []}


def _normalize_ner_output(out) -> Dict[str, list]:
    """
    Accepts dict or JSON str.
    Supports:
      - { "phrases": [ { "phrase": str, "is_ne": bool, ... } ] }
      - { "entities": [ { "name": str, "type": "NE|Literal|Noise", "keep": bool, ... } ] }
    Returns: { "phrases": [ { "phrase": str, "is_ne": bool } ] }
    """
    obj = _parse_obj(out)
    if isinstance(obj.get("phrases"), list):
        got = []
        for ph in obj["phrases"]:
            phrase = ph.get("phrase")
            is_ne = bool(ph.get("is_ne"))
            if isinstance(phrase, str):
                got.append({"phrase": phrase, "is_ne": is_ne})
        return {"phrases": got}

    ents = obj.get("entities")
    if isinstance(ents, list):
        mapped = []
        for e in ents:
            name = e.get("name") or e.get("phrase")
            etype = (e.get("type") or "").strip().lower()
            keep = e.get("keep")
            is_ne = (etype == "ne") or (keep is True)
            if isinstance(name, str):
                mapped.append({"phrase": name, "is_ne": bool(is_ne)})
        return {"phrases": mapped}

    return {"phrases": []}


# ===================== Core helpers =====================

def _route_facts(args, facts: List[dict], hop: int, model_name: str):
    acc, rej, objs = [], [], []
    for f in facts:
        s, p, o = f.get("subject"), f.get("predicate"), f.get("object")
        if not (isinstance(s, str) and isinstance(p, str) and isinstance(o, str)):
            continue
        conf = f.get("confidence")
        acc.append((s, p, o, hop, model_name, args.elicitation_strategy,
                    conf if isinstance(conf, (int, float)) else None))
        objs.append(o)
    return acc, rej, objs

def _filter_ner_candidates(objs: List[str]) -> List[str]:
    return sorted({o for o in objs if isinstance(o, str) and 1 <= len(o.split()) <= 6})

def _enqueue_next(qdb, paths, phrases: List[str], hop: int, max_depth: int):
    if not phrases:
        return
    next_hop = hop + 1
    if next_hop > max_depth:
        return
    enqueue_subjects(qdb, ((s, next_hop) for s in phrases))
    for s in phrases:
        _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": next_hop})


# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser(description="Minimal, synchronous crawler (prompt+API sanity).")
    ap.add_argument("--seed", required=True)
    ap.add_argument("--output-dir", default=None)

    # Strategies / domain
    ap.add_argument("--elicitation-strategy", default="baseline", choices=["baseline","icl","dont_know","calibrate"])
    ap.add_argument("--ner-strategy", default="baseline", choices=["baseline","icl","dont_know","calibrate"])
    ap.add_argument("--domain", default="general", choices=["general","topic"])
    ap.add_argument("--topic", default=None)

    # Depth / batching
    ap.add_argument("--max-depth", type=int, default=settings.MAX_DEPTH)
    ap.add_argument("--ner-batch-size", type=int, default=settings.NER_BATCH_SIZE)
    ap.add_argument("--max-facts-hint", default=str(settings.MAX_FACTS_HINT))
    ap.add_argument("--conf-threshold", type=float, default=0.7)

    # Models
    ap.add_argument("--elicit-model-key", default=settings.ELICIT_MODEL_KEY)
    ap.add_argument("--ner-model-key", default=settings.NER_MODEL_KEY)

    # Sampler knobs (for non-Responses models)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)

    # Hard cap on number of elicited subjects
    ap.add_argument("--max-subjects", type=int, default=0,
                    help="Stop after eliciting this many subjects (0 = unlimited).")

    # OpenAI Responses API extras
    ap.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        default=None,
        help="Only for OpenAI Responses API models (e.g., gpt-5-*). Overrides settings.extra_inputs.reasoning.effort."
    )
    ap.add_argument(
        "--verbosity",
        choices=["low", "medium", "high"],
        default=None,
        help="Only for OpenAI Responses API models (e.g., gpt-5-*). Overrides settings.extra_inputs.text.verbosity."
    )

    # Resume
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reset-working", action="store_true")

    # Debug
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    if args.domain == "topic" and not args.topic:
        raise SystemExit("When --domain topic, you must also pass --topic (e.g., --topic entertainment).")

    out_dir = _ensure_output_dir(args.output_dir)
    paths = _build_paths(out_dir)
    _dbg(f"[simple] output_dir: {out_dir}")

    # DBs
    qdb = open_queue_db(paths["queue_sqlite"])
    fdb = open_facts_db(paths["facts_sqlite"])

    # Seed or resume
    if args.resume and queue_has_rows(qdb):
        if args.reset_working:
            n = reset_working_to_pending(qdb)
            _dbg(f"[simple] resume: reset {n} 'working' → 'pending'")
        d0, w0, p0, t0 = _counts(qdb, args.max_depth)
        _dbg(f"[simple] resume: queue found: done={d0} working={w0} pending={p0} total={t0}")
    else:
        enqueue_subjects(qdb, [(args.seed, 0)])
        _dbg(f"[simple] seeded: {args.seed}")

    # Build LLMs
    el_cfg = settings.MODELS[args.elicit_model_key].model_copy(deep=True)
    ner_cfg = settings.MODELS[args.ner_model_key].model_copy(deep=True)

    for cfg in (el_cfg, ner_cfg):
        if getattr(cfg, "use_responses_api", False):
            # Responses API models ignore sampler knobs; ensure extra_inputs baseline exists
            cfg.temperature = None
            cfg.top_p = None
            cfg.top_k = None
            if cfg.extra_inputs is None:
                cfg.extra_inputs = {}
            if "reasoning" not in cfg.extra_inputs or not isinstance(cfg.extra_inputs["reasoning"], dict):
                cfg.extra_inputs["reasoning"] = {}
            if "text" not in cfg.extra_inputs or not isinstance(cfg.extra_inputs["text"], dict):
                cfg.extra_inputs["text"] = {}
            if args.reasoning_effort:
                cfg.extra_inputs["reasoning"]["effort"] = args.reasoning_effort
            if args.verbosity:
                cfg.extra_inputs["text"]["verbosity"] = args.verbosity
        else:
            # Non-Responses models: allow sampler overrides
            if args.temperature is not None: cfg.temperature = args.temperature
            if args.top_p is not None: cfg.top_p = args.top_p
            if args.top_k is not None: cfg.top_k = args.top_k

        if args.max_tokens is not None:
            cfg.max_tokens = args.max_tokens

    el_llm = make_llm_from_config(el_cfg)
    ner_llm = make_llm_from_config(ner_cfg)

    start = time.time()
    subjects_elicited_total = 0  # hard cap counter

    # === main loop ===
    while True:
        # Hard cap check before fetching more work
        if args.max_subjects and subjects_elicited_total >= args.max_subjects:
            _dbg(f"[simple] max-subjects reached ({subjects_elicited_total}); stopping.")
            break

        nxt = _fetch_one_pending(qdb, args.max_depth)
        if not nxt:
            d, w, p, t = _counts(qdb, args.max_depth)
            if t == 0:
                _dbg("[simple] nothing to do.")
            else:
                _dbg(f"[simple] queue drained: done={d} working={w} pending={p} total={t}")
            break

        subject, hop = nxt
        _dbg(f"[simple] eliciting '{subject}' (hop={hop})")

        # ------ ELICITATION ------
        try:
            user_prompt = _render_elicitation(args.elicitation_strategy, subject, args.max_facts_hint, args.domain, args.topic)

            if args.debug or _should_debug_prompts():
                print("\n--- ELICITATION PROMPT ---\n", user_prompt, "\n--------------------------\n", flush=True)

            el_schema = ELICIT_SCHEMA_CAL if (args.elicitation_strategy == "calibrate") else ELICIT_SCHEMA_BASE
            el_messages = [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": user_prompt},
            ]
            resp = el_llm(el_messages, json_schema=el_schema)

            # Normalize to {"facts":[...]}
            norm = _normalize_elicitation_output(resp)
            if args.debug:
                raw_show = resp if isinstance(resp, str) else json.dumps(resp, ensure_ascii=False)
                print("[debug] raw elicitation resp:", str(raw_show)[:1200])
                print("[debug] normalized facts:", json.dumps(norm, ensure_ascii=False)[:600])

            facts = norm.get("facts", [])

            # route
            acc, rej, objs = _route_facts(args, facts, hop, el_cfg.model)
            write_triples_accepted(fdb, acc)
            write_triples_sink(fdb, rej)
            for s, p, o, _, m, strat, c in acc:
                _append_jsonl(paths["facts_jsonl"], {
                    "subject": s, "predicate": p, "object": o,
                    "hop": hop, "model": m, "strategy": strat, "confidence": c
                })

            # ------ NER (batched over objects) ------
            cand = _filter_ner_candidates(objs)
            next_subjects: List[str] = []
            i = 0
            while i < len(cand):
                chunk = cand[i: i + args.ner_batch_size]
                ner_prompt = _render_ner(args.ner_strategy, chunk, args.domain, args.topic)

                if args.debug or _should_debug_prompts():
                    print("\n--- NER PROMPT ---\n", ner_prompt, "\n------------------\n", flush=True)

                ner_schema = NER_SCHEMA_CAL if (args.ner_strategy == "calibrate") else NER_SCHEMA_BASE
                ner_messages = [
                    {"role": "system", "content": "Return JSON only."},
                    {"role": "user", "content": ner_prompt},
                ]
                out = ner_llm(ner_messages, json_schema=ner_schema)
                norm_ner = _normalize_ner_output(out)

                if args.debug:
                    raw_show = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
                    print("[debug] raw NER resp:", str(raw_show)[:1200])
                    print("[debug] normalized NER:", json.dumps(norm_ner, ensure_ascii=False)[:600])

                for ph in norm_ner.get("phrases", []):
                    phrase = ph.get("phrase")
                    is_ne = bool(ph.get("is_ne"))
                    _append_jsonl(paths["ner_jsonl"], {
                        "parent_subject": subject, "hop": hop,
                        "phrase": phrase, "is_ne": is_ne,
                        "ner_model": ner_cfg.model, "ner_strategy": args.ner_strategy,
                        "domain": args.domain, "topic": args.topic
                    })
                    if is_ne and isinstance(phrase, str):
                        next_subjects.append(phrase)

                i += args.ner_batch_size

            _enqueue_next(qdb, paths, next_subjects, hop, args.max_depth)

            _mark_done(qdb, subject)
            subjects_elicited_total += 1  # increment cap counter

            # If we hit the cap right after finishing this subject, stop
            if args.max_subjects and subjects_elicited_total >= args.max_subjects:
                _dbg(f"[simple] max-subjects reached ({subjects_elicited_total}); stopping.")
                break

        except KeyboardInterrupt:
            n = reset_working_to_pending(qdb)
            print(f"\n[simple] Interrupted. reset {n} 'working' → 'pending' for resume.")
            break
        except Exception:
            # log + retry
            with open(paths["errors_log"], "a", encoding="utf-8") as ef:
                ef.write(f"[{datetime.datetime.now().isoformat()}] subject={subject}\n{traceback.format_exc()}\n")
            qdb.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=?", (subject,))
            qdb.commit()

    # Final snapshots
    # queue.json
    conn = sqlite3.connect(paths["queue_sqlite"])
    cur = conn.cursor()
    cur.execute("SELECT subject, hop, status, retries, created_at FROM queue ORDER BY hop, subject")
    rows = cur.fetchall()
    with open(paths["queue_json"], "w", encoding="utf-8") as f:
        json.dump(
            [{"subject": s, "hop": h, "status": st, "retries": r, "created_at": ts} for (s, h, st, r, ts) in rows],
            f, ensure_ascii=False, indent=2
        )
    conn.close()

    # facts.json
    conn = sqlite3.connect(paths["facts_sqlite"])
    cur = conn.cursor()
    cur.execute(
        "SELECT subject, predicate, object, hop, model_name, strategy, confidence "
        "FROM triples_accepted ORDER BY subject, predicate, object"
    )
    rows_acc = cur.fetchall()
    cur.execute(
        "SELECT subject, predicate, object, hop, model_name, strategy, confidence, reason "
        "FROM triples_sink ORDER BY subject, predicate, object"
    )
    rows_sink = cur.fetchall()
    with open(paths["facts_json"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "accepted": [
                    {"subject": s, "predicate": p, "object": o, "hop": h,
                     "model": m, "strategy": st, "confidence": c}
                    for (s, p, o, h, m, st, c) in rows_acc
                ],
                "sink": [
                    {"subject": s, "predicate": p, "object": o, "hop": h,
                     "model": m, "strategy": st, "confidence": c, "reason": r}
                    for (s, p, o, h, m, st, c, r) in rows_sink
                ],
            },
            f, ensure_ascii=False, indent=2
        )
    conn.close()

    dur = time.time() - start
    print(f"[simple] finished in {dur:.1f}s → outputs in {out_dir}")
    print(f"[simple] queue.json  : {paths['queue_json']}")
    print(f"[simple] facts.json  : {paths['facts_json']}")
    print(f"[simple] facts.jsonl : {paths['facts_jsonl']}")
    print(f"[simple] ner log     : {paths['ner_jsonl']}")
    print(f"[simple] errors.log  : {paths['errors_log']}")


if __name__ == "__main__":
    main()
