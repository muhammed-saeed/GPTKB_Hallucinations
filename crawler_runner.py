# crawler_concurrent.py
from __future__ import annotations
import argparse
import datetime
import json
import os
import re
import sqlite3
import time
import traceback
from typing import Dict, List, Tuple, Set

from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from settings import (
    settings,
    ELICIT_SCHEMA_BASE,
    ELICIT_SCHEMA_CAL,
    NER_SCHEMA_BASE,
    NER_SCHEMA_CAL,
)
from prompter_parser import get_prompt_messages
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

# -------------------- logging / paths --------------------

def _dbg(msg: str):
    print(msg, flush=True)

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
        "lowconf_json": os.path.join(out_dir, "facts_lowconf.json"),
        "lowconf_jsonl": os.path.join(out_dir, "facts_lowconf.jsonl"),
    }

def _append_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# -------------------- DB helpers (atomic claim) --------------------

def _fetch_one_pending(conn: sqlite3.Connection, max_depth: int) -> Tuple[str, int] | None:
    cur = conn.cursor()
    try:
        cur.execute(
            """
            UPDATE queue
               SET status='working'
             WHERE rowid = (
                SELECT rowid
                  FROM queue
                 WHERE status='pending' AND hop<=?
              ORDER BY hop ASC, created_at ASC
                 LIMIT 1
             )
         RETURNING subject, hop
            """,
            (max_depth,),
        )
        row = cur.fetchone()
        conn.commit()
        return (row[0], row[1]) if row else None
    except sqlite3.OperationalError:
        cur.execute("BEGIN IMMEDIATE")
        cur.execute(
            "SELECT rowid, subject, hop FROM queue WHERE status='pending' AND hop<=? "
            "ORDER BY hop ASC, created_at ASC LIMIT 1",
            (max_depth,),
        )
        row = cur.fetchone()
        if not row:
            conn.commit()
            return None
        rowid, subject, hop = row
        cur.execute("UPDATE queue SET status='working' WHERE rowid=? AND status='pending'", (rowid,))
        changed = cur.rowcount
        conn.commit()
        if changed == 0:
            return None
        return subject, hop

def _fetch_many_pending(conn: sqlite3.Connection, max_depth: int, limit: int) -> List[Tuple[str, int]]:
    got: List[Tuple[str, int]] = []
    for _ in range(limit):
        one = _fetch_one_pending(conn, max_depth)
        if not one:
            break
        got.append(one)
    return got

def _mark_done(conn: sqlite3.Connection, subject: str):
    conn.execute("UPDATE queue SET status='done' WHERE subject=? AND status='working'", (subject,))
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

# -------------------- Output normalization / robust parsing --------------------

def _parse_obj(maybe_json) -> dict:
    if isinstance(maybe_json, dict):
        return maybe_json
    if isinstance(maybe_json, str):
        try:
            return json.loads(maybe_json)
        except Exception:
            return {}
    return {}

def _normalize_ner_output(out) -> Dict[str, list]:
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

def _unwrap_text(resp):
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        for k in ("text", "output_text", "content", "message", "response"):
            v = resp.get(k)
            if isinstance(v, str):
                return v
        ch = resp.get("choices")
        if isinstance(ch, list) and ch:
            c0 = ch[0] or {}
            msg = c0.get("message") or {}
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
            if isinstance(c0.get("text"), str):
                return c0["text"]
        raw = resp.get("raw")
        if isinstance(raw, str):
            return raw
        if isinstance(raw, dict):
            return _unwrap_text(raw)
    return ""

def _extract_json_block(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    starts = [m.start() for m in re.finditer(r"\{", text)]
    ends = [m.start() for m in re.finditer(r"\}", text)]
    for i in range(len(starts)):
        for j in range(len(ends)-1, -1, -1):
            if ends[j] <= starts[i]:
                continue
            chunk = text[starts[i]:ends[j]+1]
            try:
                return json.loads(chunk)
            except Exception:
                continue
    return {}

def _extract_facts_from_resp(resp) -> List[dict]:
    if isinstance(resp, dict):
        for key in ("facts", "triples"):
            val = resp.get(key)
            if isinstance(val, list):
                return [t for t in val if isinstance(t, dict)]
    text = _unwrap_text(resp)
    obj = _extract_json_block(text)
    if isinstance(obj, dict):
        for key in ("facts", "triples"):
            val = obj.get(key)
            if isinstance(val, list):
                return [t for t in val if isinstance(t, dict)]
    return []

# -------------------- facts routing + NER helpers --------------------

def _route_facts(args, facts: List[dict], hop: int, model_name: str):
    acc, lowconf, objs = [], [], []
    use_threshold = (args.elicitation_strategy == "calibrate")
    thr = float(args.conf_threshold)

    for f in facts:
        s, p, o = f.get("subject"), f.get("predicate"), f.get("object")
        if not (isinstance(s, str) and isinstance(p, str) and isinstance(o, str)):
            continue
        conf = f.get("confidence")

        if use_threshold and isinstance(conf, (int, float)):
            if conf < thr:
                lowconf.append({
                    "subject": s, "predicate": p, "object": o,
                    "hop": hop, "model": model_name,
                    "strategy": args.elicitation_strategy,
                    "confidence": float(conf),
                    "threshold": thr
                })
                continue

        acc.append((s, p, o, hop, model_name, args.elicitation_strategy,
                    conf if isinstance(conf, (int, float)) else None))
        objs.append(o)

    return acc, lowconf, objs

def _filter_ner_candidates(objs: List[str]) -> List[str]:
    return sorted({o for o in objs if isinstance(o, str) and 1 <= len(o.split()) <= 6})

_NE_ORG_KEYWORDS = (
    "University", "Institute", "College", "Academy", "Society", "Laboratory",
    "Office", "Department", "Institution", "Council", "Association", "Company",
    "Corporation", "Committee", "Foundation"
)
_NE_AWARD_KEYWORDS = ("Medal", "Prize", "Award")
_NE_WORK_KEYWORDS  = ("Analyzer", "Memex", "Web", "Think", "Frontier")

_date_rx = re.compile(r"^\d{4}([-/]\d{2}){0,2}$|^(January|February|March|April|May|June|July|August|September|October|November|December)\b", re.I)
_url_rx  = re.compile(r"^https?://", re.I)

def _is_date_like(s: str) -> bool:
    return bool(_date_rx.search(s))

def _is_literal_like(s: str) -> bool:
    if _url_rx.search(s): return True
    if s.isdigit(): return True
    if s.strip().lower() in {"human","engineer","inventor","person","male","female"}:
        return True
    return False

def _titlecase_ratio(s: str) -> float:
    words = [w for w in re.split(r"\s+", s.strip()) if w]
    if not words: return 0.0
    caps = sum(1 for w in words if w[:1].isupper())
    return caps / len(words)

def _maybe_is_ne_heuristic(phrase: str) -> bool:
    if not isinstance(phrase, str): return False
    p = phrase.strip()
    if not p: return False
    if _is_date_like(p) or _is_literal_like(p): return False
    if " " not in p and p.islower(): return False
    if any(k in p for k in _NE_ORG_KEYWORDS + _NE_AWARD_KEYWORDS + _NE_WORK_KEYWORDS):
        return True
    if _titlecase_ratio(p) >= 0.6: return True
    if " " in p and not p.islower(): return True
    return False

# -------------------- Prompt helpers --------------------

def _ensure_json_keyword_in_msgs(msgs: List[dict], shape_hint: str):
    """
    DeepSeek requires the literal word 'json' in the prompt when using response_format=json.
    This appends a tiny system line if neither system/user messages contain 'json'.
    Harmless for other providers.
    """
    has_json = False
    for m in msgs:
        c = m.get("content") or ""
        if isinstance(c, str) and "json" in c.lower():
            has_json = True
            break
    if not has_json:
        msgs.append({
            "role": "system",
            "content": f"Output ONLY JSON; shape: {shape_hint}",
        })

# -------------------- Provider helpers --------------------

def _is_openai_model(cfg) -> bool:
    prov = (getattr(cfg, "provider", "") or "").lower()
    if "openai" in prov:
        return True
    name = (getattr(cfg, "model", "") or "").lower()
    return "openai" in name or name.startswith("gpt-")

# -------------------- small builders --------------------

def _build_elicitation_messages(args, subject: str) -> List[dict]:
    return get_prompt_messages(
        args.elicitation_strategy, "elicitation",
        domain=args.domain,
        variables=dict(
            subject_name=subject,
            root_subject=args.seed,
            max_facts_hint=args.max_facts_hint,
        ),
    )

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser(description="Concurrent/Batch crawler (elicitation + NER).")
    ap.add_argument("--seed", required=True)
    ap.add_argument("--output-dir", default=None)

    # Strategies / domain
    ap.add_argument("--elicitation-strategy", default="baseline", choices=["baseline","icl","dont_know","calibrate"])
    ap.add_argument("--ner-strategy", default="baseline", choices=["baseline","icl","dont_know","calibrate"])
    ap.add_argument("--domain", default="general", choices=["general","topic"])

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

    # Hard cap
    ap.add_argument("--max-subjects", type=int, default=0)

    # Responses API extras
    ap.add_argument("--reasoning-effort", choices=["minimal","low","medium","high"], default=None)
    ap.add_argument("--verbosity", choices=["low","medium","high"], default=None)

    # Resume
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reset-working", action="store_true")

    # Debug
    ap.add_argument("--debug", action="store_true")

    # Batching / concurrency controls
    ap.add_argument("--batch-size", type=int, default=8, help="OpenAI: subjects per batch request")
    ap.add_argument("--concurrency", type=int, default=8, help="Non-OpenAI: parallel elicitation calls")
    ap.add_argument("--max-inflight", type=int, default=None, help="Upper bound on outstanding elicitation calls")
    ap.add_argument("--timeout", type=float, default=90.0, help="Per-request timeout (seconds)")

    args = ap.parse_args()

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

    # Respect per-provider rules
    for cfg in (el_cfg, ner_cfg):
        if getattr(cfg, "use_responses_api", False):
            cfg.temperature = None
            cfg.top_p = None
            cfg.top_k = None
            if cfg.extra_inputs is None:
                cfg.extra_inputs = {}
            cfg.extra_inputs.setdefault("reasoning", {})
            cfg.extra_inputs.setdefault("text", {})
            if args.reasoning_effort:
                cfg.extra_inputs["reasoning"]["effort"] = args.reasoning_effort
            if args.verbosity:
                cfg.extra_inputs["text"]["verbosity"] = args.verbosity
        else:
            if args.temperature is not None: cfg.temperature = args.temperature
            if args.top_p is not None: cfg.top_p = args.top_p
            if args.top_k is not None: cfg.top_k = args.top_k
        if args.max_tokens is not None:
            cfg.max_tokens = args.max_tokens
        if cfg.max_tokens is None:
            cfg.max_tokens = 2048

        # Pass timeout if supported by the client wrapper
        if hasattr(cfg, "request_timeout"):
            cfg.request_timeout = args.timeout
        elif hasattr(cfg, "timeout"):
            cfg.timeout = args.timeout

    el_llm = make_llm_from_config(el_cfg)
    ner_llm = make_llm_from_config(ner_cfg)

    is_openai_el = _is_openai_model(el_cfg)
    if args.max_inflight is None:
        args.max_inflight = (args.batch_size if is_openai_el else args.concurrency)

    start = time.time()
    subjects_elicited_total = 0
    lowconf_accum: List[dict] = []
    seen_facts: Set[Tuple[str, str, str, int]] = set()

    # ------------ worker for non-OpenAI providers (DeepSeek/Replicate etc.) ------------
    def _elicitation_and_ner_non_openai(subject: str, hop: int):
        try:
            # ---------- ELICIT ----------
            el_messages = _build_elicitation_messages(args, subject)
            _ensure_json_keyword_in_msgs(
                el_messages,
                shape_hint='{"facts":[{"subject":"...","predicate":"...","object":"..."}]}'
            )
            if args.debug:
                print("\n--- ELICITATION MESSAGES ---")
                for m in el_messages: print(m["role"].upper()+":", m["content"][:4000])
                print("----------------------------\n")

            el_schema = ELICIT_SCHEMA_CAL if (args.elicitation_strategy == "calibrate") else ELICIT_SCHEMA_BASE
            try:
                resp = el_llm(el_messages, json_schema=el_schema)
            except TypeError:
                resp = el_llm(el_messages)
            except Exception as e:
                if "json" in str(e).lower() or "response_format" in str(e).lower():
                    resp = el_llm(el_messages)
                else:
                    raise

            facts = _extract_facts_from_resp(resp)
            if not facts:
                try:
                    resp2 = el_llm(el_messages)
                    facts = _extract_facts_from_resp(resp2)
                except Exception:
                    pass

            if not facts:
                write_triples_sink(
                    fdb,
                    [(
                        subject, "__empty__", "__empty__", hop,
                        el_cfg.model, args.elicitation_strategy, None,
                        "empty_or_unparseable_output"
                    )]
                )

            acc, lowconf, _ = _route_facts(args, facts, hop, el_cfg.model)
            write_triples_accepted(fdb, acc)
            for s, p, o, _, m, strat, c in acc:
                key = (s, p, o, hop)
                if key not in seen_facts:
                    seen_facts.add(key)
                    _append_jsonl(paths["facts_jsonl"], {
                        "subject": s, "predicate": p, "object": o,
                        "hop": hop, "model": m, "strategy": strat, "confidence": c
                    })
            if lowconf:
                for item in lowconf:
                    _append_jsonl(paths["lowconf_jsonl"], item)
                lowconf_accum.extend(lowconf)

            # ---------- NER ----------
            cand = _filter_ner_candidates([t.get("object") for t in facts if isinstance(t, dict)])
            next_subjects: List[str] = []
            i = 0
            while i < len(cand):
                chunk = cand[i: i + args.ner_batch_size]
                ner_messages = get_prompt_messages(
                    args.ner_strategy, "ner",
                    domain=args.domain,
                    variables=dict(
                        phrases_block="\n".join(chunk),
                        root_subject=args.seed,
                    ),
                )
                _ensure_json_keyword_in_msgs(
                    ner_messages,
                    shape_hint='{"phrases":[{"phrase":"...","is_ne":true}]}'
                )
                if args.debug:
                    print("\n--- NER MESSAGES ---")
                    for m in ner_messages: print(m["role"].upper()+":", m["content"][:4000])
                    print("---------------------\n")

                ner_schema = NER_SCHEMA_CAL if (args.ner_strategy == "calibrate") else NER_SCHEMA_BASE
                try:
                    out = ner_llm(ner_messages, json_schema=ner_schema)
                except TypeError:
                    out = ner_llm(ner_messages)
                except Exception as e:
                    if "json" in str(e).lower() or "response_format" in str(e).lower():
                        out = ner_llm(ner_messages)
                    else:
                        raise

                norm_ner = _normalize_ner_output(out)
                decisions = norm_ner.get("phrases", [])
                if not decisions:
                    try:
                        out2 = ner_llm(ner_messages)
                        norm2 = _normalize_ner_output(out2)
                        decisions = norm2.get("phrases", []) or []
                    except Exception:
                        decisions = []
                if not decisions:
                    decisions = [{"phrase": ph, "is_ne": _maybe_is_ne_heuristic(ph)} for ph in chunk]

                for d in decisions:
                    phrase = d.get("phrase")
                    is_ne = bool(d.get("is_ne"))
                    _append_jsonl(paths["ner_jsonl"], {
                        "parent_subject": subject, "hop": hop,
                        "phrase": phrase, "is_ne": is_ne,
                        "ner_model": ner_cfg.model, "ner_strategy": args.ner_strategy,
                        "domain": args.domain, "root_subject": args.seed if (args.domain == "topic") else None,
                        "source": "model" if d in norm_ner.get("phrases", []) else "fallback"
                    })
                    if is_ne and isinstance(phrase, str):
                        next_subjects.append(phrase)
                i += args.ner_batch_size

            if next_subjects:
                enqueue_subjects(qdb, ((s, hop + 1) for s in next_subjects if hop + 1 <= args.max_depth))
                for s in next_subjects:
                    if hop + 1 <= args.max_depth:
                        _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": hop + 1})

            _mark_done(qdb, subject)
            return (subject, hop, None)

        except Exception:
            with open(paths["errors_log"], "a", encoding="utf-8") as ef:
                ef.write(f"[{datetime.datetime.now().isoformat()}] subject={subject}\n{traceback.format_exc()}\n")
            qdb.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=?", (subject,))
            qdb.commit()
            return (subject, hop, "error")

    # ------------- main scheduling loop -------------
    while True:
        if args.max_subjects and subjects_elicited_total >= args.max_subjects:
            _dbg(f"[simple] max-subjects reached ({subjects_elicited_total}); stopping.")
            break

        claim_n = min(args.max_inflight, (args.batch_size if is_openai_el else args.concurrency))
        batch = _fetch_many_pending(qdb, args.max_depth, max(1, claim_n))
        if not batch:
            d, w, p, t = _counts(qdb, args.max_depth)
            if t == 0:
                _dbg("[simple] nothing to do.")
            else:
                _dbg(f"[simple] queue drained: done={d} working={w} pending={p} total={t}")
            break

        if is_openai_el:
            # -------- OpenAI batched path --------
            subjects = [s for (s, _hop) in batch]
            hops     = [h for (_s, h) in batch]
            _dbg(f"[simple] OpenAI batch: {len(subjects)} subjects")

            messages_list = []
            for s in subjects:
                msgs = _build_elicitation_messages(args, s)
                # OpenAI doesn't require this, but harmless if present.
                _ensure_json_keyword_in_msgs(
                    msgs,
                    shape_hint='{"facts":[{"subject":"...","predicate":"...","object":"..."}]}'
                )
                messages_list.append(msgs)

            el_schema = ELICIT_SCHEMA_CAL if (args.elicitation_strategy == "calibrate") else ELICIT_SCHEMA_BASE

            try:
                if hasattr(el_llm, "batch"):
                    try:
                        resp_list = el_llm.batch(messages_list, json_schema=el_schema, timeout=args.timeout)
                    except TypeError:
                        resp_list = el_llm.batch(messages_list, json_schema=el_schema)
                else:
                    # Per-item fallback (still fine)
                    resp_list = []
                    for msgs in messages_list:
                        try:
                            resp_list.append(el_llm(msgs, json_schema=el_schema, timeout=args.timeout))
                        except TypeError:
                            resp_list.append(el_llm(msgs, json_schema=el_schema))
            except Exception:
                # Batch failed → revert to pending so they can be retried
                for subject, _hop in batch:
                    qdb.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=?", (subject,))
                qdb.commit()
                _dbg("[warn] OpenAI batch call failed; reverted claimed items to pending.")
                continue

            # Handle each result serially
            for (subject, hop), resp in zip(batch, resp_list):
                try:
                    facts = _extract_facts_from_resp(resp)

                    if not facts:
                        # Try non-schema retry for safety
                        try:
                            resp2 = el_llm(_build_elicitation_messages(args, subject))
                            facts = _extract_facts_from_resp(resp2)
                        except Exception:
                            pass

                    if not facts:
                        write_triples_sink(
                            fdb,
                            [(
                                subject, "__empty__", "__empty__", hop,
                                el_cfg.model, args.elicitation_strategy, None,
                                "empty_or_unparseable_output"
                            )]
                        )

                    acc, lowconf, _ = _route_facts(args, facts, hop, el_cfg.model)
                    write_triples_accepted(fdb, acc)

                    for s, p, o, _, m, strat, c in acc:
                        key = (s, p, o, hop)
                        if key not in seen_facts:
                            seen_facts.add(key)
                            _append_jsonl(paths["facts_jsonl"], {
                                "subject": s, "predicate": p, "object": o,
                                "hop": hop, "model": m, "strategy": strat, "confidence": c
                            })

                    if lowconf:
                        for item in lowconf:
                            _append_jsonl(paths["lowconf_jsonl"], item)
                        lowconf_accum.extend(lowconf)

                    # ---------- NER ----------
                    cand = _filter_ner_candidates([t.get("object") for t in facts if isinstance(t, dict)])
                    next_subjects: List[str] = []
                    i = 0
                    while i < len(cand):
                        chunk = cand[i: i + args.ner_batch_size]
                        ner_messages = get_prompt_messages(
                            args.ner_strategy, "ner",
                            domain=args.domain,
                            variables=dict(
                                phrases_block="\n".join(chunk),
                                root_subject=args.seed,
                            ),
                        )
                        # Add json hint (harmless on OpenAI; needed if you swap ner_model to deepseek)
                        _ensure_json_keyword_in_msgs(
                            ner_messages,
                            shape_hint='{"phrases":[{"phrase":"...","is_ne":true}]}'
                        )
                        ner_schema = NER_SCHEMA_CAL if (args.ner_strategy == "calibrate") else NER_SCHEMA_BASE
                        out = ner_llm(ner_messages, json_schema=ner_schema)
                        norm_ner = _normalize_ner_output(out)
                        for ph in norm_ner.get("phrases", []):
                            phrase = ph.get("phrase")
                            is_ne = bool(ph.get("is_ne"))
                            _append_jsonl(paths["ner_jsonl"], {
                                "parent_subject": subject, "hop": hop,
                                "phrase": phrase, "is_ne": is_ne,
                                "ner_model": ner_cfg.model, "ner_strategy": args.ner_strategy,
                                "domain": args.domain, "root_subject": args.seed if (args.domain == "topic") else None,
                            })
                            if is_ne and isinstance(phrase, str):
                                next_subjects.append(phrase)
                        i += args.ner_batch_size

                    if next_subjects:
                        enqueue_subjects(qdb, ((s, hop + 1) for s in next_subjects if hop + 1 <= args.max_depth))
                        for s in next_subjects:
                            if hop + 1 <= args.max_depth:
                                _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": hop + 1})

                    _mark_done(qdb, subject)
                    subjects_elicited_total += 1
                    if args.max_subjects and subjects_elicited_total >= args.max_subjects:
                        _dbg(f"[simple] max-subjects reached ({subjects_elicited_total}); stopping.")
                        break

                except Exception:
                    with open(paths["errors_log"], "a", encoding="utf-8") as ef:
                        ef.write(f"[{datetime.datetime.now().isoformat()}] subject={subject}\n{traceback.format_exc()}\n")
                    qdb.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=?", (subject,))
                    qdb.commit()

        else:
            # -------- Non-OpenAI concurrent path (DeepSeek / Replicate) --------
            _dbg(f"[simple] concurrent elicitation: {len(batch)} subjects, workers={min(args.concurrency, len(batch))}")

            results = []
            with ThreadPoolExecutor(max_workers=min(args.concurrency, len(batch))) as pool:
                futs = [pool.submit(_elicitation_and_ner_non_openai, s, h) for (s, h) in batch]
                for fut in as_completed(futs):
                    results.append(fut.result())

            for _s, _h, err in results:
                if err is None:
                    subjects_elicited_total += 1
                    if args.max_subjects and subjects_elicited_total >= args.max_subjects:
                        _dbg(f"[simple] max-subjects reached ({subjects_elicited_total}); stopping.")
                        break

    # ----- Final snapshots -----
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

    with open(paths["lowconf_json"], "w", encoding="utf-8") as f:
        json.dump({"below_threshold": lowconf_accum}, f, ensure_ascii=False, indent=2)

    dur = time.time() - start
    print(f"[simple] finished in {dur:.1f}s → outputs in {out_dir}")
    print(f"[simple] queue.json        : {paths['queue_json']}")
    print(f"[simple] facts.json        : {paths['facts_json']}")
    print(f"[simple] facts.jsonl       : {paths['facts_jsonl']}")
    print(f"[simple] lowconf.json      : {paths['lowconf_json']}")
    print(f"[simple] lowconf.jsonl     : {paths['lowconf_jsonl']}")
    print(f"[simple] ner log           : {paths['ner_jsonl']}")
    print(f"[simple] errors.log        : {paths['errors_log']}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[simple] Interrupted.")
