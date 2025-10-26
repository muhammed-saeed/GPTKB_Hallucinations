# crawler_hybrid.py (GPTKB2) — OpenAI Batch (Chat or Responses) + Concurrency (DeepSeek/Replicate)
from __future__ import annotations
import argparse
import asyncio
import datetime
import json
import os
import sqlite3
import time
import traceback
from typing import Dict, List, Tuple, Optional

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

# OpenAI client is only needed for OpenAI Batch mode
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()

# ===================== Prompt rendering =====================

def render_elicitation(
    strategy: str,
    subject_name: str,
    max_facts_hint: str,
    domain: str = "general",
    topic: str | None = None,
) -> str:
    tpl = get_prompt_template(strategy, "elicitation", domain=domain, topic=topic)
    return tpl.render(subject_name=subject_name, max_facts_hint=max_facts_hint)

def render_ner(
    strategy: str,
    lines: List[str],
    domain: str = "general",
    topic: str | None = None,
) -> str:
    tpl = get_prompt_template(strategy, "ner", domain=domain, topic=topic)
    return tpl.render(lines="\n".join(lines))

# ===================== IO helpers =====================

def ensure_output_dir(base_dir: str | None) -> str:
    if base_dir:
        out = base_dir
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join("runs", ts)
    os.makedirs(out, exist_ok=True)
    return os.path.abspath(out)

def build_paths(output_dir: str) -> dict:
    return {
        "queue_sqlite": os.path.join(output_dir, "queue.sqlite"),
        "facts_sqlite": os.path.join(output_dir, "facts.sqlite"),
        "queue_jsonl": os.path.join(output_dir, "queue.jsonl"),
        "facts_jsonl": os.path.join(output_dir, "facts.jsonl"),
        "queue_json": os.path.join(output_dir, "queue.json"),
        "facts_json": os.path.join(output_dir, "facts.json"),
        "run_meta": os.path.join(output_dir, "run_meta.json"),
        "errors_log": os.path.join(output_dir, "errors.log"),
        "ner_jsonl": os.path.join(output_dir, "ner_decisions.jsonl"),
        "ner_stats": os.path.join(output_dir, "ner_stats.json"),
        # Batch artifacts
        "batches_dir": os.path.join(output_dir, "batches"),
        "tmp": os.path.join(output_dir, "tmp"),
    }

def write_run_meta(path: str, meta: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def append_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def dump_queue_json(queue_db_path: str, out_json: str, max_depth: int):
    conn = sqlite3.connect(queue_db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT subject, hop, status, retries, created_at "
        "FROM queue WHERE hop<=? ORDER BY hop, subject",
        (max_depth,),
    )
    rows = cur.fetchall()
    data = [
        {"subject": s, "hop": h, "status": st, "retries": r, "created_at": ts}
        for (s, h, st, r, ts) in rows
    ]
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def dump_facts_json(facts_db_path: str, out_json: str):
    conn = sqlite3.connect(facts_db_path)
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

    data = {
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
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ===================== DB ops =====================

def fetch_pending_batch(conn: sqlite3.Connection, max_depth: int, limit: int) -> List[Tuple[str, int]]:
    cur = conn.cursor()
    cur.execute(
        "SELECT subject, hop FROM queue WHERE status='pending' AND hop<=? LIMIT ?",
        (max_depth, limit),
    )
    rows = cur.fetchall()
    if not rows:
        return []
    cur.executemany("UPDATE queue SET status='working' WHERE subject=?", [(r[0],) for r in rows])
    conn.commit()
    return [(s, h) for (s, h) in rows]

def fetch_one_pending(conn: sqlite3.Connection, max_depth: int) -> Tuple[str, int] | None:
    cur = conn.cursor()
    cur.execute("SELECT subject, hop FROM queue WHERE status='pending' AND hop<=? LIMIT 1", (max_depth,))
    row = cur.fetchone()
    if not row:
        return None
    s, h = row
    cur.execute("UPDATE queue SET status='working' WHERE subject=?", (s,))
    conn.commit()
    return s, h

def mark_done(conn: sqlite3.Connection, subjects: List[str] | str):
    cur = conn.cursor()
    if isinstance(subjects, str):
        cur.execute("UPDATE queue SET status='done' WHERE subject=?", (subjects,))
    else:
        cur.executemany("UPDATE queue SET status='done' WHERE subject=?", [(s,) for s in subjects])
    conn.commit()

def counts(conn: sqlite3.Connection, max_depth: int):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(1) FROM queue WHERE status='done' AND hop<=?", (max_depth,))
    done = cur.fetchone()[0]
    cur.execute("SELECT COUNT(1) FROM queue WHERE status='working' AND hop<=?", (max_depth,))
    working = cur.fetchone()[0]
    cur.execute("SELECT COUNT(1) FROM queue WHERE status='pending' AND hop<=?", (max_depth,))
    pending = cur.fetchone()[0]
    return done, working, pending, done + working + pending

# ===================== LLM call builders =====================

def call_elicitation_build_messages(strategy: str, subject: str, hint: str, domain: str, topic: str | None):
    messages = [
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": render_elicitation(strategy, subject, hint, domain=domain, topic=topic)},
    ]
    schema = ELICIT_SCHEMA_CAL if strategy == "calibrate" else ELICIT_SCHEMA_BASE
    return messages, schema

def call_ner_build_messages(strategy: str, lines: List[str], domain: str, topic: str | None):
    messages = [
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": render_ner(strategy, lines, domain=domain, topic=topic)},
    ]
    schema = NER_SCHEMA_CAL if strategy == "calibrate" else NER_SCHEMA_BASE
    return messages, schema

# ===================== Async rate limiter =====================

class AsyncRateLimiter:
    def __init__(self, target_rpm: Optional[int]):
        self.target_rpm = target_rpm or 0
        self._lock = asyncio.Lock()
        self._last_ts = 0.0
        self._interval = 60.0 / self.target_rpm if self.target_rpm > 0 else 0.0

    async def wait(self):
        if self._interval <= 0.0:
            return
        async with self._lock:
            now = time.time()
            wait_for = max(0.0, (self._last_ts + self._interval) - now)
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            self._last_ts = time.time()

# ===================== Elicitation/NER helpers =====================

def _route_facts(args, facts, hop, model_name):
    acc, rej, objs = [], [], []
    for f in facts or []:
        s, p, o = f.get("subject"), f.get("predicate"), f.get("object")
        if not (isinstance(s, str) and isinstance(p, str) and isinstance(o, str)):
            continue
        conf = f.get("confidence")
        if args.elicitation_strategy == "calibrate" and isinstance(conf, (int, float)):
            if conf >= args.conf_threshold:
                acc.append((s, p, o, hop, model_name, args.elicitation_strategy, conf))
            else:
                rej.append((s, p, o, hop, model_name, args.elicitation_strategy, conf, "below_threshold"))
        else:
            acc.append((s, p, o, hop, model_name, args.elicitation_strategy,
                        conf if isinstance(conf, (int, float)) else None))
        objs.append(o)
    return acc, rej, objs

def _filter_ner_candidates(objs: List[str]) -> List[str]:
    return sorted({o for o in objs if isinstance(o, str) and 1 <= len(o.split()) <= 6})

def _enqueue_next(qdb, paths, next_subjects: List[str], hop: int, max_depth: int):
    if not next_subjects:
        return
    next_hop = hop + 1
    if next_hop > max_depth:
        return
    enqueue_subjects(qdb, ((s2, next_hop) for s2 in next_subjects))
    for s2 in next_subjects:
        append_jsonl(paths["queue_jsonl"], {"subject": s2, "hop": next_hop})

# ===================== OpenAI Batch mode helpers =====================

def ensure_openai_client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("openai package not installed. `pip install openai`")
    return OpenAI()

def _responses_extract_json_from_body(body: Dict[str, any]) -> Tuple[Optional[dict], Optional[str]]:
    # 1) output_text
    txt = body.get("output_text")
    if isinstance(txt, str) and txt.strip():
        try:
            return json.loads(txt), None
        except Exception as e:
            return None, f"Content not JSON: {e} | raw={txt[:800]}"
    # 2) output[].content[].text
    out = body.get("output")
    if isinstance(out, list):
        for item in out:
            content = item.get("content")
            if isinstance(content, list):
                for piece in content:
                    t = piece.get("text") if isinstance(piece, dict) else None
                    if isinstance(t, str) and t.strip():
                        try:
                            return json.loads(t), None
                        except Exception as e:
                            return None, f"Content not JSON: {e} | raw={t[:800]}"
    # 3) fallback
    try:
        ctext = (((body.get("choices") or [])[0]).get("message") or {}).get("content", "")
        if isinstance(ctext, str) and ctext.strip():
            try:
                return json.loads(ctext), None
            except Exception as e:
                return None, f"Content not JSON: {e} | raw={ctext[:800]}"
    except Exception:
        pass
    return None, json.dumps(body)[:1200]

def parse_openai_batch_output_line(line: str, is_responses: bool) -> Tuple[str, Optional[dict], Optional[str]]:
    try:
        obj = json.loads(line)
    except Exception as e:
        return ("", None, f"JSON decode error: {e}")
    custom_id = obj.get("custom_id", "")
    body = ((obj.get("response") or {}).get("body")) or {}
    if is_responses:
        parsed, err = _responses_extract_json_from_body(body)
        return (custom_id, parsed, err)
    # Chat Completions
    try:
        content = (((body.get("choices") or [])[0]).get("message") or {}).get("content", "")
    except Exception:
        content = ""
    if not content:
        return (custom_id, None, json.dumps(body)[:1200])
    try:
        parsed = json.loads(content)
        return (custom_id, parsed, None)
    except Exception as e:
        return (custom_id, None, f"Content not JSON: {e} | raw={content[:800]}")

def _openai_direct_chat(oa: OpenAI, body: dict) -> dict:
    resp = oa.chat.completions.create(**body)
    try:
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        return {}

def _openai_direct_responses(oa: OpenAI, body: dict) -> dict:
    resp = oa.responses.create(**body)
    try:
        body_json = json.loads(resp.model_dump_json())
    except Exception:
        body_json = json.loads(json.dumps(resp, default=lambda o: getattr(o, "__dict__", str(o))))
    parsed, _err = _responses_extract_json_from_body(body_json)
    return parsed or {}

def build_elicitation_request_line(
    *,
    model_name: str,
    strategy: str,
    subject: str,
    hop: int,
    max_facts_hint: str,
    domain: str,
    topic: Optional[str],
    temperature: float,
    top_p: float,
    max_tokens: int,
    use_cal_schema: bool,
    use_responses_api: bool,
    effort: Optional[str] = None,
    verbosity: Optional[str] = None,
) -> dict:
    user_prompt = render_elicitation(strategy, subject, max_facts_hint, domain, topic)
    custom_id = f"elic::{subject}::hop={hop}"

    if use_responses_api:
        user_prompt = "SYSTEM: Return ONLY valid JSON. No prose, no markdown fences.\n\n" + user_prompt
        body = {
            "model": model_name,
            "input": [{
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            }],
            "max_output_tokens": max_tokens,
            "text": {"format": {"type": "text"}},
        }
        if effort:
            body["reasoning"] = {"effort": effort}
        if verbosity:
            body["text"]["verbosity"] = verbosity
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }

    schema = ELICIT_SCHEMA_CAL if use_cal_schema else ELICIT_SCHEMA_BASE
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "schema", "schema": schema},
            },
        },
    }

# ===================== OpenAI Batch / Concurrency runners =====================

def run_openai_batch_mode(args, paths, qdb, fdb, el_cfg, ner_cfg):
    oa = ensure_openai_client()
    os.makedirs(paths["batches_dir"], exist_ok=True)

    ner_llm = make_llm_from_config(ner_cfg)

    inflight: Dict[str, Dict[str, object]] = {}
    subjects_elicited_total = 0
    start = time.time()
    last_log = start

    print("[hybrid] Mode: OpenAI Batch (elicitation) + sync NER (batched per subject)")

    try:
        while True:
            if args.max_subjects and subjects_elicited_total >= args.max_subjects:
                print(f"[batch] reached --max-subjects={args.max_subjects}; stopping submissions")

            # --------------- aggregation window ---------------
            agg_started_at = time.time()
            pending_acc: List[Tuple[str, int]] = []

            while len(inflight) < args.max_inflight:
                # top up accumulator
                need = max(args.batch_size - len(pending_acc), 0)
                if need > 0:
                    rows = fetch_pending_batch(qdb, args.max_depth, need)
                    pending_acc.extend(rows)

                # Got enough to submit a batch?
                if len(pending_acc) >= max(args.openai_batch_min, 1):
                    rows_to_submit = pending_acc[:args.batch_size]
                    pending_acc = pending_acc[args.batch_size:]

                    # Batch lines
                    lines = []
                    id_map: Dict[str, Tuple[str, int]] = {}
                    use_responses_api = bool(getattr(el_cfg, "use_responses_api", False))

                    ex = el_cfg.extra_inputs or {}
                    default_effort = (args.gpt5_effort or (ex.get("reasoning", {}) or {}).get("effort")
                                      or ("minimal" if use_responses_api else None))
                    default_verbosity = (args.gpt5_verbosity or (ex.get("text", {}) or {}).get("verbosity")
                                         or ("low" if use_responses_api else None))

                    for (subject, hop) in rows_to_submit:
                        line = build_elicitation_request_line(
                            model_name=el_cfg.model,
                            strategy=args.elicitation_strategy,
                            subject=subject,
                            hop=hop,
                            max_facts_hint=args.max_facts_hint,
                            domain=args.domain,
                            topic=args.topic,
                            temperature=el_cfg.temperature or 0.0,
                            top_p=el_cfg.top_p or 1.0,
                            max_tokens=el_cfg.max_tokens or (args.max_tokens or 1024),
                            use_cal_schema=(args.elicitation_strategy == "calibrate"),
                            use_responses_api=use_responses_api,
                            effort=default_effort,
                            verbosity=default_verbosity,
                        )
                        lines.append(line)
                        id_map[line["custom_id"]] = (subject, hop)

                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    req_path = os.path.join(paths["batches_dir"], f"elic_req_{ts}_{len(inflight)}.jsonl")
                    with open(req_path, "w", encoding="utf-8") as f:
                        for obj in lines:
                            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

                    batch_endpoint = "/v1/responses" if getattr(el_cfg, "use_responses_api", False) else "/v1/chat/completions"

                    input_file = oa.files.create(file=open(req_path, "rb"), purpose="batch")
                    batch = oa.batches.create(
                        input_file_id=input_file.id,
                        endpoint=batch_endpoint,
                        completion_window=args.completion_window,
                        metadata={"description": "KB elicitation batch"},
                    )
                    inflight[batch.id] = {"manifest": req_path, "map": id_map, "is_responses": getattr(el_cfg, "use_responses_api", False)}
                    if args.debug:
                        print(f"[batch] submitted batch: {batch.id} subjects={len(lines)} inflight={len(inflight)} endpoint={batch_endpoint}")
                    continue

                waited = time.time() - agg_started_at
                if waited < args.openai_batch_timeout:
                    time.sleep(0.5)
                    continue

                if pending_acc and args.openai_fastpath_max > 0:
                    direct_rows = pending_acc[:args.openai_fastpath_max]
                    pending_acc = pending_acc[len(direct_rows):]

                    if args.debug:
                        print(f"[fastpath] sending {len(direct_rows)} direct calls")

                    for (subject, hop) in direct_rows:
                        use_responses_api = bool(getattr(el_cfg, "use_responses_api", False))
                        if use_responses_api:
                            user_prompt = render_elicitation(
                                args.elicitation_strategy, subject, args.max_facts_hint, args.domain, args.topic
                            )
                            user_prompt = "SYSTEM: Return ONLY valid JSON. No prose, no markdown fences.\n\n" + user_prompt

                            ex = el_cfg.extra_inputs or {}
                            effort = args.gpt5_effort or (ex.get("reasoning", {}) or {}).get("effort") or "minimal"
                            verbosity = args.gpt5_verbosity or (ex.get("text", {}) or {}).get("verbosity") or "low"

                            body = {
                                "model": el_cfg.model,
                                "input": [{
                                    "role": "user",
                                    "content": [{"type": "input_text", "text": user_prompt}],
                                }],
                                "max_output_tokens": el_cfg.max_tokens or (args.max_tokens or 1024),
                                "text": {"format": {"type": "text"}, "verbosity": verbosity},
                                "reasoning": {"effort": effort},
                            }

                            try:
                                out = _openai_direct_responses(oa, body) or {}
                            except Exception:
                                qdb.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=?", (subject,))
                                qdb.commit()
                                with open(paths["errors_log"], "a", encoding="utf-8") as ef:
                                    ef.write(f"[{datetime.datetime.now().isoformat()}] subject={subject} fastpath\n{traceback.format_exc()}\n")
                                continue
                        else:
                            user_prompt = render_elicitation(
                                args.elicitation_strategy, subject, args.max_facts_hint, args.domain, args.topic
                            )
                            schema = ELICIT_SCHEMA_CAL if (args.elicitation_strategy == "calibrate") else ELICIT_SCHEMA_BASE
                            body = {
                                "model": el_cfg.model,
                                "messages": [
                                    {"role": "system", "content": "Return JSON only."},
                                    {"role": "user", "content": user_prompt},
                                ],
                                "temperature": el_cfg.temperature or 0.0,
                                "top_p": el_cfg.top_p or 1.0,
                                "max_tokens": el_cfg.max_tokens or (args.max_tokens or 1024),
                                "response_format": {
                                    "type": "json_schema",
                                    "json_schema": {"name": "schema", "schema": schema},
                                },
                            }
                            try:
                                out = _openai_direct_chat(oa, body) or {}
                            except Exception:
                                qdb.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=?", (subject,))
                                qdb.commit()
                                with open(paths["errors_log"], "a", encoding="utf-8") as ef:
                                    ef.write(f"[{datetime.datetime.now().isoformat()}] subject={subject} fastpath\n{traceback.format_exc()}\n")
                                continue

                        facts = out.get("facts", []) or []
                        acc, rej, objs = _route_facts(args, facts, hop, el_cfg.model)
                        write_triples_accepted(fdb, acc)
                        write_triples_sink(fdb, rej)

                        for s, p, o, _, m, strat, c in acc:
                            append_jsonl(paths["facts_jsonl"], {
                                "subject": s, "predicate": p, "object": o,
                                "hop": hop, "model": m, "strategy": strat, "confidence": c
                            })
                        for s, p, o, _, m, strat, c, reason in rej:
                            append_jsonl(paths["facts_jsonl"], {
                                "subject": s, "predicate": p, "object": o,
                                "hop": hop, "model": m, "strategy": strat,
                                "confidence": c, "reason": reason
                            })

                        cand = _filter_ner_candidates([f.get("object") for f in facts if isinstance(f.get("object"), str)])
                        next_subjects: List[str] = []
                        i = 0
                        while i < len(cand):
                            chunk = cand[i: i + args.ner_batch_size]
                            ner_msgs, ner_schema = call_ner_build_messages(args.ner_strategy, chunk, args.domain, args.topic)
                            out_ner = ner_llm(ner_msgs, json_schema=ner_schema)
                            for ph in (out_ner.get("phrases") or []):
                                phrase = ph.get("phrase")
                                is_ne = bool(ph.get("is_ne"))
                                conf = ph.get("confidence") if isinstance(ph.get("confidence"), (int, float)) else None
                                append_jsonl(paths["ner_jsonl"], {
                                    "parent_subject": subject, "hop": hop,
                                    "phrase": phrase, "is_ne": is_ne, "confidence": conf,
                                    "ner_model": ner_cfg.model, "ner_strategy": args.ner_strategy,
                                    "domain": args.domain, "topic": args.topic
                                })
                                if is_ne and isinstance(phrase, str):
                                    next_subjects.append(phrase)
                            i += args.ner_batch_size

                        _enqueue_next(qdb, paths, next_subjects, hop, args.max_depth)
                        mark_done(qdb, subject)
                        subjects_elicited_total += 1

                break  # leave aggregation loop to poll inflight

            d, w, p, t = counts(qdb, args.max_depth)
            if not inflight and p == 0 and w == 0:
                print(f"[batch] queue drained: done={d} working={w} pending={p} total={t}")
                break

            # ------------------- Poll inflight batches -------------------
            to_remove = []
            for bid, meta in list(inflight.items()):
                try:
                    b = oa.batches.retrieve(bid)
                except Exception:
                    with open(paths["errors_log"], "a", encoding="utf-8") as ef:
                        ef.write(f"[{datetime.datetime.now().isoformat()}] batch={bid}\n{traceback.format_exc()}\n")
                    continue

                if args.debug:
                    try:
                        rc = getattr(b, "request_counts", None)
                        print(f"[batch] id={bid} status={b.status} output_file_id={b.output_file_id} request_counts={rc}")
                    except Exception:
                        pass

                if b.status in ("created", "validating", "in_progress", "finalizing"):
                    continue

                if b.status not in ("completed", "failed", "canceled", "expired"):
                    continue

                try:
                    err_fid = getattr(b, "error_file_id", None)
                    if err_fid:
                        err_bytes = oa.files.content(err_fid).content
                        err_path = os.path.join(paths["batches_dir"], f"{bid}_errors.jsonl")
                        with open(err_path, "wb") as ef:
                            ef.write(err_bytes)
                        print(f"[batch] id={bid} wrote errors to {err_path}")
                except Exception:
                    pass

                id_map: Dict[str, Tuple[str, int]] = meta["map"]
                subjects_this_batch = [s for (s, _) in id_map.values()]
                is_responses = bool(meta.get("is_responses"))

                if b.status != "completed":
                    for s in subjects_this_batch:
                        qdb.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=?", (s,))
                    qdb.commit()
                    to_remove.append(bid)
                    if args.debug:
                        print(f"[batch] batch {bid} status={b.status} -> returned {len(subjects_this_batch)} subjects to pending")
                    continue

                # Completed: download and parse results
                try:
                    out_bytes = oa.files.content(b.output_file_id).content
                except Exception:
                    out_bytes = b""
                out_path = os.path.join(paths["batches_dir"], f"{bid}_results.jsonl")
                with open(out_path, "wb") as f:
                    f.write(out_bytes)

                accepted_all, rejected_all = [], []
                ner_jobs: List[Tuple[str, int, List[str]]] = []

                with open(out_path, "r", encoding="utf-8") as rf:
                    for line in rf:
                        custom_id, parsed, err = parse_openai_batch_output_line(line, is_responses=is_responses)
                        subj, hop = id_map.get(custom_id, ("", 0))
                        if not subj:
                            continue
                        if err or not parsed:
                            qdb.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=?", (subj,))
                            qdb.commit()
                            with open(paths["errors_log"], "a", encoding="utf-8") as ef:
                                ef.write(f"[{datetime.datetime.now().isoformat()}] subject={subj} batch={bid}\n{err}\n")
                            continue

                        facts = parsed.get("facts", []) or []
                        acc, rej, objs = _route_facts(args, facts, hop, el_cfg.model)
                        accepted_all.extend(acc)
                        rejected_all.extend(rej)

                        for s, p, o, _, m, strat, c in acc:
                            append_jsonl(paths["facts_jsonl"], {"subject": s, "predicate": p, "object": o,
                                                                "hop": hop, "model": m, "strategy": strat, "confidence": c})
                        for s, p, o, _, m, strat, c, reason in rej:
                            append_jsonl(paths["facts_jsonl"], {"subject": s, "predicate": p, "object": o,
                                                                "hop": hop, "model": m, "strategy": strat,
                                                                "confidence": c, "reason": reason})

                        cand = _filter_ner_candidates(objs)
                        ner_jobs.append((subj, hop, cand))

                write_triples_accepted(fdb, accepted_all)
                write_triples_sink(fdb, rejected_all)

                for parent_subject, hop, cand in ner_jobs:
                    if not cand:
                        continue
                    if args.debug:
                        print(f"[batch] NER {len(cand)} candidates (hop={hop}) for '{parent_subject}'")
                    i = 0
                    next_subjects: List[str] = []
                    while i < len(cand):
                        chunk = cand[i: i + args.ner_batch_size]
                        ner_msgs, ner_schema = call_ner_build_messages(args.ner_strategy, chunk, args.domain, args.topic)
                        out = ner_llm(ner_msgs, json_schema=ner_schema)
                        phrases = out.get("phrases", []) or []
                        for ph in phrases:
                            phrase = ph.get("phrase")
                            is_ne = bool(ph.get("is_ne"))
                            conf = ph.get("confidence") if isinstance(ph.get("confidence"), (int, float)) else None
                            append_jsonl(paths["ner_jsonl"], {"parent_subject": parent_subject, "hop": hop,
                                                              "phrase": phrase, "is_ne": is_ne, "confidence": conf,
                                                              "ner_model": ner_cfg.model, "ner_strategy": args.ner_strategy,
                                                              "domain": args.domain, "topic": args.topic})
                            if is_ne and isinstance(phrase, str):
                                next_subjects.append(phrase)
                        i += args.ner_batch_size

                    _enqueue_next(qdb, paths, next_subjects, hop, args.max_depth)

                mark_done(qdb, subjects_this_batch)
                subjects_elicited_total += len(subjects_this_batch)
                to_remove.append(bid)

            for bid in to_remove:
                inflight.pop(bid, None)

            now = time.time()
            if now - last_log >= 2.0:
                d, w, p, t = counts(qdb, args.max_depth)
                print(f"[batch] progress: done={d} working={w} pending={p} inflight={len(inflight)} total={t} "
                      f"| elicited_total={subjects_elicited_total}")
                last_log = now

            time.sleep(max(1, args.poll_interval))

    except KeyboardInterrupt:
        n = reset_working_to_pending(qdb)
        print(f"\n[batch] Interrupted. reset {n} 'working' → 'pending' for resume.")

# ===================== Concurrency mode (DeepSeek / Replicate) =====================

async def run_concurrency_mode(args, paths, qdb, fdb, el_cfg, ner_cfg):
    el_llm = make_llm_from_config(el_cfg)
    ner_llm = make_llm_from_config(ner_cfg)
    limiter = AsyncRateLimiter(args.target_rpm)

    inflight: set[asyncio.Task] = set()
    subjects_elicited_total = 0
    subjects_elicited_lock = asyncio.Lock()

    print(f"[hybrid] Mode: Concurrency (elicitation & NER) provider={el_cfg.provider} | "
          f"concurrency={args.concurrency} target_rpm={args.target_rpm or 0}")

    async def process_subject(subject: str, hop: int):
        nonlocal subjects_elicited_total
        try:
            # --- ELICITATION ---
            if args.debug:
                print(f"[conc] eliciting '{subject}' (hop={hop})")
            el_msgs, el_schema = call_elicitation_build_messages(
                args.elicitation_strategy, subject, args.max_facts_hint, args.domain, args.topic
            )
            await limiter.wait()
            # IMPORTANT: pass json_schema by name to avoid positional errors
            resp = await asyncio.to_thread(el_llm, el_msgs, json_schema=el_schema)

            acc, rej, objs = _route_facts(args, resp.get("facts", []) or [], hop, el_cfg.model)
            await asyncio.to_thread(write_triples_accepted, fdb, acc)
            await asyncio.to_thread(write_triples_sink, fdb, rej)

            for s, p, o, _, m, strat, c in acc:
                append_jsonl(paths["facts_jsonl"], {
                    "subject": s, "predicate": p, "object": o, "hop": hop,
                    "model": m, "strategy": strat, "confidence": c
                })
            for s, p, o, _, m, strat, c, reason in rej:
                append_jsonl(paths["facts_jsonl"], {
                    "subject": s, "predicate": p, "object": o, "hop": hop,
                    "model": m, "strategy": strat, "confidence": c, "reason": reason
                })

            # --- NER (batched, per subject) ---
            cand = _filter_ner_candidates(objs)
            next_subjects: List[str] = []
            i = 0
            while i < len(cand):
                chunk = cand[i: i + args.ner_batch_size]
                ner_msgs, ner_schema = call_ner_build_messages(args.ner_strategy, chunk, args.domain, args.topic)
                await limiter.wait()
                out = await asyncio.to_thread(ner_llm, ner_msgs, json_schema=ner_schema)
                for ph in (out.get("phrases") or []):
                    phrase = ph.get("phrase")
                    is_ne = bool(ph.get("is_ne"))
                    conf = ph.get("confidence") if isinstance(ph.get("confidence"), (int, float)) else None
                    append_jsonl(paths["ner_jsonl"], {
                        "parent_subject": subject, "hop": hop,
                        "phrase": phrase, "is_ne": is_ne, "confidence": conf,
                        "ner_model": ner_cfg.model, "ner_strategy": args.ner_strategy,
                        "domain": args.domain, "topic": args.topic
                    })
                    if is_ne and isinstance(phrase, str):
                        next_subjects.append(phrase)
                i += args.ner_batch_size

            _enqueue_next(qdb, paths, next_subjects, hop, args.max_depth)
            await asyncio.to_thread(mark_done, qdb, subject)

            async with subjects_elicited_lock:
                subjects_elicited_total += 1

        except Exception:
            with open(paths["errors_log"], "a", encoding="utf-8") as ef:
                ef.write(f"[{datetime.datetime.now().isoformat()}] subject={subject}\n{traceback.format_exc()}\n")
            def _retry():
                qdb.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=?", (subject,))
                qdb.commit()
            await asyncio.to_thread(_retry)

    # -------- Dispatcher loop --------
    start = time.time()
    last_log = start

    try:
        while True:
            now = time.time()
            if now - last_log >= 2.0:
                d, w, p, t = counts(qdb, args.max_depth)
                print(f"[conc] progress: inflight={len(inflight)} done={d} working={w} pending={p} total={t} ")
                last_log = now

            if args.max_subjects and subjects_elicited_total >= args.max_subjects:
                if len(inflight) == 0:
                    print(f"[conc] reached --max-subjects={args.max_subjects}; stopping.")
                    break
                await asyncio.sleep(0.2)
                n = await asyncio.to_thread(reset_working_to_pending, qdb)
                continue

            capacity = args.concurrency - len(inflight)
            if args.max_subjects:
                remaining = args.max_subjects - subjects_elicited_total
                capacity = min(capacity, max(0, remaining))

            if capacity > 0:
                rows = await asyncio.to_thread(fetch_pending_batch, qdb, args.max_depth, capacity)
                for (subject, hop) in rows:
                    t = asyncio.create_task(process_subject(subject, hop))
                    inflight.add(t)
                    t.add_done_callback(lambda tt: inflight.discard(tt))

            d, w, p, t = counts(qdb, args.max_depth)
            if p == 0 and len(inflight) == 0:
                print(f"[conc] queue drained: done={d} working={w} pending={p} total={t}")
                break

            await asyncio.sleep(0.05)

    except KeyboardInterrupt:
        n = reset_working_to_pending(qdb)
        print(f"\n[conc] Interrupted. reset {n} 'working' → 'pending' for resume.")

# ===================== main =====================

def main():
    ap = argparse.ArgumentParser(
        description="Hybrid crawler: OpenAI Batch (Chat or Responses) OR Concurrency for DeepSeek/Replicate. NER batched per subject. Resumable."
    )
    ap.add_argument("--seed", required=True)
    ap.add_argument("--output-dir", default=None)

    # Strategies
    ap.add_argument("--elicitation-strategy", default="baseline", choices=["baseline","icl","dont_know","calibrate"])
    ap.add_argument("--ner-strategy", default="baseline", choices=["baseline","icl","dont_know","calibrate"])

    # Domain / Topic
    ap.add_argument("--domain", default="general", choices=["general","topic"])
    ap.add_argument("--topic", default=None)

    # Limits
    ap.add_argument("--max-depth", type=int, default=settings.MAX_DEPTH)
    ap.add_argument("--max-subjects", type=int, default=0, help="Optional hard cap on total elicited subjects (0=disabled)")

    # NER batching
    ap.add_argument("--ner-batch-size", type=int, default=settings.NER_BATCH_SIZE)
    ap.add_argument("--max-facts-hint", default=str(settings.MAX_FACTS_HINT))
    ap.add_argument("--conf-threshold", type=float, default=0.7)

    # Models
    ap.add_argument("--elicit-model-key", default=settings.ELICIT_MODEL_KEY)
    ap.add_argument("--ner-model-key", default=settings.NER_MODEL_KEY)

    # Sampler knobs (for Chat providers; GPT-5 ignores these)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=None, help="Max output tokens; used for both Chat and GPT-5 Responses.")

    # Batch controls (OpenAI only)
    ap.add_argument("--batch-size", type=int, default=50, help="OpenAI Batch: subjects per batch")
    ap.add_argument("--max-inflight", type=int, default=3, help="OpenAI Batch: max in-flight batches")
    ap.add_argument("--poll-interval", type=int, default=20, help="OpenAI Batch: seconds between polls")
    ap.add_argument("--completion-window", default="24h", help="OpenAI Batch completion window (e.g., '24h')")

    # OpenAI batch tuning + fast-path
    ap.add_argument("--openai-batch-min", type=int, default=5,
                    help="Minimum requests to prefer Batch; below this and after timeout, use direct API fast path.")
    ap.add_argument("--openai-batch-timeout", type=int, default=15,
                    help="Seconds to wait while aggregating before deciding to submit (Batch or fast path).")
    ap.add_argument("--openai-fastpath-max", type=int, default=10,
                    help="Max number of direct (non-Batch) requests to send per fast-path cycle.")

    # Concurrency controls (non-OpenAI providers)
    ap.add_argument("--concurrency", type=int, default=10, help="Number of async workers (non-OpenAI mode)")
    ap.add_argument("--target-rpm", type=int, default=0, help="Global requests/minute cap (0=unlimited)")

    # NEW: GPT-5 Responses controls
    ap.add_argument("--gpt5-effort", choices=["minimal", "low", "medium", "high"], default=None,
                    help="Responses API: reasoning.effort for GPT-5 family.")
    ap.add_argument("--gpt5-verbosity", choices=["low", "medium", "high"], default=None,
                    help="Responses API: text.verbosity for GPT-5 family.")

    # Resume
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reset-working", action="store_true")

    # Debug
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    if args.domain == "topic" and not args.topic:
        raise SystemExit("When --domain topic, also pass --topic (e.g., --topic entertainment).")

    out_dir = ensure_output_dir(args.output_dir)
    paths = build_paths(out_dir)
    os.makedirs(paths["tmp"], exist_ok=True)
    os.makedirs(paths["batches_dir"], exist_ok=True)
    print("[hybrid] output_dir:", out_dir, flush=True)

    qdb = open_queue_db(paths["queue_sqlite"])
    fdb = open_facts_db(paths["facts_sqlite"])

    if args.resume and queue_has_rows(qdb):
        if args.reset_working:
            n = reset_working_to_pending(qdb)
            print(f"[hybrid] resume: reset {n} 'working' → 'pending'")
        d0, w0, p0, t0 = counts(qdb, args.max_depth)
        print(f"[hybrid] resume: queue found: done={d0} working={w0} pending={p0} total={t0}")
    else:
        enqueue_subjects(qdb, [(args.seed, 0)])
        print("[hybrid] seeded queue with:", args.seed)

    el_cfg = settings.MODELS[args.elicit_model_key].model_copy(deep=True)
    ner_cfg = settings.MODELS[args.ner_model_key].model_copy(deep=True)
    for cfg in (el_cfg, ner_cfg):
        if args.temperature is not None: cfg.temperature = args.temperature
        if args.top_p is not None: cfg.top_p = args.top_p
        if args.top_k is not None: cfg.top_k = args.top_k
        if args.max_tokens is not None: cfg.max_tokens = args.max_tokens

    # Decide mode (Batch on OpenAI providers with no custom base_url)
    is_openai_batch = (el_cfg.provider in ("openai", "openai_compatible")) and (not el_cfg.base_url)

    if not is_openai_batch:
        if (args.batch_size and args.batch_size != 0) or (args.max_inflight and args.max_inflight != 0):
            print("[hybrid][warn] Non-OpenAI provider detected "
                  f"(provider={el_cfg.provider}, base_url={el_cfg.base_url}). "
                  "Ignoring --batch-size / --max-inflight and switching to concurrency mode.\n"
                  "Tip: use --concurrency and --target-rpm to control throughput.",
                  flush=True)
        args.batch_size = 0
        args.max_inflight = 0

    write_run_meta(paths["run_meta"], {
        "seed": args.seed,
        "max_depth": args.max_depth,
        "max_subjects": args.max_subjects,
        "ner_batch_size": args.ner_batch_size,
        "elicitation_strategy": args.elicitation_strategy,
        "ner_strategy": args.ner_strategy,
        "elicit_model_key": args.elicit_model_key,
        "ner_model_key": args.ner_model_key,
        "temperature": el_cfg.temperature,
        "top_p": el_cfg.top_p,
        "top_k": ner_cfg.top_k,
        "max_tokens": el_cfg.max_tokens,
        "conf_threshold": args.conf_threshold,
        "output_dir": out_dir,
        "mode": "openai_batch" if is_openai_batch else "concurrency",
        "resume": args.resume,
        "reset_working": args.reset_working,
        "domain": args.domain,
        "topic": args.topic,
        "batch_size": args.batch_size,
        "max_inflight": args.max_inflight,
        "poll_interval": args.poll_interval,
        "completion_window": args.completion_window,
        "concurrency": args.concurrency,
        "target_rpm": args.target_rpm,
        "openai_batch_min": args.openai_batch_min,
        "openai_batch_timeout": args.openai_batch_timeout,
        "openai_fastpath_max": args.openai_fastpath_max,
        "gpt5_effort": args.gpt5_effort,
        "gpt5_verbosity": args.gpt5_verbosity,
    })

    if is_openai_batch:
        run_openai_batch_mode(args, paths, qdb, fdb, el_cfg, ner_cfg)
    else:
        try:
            asyncio.run(run_concurrency_mode(args, paths, qdb, fdb, el_cfg, ner_cfg))
        except KeyboardInterrupt:
            n = reset_working_to_pending(qdb)
            print(f"\n[conc] Interrupted. reset {n} 'working' → 'pending' for resume.")

    dump_queue_json(paths["queue_sqlite"], paths["queue_json"], args.max_depth)
    dump_facts_json(paths["facts_sqlite"], paths["facts_json"])

    print(f"[hybrid] outputs in: {out_dir}")
    print(f"[hybrid] JSON snapshots: {paths['queue_json']} | {paths['facts_json']}")
    print(f"[hybrid] Streams: {paths['queue_jsonl']} | {paths['facts_jsonl']}")
    print(f"[hybrid] NER log: {paths['ner_jsonl']}")
    print(f"[hybrid] SQLite:  {paths['queue_sqlite']} | {paths['facts_sqlite']}")
    print(f"[hybrid] Errors:  {paths['errors_log']}")

if __name__ == "__main__":
    main()
