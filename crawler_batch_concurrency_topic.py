# crawler_simple_openai_claude_claude.py
from __future__ import annotations
import argparse
import datetime
import json
import os
import sqlite3
import time
import traceback
from typing import Any, Dict, List, Tuple, Optional
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

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

    # NEW: pass a root_subject to templates.
    # In topic-mode we anchor to the initial seed, stored in env by main().
    # Otherwise we just use the current subject.
    root_subject = os.getenv("ROOT_SUBJECT") if domain == "topic" else subject_name

    txt = tpl.render(
        subject_name=subject_name,
        subject=subject_name,
        seed=subject_name,            # legacy compat
        max_facts_hint=max_facts_hint,
        hint=max_facts_hint,          # legacy compat
        # topic-aware context
        domain=domain,
        topic=topic,
        root_subject=root_subject,    # <- templates can gate facts to this topic
    )
    return txt

def _render_ner(strategy: str, lines: List[str],
                domain: str = "general", topic: str | None = None) -> str:
    tpl = get_prompt_template(strategy, "ner", domain=domain, topic=topic)

    # NEW: same root_subject logic as elicitation
    root_subject = os.getenv("ROOT_SUBJECT") if domain == "topic" else ""

    txt = tpl.render(
        lines="\n".join(lines),
        phrases="\n".join(lines),
        phrases_block="\n".join(lines),
        # topic-aware context
        domain=domain,
        topic=topic,
        root_subject=root_subject,
    )
    return txt

# ====== provider-aware prompt sanitization ======

def _needs_prompt_sanitization(cfg) -> bool:
    try:
        provider = (getattr(cfg, "provider", "") or "").lower()
        return provider in ("replicate", "deepseek")
    except Exception:
        return False

def _sanitize_prompt_for_llm_contract(prompt: str) -> str:
    lines = []
    for ln in (prompt or "").splitlines():
        s = ln.strip()
        if s.startswith(("SYSTEM:", "SCHEMA-HINT:", "EXAMPLE:", "USER:", "ASSISTANT:")):
            continue
        lines.append(ln)
    lines.append("\nReturn ONLY valid JSON; no prose; no markdown; no code fences.")
    return "\n".join(lines).strip()

def _is_openai(cfg) -> bool:
    try:
        return (getattr(cfg, "provider", "") or "").lower() in ("openai", "openai_compatible")
    except Exception:
        return False

# ===================== IO helpers =====================

def _ensure_output_dir(base_dir: str | None) -> str:
    out = base_dir or os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    Path(out, "batches").mkdir(parents=True, exist_ok=True)
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
        "batch_state": os.path.join(out_dir, "batch_state.json"),
        "batches_dir": os.path.join(out_dir, "batches"),
    }

def _append_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ===================== DB helpers =====================

def _fetch_many_pending(conn: sqlite3.Connection, max_depth: int, limit: int) -> List[Tuple[str, int]]:
    cur = conn.cursor()
    cur.execute("SELECT subject, hop FROM queue WHERE status='pending' AND hop<=? LIMIT ?", (max_depth, limit))
    rows = cur.fetchall()
    if not rows:
        return []
    cur.executemany("UPDATE queue SET status='working' WHERE subject=?", [(s,) for s, _ in rows])
    conn.commit()
    return rows

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

# ===================== Normalizers & salvage =====================

def _parse_obj(maybe_json) -> dict:
    if isinstance(maybe_json, dict):
        return maybe_json
    if isinstance(maybe_json, str):
        try:
            return json.loads(maybe_json)
        except Exception:
            return {}
    return {}

def _normalize_elicitation_output(out) -> Dict[str, list]:
    obj = _parse_obj(out)
    facts = obj.get("facts")
    if isinstance(facts, list):
        return {"facts": [t for t in facts if isinstance(t, dict)]}
    triples = obj.get("triples")
    if isinstance(triples, list):
        return {"facts": [t for t in triples if isinstance(t, dict)]}
    return {"facts": []}

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

def _salvage_facts_from_raw(raw: str) -> List[dict]:
    if not isinstance(raw, str) or not raw:
        return []
    t = raw.strip()
    if t.startswith("```"):
        nl = t.find("\n")
        if nl != -1:
            t = t[nl + 1 :].strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    key_idx = t.find('"facts"')
    if key_idx == -1:
        key_idx = t.find("'facts'")
        if key_idx == -1:
            return []
    arr_start = t.find("[", key_idx)
    if arr_start == -1:
        return []
    i = arr_start + 1
    n = len(t)
    buf = []
    obj_depth = 0
    in_str = False
    esc = False
    facts: List[dict] = []

    def _try_flush(seg: str):
        seg = seg.strip()
        if not seg:
            return
        if seg and seg[0] == ",":
            seg = seg[1:].lstrip()
        if seg and seg[-1] == ",":
            seg = seg[:-1].rstrip()
        if seg.count("{") > seg.count("}"):
            seg = seg + "}"
        try:
            obj = json.loads(seg)
            if isinstance(obj, dict):
                s, p, o = obj.get("subject"), obj.get("predicate"), obj.get("object")
                if isinstance(s, str) and isinstance(p, str) and isinstance(o, str):
                    facts.append(obj)
        except Exception:
            pass

    while i < n:
        ch = t[i]
        if in_str:
            buf.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            buf.append(ch); i += 1; continue
        if ch == "{":
            obj_depth += 1; buf.append(ch); i += 1; continue
        if ch == "}":
            obj_depth -= 1; buf.append(ch); i += 1
            if obj_depth == 0 and buf:
                _try_flush("".join(buf)); buf = []
            continue
        if ch == "]" and obj_depth == 0:
            break
        buf.append(ch); i += 1

    if buf and obj_depth >= 0:
        _try_flush("".join(buf))
    return facts

def _loose_json_from_raw(raw: str) -> dict:
    if not raw or not isinstance(raw, str):
        return {}
    t = raw.strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    if t.startswith("```"):
        nl = t.find("\n")
        t2 = t[nl + 1 :].strip() if nl != -1 else t
        if t2.endswith("```"):
            t2 = t2[:-3].strip()
    else:
        t2 = t
    try:
        obj = json.loads(t2)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    try:
        t3 = (t2.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r").replace('\\"', '"'))
        obj = json.loads(t3)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = t2.find("{")
    if start != -1:
        depth = 0; in_str = False; esc = False
        for i in range(start, len(t2)):
            ch = t2[i]
            if in_str:
                if esc: esc = False
                elif ch == "\\": esc = True
                elif ch == '"': in_str = False
                continue
            else:
                if ch == '"': in_str = True; continue
                if ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        cand = t2[start : i + 1]
                        try:
                            obj = json.loads(cand)
                            if isinstance(obj, dict):
                                return obj
                        except Exception:
                            break
    facts = _salvage_facts_from_raw(t2)
    if facts:
        return {"facts": facts}
    return {}

# ===================== Routing & enqueue helpers =====================

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

def _enqueue_next(qdb, paths, phrases: List[str], hop: int, max_depth: int):
    if not phrases:
        return
    next_hop = hop + 1
    if next_hop > max_depth:
        return
    enqueue_subjects(qdb, ((s, next_hop) for s in phrases))
    for s in phrases:
        _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": next_hop})

# ===================== OpenAI Batch manager =====================

_STATUS_IN_PROGRESS = {"created", "validating", "in_progress", "finalizing", "parsing"}

def _state_load(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"batches": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"batches": {}}

def _state_save(path: str, obj: Dict[str, Any]):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _build_openai_chat_body(model: str, messages: List[Dict[str,str]], json_schema: Dict[str,Any], max_tokens: int):
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "schema", "schema": json_schema},
        },
    }

def _create_elicitation_batch(client: OpenAI, *, batches_dir: str, el_cfg, subjects: List[Tuple[str,int]], args) -> dict:
    """
    Create a single OpenAI batch from a list of (subject, hop).
    Returns dict with metadata to persist into state.
    """
    if not subjects:
        return {}
    req_lines = []
    subj_index = []
    for subject, hop in subjects:
        user_prompt = _render_elicitation(args.elicitation_strategy, subject, args.max_facts_hint, args.domain, args.topic)
        el_schema = ELICIT_SCHEMA_CAL if (args.elicitation_strategy == "calibrate") else ELICIT_SCHEMA_BASE
        messages = [{"role": "system", "content": "Return JSON only."},
                    {"role": "user", "content": user_prompt}]
        body = _build_openai_chat_body(el_cfg.model, messages, el_schema, el_cfg.max_tokens or 1024)
        req_lines.append(json.dumps({
            "custom_id": subject,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body
        }, ensure_ascii=False))
        subj_index.append({"subject": subject, "hop": hop})

    # write local requests.jsonl
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    req_path = os.path.join(batches_dir, f"elicitation_{ts}_requests.jsonl")
    with open(req_path, "w", encoding="utf-8") as f:
        f.write("\n".join(req_lines) + "\n")

    # upload file & create batch
    up = client.files.create(file=open(req_path, "rb"), purpose="batch")
    bj = client.batches.create(
        input_file_id=up.id,
        endpoint="/v1/chat/completions",
        completion_window=args.openai_batch_window,
        metadata={"description": "Knowledge Elicitation"},
    )

    return {
        "id": bj.id,
        "type": "elicitation",
        "status": bj.status,
        "input_file_id": up.id,
        "output_file_id": None,
        "subjects": subj_index,
        "requests_path": req_path,
        "results_path": None,
    }

def _poll_and_update_batches(client: OpenAI, state: Dict[str, Any]) -> None:
    """Refresh status for in-progress batches; update output_file_id when available."""
    for bid, meta in list(state.get("batches", {}).items()):
        if meta.get("status") in _STATUS_IN_PROGRESS:
            bj = client.batches.retrieve(bid)
            meta["status"] = bj.status
            meta["input_file_id"] = bj.input_file_id
            meta["output_file_id"] = bj.output_file_id or meta.get("output_file_id")

def _download_output_if_ready(client: OpenAI, batches_dir: str, meta: dict) -> Optional[str]:
    if not meta.get("output_file_id"):
        return None
    if meta.get("results_path"):
        return meta["results_path"]
    content = client.files.content(meta["output_file_id"]).read()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = os.path.join(batches_dir, f"{meta['type']}_{ts}_results.jsonl")
    with open(out_path, "wb") as f:
        f.write(content)
    meta["results_path"] = out_path
    return out_path

def _process_completed_elicitation_batch(meta: dict, *, paths, qdb, fdb, el_cfg, ner_cfg, ner_llm, args, write_lock: Lock):
    """
    Parse a completed elicitation batch, write facts, run NER on found phrases,
    enqueue next subjects, and mark current subjects done.
    """
    results_path = meta.get("results_path")
    if not results_path or not os.path.exists(results_path):
        return

    subj_hop = {x["subject"]: x["hop"] for x in meta["subjects"]}

    next_subjects_by_hop: Dict[int, List[str]] = {}
    lowconf_all: List[dict] = []

    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue

            custom_id = item.get("custom_id")
            resp = item.get("response", {})
            choice = ((resp.get("body") or {}).get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content") or ""

            # parse JSON or salvage
            try:
                obj = json.loads(text)
            except Exception:
                obj = _loose_json_from_raw(text)

            norm = _normalize_elicitation_output(obj)
            facts = norm.get("facts", [])
            hop = int(subj_hop.get(custom_id, 0))

            acc, lowconf, objs = _route_facts(args, facts, hop, el_cfg.model)

            # write accepted & logs
            with write_lock:
                write_triples_accepted(fdb, acc)
                for s, p, o, _, m, strat, c in acc:
                    _append_jsonl(paths["facts_jsonl"], {
                        "subject": s, "predicate": p, "object": o,
                        "hop": hop, "model": m, "strategy": strat, "confidence": c,
                        # NEW: persist topic context into logs
                        "domain": args.domain, "topic": args.topic,
                        "root_subject": os.getenv("ROOT_SUBJECT")
                    })
                write_triples_sink(fdb, [])

            if lowconf:
                lowconf_all.extend(lowconf)

            # prepare NER candidates bucketed by next hop
            cand = _filter_ner_candidates(objs)
            if cand:
                next_subjects_by_hop.setdefault(hop + 1, []).extend(cand)

            # mark this subject done
            _mark_done(qdb, custom_id)

    # NER: for each hop, chunk and classify; then enqueue the positives
    for hop, phrases in next_subjects_by_hop.items():
        i = 0
        while i < len(phrases):
            chunk = phrases[i: i + args.ner_batch_size]
            ner_prompt = _render_ner(args.ner_strategy, chunk, args.domain, args.topic)
            if _needs_prompt_sanitization(ner_cfg):
                ner_prompt = _sanitize_prompt_for_llm_contract(ner_prompt)
            ner_schema = NER_SCHEMA_CAL if (args.ner_strategy == "calibrate") else NER_SCHEMA_BASE
            ner_messages = [{"role":"system","content":"Return JSON only."},
                            {"role":"user","content":ner_prompt}]
            out = ner_llm(ner_messages, json_schema=ner_schema)
            if isinstance(out, dict) and "_raw" in out and out["_raw"]:
                loose = _loose_json_from_raw(out["_raw"])
                if loose:
                    out = loose
            norm_ner = _normalize_ner_output(out)

            next_subjects: List[str] = []
            with write_lock:
                for ph in norm_ner.get("phrases", []):
                    phrase = ph.get("phrase")
                    is_ne = bool(ph.get("is_ne"))
                    _append_jsonl(paths["ner_jsonl"], {
                        "parent_subject": "<batch>", "hop": hop - 1,
                        "phrase": phrase, "is_ne": is_ne,
                        "ner_model": ner_cfg.model, "ner_strategy": args.ner_strategy,
                        # NEW: include topic context
                        "domain": args.domain, "topic": args.topic,
                        "root_subject": os.getenv("ROOT_SUBJECT")
                    })
                    if is_ne and isinstance(phrase, str):
                        next_subjects.append(phrase)

            # Note: _enqueue_next computes next_hop = (hop_arg) + 1, so pass hop-1 here.
            _enqueue_next(qdb, paths, next_subjects, hop - 1, args.max_depth)
            i += args.ner_batch_size

    # write low-conf after everything (single file append)
    if lowconf_all:
        with write_lock:
            for item in lowconf_all:
                _append_jsonl(paths["lowconf_jsonl"], item)

def _openai_batch_loop(args, paths, qdb, fdb, el_cfg, ner_cfg, ner_llm):
    """
    Queue manager:
      - maintain limited number of outstanding batches
      - create batches from pending subjects
      - poll statuses and parse just-completed results
      - stop when queue drains (or max-subjects queued)
    """
    client = OpenAI()
    write_lock = Lock()
    state = _state_load(paths["batch_state"])

    subjects_queued_total = 0

    def outstanding_ids() -> List[str]:
        return [bid for bid, meta in state["batches"].items() if meta.get("status") in _STATUS_IN_PROGRESS]

    while True:
        # 1) refresh statuses
        _poll_and_update_batches(client, state)
        _state_save(paths["batch_state"], state)

        # 2) collect completed elicitation batches to parse
        completed_ids = [bid for bid, m in state["batches"].items() if m.get("status") == "completed" and m.get("type") == "elicitation"]
        for bid in completed_ids:
            meta = state["batches"][bid]
            # download results (if not already)
            _download_output_if_ready(client, paths["batches_dir"], meta)
            # parse & commit & NER & enqueue-next
            _process_completed_elicitation_batch(
                meta, paths=paths, qdb=qdb, fdb=fdb,
                el_cfg=el_cfg, ner_cfg=ner_cfg, ner_llm=ner_llm, args=args, write_lock=write_lock
            )
            # mark parsed
            meta["status"] = "parsed"
            _state_save(paths["batch_state"], state)

        # 3) check if we can create new batches
        out_ids = outstanding_ids()
        if len(out_ids) < args.openai_batch_queue:
            slots = args.openai_batch_queue - len(out_ids)
            # claim up to (slots * batch_size) subjects, but submit in batches of batch_size
            max_to_claim = min(
                args.openai_batch_size * slots,
                (args.max_subjects - subjects_queued_total) if args.max_subjects else args.openai_batch_size * slots
            )
            if max_to_claim > 0:
                claims = _fetch_many_pending(qdb, args.max_depth, limit=max_to_claim)
                if claims:
                    # split into chunks of batch_size
                    for i in range(0, len(claims), args.openai_batch_size):
                        chunk = claims[i:i + args.openai_batch_size]
                        meta = _create_elicitation_batch(
                            client,
                            batches_dir=paths["batches_dir"],
                            el_cfg=el_cfg,
                            subjects=chunk,
                            args=args,
                        )
                        if meta:
                            state["batches"][meta["id"]] = meta
                            _state_save(paths["batch_state"], state)
                            subjects_queued_total += len(chunk)
                            if args.max_subjects and subjects_queued_total >= args.max_subjects:
                                break

        # 4) drain logic
        d, w, p, t = _counts(qdb, args.max_depth)
        out_ids = outstanding_ids()
        has_inflight = bool(out_ids)
        has_pending = p > 0
        has_to_parse = any(m.get("status") == "completed" for m in state["batches"].values())
        if not has_inflight and not has_pending and not has_to_parse:
            if t == 0:
                _dbg("[simple] nothing to do.")
            else:
                _dbg(f"[simple] queue drained: done={d} working={w} pending={p} total={t}")
            break

        time.sleep(args.openai_batch_poll)

# ===================== Threaded worker (non-OpenAI path) =====================

def _process_one(args, paths, fdb, el_cfg, ner_cfg, el_llm, ner_llm, subject, hop, write_lock: Lock):
    """Single-subject pipeline; returns (next_subjects, lowconf_items)."""
    # ---------- ELICITATION ----------
    user_prompt = _render_elicitation(args.elicitation_strategy, subject, args.max_facts_hint, args.domain, args.topic)
    if _needs_prompt_sanitization(el_cfg):
        user_prompt = _sanitize_prompt_for_llm_contract(user_prompt)
    if args.debug or _should_debug_prompts():
        print("\n--- ELICITATION PROMPT ---\n", user_prompt, "\n--------------------------\n", flush=True)

    el_schema = ELICIT_SCHEMA_CAL if (args.elicitation_strategy == "calibrate") else ELICIT_SCHEMA_BASE
    el_messages = [
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": user_prompt},
    ]
    resp = el_llm(el_messages, json_schema=el_schema)

    raw_blob = None
    if isinstance(resp, dict) and "_raw" in resp and resp["_raw"]:
        raw_blob = str(resp["_raw"])
        if args.debug:
            print("[debug] elicitation _raw (first 1200):", raw_blob[:1200])
        loose = _loose_json_from_raw(raw_blob)
        if loose:
            resp = loose

    norm = _normalize_elicitation_output(resp)
    if args.debug:
        raw_show = resp if isinstance(resp, str) else json.dumps(resp, ensure_ascii=False)
        print("[debug] raw elicitation resp:", str(raw_show)[:1200])
        print("[debug] normalized facts:", json.dumps(norm, ensure_ascii=False)[:600])

    facts = norm.get("facts", [])
    if (not facts) and raw_blob:
        salvaged = _salvage_facts_from_raw(raw_blob)
        if salvaged:
            facts = salvaged
            if args.debug:
                print(f"[debug] salvaged {len(facts)} facts from truncated output")

    acc, lowconf, objs = _route_facts(args, facts, hop, el_cfg.model)

    with write_lock:
        write_triples_accepted(fdb, acc)
        for s, p, o, _, m, strat, c in acc:
            _append_jsonl(paths["facts_jsonl"], {
                "subject": s, "predicate": p, "object": o,
                "hop": hop, "model": m, "strategy": strat, "confidence": c,
                # NEW: include topic context
                "domain": args.domain, "topic": args.topic,
                "root_subject": os.getenv("ROOT_SUBJECT")
            })
        write_triples_sink(fdb, [])

    # ---------- NER ----------
    cand = _filter_ner_candidates(objs)
    next_subjects: List[str] = []

    i = 0
    while i < len(cand):
        chunk = cand[i: i + args.ner_batch_size]
        ner_prompt = _render_ner(args.ner_strategy, chunk, args.domain, args.topic)
        if _needs_prompt_sanitization(ner_cfg):
            ner_prompt = _sanitize_prompt_for_llm_contract(ner_prompt)

        if args.debug or _should_debug_prompts():
            print("\n--- NER PROMPT ---\n", ner_prompt, "\n------------------\n", flush=True)

        ner_schema = NER_SCHEMA_CAL if (args.ner_strategy == "calibrate") else NER_SCHEMA_BASE
        ner_messages = [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": ner_prompt},
        ]
        out = ner_llm(ner_messages, json_schema=ner_schema)

        if isinstance(out, dict) and "_raw" in out and out["_raw"]:
            loose = _loose_json_from_raw(out["_raw"])
            if loose:
                out = loose

        norm_ner = _normalize_ner_output(out)

        if args.debug:
            raw_show = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
            print("[debug] raw NER resp:", str(raw_show)[:1200])
            print("[debug] normalized NER:", json.dumps(norm_ner, ensure_ascii=False)[:600])

        with write_lock:
            for ph in norm_ner.get("phrases", []):
                phrase = ph.get("phrase")
                is_ne = bool(ph.get("is_ne"))
                _append_jsonl(paths["ner_jsonl"], {
                    "parent_subject": subject, "hop": hop,
                    "phrase": phrase, "is_ne": is_ne,
                    "ner_model": ner_cfg.model, "ner_strategy": args.ner_strategy,
                    # NEW: include topic context
                    "domain": args.domain, "topic": args.topic,
                    "root_subject": os.getenv("ROOT_SUBJECT")
                })
                if is_ne and isinstance(phrase, str):
                    next_subjects.append(phrase)

        i += args.ner_batch_size

    return next_subjects, lowconf

# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser(description="Concurrent crawler with OpenAI Batch + Replicate/Grok/Gemini support.")
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

    # Sampler knobs (non-Responses)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)

    # Responses API extras
    ap.add_argument("--reasoning-effort", choices=["minimal","low","medium","high"], default=None)
    ap.add_argument("--verbosity", choices=["low","medium","high"], default=None)

    # Limits & resume
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reset-working", action="store_true")

    # Concurrency & OpenAI batch
    ap.add_argument("--concurrency", type=int, default=int(os.getenv("CONCURRENCY", settings.CONCURRENCY)))
    ap.add_argument("--openai-batch", action="store_true",
                    help="Use OpenAI Batch API for elicitation when provider is OpenAI.")
    ap.add_argument("--openai-batch-size", type=int, default=10,
                    help="Max subjects per OpenAI batch job.")
    ap.add_argument("--openai-batch-queue", type=int, default=4,
                    help="Max number of outstanding OpenAI batches.")
    ap.add_argument("--openai-batch-window", default="24h",
                    help="OpenAI batch completion window (e.g., 24h).")
    ap.add_argument("--openai-batch-poll", type=int, default=15,
                    help="Seconds between batch status polls.")

    # Debug
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    # if args.domain == "topic" and not args.topic:
    #     raise SystemExit("When --domain topic, you must also pass --topic (e.g., --topic entertainment).")

    # NEW: set ROOT_SUBJECT env for topic-mode so renderers can read it without changing signatures
    if args.domain == "topic":
        os.environ["ROOT_SUBJECT"] = args.seed
    else:
        os.environ.pop("ROOT_SUBJECT", None)

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
        if _needs_prompt_sanitization(cfg) and (cfg.max_tokens is None):
            cfg.max_tokens = 1024

    el_llm = make_llm_from_config(el_cfg)
    ner_llm = make_llm_from_config(ner_cfg)

    start = time.time()
    subjects_elicited_total = 0

    # === mode switch: OpenAI Batch vs threaded ===
    if args.openai_batch and _is_openai(el_cfg):
        _openai_batch_loop(args, paths, qdb, fdb, el_cfg, ner_cfg, ner_llm)
    else:
        # ---- threaded mode ----
        write_lock = Lock()
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            while True:
                if args.max_subjects and subjects_elicited_total >= args.max_subjects:
                    _dbg(f"[simple] max-subjects reached ({subjects_elicited_total}); stopping.")
                    break

                # claim at most 'concurrency' subjects
                batch = _fetch_many_pending(qdb, args.max_depth, limit=args.concurrency)
                if not batch:
                    d, w, p, t = _counts(qdb, args.max_depth)
                    if t == 0:
                        _dbg("[simple] nothing to do.")
                    else:
                        _dbg(f"[simple] queue drained: done={d} working={w} pending={p} total={t}")
                    break

                fut_map = {}
                for (subject, hop) in batch:
                    fut = pool.submit(
                        _process_one,
                        args, paths, fdb, el_cfg, ner_cfg, el_llm, ner_llm,
                        subject, hop, write_lock
                    )
                    fut_map[fut] = (subject, hop)

                for fut in as_completed(fut_map):
                    subject, hop = fut_map[fut]
                    try:
                        next_subjects, lowconf = fut.result()
                        with write_lock:
                            _enqueue_next(qdb, paths, next_subjects, hop, args.max_depth)
                            _mark_done(qdb, subject)
                            for item in lowconf:
                                _append_jsonl(paths["lowconf_jsonl"], item)
                        subjects_elicited_total += 1
                    except KeyboardInterrupt:
                        n = reset_working_to_pending(qdb)
                        print(f"\n[simple] Interrupted. reset {n} 'working' → 'pending' for resume.")
                        raise
                    except Exception:
                        with open(paths["errors_log"], "a", encoding="utf-8") as ef:
                            ef.write(f"[{datetime.datetime.now().isoformat()}] subject={subject}\n{traceback.format_exc()}\n")
                        qdb.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=?", (subject,))
                        qdb.commit()

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

    dur = time.time() - start
    print(f"[simple] finished in {dur:.1f}s → outputs in {out_dir}")
    print(f"[simple] queue.json        : {paths['queue_json']}")
    print(f"[simple] facts.json        : {paths['facts_json']}")
    print(f"[simple] facts.jsonl       : {paths['facts_jsonl']}")
    print(f"[simple] lowconf.json      : {paths['lowconf_json']}")
    print(f"[simple] lowconf.jsonl     : {paths['lowconf_jsonl']}")
    print(f"[simple] ner log           : {paths['ner_jsonl']}")
    print(f"[simple] errors.log        : {paths['errors_log']}")
    print(f"[simple] batch_state       : {paths['batch_state']}")
    print(f"[simple] batches dir       : {paths['batches_dir']}")

if __name__ == "__main__":
    main()

