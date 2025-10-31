# # crawler_simple.py
# from __future__ import annotations
# import argparse
# import datetime
# import json
# import os
# import sqlite3
# import time
# import traceback
# from typing import Dict, List, Tuple

# from concurrent.futures import ThreadPoolExecutor, as_completed
# import threading

# from dotenv import load_dotenv

# from settings import (
#     settings,
#     ELICIT_SCHEMA_BASE,
#     ELICIT_SCHEMA_CAL,
#     NER_SCHEMA_BASE,
#     NER_SCHEMA_CAL,
# )
# from prompter_parser import get_prompt_messages
# from llm.factory import make_llm_from_config
# from db_models import (
#     open_queue_db,
#     open_facts_db,
#     enqueue_subjects,
#     write_triples_accepted,
#     write_triples_sink,
#     queue_has_rows,
#     reset_working_to_pending,
# )

# load_dotenv()


# def _dbg(msg: str):
#     print(msg, flush=True)


# def _ensure_output_dir(base_dir: str | None) -> str:
#     out = base_dir or os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
#     os.makedirs(out, exist_ok=True)
#     return out


# def _build_paths(out_dir: str) -> dict:
#     return {
#         "queue_sqlite": os.path.join(out_dir, "queue.sqlite"),
#         "facts_sqlite": os.path.join(out_dir, "facts.sqlite"),
#         "queue_jsonl": os.path.join(out_dir, "queue.jsonl"),
#         "facts_jsonl": os.path.join(out_dir, "facts.jsonl"),
#         "queue_json": os.path.join(out_dir, "queue.json"),
#         "facts_json": os.path.join(out_dir, "facts.json"),
#         "errors_log": os.path.join(out_dir, "errors.log"),
#         "ner_jsonl": os.path.join(out_dir, "ner_decisions.jsonl"),
#         "lowconf_json": os.path.join(out_dir, "facts_lowconf.json"),
#         "lowconf_jsonl": os.path.join(out_dir, "facts_lowconf.jsonl"),
#     }


# def _append_jsonl(path: str, obj: dict):
#     with open(path, "a", encoding="utf-8") as f:
#         f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# # -------------------- DB helpers --------------------

# def _fetch_one_pending(conn: sqlite3.Connection, max_depth: int) -> Tuple[str, int] | None:
#     cur = conn.cursor()
#     # Deterministic claim (oldest hop first), simple 2-step claim for compatibility
#     cur.execute(
#         "SELECT subject, hop FROM queue WHERE status='pending' AND hop<=? ORDER BY hop ASC, created_at ASC LIMIT 1",
#         (max_depth,)
#     )
#     row = cur.fetchone()
#     if not row:
#         return None
#     s, h = row
#     cur.execute("UPDATE queue SET status='working' WHERE subject=?", (s,))
#     conn.commit()
#     return s, h


# def _fetch_many_pending(conn: sqlite3.Connection, max_depth: int, limit: int) -> List[Tuple[str, int]]:
#     got: List[Tuple[str, int]] = []
#     for _ in range(limit):
#         one = _fetch_one_pending(conn, max_depth)
#         if not one:
#             break
#         got.append(one)
#     return got


# def _mark_done(conn: sqlite3.Connection, subject: str):
#     conn.execute("UPDATE queue SET status='done' WHERE subject=?", (subject,))
#     conn.commit()


# def _counts(conn: sqlite3.Connection, max_depth: int):
#     cur = conn.cursor()
#     cur.execute("SELECT COUNT(1) FROM queue WHERE status='done' AND hop<=?", (max_depth,))
#     done = cur.fetchone()[0]
#     cur.execute("SELECT COUNT(1) FROM queue WHERE status='working' AND hop<=?", (max_depth,))
#     working = cur.fetchone()[0]
#     cur.execute("SELECT COUNT(1) FROM queue WHERE status='pending' AND hop<=?", (max_depth,))
#     pending = cur.fetchone()[0]
#     return done, working, pending, done + working + pending


# # -------------------- Output normalization --------------------

# def _parse_obj(maybe_json) -> dict:
#     if isinstance(maybe_json, dict):
#         return maybe_json
#     if isinstance(maybe_json, str):
#         try:
#             return json.loads(maybe_json)
#         except Exception as e:
#             _dbg(f"[warn] JSON parse failed: {e}; head={maybe_json[:200]!r}")
#             return {}
#     return {}

# def _normalize_elicitation_output(out) -> Dict[str, list]:
#     obj = _parse_obj(out)
#     facts = obj.get("facts")
#     if isinstance(facts, list):
#         return {"facts": [t for t in facts if isinstance(t, dict)]}
#     triples = obj.get("triples")
#     if isinstance(triples, list):
#         return {"facts": [t for t in triples if isinstance(t, dict)]}
#     return {"facts": []}

# def _normalize_ner_output(out) -> Dict[str, list]:
#     obj = _parse_obj(out)
#     if isinstance(obj.get("phrases"), list):
#         got = []
#         for ph in obj["phrases"]:
#             phrase = ph.get("phrase")
#             is_ne = bool(ph.get("is_ne"))
#             if isinstance(phrase, str):
#                 got.append({"phrase": phrase, "is_ne": is_ne})
#         return {"phrases": got}
#     ents = obj.get("entities")
#     if isinstance(ents, list):
#         mapped = []
#         for e in ents:
#             name = e.get("name") or e.get("phrase")
#             etype = (e.get("type") or "").strip().lower()
#             keep = e.get("keep")
#             is_ne = (etype == "ne") or (keep is True)
#             if isinstance(name, str):
#                 mapped.append({"phrase": name, "is_ne": bool(is_ne)})
#         return {"phrases": mapped}
#     return {"phrases": []}

# def _route_facts(args, facts: List[dict], hop: int, model_name: str):
#     acc, lowconf, objs = [], [], []
#     use_threshold = (args.elicitation_strategy == "calibrate")
#     thr = float(args.conf_threshold)

#     for f in facts:
#         s, p, o = f.get("subject"), f.get("predicate"), f.get("object")
#         if not (isinstance(s, str) and isinstance(p, str) and isinstance(o, str)):
#             continue
#         conf = f.get("confidence")

#         if use_threshold and isinstance(conf, (int, float)):
#             if conf < thr:
#                 lowconf.append({
#                     "subject": s, "predicate": p, "object": o,
#                     "hop": hop, "model": model_name,
#                     "strategy": args.elicitation_strategy,
#                     "confidence": float(conf),
#                     "threshold": thr
#                 })
#                 continue

#         acc.append((s, p, o, hop, model_name, args.elicitation_strategy,
#                     conf if isinstance(conf, (int, float)) else None))
#         objs.append(o)

#     return acc, lowconf, objs

# def _filter_ner_candidates(objs: List[str]) -> List[str]:
#     return sorted({o for o in objs if isinstance(o, str) and 1 <= len(o.split()) <= 6})

# def _enqueue_next(qdb, paths, phrases: List[str], hop: int, max_depth: int):
#     if not phrases:
#         return
#     next_hop = hop + 1
#     if next_hop > max_depth:
#         return
#     enqueue_subjects(qdb, ((s, next_hop) for s in phrases))
#     for s in phrases:
#         _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": next_hop})


# # -------------------- Provider helpers --------------------

# def _is_openai_model(cfg) -> bool:
#     prov = (getattr(cfg, "provider", "") or "").lower()
#     if "openai" in prov:
#         return True
#     name = (getattr(cfg, "model", "") or "").lower()
#     return "openai" in name or name.startswith("gpt-")


# def _build_elicitation_messages(args, subject: str) -> List[dict]:
#     return get_prompt_messages(
#         args.elicitation_strategy, "elicitation",
#         domain=args.domain,
#         variables=dict(
#             subject_name=subject,
#             root_subject=args.seed,          # use seed as the topic anchor when domain == "topic"
#             max_facts_hint=args.max_facts_hint,
#         ),
#     )


# # -------------------- Main --------------------

# def main():
#     ap = argparse.ArgumentParser(description="Simple crawler (system+user prompts from JSON).")
#     ap.add_argument("--seed", required=True)
#     ap.add_argument("--output-dir", default=None)

#     # Strategies / domain
#     ap.add_argument("--elicitation-strategy", default="baseline", choices=["baseline","icl","dont_know","calibrate"])
#     ap.add_argument("--ner-strategy", default="baseline", choices=["baseline","icl","dont_know","calibrate"])
#     ap.add_argument("--domain", default="general", choices=["general","topic"])

#     # Depth / batching
#     ap.add_argument("--max-depth", type=int, default=settings.MAX_DEPTH)
#     ap.add_argument("--ner-batch-size", type=int, default=settings.NER_BATCH_SIZE)
#     ap.add_argument("--max-facts-hint", default=str(settings.MAX_FACTS_HINT))
#     ap.add_argument("--conf-threshold", type=float, default=0.7)

#     # Models
#     ap.add_argument("--elicit-model-key", default=settings.ELICIT_MODEL_KEY)
#     ap.add_argument("--ner-model-key", default=settings.NER_MODEL_KEY)

#     # Sampler knobs (for non-Responses models)
#     ap.add_argument("--temperature", type=float, default=None)
#     ap.add_argument("--top-p", type=float, default=None)
#     ap.add_argument("--top-k", type=int, default=None)
#     ap.add_argument("--max-tokens", type=int, default=None)

#     # Hard cap
#     ap.add_argument("--max-subjects", type=int, default=0)

#     # Responses API extras
#     ap.add_argument("--reasoning-effort", choices=["minimal","low","medium","high"], default=None)
#     ap.add_argument("--verbosity", choices=["low","medium","high"], default=None)

#     # Resume
#     ap.add_argument("--resume", action="store_true")
#     ap.add_argument("--reset-working", action="store_true")

#     # Debug
#     ap.add_argument("--debug", action="store_true")

#     # NEW: Batching / concurrency controls
#     ap.add_argument("--batch-size", type=int, default=None, help="OpenAI-only: number of subjects per batch request")
#     ap.add_argument("--concurrency", type=int, default=None, help="Non-OpenAI: max parallel elicitation calls")
#     ap.add_argument("--max-inflight", type=int, default=None, help="Upper bound on outstanding elicitation calls")
#     ap.add_argument("--timeout", type=float, default=90.0, help="Per-request timeout in seconds")

#     args = ap.parse_args()

#     out_dir = _ensure_output_dir(args.output_dir)
#     paths = _build_paths(out_dir)
#     _dbg(f"[simple] output_dir: {out_dir}")

#     # DBs
#     qdb = open_queue_db(paths["queue_sqlite"])
#     fdb = open_facts_db(paths["facts_sqlite"])

#     # Seed or resume
#     if args.resume and queue_has_rows(qdb):
#         if args.reset_working:
#             n = reset_working_to_pending(qdb)
#             _dbg(f"[simple] resume: reset {n} 'working' → 'pending'")
#         d0, w0, p0, t0 = _counts(qdb, args.max_depth)
#         _dbg(f"[simple] resume: queue found: done={d0} working={w0} pending={p0} total={t0}")
#     else:
#         enqueue_subjects(qdb, [(args.seed, 0)])
#         _dbg(f"[simple] seeded: {args.seed}")

#     # Build LLMs
#     el_cfg = settings.MODELS[args.elicit_model_key].model_copy(deep=True)
#     ner_cfg = settings.MODELS[args.ner_model_key].model_copy(deep=True)

#     # Respect per-provider rules
#     for cfg in (el_cfg, ner_cfg):
#         if getattr(cfg, "use_responses_api", False):
#             cfg.temperature = None
#             cfg.top_p = None
#             cfg.top_k = None
#             if cfg.extra_inputs is None:
#                 cfg.extra_inputs = {}
#             cfg.extra_inputs.setdefault("reasoning", {})
#             cfg.extra_inputs.setdefault("text", {})
#             if args.reasoning_effort:
#                 cfg.extra_inputs["reasoning"]["effort"] = args.reasoning_effort
#             if args.verbosity:
#                 cfg.extra_inputs["text"]["verbosity"] = args.verbosity
#         else:
#             if args.temperature is not None: cfg.temperature = args.temperature
#             if args.top_p is not None: cfg.top_p = args.top_p
#             if args.top_k is not None: cfg.top_k = args.top_k
#         if args.max_tokens is not None:
#             cfg.max_tokens = args.max_tokens
#         if cfg.max_tokens is None:
#             cfg.max_tokens = 2048

#         # Try to push timeout to wrappers that support it
#         if hasattr(cfg, "request_timeout"):
#             cfg.request_timeout = args.timeout
#         elif hasattr(cfg, "timeout"):
#             cfg.timeout = args.timeout

#     el_llm = make_llm_from_config(el_cfg)
#     ner_llm = make_llm_from_config(ner_cfg)

#     # Validate batching/concurrency depending on provider
#     is_openai_el = _is_openai_model(el_cfg)
#     if is_openai_el:
#         if not args.batch_size or args.batch_size <= 0:
#             raise SystemExit("--batch-size is required and must be > 0 when elicitation model is OpenAI")
#     else:
#         if not args.concurrency or args.concurrency <= 0:
#             raise SystemExit("--concurrency is required and must be > 0 when elicitation model is not OpenAI")

#     if args.max_inflight is None:
#         args.max_inflight = (args.batch_size if is_openai_el else args.concurrency)

#     start = time.time()
#     subjects_elicited_total = 0
#     lowconf_accum: List[dict] = []

#     lock = threading.Lock()  # reserved if you later move DB writes into worker threads

#     while True:
#         if args.max_subjects and subjects_elicited_total >= args.max_subjects:
#             _dbg(f"[simple] max-subjects reached ({subjects_elicited_total}); stopping.")
#             break

#         # Decide how many to pick this iteration
#         claim_n = min(args.max_inflight, (args.batch_size if is_openai_el else args.concurrency))

#         # Claim work
#         batch = _fetch_many_pending(qdb, args.max_depth, claim_n)
#         if not batch:
#             d, w, p, t = _counts(qdb, args.max_depth)
#             if t == 0:
#                 _dbg("[simple] nothing to do.")
#             else:
#                 _dbg(f"[simple] queue drained: done={d} working={w} pending={p} total={t}")
#             break

#         if is_openai_el:
#             # -------- OpenAI batched path --------
#             subjects = [s for (s, _hop) in batch]
#             hops     = [h for (_s, h) in batch]
#             _dbg(f"[simple] OpenAI batch: {len(subjects)} subjects")

#             # Build per-item messages
#             messages_list = [_build_elicitation_messages(args, s) for s in subjects]
#             el_schema = ELICIT_SCHEMA_CAL if (args.elicitation_strategy == "calibrate") else ELICIT_SCHEMA_BASE

#             # Try true batch call first; fall back to per-item calls if not supported
#             try:
#                 if hasattr(el_llm, "batch"):
#                     try:
#                         resp_list = el_llm.batch(messages_list, json_schema=el_schema, timeout=args.timeout)
#                     except TypeError:
#                         resp_list = el_llm.batch(messages_list, json_schema=el_schema)
#                 else:
#                     resp_list = []
#                     for msgs in messages_list:
#                         try:
#                             resp_list.append(el_llm(msgs, json_schema=el_schema, timeout=args.timeout))
#                         except TypeError:
#                             resp_list.append(el_llm(msgs, json_schema=el_schema))
#             except Exception:
#                 # If the whole batch call fails, revert all to pending and continue
#                 for subject, _hop in batch:
#                     qdb.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=?", (subject,))
#                 qdb.commit()
#                 _dbg("[warn] OpenAI batch call failed; reverted claimed items to pending.")
#                 continue

#             # Handle each result sequentially (DB writes on main thread)
#             for (subject, hop), resp in zip(batch, resp_list):
#                 try:
#                     obj = resp if isinstance(resp, dict) else _parse_obj(resp)
#                     facts = []
#                     if isinstance(obj.get("facts"), list):
#                         facts = [t for t in obj["facts"] if isinstance(t, dict)]
#                     elif isinstance(obj.get("triples"), list):
#                         facts = [t for t in obj["triples"] if isinstance(t, dict)]

#                     acc, lowconf, _objs = _route_facts(args, facts, hop, el_cfg.model)
#                     write_triples_accepted(fdb, acc)

#                     for s, p, o, _, m, strat, c in acc:
#                         _append_jsonl(paths["facts_jsonl"], {
#                             "subject": s, "predicate": p, "object": o,
#                             "hop": hop, "model": m, "strategy": strat, "confidence": c
#                         })

#                     if lowconf:
#                         for item in lowconf:
#                             _append_jsonl(paths["lowconf_jsonl"], item)
#                         lowconf_accum.extend(lowconf)

#                     # ---------- NER ----------
#                     cand = _filter_ner_candidates([t.get("object") for t in facts if isinstance(t, dict)])
#                     next_subjects: List[str] = []
#                     i = 0
#                     while i < len(cand):
#                         chunk = cand[i: i + args.ner_batch_size]
#                         ner_messages = get_prompt_messages(
#                             args.ner_strategy, "ner",
#                             domain=args.domain,
#                             variables=dict(
#                                 phrases_block="\n".join(chunk),
#                                 root_subject=args.seed,
#                             ),
#                         )
#                         ner_schema = NER_SCHEMA_CAL if (args.ner_strategy == "calibrate") else NER_SCHEMA_BASE
#                         out = ner_llm(ner_messages, json_schema=ner_schema)
#                         norm_ner = _normalize_ner_output(out)
#                         for ph in norm_ner.get("phrases", []):
#                             phrase = ph.get("phrase")
#                             is_ne = bool(ph.get("is_ne"))
#                             _append_jsonl(paths["ner_jsonl"], {
#                                 "parent_subject": subject, "hop": hop,
#                                 "phrase": phrase, "is_ne": is_ne,
#                                 "ner_model": ner_cfg.model, "ner_strategy": args.ner_strategy,
#                                 "domain": args.domain, "root_subject": args.seed if (args.domain == "topic") else None,
#                             })
#                             if is_ne and isinstance(phrase, str):
#                                 next_subjects.append(phrase)
#                         i += args.ner_batch_size

#                     _enqueue_next(qdb, paths, next_subjects, hop, args.max_depth)
#                     _mark_done(qdb, subject)
#                     subjects_elicited_total += 1

#                     if args.max_subjects and subjects_elicited_total >= args.max_subjects:
#                         _dbg(f"[simple] max-subjects reached ({subjects_elicited_total}); stopping.")
#                         break

#                 except Exception:
#                     with open(paths["errors_log"], "a", encoding="utf-8") as ef:
#                         ef.write(f"[{datetime.datetime.now().isoformat()}] subject={subject}\n{traceback.format_exc()}\n")
#                     qdb.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=?", (subject,))
#                     qdb.commit()

#         else:
#             # -------- Non-OpenAI concurrent path --------
#             _dbg(f"[simple] concurrent elicitation: {len(batch)} subjects, workers={args.concurrency}")

#             def _worker_call(inp):
#                 subject, hop = inp
#                 try:
#                     msgs = _build_elicitation_messages(args, subject)
#                     el_schema = ELICIT_SCHEMA_CAL if (args.elicitation_strategy == "calibrate") else ELICIT_SCHEMA_BASE
#                     try:
#                         resp = el_llm(msgs, json_schema=el_schema, timeout=args.timeout)
#                     except TypeError:
#                         resp = el_llm(msgs, json_schema=el_schema)
#                     return (subject, hop, None, resp)
#                 except Exception as e:
#                     return (subject, hop, e, None)

#             results = []
#             with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
#                 fut_map = {pool.submit(_worker_call, item): item for item in batch}
#                 for fut in as_completed(fut_map):
#                     try:
#                         results.append(fut.result())
#                     except Exception as e:
#                         # Shouldn't happen because worker catches, but guard anyway
#                         subject, hop = fut_map[fut]
#                         results.append((subject, hop, e, None))

#             # Serialize DB writes
#             for (subject, hop, err, resp) in results:
#                 if err is not None:
#                     with open(paths["errors_log"], "a", encoding="utf-8") as ef:
#                         ef.write(f"[{datetime.datetime.now().isoformat()}] subject={subject}\n{traceback.format_exc()}\n")
#                     qdb.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=?", (subject,))
#                     qdb.commit()
#                     continue

#                 try:
#                     obj = resp if isinstance(resp, dict) else _parse_obj(resp)
#                     facts = []
#                     if isinstance(obj.get("facts"), list):
#                         facts = [t for t in obj["facts"] if isinstance(t, dict)]
#                     elif isinstance(obj.get("triples"), list):
#                         facts = [t for t in obj["triples"] if isinstance(t, dict)]

#                     acc, lowconf, _objs = _route_facts(args, facts, hop, el_cfg.model)
#                     write_triples_accepted(fdb, acc)

#                     for s, p, o, _, m, strat, c in acc:
#                         _append_jsonl(paths["facts_jsonl"], {
#                             "subject": s, "predicate": p, "object": o,
#                             "hop": hop, "model": m, "strategy": strat, "confidence": c
#                         })

#                     if lowconf:
#                         for item in lowconf:
#                             _append_jsonl(paths["lowconf_jsonl"], item)
#                         lowconf_accum.extend(lowconf)

#                     # ---------- NER ----------
#                     cand = _filter_ner_candidates([t.get("object") for t in facts if isinstance(t, dict)])
#                     next_subjects: List[str] = []
#                     i = 0
#                     while i < len(cand):
#                         chunk = cand[i: i + args.ner_batch_size]
#                         ner_messages = get_prompt_messages(
#                             args.ner_strategy, "ner",
#                             domain=args.domain,
#                             variables=dict(
#                                 phrases_block="\n".join(chunk),
#                                 root_subject=args.seed,
#                             ),
#                         )
#                         ner_schema = NER_SCHEMA_CAL if (args.ner_strategy == "calibrate") else NER_SCHEMA_BASE
#                         out = ner_llm(ner_messages, json_schema=ner_schema)
#                         norm_ner = _normalize_ner_output(out)
#                         for ph in norm_ner.get("phrases", []):
#                             phrase = ph.get("phrase")
#                             is_ne = bool(ph.get("is_ne"))
#                             _append_jsonl(paths["ner_jsonl"], {
#                                 "parent_subject": subject, "hop": hop,
#                                 "phrase": phrase, "is_ne": is_ne,
#                                 "ner_model": ner_cfg.model, "ner_strategy": args.ner_strategy,
#                                 "domain": args.domain, "root_subject": args.seed if (args.domain == "topic") else None,
#                             })
#                             if is_ne and isinstance(phrase, str):
#                                 next_subjects.append(phrase)
#                         i += args.ner_batch_size

#                     _enqueue_next(qdb, paths, next_subjects, hop, args.max_depth)
#                     _mark_done(qdb, subject)
#                     subjects_elicited_total += 1

#                     if args.max_subjects and subjects_elicited_total >= args.max_subjects:
#                         _dbg(f"[simple] max-subjects reached ({subjects_elicited_total}); stopping.")
#                         break

#                 except Exception:
#                     with open(paths["errors_log"], "a", encoding="utf-8") as ef:
#                         ef.write(f"[{datetime.datetime.now().isoformat()}] subject={subject}\n{traceback.format_exc()}\n")
#                     qdb.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=?", (subject,))
#                     qdb.commit()

#     # ----- Final snapshots -----
#     conn = sqlite3.connect(paths["queue_sqlite"])
#     cur = conn.cursor()
#     cur.execute("SELECT subject, hop, status, retries, created_at FROM queue ORDER BY hop, subject")
#     rows = cur.fetchall()
#     with open(paths["queue_json"], "w", encoding="utf-8") as f:
#         json.dump(
#             [{"subject": s, "hop": h, "status": st, "retries": r, "created_at": ts} for (s, h, st, r, ts) in rows],
#             f, ensure_ascii=False, indent=2
#         )
#     conn.close()

#     conn = sqlite3.connect(paths["facts_sqlite"])
#     cur = conn.cursor()
#     cur.execute(
#         "SELECT subject, predicate, object, hop, model_name, strategy, confidence "
#         "FROM triples_accepted ORDER BY subject, predicate, object"
#     )
#     rows_acc = cur.fetchall()
#     cur.execute(
#         "SELECT subject, predicate, object, hop, model_name, strategy, confidence, reason "
#         "FROM triples_sink ORDER BY subject, predicate, object"
#     )
#     rows_sink = cur.fetchall()
#     with open(paths["facts_json"], "w", encoding="utf-8") as f:
#         json.dump(
#             {
#                 "accepted": [
#                     {"subject": s, "predicate": p, "object": o, "hop": h,
#                      "model": m, "strategy": st, "confidence": c}
#                     for (s, p, o, h, m, st, c) in rows_acc
#                 ],
#                 "sink": [
#                     {"subject": s, "predicate": p, "object": o, "hop": h,
#                      "model": m, "strategy": st, "confidence": c, "reason": r}
#                     for (s, p, o, h, m, st, c, r) in rows_sink
#                 ],
#             },
#             f, ensure_ascii=False, indent=2
#         )
#     conn.close()

#     with open(paths["lowconf_json"], "w", encoding="utf-8") as f:
#         json.dump({"below_threshold": lowconf_accum}, f, ensure_ascii=False, indent=2)

#     dur = time.time() - start
#     print(f"[simple] finished in {dur:.1f}s → outputs in {out_dir}")
#     print(f"[simple] queue.json        : {paths['queue_json']}")
#     print(f"[simple] facts.json        : {paths['facts_json']}")
#     print(f"[simple] facts.jsonl       : {paths['facts_jsonl']}")
#     print(f"[simple] lowconf.json      : {paths['lowconf_json']}")
#     print(f"[simple] lowconf.jsonl     : {paths['lowconf_jsonl']}")
#     print(f"[simple] ner log           : {paths['ner_jsonl']}")
#     print(f"[simple] errors.log        : {paths['errors_log']}")

# if __name__ == "__main__":
#     main()

# crawler_simple.py
from __future__ import annotations
import argparse
import datetime
import json
import os
import sqlite3
import time
import traceback
from typing import Dict, List, Tuple, Iterable, Set

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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


# -------------------- DB helpers --------------------

def _fetch_one_pending(conn: sqlite3.Connection, max_depth: int) -> Tuple[str, int] | None:
    """
    Atomically claim the oldest pending item (hop ASC, created_at ASC) by flipping status to 'working'.
    Uses UPDATE ... RETURNING on SQLite >= 3.35. Fallback grabs a write lock via BEGIN IMMEDIATE.
    """
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
        # Older SQLite fallback: take a write lock so SELECT+UPDATE is effectively atomic.
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
    # Harden: only move items that are currently 'working' to 'done'
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


# -------------------- Output normalization --------------------

def _parse_obj(maybe_json) -> dict:
    if isinstance(maybe_json, dict):
        return maybe_json
    if isinstance(maybe_json, str):
        try:
            return json.loads(maybe_json)
        except Exception as e:
            _dbg(f"[warn] JSON parse failed: {e}; head={maybe_json[:200]!r}")
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


# -------------------- Provider helpers --------------------

def _is_openai_model(cfg) -> bool:
    prov = (getattr(cfg, "provider", "") or "").lower()
    if "openai" in prov:
        return True
    name = (getattr(cfg, "model", "") or "").lower()
    return "openai" in name or name.startswith("gpt-")


def _build_elicitation_messages(args, subject: str) -> List[dict]:
    return get_prompt_messages(
        args.elicitation_strategy, "elicitation",
        domain=args.domain,
        variables=dict(
            subject_name=subject,
            root_subject=args.seed,          # use seed as the topic anchor when domain == "topic"
            max_facts_hint=args.max_facts_hint,
        ),
    )


# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser(description="Simple crawler (system+user prompts from JSON).")
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

    # NEW: Batching / concurrency controls
    ap.add_argument("--batch-size", type=int, default=None, help="OpenAI-only: number of subjects per batch request")
    ap.add_argument("--concurrency", type=int, default=None, help="Non-OpenAI: max parallel elicitation calls")
    ap.add_argument("--max-inflight", type=int, default=None, help="Upper bound on outstanding elicitation calls")
    ap.add_argument("--timeout", type=float, default=90.0, help="Per-request timeout in seconds")

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

        # Try to push timeout to wrappers that support it
        if hasattr(cfg, "request_timeout"):
            cfg.request_timeout = args.timeout
        elif hasattr(cfg, "timeout"):
            cfg.timeout = args.timeout

    el_llm = make_llm_from_config(el_cfg)
    ner_llm = make_llm_from_config(ner_cfg)

    # Validate batching/concurrency depending on provider
    is_openai_el = _is_openai_model(el_cfg)
    if is_openai_el:
        if not args.batch_size or args.batch_size <= 0:
            raise SystemExit("--batch-size is required and must be > 0 when elicitation model is OpenAI")
    else:
        if not args.concurrency or args.concurrency <= 0:
            raise SystemExit("--concurrency is required and must be > 0 when elicitation model is not OpenAI")

    if args.max_inflight is None:
        args.max_inflight = (args.batch_size if is_openai_el else args.concurrency)

    start = time.time()
    subjects_elicited_total = 0
    lowconf_accum: List[dict] = []

    lock = threading.Lock()  # reserved if you later move DB writes into worker threads

    # Run-level JSONL dedupe (keeps facts.jsonl tidy even if a subject is retried)
    seen_facts: Set[Tuple[str, str, str, int]] = set()

    while True:
        if args.max_subjects and subjects_elicited_total >= args.max_subjects:
            _dbg(f"[simple] max-subjects reached ({subjects_elicited_total}); stopping.")
            break

        # Decide how many to pick this iteration
        claim_n = min(args.max_inflight, (args.batch_size if is_openai_el else args.concurrency))

        # Claim work
        batch = _fetch_many_pending(qdb, args.max_depth, claim_n)
        if not batch:
            d, w, p, t = _counts(qdb, args.max_depth)
            if t == 0:
                _dbg("[simple] nothing to do.")
            else:
                _dbg(f"[simple] queue drained: done={d} working={w} pending={p} total={t}")
            break

        if is_openai_el:
            # -------- OpenAI batch: single-threaded claim, no duplication issue --------
            subjects = [s for (s, _hop) in batch]
            hops     = [h for (_s, h) in batch]
            _dbg(f"[simple] OpenAI batch: {len(subjects)} subjects")

            # Build per-item messages
            messages_list = [_build_elicitation_messages(args, s) for s in subjects]
            el_schema = ELICIT_SCHEMA_CAL if (args.elicitation_strategy == "calibrate") else ELICIT_SCHEMA_BASE

            # Try true batch call first; fall back to per-item calls if not supported
            try:
                if hasattr(el_llm, "batch"):
                    try:
                        resp_list = el_llm.batch(messages_list, json_schema=el_schema, timeout=args.timeout)
                    except TypeError:
                        resp_list = el_llm.batch(messages_list, json_schema=el_schema)
                else:
                    resp_list = []
                    for msgs in messages_list:
                        try:
                            resp_list.append(el_llm(msgs, json_schema=el_schema, timeout=args.timeout))
                        except TypeError:
                            resp_list.append(el_llm(msgs, json_schema=el_schema))
            except Exception:
                # If the whole batch call fails, revert all to pending and continue
                for subject, _hop in batch:
                    qdb.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=?", (subject,))
                qdb.commit()
                _dbg("[warn] OpenAI batch call failed; reverted claimed items to pending.")
                continue

            # Handle each result sequentially (DB writes on main thread)
            for (subject, hop), resp in zip(batch, resp_list):
                try:
                    obj = resp if isinstance(resp, dict) else _parse_obj(resp)
                    facts = []
                    if isinstance(obj.get("facts"), list):
                        facts = [t for t in obj["facts"] if isinstance(t, dict)]
                    elif isinstance(obj.get("triples"), list):
                        facts = [t for t in obj["triples"] if isinstance(t, dict)]

                    acc, lowconf, _objs = _route_facts(args, facts, hop, el_cfg.model)
                    write_triples_accepted(fdb, acc)

                    # Append JSONL once per unique (s,p,o,hop) for this run
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

                    _enqueue_next(qdb, paths, next_subjects, hop, args.max_depth)
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
            # -------- Non-OpenAI concurrent path (fixed) --------
            _dbg(f"[simple] concurrent elicitation: {len(batch)} subjects, workers={min(args.concurrency, len(batch))}")

            def _worker_call(inp):
                subject, hop = inp
                try:
                    msgs = _build_elicitation_messages(args, subject)
                    el_schema = ELICIT_SCHEMA_CAL if (args.elicitation_strategy == "calibrate") else ELICIT_SCHEMA_BASE
                    try:
                        resp = el_llm(msgs, json_schema=el_schema, timeout=args.timeout)
                    except TypeError:
                        resp = el_llm(msgs, json_schema=el_schema)
                    return (subject, hop, None, resp)
                except Exception as e:
                    return (subject, hop, e, None)

            results = []
            # Right-size the pool to the actual batch size to reduce contention
            with ThreadPoolExecutor(max_workers=min(args.concurrency, len(batch))) as pool:
                fut_map = {pool.submit(_worker_call, item): item for item in batch}
                for fut in as_completed(fut_map):
                    try:
                        results.append(fut.result())
                    except Exception as e:
                        # Shouldn't happen because worker catches, but guard anyway
                        subject, hop = fut_map[fut]
                        results.append((subject, hop, e, None))

            # Serialize DB writes
            for (subject, hop, err, resp) in results:
                if err is not None:
                    with open(paths["errors_log"], "a", encoding="utf-8") as ef:
                        ef.write(f"[{datetime.datetime.now().isoformat()}] subject={subject}\n{traceback.format_exc()}\n")
                    qdb.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=?", (subject,))
                    qdb.commit()
                    continue

                try:
                    obj = resp if isinstance(resp, dict) else _parse_obj(resp)
                    facts = []
                    if isinstance(obj.get("facts"), list):
                        facts = [t for t in obj["facts"] if isinstance(t, dict)]
                    elif isinstance(obj.get("triples"), list):
                        facts = [t for t in obj["triples"] if isinstance(t, dict)]

                    acc, lowconf, _objs = _route_facts(args, facts, hop, el_cfg.model)
                    write_triples_accepted(fdb, acc)

                    # Append JSONL once per unique (s,p,o,hop) for this run
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
                    try:
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
                    except Exception:
                        with open(paths["errors_log"], "a", encoding="utf-8") as ef:
                            ef.write(f"[{datetime.datetime.now().isoformat()}] NER failed for subject={subject}\n{traceback.format_exc()}\n")

                    _enqueue_next(qdb, paths, next_subjects, hop, args.max_depth)
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
    main()
