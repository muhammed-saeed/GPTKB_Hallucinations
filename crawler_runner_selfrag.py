# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Resilient GPTKB-style crawler with optional Self-RAG and recovery paths.

# Key improvements:
# - Auto-detect elicitation/NER models if model keys are missing in settings.
# - If elicitation yields 0 facts, try: (a) relaxed fallback prompt, (b) optional last-chance Self-RAG grounding.
# - If schema call returns empty, automatically retry once without schema.
# - Persist raw completions for debugging to runs/.../tmp/elicitation_raw_<subject>.txt
# """

# from __future__ import annotations

# import argparse, datetime, json, os, re, sqlite3, threading, time, traceback
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from typing import Dict, List, Tuple, Set, Optional

# from dotenv import load_dotenv
# load_dotenv()

# # ---------- locks & tiny utils ----------
# _jsonl_lock = threading.Lock()
# _seen_facts_lock = threading.Lock()
# _lowconf_lock = threading.Lock()
# _ner_lowconf_lock = threading.Lock()

# def _append_jsonl(path: str, obj: dict):
#     line = json.dumps(obj, ensure_ascii=False) + "\n"
#     with _jsonl_lock:
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         with open(path, "a", encoding="utf-8") as f:
#             f.write(line)

# def _dbg(msg: str): print(msg, flush=True)

# def _print_messages(tag: str, msgs: List[dict], limit: int | None = None):
#     print(f"\n--- {tag} MESSAGES ({len(msgs)}) ---")
#     for i, m in enumerate(msgs, 1):
#         role = (m.get("role") or "").upper()
#         content = m.get("content")
#         if isinstance(content, str) and limit:
#             content = (content[:limit] + "…") if len(content) > limit else content
#         print(f"[{i:02d}] {role}: {content if isinstance(content, str) else content}")
#     print(f"--- END {tag} ---\n")

# def _print_enqueue_summary(results: List[Tuple[str,int,str]]):
#     if not results:
#         print("[enqueue] (no results)")
#         return
#     ins = sum(1 for *_r, out in results if out == "inserted")
#     red = sum(1 for *_r, out in results if out == "hop_reduced")
#     ign = sum(1 for *_r, out in results if out == "ignored")
#     print(f"[enqueue] inserted={ins} hop_reduced={red} ignored={ign}")

# # ---------- repo imports ----------
# from processing_queue import (
#     init_cache as procq_init_cache,
#     enqueue_subjects_processed as procq_enqueue,
#     DEFAULT_LEADING_ARTICLES as PROCQ_LEADING,
#     get_thread_queue_conn as procq_get_thread_conn,
# )
# from settings import (
#     settings,
#     ELICIT_SCHEMA_BASE, ELICIT_SCHEMA_CAL,
#     NER_SCHEMA_BASE,   NER_SCHEMA_CAL,
# )
# from prompter_parser import get_prompt_messages
# from llm.factory import make_llm_from_config
# from db_models import (
#     open_queue_db, open_facts_db,
#     write_triples_accepted, write_triples_sink,
#     queue_has_rows, reset_working_to_pending,
# )

# # JSON extractor
# from llm.json_utils import best_json

# # ---------- paths ----------
# def _ensure_output_dir(base_dir: Optional[str]) -> str:
#     out = base_dir or os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
#     os.makedirs(out, exist_ok=True)
#     return out

# def _build_paths(out_dir: str) -> dict:
#     tmp = os.path.join(out_dir, "tmp")
#     os.makedirs(tmp, exist_ok=True)
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
#         "ner_lowconf_jsonl": os.path.join(out_dir, "ner_lowconf.jsonl"),
#         "ner_lowconf_json": os.path.join(out_dir, "ner_lowconf.json"),
#         "run_meta_json": os.path.join(out_dir, "run_meta.json"),
#         "tmp_dir": tmp,
#     }

# # ---------- per-thread sqlite ----------
# _thread_local = threading.local()

# def get_thread_queue_conn(db_path: str) -> sqlite3.Connection:
#     return procq_get_thread_conn(db_path)

# def get_thread_facts_conn(db_path: str) -> sqlite3.Connection:
#     key = f"facts_conn__{db_path}"
#     conn = getattr(_thread_local, key, None)
#     if conn is None:
#         conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
#         conn.execute("PRAGMA journal_mode=WAL;")
#         conn.execute("PRAGMA synchronous=NORMAL;")
#         conn.execute("PRAGMA busy_timeout=5000;")
#         conn.execute("PRAGMA temp_store=MEMORY;")
#         setattr(_thread_local, key, conn)
#     return conn

# def mark_done_threadsafe(queue_db_path: str, subject: str, hop: int):
#     conn = get_thread_queue_conn(queue_db_path)
#     with conn:
#         conn.execute("UPDATE queue SET status='done' WHERE subject=? AND hop=? AND status='working'", (subject, hop))

# def mark_pending_on_error(queue_db_path: str, subject: str, hop: int):
#     conn = get_thread_queue_conn(queue_db_path)
#     with conn:
#         conn.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=? AND hop=? AND status='working'", (subject, hop))

# # ---------- unwrap & salvage ----------
# def _unwrap_text(resp):
#     if isinstance(resp, str): return resp
#     if isinstance(resp, dict):
#         for k in ("text","output_text","content","message","response"):
#             v = resp.get(k)
#             if isinstance(v, str): return v
#         ch = resp.get("choices")
#         if isinstance(ch, list) and ch:
#             c0 = ch[0] or {}
#             msg = c0.get("message") or {}
#             if isinstance(msg, dict) and isinstance(msg.get("content"), str):
#                 return msg["content"]
#             if isinstance(c0.get("text"), str): return c0["text"]
#         if isinstance(resp.get("_raw"), str): return resp["_raw"]
#         if isinstance(resp.get("raw"), str):  return resp["raw"]
#         if isinstance(resp.get("raw"), dict): return _unwrap_text(resp["raw"])
#     return ""

# def _extract_json_block(text: str):
#     obj = best_json(text)
#     return obj if isinstance(obj, (dict, list)) else {}

# def _normalize_fact_keys(d: dict) -> dict | None:
#     if not isinstance(d, dict): return None
#     key_map = {
#         "subject": ["subject","subj","s","head","h"],
#         "predicate": ["predicate","pred","p","relation","rel","r"],
#         "object": ["object","obj","o","tail","t","value","val"],
#         "confidence": ["confidence","conf","c","score","prob"]
#     }
#     out = {}
#     for std, alts in key_map.items():
#         for k in alts:
#             if k in d and isinstance(d[k], (str, float, int)):
#                 out[std] = d[k]; break
#     s,p,o = out.get("subject"), out.get("predicate"), out.get("object")
#     if not (isinstance(s,str) and isinstance(p,str) and isinstance(o,str)): return None
#     if "confidence" in out:
#         try: out["confidence"] = float(out["confidence"])
#         except Exception: out["confidence"] = None
#     else:
#         out["confidence"] = None
#     return out

# _TRIPLE_OBJ_RX = re.compile(r"\{[^{}]*?(\"subject\"|\"subj\"|\"s\"|\"head\")[^{}]*?\}", re.I)
# _FLEX_TRIPLE_RX = re.compile(r"\{[^{}]*\}", re.S)

# def _salvage_facts_from_text(text: str) -> List[dict]:
#     salvaged: List[dict] = []
#     obj = _extract_json_block(text)
#     if obj:
#         if isinstance(obj, dict):
#             for key in ("facts","triples"):
#                 val = obj.get(key)
#                 if isinstance(val, list):
#                     for item in val:
#                         norm = _normalize_fact_keys(item)
#                         if norm: salvaged.append(norm)
#             if not salvaged:
#                 norm = _normalize_fact_keys(obj)
#                 if norm: salvaged.append(norm)
#         elif isinstance(obj, list):
#             for item in obj:
#                 norm = _normalize_fact_keys(item)
#                 if norm: salvaged.append(norm)
#     if not salvaged:
#         for m in _TRIPLE_OBJ_RX.finditer(text or ""):
#             chunk = m.group(0)
#             try:
#                 d = json.loads(chunk)
#                 norm = _normalize_fact_keys(d)
#                 if norm: salvaged.append(norm)
#             except Exception:
#                 patched = chunk + "}" * max(0, chunk.count("{") - chunk.count("}"))
#                 try:
#                     d = json.loads(patched)
#                     norm = _normalize_fact_keys(d)
#                     if norm: salvaged.append(norm)
#                 except Exception:
#                     continue
#     if not salvaged:
#         for m in _FLEX_TRIPLE_RX.finditer(text or ""):
#             try:
#                 d = json.loads(m.group(0))
#             except Exception:
#                 continue
#             norm = _normalize_fact_keys(d)
#             if norm: salvaged.append(norm)
#     return [
#         {"subject": t["subject"], "predicate": t["predicate"], "object": t["object"], "confidence": t.get("confidence")}
#         for t in salvaged
#     ]

# # ---------- NER heuristics ----------
# _date_rx = re.compile(r"^\d{4}([-/]\d{2}){0,2}$|^(January|February|March|April|May|June|July|August|September|October|November|December)\b", re.I)
# _url_rx  = re.compile(r"^https?://", re.I)
# def _is_date_like(s:str)->bool: return bool(_date_rx.search(s or ""))
# def _is_literal_like(s:str)->bool:
#     s = s or ""
#     if _url_rx.search(s): return True
#     if s.isdigit(): return True
#     if s.strip().lower() in {"human","engineer","inventor","person","male","female"}: return True
#     return False
# def _titlecase_ratio(s:str)->float:
#     words = [w for w in re.split(r"\s+", (s or "").strip()) if w]
#     if not words: return 0.0
#     caps = sum(1 for w in words if w[:1].isupper())
#     return caps/len(words)
# _variant_rx = re.compile(r"[\(\)\[\]\{\}:–—\-]")
# def _norm(s:str)->str: return re.sub(r"\s+"," ",(s or "")).strip().lower()
# def _is_subject_variant(phrase:str, subject:str)->bool:
#     ps, ss = _norm(phrase), _norm(subject)
#     if not ps or not ss: return False
#     if ps == ss: return True
#     if ps.startswith(ss+" (") or ps.startswith(ss+" -") or ps.startswith(ss+":"): return True
#     if _variant_rx.sub("", ps) == _variant_rx.sub("", ss): return True
#     if ps.startswith(ss) and any(ch in ps[len(ss):len(ss)+3] for ch in "():-—–[]{}"): return True
#     return False
# def _maybe_is_ne_heuristic(phrase:str)->bool:
#     if not isinstance(phrase,str): return False
#     p = phrase.strip()
#     if not p: return False
#     if _is_date_like(p) or _is_literal_like(p): return False
#     if " " not in p and p.islower(): return False
#     if _titlecase_ratio(p) >= 0.6: return True
#     if " " in p and not p.islower(): return True
#     return False
# def _filter_ner_candidates(objs: List[str], subject: Optional[str]=None)->List[str]:
#     uniq:Set[str] = set()
#     for o in objs:
#         if not isinstance(o,str): continue
#         o2 = (o or "").strip()
#         if not o2: continue
#         if len(o2.split())>6: continue
#         if subject and _is_subject_variant(o2, subject): continue
#         if _is_date_like(o2) or _is_literal_like(o2): continue
#         uniq.add(o2)
#     return sorted(uniq)

# # ---------- Self-RAG ----------
# SELF_RAG_SCHEMA = {
#     "type": "object",
#     "additionalProperties": False,
#     "properties": {
#         "summary": {"type": "string"},
#         "aliases": {"type": "array", "items": {"type": "string"}},
#         "salient_facts": {
#             "type": "array",
#             "items": {
#                 "type":"object",
#                 "additionalProperties": False,
#                 "properties":{
#                     "predicate":{"type":"string"},
#                     "object":{"type":"string"},
#                     "confidence":{"type":"number"}
#                 },
#                 "required":["predicate","object"]
#             }
#         }
#     },
#     "required": ["summary","salient_facts"]
# }

# def _ensure_json_keyword_in_msgs(msgs: List[dict], shape_hint: str):
#     has_json = any(isinstance(m.get("content"),str) and "json" in (m.get("content") or "").lower() for m in msgs)
#     if not has_json:
#         msgs.insert(0, {"role":"system","content":f"Output ONLY JSON; shape: {shape_hint}"})

# def _build_selfrag_messages(subject:str, root_subject:str) -> List[dict]:
#     sys = (
#         "You are a concise grounding assistant. Return STRICT JSON only: "
#         '{"summary":"...", "aliases":["..."], "salient_facts":[{"predicate":"...", "object":"...", "confidence":0.0}]}'
#     )
#     user = f"Subject: {subject}\nDomain focus: {root_subject}\nReturn only JSON."
#     return [{"role":"system","content":sys},{"role":"user","content":user}]

# def _inject_selfrag_context_into_elicitation(elicitation_msgs: List[dict], subject: str, context: dict):
#     summary = (context.get("summary") or "").strip()
#     aliases = ", ".join(context.get("aliases") or [])
#     facts = context.get("salient_facts") or []
#     fact_lines = []
#     for f in facts[:16]:
#         p = (f.get("predicate") or "").strip()
#         o = (f.get("object") or "").strip()
#         c = f.get("confidence")
#         if p and o:
#             if isinstance(c,(int,float)):
#                 fact_lines.append(f'- {subject} — {p} — {o} (c={c:.2f})')
#             else:
#                 fact_lines.append(f'- {subject} — {p} — {o}')
#     ctx_txt = (
#         "CONTEXT (self-RAG grounding; use if uncertain; do not quote directly):\n"
#         f"Summary: {summary}\nAliases: {aliases}\nSalient facts:\n" + ("\n".join(fact_lines) if fact_lines else "(none)")
#     )
#     elicitation_msgs.insert(0, {"role":"system","content":ctx_txt})

# # ---------- routing ----------
# def _route_facts(args, facts: List[dict], hop:int, model_name:str):
#     acc, lowconf, objs = [], [], []
#     use_thr = (args.elicitation_strategy == "calibrate")
#     thr = float(args.conf_threshold)
#     for f in facts:
#         s, p, o = f.get("subject"), f.get("predicate"), f.get("object")
#         if not (isinstance(s,str) and isinstance(p,str) and isinstance(o,str)): continue
#         conf = f.get("confidence")
#         if conf is None and args.force_missing_conf_zero:
#             conf = 0.0
#         if use_thr and isinstance(conf,(int,float)) and conf < thr:
#             lowconf.append({
#                 "subject": s, "predicate": p, "object": o,
#                 "hop": hop, "model": model_name, "strategy": args.elicitation_strategy,
#                 "confidence": float(conf), "threshold": thr
#             })
#             continue
#         acc.append((s,p,o,hop,model_name,args.elicitation_strategy, float(conf) if isinstance(conf,(int,float)) else None))
#         objs.append(o)
#     return acc, lowconf, objs

# # ---------- prompts (primary + relaxed fallback) ----------
# def _primary_elicit_prompt(subject: str, root: str, max_facts_hint: str) -> List[dict]:
#     sys = (
#         "Extract factual triples about the subject. STRICT JSON: "
#         '{"facts":[{"subject":"%s","predicate":"<predicate>","object":"<object>","confidence":0.0}]}\n'
#         "Rules: verifiable, concise, no speculation; aim 30–60 facts for major entities; confidence ∈ [0,1]."
#     ) % subject
#     user = f"Subject: {subject}\nRoot context: {root}\nMax facts hint: {max_facts_hint}\nReturn only JSON."
#     return [{"role":"system","content":sys},{"role":"user","content":user}]

# def _relaxed_elicit_prompt(subject: str, root: str) -> List[dict]:
#     sys = (
#         "Return STRICT JSON only. Schema: "
#         '{"facts":[{"subject":"string","predicate":"string","object":"string","confidence":0.0}]}\n'
#         "Keep outputs concise and well-known; if unsure, omit the fact."
#     )
#     user = f"Subject: {subject}\nContext: {root}\nOnly JSON."
#     return [{"role":"system","content":sys},{"role":"user","content":user}]

# def _build_elicitation_messages(args, subject:str)->List[dict]:
#     # Try prompt pack
#     try:
#         msgs = get_prompt_messages(
#             args.elicitation_strategy, "elicitation",
#             domain=args.domain,
#             variables=dict(subject_name=subject, root_subject=args.seed, max_facts_hint=args.max_facts_hint),
#         )
#     except Exception:
#         msgs = []
#     if not msgs:
#         msgs = _primary_elicit_prompt(subject, args.seed, args.max_facts_hint)
#     if getattr(args,"footer_mode",False):
#         footer = ("\n\nImportant: If famous, aim ~50 distinct triples; else ~10. Verifiable only.")
#         for m in msgs:
#             if m.get("role")=="system":
#                 m["content"] = (m.get("content") or "") + footer
#                 break
#         else:
#             msgs.insert(0, {"role":"system","content":footer})
#     return msgs

# # ---------- main ----------
# def main():
#     ap = argparse.ArgumentParser(description="Crawler with optional Self-RAG and recovery.")
#     ap.add_argument("--seed", required=True)
#     ap.add_argument("--output-dir", default=None)

#     ap.add_argument("--elicitation-strategy", default="baseline", choices=["baseline","icl","dont_know","calibrate"])
#     ap.add_argument("--ner-strategy", default="baseline", choices=["baseline","icl","dont_know","calibrate"])
#     ap.add_argument("--domain", default="general", choices=["general","topic"])

#     ap.add_argument("--max-depth", type=int, default=settings.MAX_DEPTH)
#     ap.add_argument("--max-subjects", type=int, default=0)
#     ap.add_argument("--ner-batch-size", type=int, default=settings.NER_BATCH_SIZE)
#     ap.add_argument("--max-facts-hint", default=str(settings.MAX_FACTS_HINT))
#     ap.add_argument("--conf-threshold", type=float, default=0.7)
#     ap.add_argument("--ner-conf-threshold", type=float, default=0.9)
#     ap.add_argument("--footer-mode", action="store_true")

#     ap.add_argument("--elicit-model-key", default=getattr(settings, "ELICIT_MODEL_KEY", None))
#     ap.add_argument("--ner-model-key",     default=getattr(settings, "NER_MODEL_KEY", None))

#     ap.add_argument("--elicit-temperature", type=float, default=0.7)
#     ap.add_argument("--ner-temperature", type=float, default=0.3)
#     ap.add_argument("--elicit-top-p", type=float, default=None)
#     ap.add_argument("--ner-top-p", type=float, default=None)
#     ap.add_argument("--elicit-top-k", type=int, default=None)
#     ap.add_argument("--ner-top-k", type=int, default=None)
#     ap.add_argument("--elicit-max-tokens", type=int, default=4096)
#     ap.add_argument("--ner-max-tokens", type=int, default=4096)

#     ap.add_argument("--batch-size", type=int, default=1)
#     ap.add_argument("--concurrency", type=int, default=8)
#     ap.add_argument("--timeout", type=float, default=90.0)
#     ap.add_argument("--max-retries", type=int, default=3)

#     ap.add_argument("--debug", action="store_true")
#     ap.add_argument("--progress-metrics", dest="progress_metrics", action="store_true", default=True)
#     ap.add_argument("--no-progress-metrics", dest="progress_metrics", action="store_false")

#     ap.add_argument("--resume", action="store_true")
#     ap.add_argument("--reset-working", action="store_true")

#     # Self-RAG toggle
#     sr = ap.add_mutually_exclusive_group()
#     sr.add_argument("--use-selfrag", dest="use_selfrag", action="store_true", help="Enable Self-RAG grounding context.")
#     sr.add_argument("--no-selfrag", dest="use_selfrag", action="store_false", help="Disable Self-RAG grounding context.")
#     ap.set_defaults(use_selfrag=False)

#     ap.add_argument("--selfrag-model-key", default=None)
#     ap.add_argument("--selfrag-max-tokens", type=int, default=512)
#     ap.add_argument("--selfrag-temperature", type=float, default=0.1)
#     ap.add_argument("--selfrag-top-p", type=float, default=None)
#     ap.add_argument("--selfrag-top-k", type=int, default=None)

#     # Recovery controls
#     ap.add_argument("--recovery-enable-selfrag", action="store_true",
#                     help="If elicitation is empty, try one last attempt with Self-RAG even if --no-selfrag was set.")
#     ap.add_argument("--recovery-skip-schema-retry", action="store_true",
#                     help="Disable the automatic 'retry without schema' on empty output.")

#     ap.add_argument("--force-missing-conf-zero", action="store_true")

#     args = ap.parse_args()

#     out_dir = _ensure_output_dir(args.output_dir)
#     paths = _build_paths(out_dir)
#     _dbg(f"[runner] output_dir: {out_dir}")

#     qdb = open_queue_db(paths["queue_sqlite"])
#     fdb = open_facts_db(paths["facts_sqlite"])
#     procq_init_cache(qdb)

#     # seed/resume
#     if args.resume:
#         if not queue_has_rows(qdb):
#             for s, kept_hop, outcome in procq_enqueue(paths["queue_sqlite"], [(args.seed, 0)], leading_articles=PROCQ_LEADING):
#                 if outcome in ("inserted","hop_reduced"):
#                     _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": kept_hop, "event": outcome})
#         else:
#             if args.reset_working:
#                 n = reset_working_to_pending(qdb)
#                 _dbg(f"[resume] reset {n} working→pending")
#     else:
#         for s, kept_hop, outcome in procq_enqueue(paths["queue_sqlite"], [(args.seed, 0)], leading_articles=PROCQ_LEADING):
#             if outcome in ("inserted","hop_reduced"):
#                 _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": kept_hop, "event": outcome})

#     # ---- build cfgs + apply stage params
#     def _apply_stage(which, cfg):
#         # Responses-API style models may ignore sampling params; keep it simple
#         t  = getattr(args, f"{which}_temperature")
#         tp = getattr(args, f"{which}_top_p")
#         tk = getattr(args, f"{which}_top_k")
#         if t  is not None: cfg.temperature = t
#         if tp is not None: cfg.top_p = tp
#         if tk is not None: cfg.top_k = tk
#         mt = getattr(args, f"{which}_max_tokens")
#         if mt is not None: cfg.max_tokens = mt
#         if getattr(cfg,"max_tokens", None) is None: cfg.max_tokens = 2048
#         if hasattr(cfg,"request_timeout"): cfg.request_timeout = args.timeout
#         elif hasattr(cfg,"timeout"):       cfg.timeout = args.timeout

#     # ---- model resolution (defines elicitation model if missing) ----
#     def _resolve_model_key(name: Optional[str], fallback_label: str) -> str:
#         if name and name in settings.MODELS:
#             return name
#         # pick first available as default
#         if getattr(settings, "MODELS", None):
#             picked = next(iter(settings.MODELS.keys()))
#             _dbg(f"[models] {fallback_label} '{name}' not found → using '{picked}'")
#             return picked
#         raise RuntimeError("settings.MODELS is empty; cannot resolve model.")

#     elicit_key = _resolve_model_key(args.elicit_model_key, "elicitation model key")
#     ner_key    = _resolve_model_key(args.ner_model_key,     "NER model key")

#     el_cfg = settings.MODELS[elicit_key].model_copy(deep=True)  # type: ignore
#     ner_cfg = settings.MODELS[ner_key].model_copy(deep=True)    # type: ignore
#     _apply_stage("elicit", el_cfg)
#     _apply_stage("ner", ner_cfg)

#     # Self-RAG cfg (defaults to elicitation model)
#     if args.selfrag_model_key:
#         if args.selfrag_model_key in settings.MODELS:
#             selfrag_cfg = settings.MODELS[args.selfrag_model_key].model_copy(deep=True)  # type: ignore
#         else:
#             _dbg(f"[models] selfrag model key '{args.selfrag_model_key}' not found → using elicitation model '{elicit_key}'")
#             selfrag_cfg = el_cfg.model_copy(deep=True)  # type: ignore
#     else:
#         selfrag_cfg = el_cfg.model_copy(deep=True)  # type: ignore

#     # dedicated sampling for selfrag
#     selfrag_cfg.temperature = args.selfrag_temperature
#     if args.selfrag_top_p is not None: selfrag_cfg.top_p = args.selfrag_top_p
#     if args.selfrag_top_k is not None: selfrag_cfg.top_k = args.selfrag_top_k
#     selfrag_cfg.max_tokens = args.selfrag_max_tokens
#     if hasattr(selfrag_cfg,"request_timeout"): selfrag_cfg.request_timeout = args.timeout
#     elif hasattr(selfrag_cfg,"timeout"):       selfrag_cfg.timeout = args.timeout

#     el_llm = make_llm_from_config(el_cfg)
#     ner_llm = make_llm_from_config(ner_cfg)
#     selfrag_llm = make_llm_from_config(selfrag_cfg) if (args.use_selfrag or args.recovery_enable_selfrag) else None

#     # progress timing
#     last_progress_ts = 0.0

#     subjects_elicited_total = 0
#     seen_facts: Set[Tuple[str,str,str,int]] = set()

#     # ---- worker ----
#     def _elicitation_and_ner(subject: str, hop: int):
#         try:
#             _dbg(f"[subject] '{subject}' hop={hop} start")

#             # (1) Self-RAG grounding (only if explicitly enabled for the main path)
#             selfrag_context = None
#             if args.use_selfrag and selfrag_llm is not None:
#                 sr_msgs = _build_selfrag_messages(subject, args.seed)
#                 _ensure_json_keyword_in_msgs(sr_msgs, shape_hint='{"summary":"...","aliases":["..."],"salient_facts":[{"predicate":"...","object":"...","confidence":0.0}]}')
#                 if args.debug: _print_messages(f"SELF-RAG for [{subject}]", sr_msgs)
#                 try:
#                     sr_resp = selfrag_llm(sr_msgs, json_schema=SELF_RAG_SCHEMA)
#                 except Exception:
#                     sr_resp = selfrag_llm(sr_msgs)
#                 sr_txt = _unwrap_text(sr_resp)
#                 sr_obj = _extract_json_block(sr_txt) if sr_txt else (sr_resp if isinstance(sr_resp, dict) else {})
#                 if isinstance(sr_obj, dict):
#                     selfrag_context = {
#                         "summary": sr_obj.get("summary") or "",
#                         "aliases": sr_obj.get("aliases") or [],
#                         "salient_facts": sr_obj.get("salient_facts") or []
#                     }
#                 _dbg(f"[selfrag] context={'ok' if selfrag_context else 'none'}")

#             # helper: one elicitation attempt (with optional schema, optional context)
#             def _one_elicit_attempt(use_schema: bool, use_relaxed_prompt: bool, context: Optional[dict]) -> Tuple[List[dict], str]:
#                 msgs = _relaxed_elicit_prompt(subject, args.seed) if use_relaxed_prompt else _build_elicitation_messages(args, subject)
#                 if context:
#                     _inject_selfrag_context_into_elicitation(msgs, subject, context)
#                 _ensure_json_keyword_in_msgs(msgs, shape_hint='{"facts":[{"subject":"...","predicate":"...","object":"...","confidence":0.0}]}')
#                 if args.debug: _print_messages(f"ELICIT for [{subject}]{' [relaxed]' if use_relaxed_prompt else ''}", msgs)
#                 schema = ELICIT_SCHEMA_CAL if (args.elicitation_strategy=="calibrate") else ELICIT_SCHEMA_BASE
#                 try:
#                     resp = el_llm(msgs, json_schema=schema) if use_schema else el_llm(msgs)
#                 except Exception:
#                     resp = el_llm(msgs)
#                 raw = _unwrap_text(resp)
#                 # save raw completion for debugging
#                 try:
#                     with open(os.path.join(paths["tmp_dir"], f"elicitation_raw_{re.sub(r'[^a-zA-Z0-9_.-]', '_', subject)}.txt"), "w", encoding="utf-8") as rf:
#                         rf.write(raw or (json.dumps(resp, ensure_ascii=False) if isinstance(resp, dict) else ""))
#                 except Exception:
#                     pass
#                 # parse
#                 obj = _extract_json_block(raw) if raw else (resp if isinstance(resp, (dict, list)) else {})
#                 facts: List[dict] = []
#                 if isinstance(obj, dict):
#                     arr = obj.get("facts"); 
#                     if isinstance(arr, list): facts = [t for t in arr if isinstance(t, dict)]
#                 elif isinstance(obj, list):
#                     facts = [t for t in obj if isinstance(t, dict)]
#                 if not facts and isinstance(raw, str) and raw:
#                     salv = _salvage_facts_from_text(raw)
#                     if salv: facts = salv
#                 return facts, raw if isinstance(raw, str) else ""
            
#             # (2) Elicitation main loop
#             facts: List[dict] = []
#             attempt = 0
#             while attempt < max(1, args.max_retries):
#                 # First try WITH schema
#                 facts, raw_text = _one_elicit_attempt(use_schema=True, use_relaxed_prompt=False, context=selfrag_context)
#                 _dbg(f"[elicitation] try={attempt+1} facts={len(facts)} (with schema)")
#                 if facts:
#                     break
#                 # If empty and schema retry enabled, try once WITHOUT schema
#                 if not args.recovery_skip_schema_retry:
#                     facts, raw_text = _one_elicit_attempt(use_schema=False, use_relaxed_prompt=False, context=selfrag_context)
#                     _dbg(f"[elicitation] try={attempt+1} facts={len(facts)} (no schema)")
#                     if facts:
#                         break
#                 attempt += 1

#             # Recovery: relaxed prompt (still honoring --no-selfrag for the main path)
#             if not facts:
#                 facts, raw_text = _one_elicit_attempt(use_schema=False, use_relaxed_prompt=True, context=None)
#                 _dbg(f"[elicitation][relaxed] facts={len(facts)}")

#             # Final recovery: allow a last-chance Self-RAG even if --no-selfrag was set
#             if not facts and args.recovery_enable_selfrag and selfrag_llm is not None:
#                 # build a fresh selfrag context for recovery
#                 sr_msgs = _build_selfrag_messages(subject, args.seed)
#                 _ensure_json_keyword_in_msgs(sr_msgs, shape_hint='{"summary":"...","aliases":["..."],"salient_facts":[{"predicate":"...","object":"...","confidence":0.0}]}')
#                 try:
#                     sr_resp = selfrag_llm(sr_msgs, json_schema=SELF_RAG_SCHEMA)
#                 except Exception:
#                     sr_resp = selfrag_llm(sr_msgs)
#                 sr_txt = _unwrap_text(sr_resp)
#                 sr_obj = _extract_json_block(sr_txt) if sr_txt else (sr_resp if isinstance(sr_resp, dict) else {})
#                 rec_ctx = sr_obj if isinstance(sr_obj, dict) else None
#                 facts, raw_text = _one_elicit_attempt(use_schema=False, use_relaxed_prompt=True, context=rec_ctx)
#                 _dbg(f"[elicitation][recovery+selfrag] facts={len(facts)}")

#             if not facts:
#                 _append_jsonl(paths["lowconf_jsonl"], {
#                     "event":"elicitation_empty_or_unparseable",
#                     "subject": subject, "hop": hop,
#                     "model": getattr(el_cfg,"model",None),
#                     "strategy": args.elicitation_strategy,
#                     "domain": args.domain, "seed": args.seed
#                 })
#                 write_triples_sink(get_thread_facts_conn(paths["facts_sqlite"]),
#                     [(subject,"__empty__","__empty__",hop, getattr(el_cfg,"model",None),args.elicitation_strategy,None,"empty_or_unparseable_output")]
#                 )
#                 _dbg(f"[elicitation] no facts for '{subject}'")

#             # route & persist
#             acc, lowconf, _ = _route_facts(args, facts, hop, getattr(el_cfg,"model",None))
#             if acc:
#                 write_triples_accepted(get_thread_facts_conn(paths["facts_sqlite"]), acc)
#                 with _seen_facts_lock:
#                     for s,p,o,_,m,st,c in acc:
#                         key = (s,p,o,hop)
#                         if key not in seen_facts:
#                             seen_facts.add(key)
#                             _append_jsonl(paths["facts_jsonl"], {
#                                 "subject": s, "predicate": p, "object": o,
#                                 "hop": hop, "model": m, "strategy": st, "confidence": c
#                             })
#             for item in lowconf:
#                 _append_jsonl(paths["lowconf_jsonl"], item)

#             # (3) NER → enqueue
#             cand = _filter_ner_candidates([t.get("object") for t in facts if isinstance(t, dict)], subject)
#             _dbg(f"[ner] candidates={len(cand)}")
#             next_subjects: List[str] = []
#             i = 0
#             while i < len(cand):
#                 chunk = cand[i: i + max(1, args.ner_batch_size)]
#                 try:
#                     ner_messages = get_prompt_messages(args.ner_strategy, "ner",
#                         domain=args.domain,
#                         variables=dict(phrases_block="\n".join(chunk), root_subject=args.seed, subject_name=subject))
#                 except Exception:
#                     ner_messages = [
#                         {"role":"system","content":
#                          'Decide which phrases are named entities. Return ONLY JSON: {"phrases":[{"phrase":"...","is_ne":true,"confidence":0.0}]}'},
#                         {"role":"user","content":"\n".join(chunk)}
#                     ]
#                 ner_schema = NER_SCHEMA_CAL if (args.ner_strategy=="calibrate") else NER_SCHEMA_BASE
#                 if args.debug: _print_messages(f"NER for [{subject}] chunk[{i}:{i+args.ner_batch_size}]", ner_messages)
#                 try:
#                     out = ner_llm(ner_messages, json_schema=ner_schema)
#                 except Exception:
#                     out = ner_llm(ner_messages)
#                 norm = out if isinstance(out, dict) else {}
#                 decisions = norm.get("phrases", []) if isinstance(norm.get("phrases"), list) else []
#                 if not decisions:
#                     decisions = [{"phrase": ph, "is_ne": _maybe_is_ne_heuristic(ph), "confidence": None} for ph in chunk]

#                 if args.ner_strategy == "calibrate":
#                     for d in decisions:
#                         if not isinstance(d.get("confidence"), (int, float)):
#                             d["confidence"] = 0.90

#                 use_thr = (args.ner_strategy=="calibrate")
#                 for d in decisions:
#                     phrase = d.get("phrase"); is_ne = bool(d.get("is_ne"))
#                     conf = d.get("confidence")
#                     try: conf = float(conf)
#                     except Exception: conf = None
#                     is_variant = _is_subject_variant(phrase, subject)
#                     if is_variant:
#                         is_ne = False; conf = 0.0 if conf is None else min(conf, 0.0)
#                     conf_ok = (isinstance(conf,(int,float)) and conf >= args.ner_conf_threshold) if use_thr else True
#                     record = {
#                         "current_entity": subject, "hop": hop, "phrase": phrase,
#                         "is_ne": is_ne, "is_variant": is_variant,
#                         "confidence": (float(conf) if isinstance(conf,(int,float)) else None),
#                         "ner_conf_threshold": float(args.ner_conf_threshold),
#                         "passed_threshold": bool(conf_ok if use_thr else True),
#                         "ner_model": getattr(ner_cfg,"model",None), "ner_strategy": args.ner_strategy,
#                         "domain": args.domain, "root_subject": args.seed, "source": "model_or_fallback"
#                     }
#                     _append_jsonl(paths["ner_jsonl"], record)
#                     if is_ne and conf_ok and not is_variant and isinstance(phrase,str):
#                         next_subjects.append(phrase)
#                 i += max(1, args.ner_batch_size)

#             if next_subjects:
#                 results = procq_enqueue(
#                     paths["queue_sqlite"],
#                     [(s, hop+1) for s in next_subjects if (args.max_depth==0 or hop+1<=args.max_depth)],
#                     leading_articles=PROCQ_LEADING
#                 )
#                 for s, kept_hop, outcome in results:
#                     if outcome in ("inserted","hop_reduced"):
#                         _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": kept_hop, "event": outcome})
#                 if args.debug:
#                     _print_enqueue_summary(results)
#             else:
#                 _dbg(f"[ner] no next subjects from '{subject}'")

#             mark_done_threadsafe(paths["queue_sqlite"], subject, hop)
#             _dbg(f"[subject] '{subject}' hop={hop} done")
#             return (subject, hop, None)
#         except Exception as e:
#             with open(paths["errors_log"], "a", encoding="utf-8") as ef:
#                 ef.write(f"[{datetime.datetime.now().isoformat()}] subject={subject}\n{traceback.format_exc()}\n")
#             _dbg(f"[error] {e}")
#             mark_pending_on_error(paths["queue_sqlite"], subject, hop)
#             return (subject, hop, "error")

#     # ------------- loop -------------
#     while True:
#         if args.progress_metrics:
#             now = time.perf_counter()
#             cur = qdb.cursor()
#             cur.execute("SELECT COUNT(1) FROM queue WHERE status='done'"); d = cur.fetchone()[0]
#             cur.execute("SELECT COUNT(1) FROM queue WHERE status='working'"); w = cur.fetchone()[0]
#             cur.execute("SELECT COUNT(1) FROM queue WHERE status='pending'"); p = cur.fetchone()[0]
#             _dbg(f"[progress] done={d} working={w} pending={p}")

#         if args.max_subjects and subjects_elicited_total >= args.max_subjects:
#             _dbg(f"[stop] max-subjects reached ({subjects_elicited_total})")
#             break

#         # claim up to concurrency
#         def _fetch_one_pending(conn: sqlite3.Connection) -> Tuple[str,int] | None:
#             cur = conn.cursor()
#             try:
#                 cur.execute("""
#                     UPDATE queue SET status='working'
#                     WHERE rowid = (SELECT rowid FROM queue WHERE status='pending' ORDER BY hop, created_at LIMIT 1)
#                     RETURNING subject, hop
#                 """)
#                 row = cur.fetchone()
#                 conn.commit()
#                 return (row[0], row[1]) if row else None
#             except sqlite3.OperationalError:
#                 cur.execute("BEGIN IMMEDIATE")
#                 cur.execute("SELECT rowid, subject, hop FROM queue WHERE status='pending' ORDER BY hop, created_at LIMIT 1")
#                 row = cur.fetchone()
#                 if not row:
#                     conn.commit(); return None
#                 rowid, subject, hop = row
#                 cur.execute("UPDATE queue SET status='working' WHERE rowid=? AND status='pending'", (rowid,))
#                 changed = cur.rowcount
#                 conn.commit()
#                 return (subject, hop) if changed else None

#         def _fetch_many_pending(conn: sqlite3.Connection, limit: int) -> List[Tuple[str,int]]:
#             got = []
#             for _ in range(max(1,limit)):
#                 one = _fetch_one_pending(conn)
#                 if not one: break
#                 got.append(one)
#             return got

#         batch = _fetch_many_pending(qdb, max(1, args.concurrency))
#         if not batch:
#             _dbg("[idle] queue drained or empty.")
#             break

#         _dbg(f"[path=concurrency] subjects={len(batch)} workers={min(args.concurrency, len(batch))}")
#         results = []
#         with ThreadPoolExecutor(max_workers=min(args.concurrency, len(batch))) as pool:
#             futs = [pool.submit(_elicitation_and_ner, s, h) for (s,h) in batch]
#             for fut in as_completed(futs):
#                 results.append(fut.result())
#         for _s,_h,err in results:
#             if err is None:
#                 subjects_elicited_total += 1
#                 if args.max_subjects and subjects_elicited_total >= args.max_subjects:
#                     _dbg(f"[stop] max-subjects reached ({subjects_elicited_total})")
#                     break

#     # ----- final snapshot + meta -----
#     with open(paths["run_meta_json"], "w", encoding="utf-8") as f:
#         json.dump({
#             "timestamp_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
#             "seed": args.seed, "domain": args.domain,
#             "elicitation_strategy": args.elicitation_strategy, "ner_strategy": args.ner_strategy,
#             "use_selfrag": bool(args.use_selfrag),
#             "models": {
#                 "elicitation": {"model": getattr(el_cfg,"model",None), "provider": getattr(el_cfg,"provider","openai")},
#                 "ner": {"model": getattr(ner_cfg,"model",None), "provider": getattr(ner_cfg,"provider","openai")},
#                 "selfrag": {"model": getattr(selfrag_cfg,"model",None), "provider": getattr(selfrag_cfg,"provider","openai")} if (args.use_selfrag or args.recovery_enable_selfrag) else None,
#             },
#             "args_raw": vars(args),
#         }, f, ensure_ascii=False, indent=2)

#     print(f"[done] finished → {out_dir}")
#     for k in ("queue_jsonl","facts_jsonl","lowconf_jsonl","ner_jsonl","ner_lowconf_jsonl","run_meta_json","errors_log"):
#         print(f"[out] {k:18}: {paths[k]}")

# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("\n[interrupt] bye")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
crawler_runner_selfrag.py
-------------------------
GPTKB-style crawler with an optional Self-RAG grounding stage.

This runner:
  1) Seeds the queue if empty
  2) (NEW) Builds a compact Self-RAG grounding context for the subject (optional)
  3) Runs elicitation to extract (subj, pred, obj) triples
  4) Runs NER on objects to enqueue next subjects
  5) Writes accepted/lowconf/ner JSONL + SQLite

Key improvements in this version:
  - Robust queue seeding (no silent 'idle')
  - --use-selfrag / --no-selfrag mutually exclusive flags
  - Self-RAG prompt/output + token usage logged to selfrag_log.jsonl
  - Per-fact lines in facts.jsonl also record self-RAG model & token usage
  - Clearer debug printing and error logging
"""

from __future__ import annotations

import argparse, datetime, json, os, re, sqlite3, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Set, Optional

from dotenv import load_dotenv
load_dotenv()

# ---------- locks & tiny utils ----------
_jsonl_lock = threading.Lock()
_seen_facts_lock = threading.Lock()
_lowconf_lock = threading.Lock()
_ner_lowconf_lock = threading.Lock()

def _append_jsonl(path: str, obj: dict):
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with _jsonl_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

def _dbg(msg: str): print(msg, flush=True)

def _print_messages(tag: str, msgs: List[dict], limit: int | None = None):
    print(f"\n--- {tag} MESSAGES ({len(msgs)}) ---")
    for i, m in enumerate(msgs, 1):
        role = (m.get("role") or "").upper()
        content = m.get("content")
        if isinstance(content, str) and limit:
            content = (content[:limit] + "…") if len(content) > limit else content
        print(f"[{i:02d}] {role}: {content if isinstance(content, str) else content}")
    print(f"--- END {tag} ---\n")

def _print_enqueue_summary(results: List[Tuple[str,int,str]]):
    if not results:
        print("[enqueue] (no results)")
        return
    ins = sum(1 for *_r, out in results if out == "inserted")
    red = sum(1 for *_r, out in results if out == "hop_reduced")
    ign = sum(1 for *_r, out in results if out == "ignored")
    print(f"[enqueue] inserted={ins} hop_reduced={red} ignored={ign}")

# ---------- repo imports ----------
from processing_queue import (
    init_cache as procq_init_cache,
    enqueue_subjects_processed as procq_enqueue,
    DEFAULT_LEADING_ARTICLES as PROCQ_LEADING,
    get_thread_queue_conn as procq_get_thread_conn,
)
from settings import (
    settings,
    ELICIT_SCHEMA_BASE, ELICIT_SCHEMA_CAL,
    NER_SCHEMA_BASE,   NER_SCHEMA_CAL,
)
from prompter_parser import get_prompt_messages
from llm.factory import make_llm_from_config
from db_models import (
    open_queue_db, open_facts_db,
    write_triples_accepted, write_triples_sink,
    queue_has_rows, reset_working_to_pending,
)

# shared JSON extractor
from llm.json_utils import best_json

# ---------- paths ----------
def _ensure_output_dir(base_dir: Optional[str]) -> str:
    out = base_dir or os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    return out

def _build_paths(out_dir: str) -> dict:
    tmp = os.path.join(out_dir, "tmp")
    os.makedirs(tmp, exist_ok=True)
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
        "ner_lowconf_jsonl": os.path.join(out_dir, "ner_lowconf.jsonl"),
        "ner_lowconf_json": os.path.join(out_dir, "ner_lowconf.json"),
        "run_meta_json": os.path.join(out_dir, "run_meta.json"),
        "selfrag_log_jsonl": os.path.join(out_dir, "selfrag_log.jsonl"),
        "tmp_dir": tmp,
    }

# ---------- per-thread sqlite ----------
_thread_local = threading.local()

def get_thread_queue_conn(db_path: str) -> sqlite3.Connection:
    return procq_get_thread_conn(db_path)

def get_thread_facts_conn(db_path: str) -> sqlite3.Connection:
    key = f"facts_conn__{db_path}"
    conn = getattr(_thread_local, key, None)
    if conn is None:
        conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        setattr(_thread_local, key, conn)
    return conn

def mark_done_threadsafe(queue_db_path: str, subject: str, hop: int):
    conn = get_thread_queue_conn(queue_db_path)
    with conn:
        conn.execute("UPDATE queue SET status='done' WHERE subject=? AND hop=? AND status='working'", (subject, hop))

def mark_pending_on_error(queue_db_path: str, subject: str, hop: int):
    conn = get_thread_queue_conn(queue_db_path)
    with conn:
        conn.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=? AND hop=? AND status='working'", (subject, hop))

def _get_retries(queue_db_path: str, subject: str, hop: int) -> int:
    conn = get_thread_queue_conn(queue_db_path)
    cur = conn.cursor()
    cur.execute("SELECT retries FROM queue WHERE subject=? AND hop=?", (subject, hop))
    row = cur.fetchone()
    return int(row[0]) if row else 0

def _inc_retries_and_pending(queue_db_path: str, subject: str, hop: int):
    conn = get_thread_queue_conn(queue_db_path)
    with conn:
        conn.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=? AND hop=?", (subject, hop))

# ---------- unwrap & salvage ----------
def _parse_obj(maybe_json) -> dict:
    if isinstance(maybe_json, dict): return maybe_json
    if isinstance(maybe_json, str):
        try: return json.loads(maybe_json)
        except Exception: return {}
    return {}

def _unwrap_text(resp):
    if isinstance(resp, str): return resp
    if isinstance(resp, dict):
        for k in ("text","output_text","content","message","response"):
            v = resp.get(k)
            if isinstance(v, str): return v
        ch = resp.get("choices")
        if isinstance(ch, list) and ch:
            c0 = ch[0] or {}
            msg = c0.get("message") or {}
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
            if isinstance(c0.get("text"), str): return c0["text"]
        if isinstance(resp.get("_raw"), str): return resp["_raw"]
        if isinstance(resp.get("raw"), str):  return resp["raw"]
        if isinstance(resp.get("raw"), dict): return _unwrap_text(resp["raw"])
        # usage payload passthrough not text
    return ""

def _extract_json_block(text: str):
    obj = best_json(text)
    return obj if isinstance(obj, (dict, list)) else {}

def _normalize_fact_keys(d: dict) -> dict | None:
    if not isinstance(d, dict): return None
    key_map = {
        "subject": ["subject","subj","s","head","h"],
        "predicate": ["predicate","pred","p","relation","rel","r"],
        "object": ["object","obj","o","tail","t","value","val"],
        "confidence": ["confidence","conf","c","score","prob"]
    }
    out = {}
    for std, alts in key_map.items():
        for k in alts:
            if k in d and isinstance(d[k], (str, float, int)):
                out[std] = d[k]
                break
    s,p,o = out.get("subject"), out.get("predicate"), out.get("object")
    if not (isinstance(s,str) and isinstance(p,str) and isinstance(o,str)):
        return None
    if "confidence" in out:
        try: out["confidence"] = float(out["confidence"])
        except Exception: out["confidence"] = None
    else:
        out["confidence"] = None
    return out

_TRIPLE_OBJ_RX = re.compile(r"\{[^{}]*?(\"subject\"|\"subj\"|\"s\"|\"head\")[^{}]*?\}", re.I)
_FLEX_TRIPLE_RX = re.compile(r"\{[^{}]*\}", re.S)

def _salvage_facts_from_text(text: str, debug=False) -> List[dict]:
    salvaged: List[dict] = []

    obj = _extract_json_block(text)
    if obj:
        if isinstance(obj, dict):
            for key in ("facts","triples"):
                val = obj.get(key)
                if isinstance(val, list):
                    for item in val:
                        norm = _normalize_fact_keys(item)
                        if norm: salvaged.append(norm)
            if not salvaged:
                norm = _normalize_fact_keys(obj)
                if norm: salvaged.append(norm)
        elif isinstance(obj, list):
            for item in obj:
                norm = _normalize_fact_keys(item)
                if norm: salvaged.append(norm)

    if not salvaged:
        for m in _TRIPLE_OBJ_RX.finditer(text or ""):
            chunk = m.group(0)
            try:
                d = json.loads(chunk)
                norm = _normalize_fact_keys(d)
                if norm: salvaged.append(norm)
            except Exception:
                patched = chunk
                open_br = chunk.count("{")
                close_br = chunk.count("}")
                patched += "}" * max(0, open_br - close_br)
                try:
                    d = json.loads(patched)
                    norm = _normalize_fact_keys(d)
                    if norm: salvaged.append(norm)
                except Exception:
                    continue

    if not salvaged:
        for m in _FLEX_TRIPLE_RX.finditer(text or ""):
            try:
                d = json.loads(m.group(0))
            except Exception:
                continue
            norm = _normalize_fact_keys(d)
            if norm:
                salvaged.append(norm)

    facts = []
    for t in salvaged:
        facts.append({
            "subject": t["subject"],
            "predicate": t["predicate"],
            "object": t["object"],
            "confidence": t.get("confidence")
        })
    return facts

def _extract_facts_from_resp(resp, debug=False) -> Tuple[List[dict], str]:
    if isinstance(resp, list):
        facts = [t for t in resp if isinstance(t, dict)]
        return facts, ""
    if isinstance(resp, dict):
        for key in ("facts","triples"):
            val = resp.get(key)
            if isinstance(val, list):
                return [t for t in val if isinstance(t, dict)], ""
    txt = _unwrap_text(resp)
    obj = _extract_json_block(txt)
    if isinstance(obj, dict):
        for key in ("facts","triples"):
            val = obj.get(key)
            if isinstance(val, list):
                return [t for t in val if isinstance(t, dict)], txt
    if isinstance(obj, list):
        return [t for t in obj if isinstance(t, dict)], txt
    return [], txt

# ---------- NER heuristics ----------
_date_rx = re.compile(r"^\d{4}([-/]\d{2}){0,2}$|^(January|February|March|April|May|June|July|August|September|October|November|December)\b", re.I)
_url_rx  = re.compile(r"^https?://", re.I)
def _is_date_like(s:str)->bool: return bool(_date_rx.search(s or ""))
def _is_literal_like(s:str)->bool:
    s = s or ""
    if _url_rx.search(s): return True
    if s.isdigit(): return True
    if s.strip().lower() in {"human","engineer","inventor","person","male","female"}: return True
    return False
def _titlecase_ratio(s:str)->float:
    words = [w for w in re.split(r"\s+", (s or "").strip()) if w]
    if not words: return 0.0
    caps = sum(1 for w in words if w[:1].isupper())
    return caps/len(words)
_variant_rx = re.compile(r"[\(\)\[\]\{\}:–—\-]")
def _norm(s:str)->str: return re.sub(r"\s+"," ",(s or "")).strip().lower()
def _is_subject_variant(phrase:str, subject:str)->bool:
    ps, ss = _norm(phrase), _norm(subject)
    if not ps or not ss: return False
    if ps == ss: return True
    if ps.startswith(ss+" (") or ps.startswith(ss+" -") or ps.startswith(ss+":"): return True
    if _variant_rx.sub("", ps) == _variant_rx.sub("", ss): return True
    if ps.startswith(ss) and any(ch in ps[len(ss):len(ss)+3] for ch in "():-—–[]{}"): return True
    return False
def _maybe_is_ne_heuristic(phrase:str)->bool:
    if not isinstance(phrase,str): return False
    p = phrase.strip()
    if not p: return False
    if _is_date_like(p) or _is_literal_like(p): return False
    if " " not in p and p.islower(): return False
    if _titlecase_ratio(p) >= 0.6: return True
    if " " in p and not p.islower(): return True
    return False
def _filter_ner_candidates(objs: List[str], subject: Optional[str]=None)->List[str]:
    uniq:Set[str] = set()
    for o in objs:
        if not isinstance(o,str): continue
        o2 = o.strip()
        if not o2: continue
        if len(o2.split())>6: continue
        if subject and _is_subject_variant(o2, subject): continue
        if _is_date_like(o2) or _is_literal_like(o2): continue
        uniq.add(o2)
    return sorted(uniq)

# ---------- Self-RAG (NEW) ----------
SELF_RAG_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string", "description": "1-3 sentences factual summary of the subject."},
        "aliases": {"type": "array", "items": {"type": "string"}},
        "salient_facts": {
            "type": "array",
            "items": {
                "type":"object",
                "additionalProperties": False,
                "properties":{
                    "predicate":{"type":"string"},
                    "object":{"type":"string"},
                    "confidence":{"type":"number"}
                },
                "required":["predicate","object"]
            }
        }
    },
    "required": ["summary","salient_facts"]
}

def _supports_reasoning_controls(cfg) -> bool:
    if not getattr(cfg, "use_responses_api", False):
        return False
    name = (getattr(cfg, "model", "") or "").lower()
    return name.startswith("gpt-5")

def _apply_reasoning_text_overrides(cfg, effort: str | None, verbosity: str | None, stage_label: str):
    if not _supports_reasoning_controls(cfg):
        return
    if cfg.extra_inputs is None:
        cfg.extra_inputs = {}
    cfg.extra_inputs.setdefault("reasoning", {})
    cfg.extra_inputs.setdefault("text", {})
    if effort is not None:
        cfg.extra_inputs["reasoning"]["effort"] = effort
    if verbosity is not None:
        cfg.extra_inputs["text"]["verbosity"] = verbosity
    _dbg(f"[model:{stage_label}] reasoning.effort={cfg.extra_inputs['reasoning'].get('effort')} "
         f"text.verbosity={cfg.extra_inputs['text'].get('verbosity')}")

def _ensure_json_keyword_in_msgs(msgs: List[dict], shape_hint: str):
    has_json = any(isinstance(m.get("content"),str) and "json" in (m.get("content") or "").lower() for m in msgs)
    if not has_json:
        msgs.insert(0, {"role":"system","content":f"Output ONLY JSON; shape: {shape_hint}"})

def _build_selfrag_messages(subject:str, root_subject:str) -> List[dict]:
    sys = (
        "You are a concise, factual grounding assistant. Given a subject entity, return STRICT JSON: "
        '{"summary":"...", "aliases":["..."], "salient_facts":[{"predicate":"...", "object":"...", "confidence":0.0}]}.\n'
        "Rules: short, verifiable, no speculation; keep 5–12 salient_facts; confidence in [0,1]."
    )
    user = (
        f"Subject: {subject}\n"
        f"Domain focus (context): {root_subject}\n"
        "Return only JSON; no markdown or prose."
    )
    return [{"role":"system","content":sys},{"role":"user","content":user}]

def _inject_selfrag_context_into_elicitation(elicitation_msgs: List[dict], subject: str, context: dict):
    """Prepend a high-priority system message that provides the grounding context."""
    summary = (context.get("summary") or "").strip()
    aliases = ", ".join(context.get("aliases") or [])
    facts = context.get("salient_facts") or []
    fact_lines = []
    for f in facts[:16]:
        p = (f.get("predicate") or "").strip()
        o = (f.get("object") or "").strip()
        c = f.get("confidence")
        if p and o:
            if isinstance(c,(int,float)):
                fact_lines.append(f'- {subject} — {p} — {o} (c={c:.2f})')
            else:
                fact_lines.append(f'- {subject} — {p} — {o}')
    ctx_txt = (
        "CONTEXT (self-RAG grounding; use when uncertain; do not quote directly):\n"
        f"Summary: {summary}\n"
        f"Aliases: {aliases}\n"
        "Salient facts:\n" + ("\n".join(fact_lines) if fact_lines else "(none)")
    )
    elicitation_msgs.insert(0, {"role":"system","content":ctx_txt})

# ---------- provider helpers ----------
def _is_openai_model(cfg)->bool:
    prov = (getattr(cfg,"provider","") or "").lower()
    if "openai" in prov: return True
    name = (getattr(cfg,"model","") or "").lower()
    return "openai" in name or name.startswith("gpt-")

# unchanged router, but optionally force missing conf to 0.0
def _route_facts(args, facts: List[dict], hop:int, model_name:str):
    acc, lowconf, objs = [], [], []
    use_thr = (args.elicitation_strategy == "calibrate")
    thr = float(args.conf_threshold)
    for f in facts:
        s, p, o = f.get("subject"), f.get("predicate"), f.get("object")
        if not (isinstance(s,str) and isinstance(p,str) and isinstance(o,str)): continue
        conf = f.get("confidence")
        if conf is None and args.force_missing_conf_zero:
            conf = 0.0
        if use_thr and isinstance(conf,(int,float)) and conf < thr:
            lowconf.append({
                "subject": s, "predicate": p, "object": o,
                "hop": hop, "model": model_name, "strategy": args.elicitation_strategy,
                "confidence": float(conf), "threshold": thr
            })
            continue
        acc.append((s,p,o,hop,model_name,args.elicitation_strategy, float(conf) if isinstance(conf,(int,float)) else None))
        objs.append(o)
    return acc, lowconf, objs

# ---------- prompts ----------
def _build_elicitation_messages(args, subject:str)->List[dict]:
    msgs = get_prompt_messages(
        args.elicitation_strategy, "elicitation",
        domain=args.domain,
        variables=dict(subject_name=subject, root_subject=args.seed, max_facts_hint=args.max_facts_hint),
    )
    if getattr(args,"footer_mode",False):
        footer = ("\n\nFinal important note:\n"
                  "If the entity is famous, aim ~50 distinct triplets; else ~10 if any exist. "
                  "Only concrete, verifiable info.")
        for m in msgs:
            if m.get("role")=="system":
                m["content"] = (m.get("content") or "") + footer
                break
        else:
            msgs.insert(0, {"role":"system","content":footer})
    return msgs

# ---------- queue fetch helpers (define BEFORE main loop) ----------
def _fetch_one_pending(conn: sqlite3.Connection) -> Tuple[str,int] | None:
    cur = conn.cursor()
    try:
        cur.execute("""
            UPDATE queue SET status='working'
            WHERE rowid = (SELECT rowid FROM queue WHERE status='pending' ORDER BY hop, created_at LIMIT 1)
            RETURNING subject, hop
        """)
        row = cur.fetchone()
        conn.commit()
        return (row[0], row[1]) if row else None
    except sqlite3.OperationalError:
        cur.execute("BEGIN IMMEDIATE")
        cur.execute("SELECT rowid, subject, hop FROM queue WHERE status='pending' ORDER BY hop, created_at LIMIT 1")
        row = cur.fetchone()
        if not row:
            conn.commit(); return None
        rowid, subject, hop = row
        cur.execute("UPDATE queue SET status='working' WHERE rowid=? AND status='pending'", (rowid,))
        changed = cur.rowcount
        conn.commit()
        return (subject, hop) if changed else None

def _fetch_many_pending(conn: sqlite3.Connection, limit: int) -> List[Tuple[str,int]]:
    got = []
    for _ in range(max(1,limit)):
        one = _fetch_one_pending(conn)
        if not one: break
        got.append(one)
    if not got:
        _dbg("[debug] No subjects found in the queue to process.")
    return got

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Crawler with Self-RAG grounding stage.")
    ap.add_argument("--seed", required=True)
    ap.add_argument("--output-dir", default=None)

    ap.add_argument("--elicitation-strategy", default="baseline", choices=["baseline","icl","dont_know","calibrate"])
    ap.add_argument("--ner-strategy", default="baseline", choices=["baseline","icl","dont_know","calibrate"])
    ap.add_argument("--domain", default="general", choices=["general","topic"])

    ap.add_argument("--max-depth", type=int, default=settings.MAX_DEPTH)
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--ner-batch-size", type=int, default=settings.NER_BATCH_SIZE)
    ap.add_argument("--max-facts-hint", default=str(settings.MAX_FACTS_HINT))
    ap.add_argument("--conf-threshold", type=float, default=0.7)
    ap.add_argument("--ner-conf-threshold", type=float, default=0.9)
    ap.add_argument("--footer-mode", action="store_true")

    ap.add_argument("--elicit-model-key", default=settings.ELICIT_MODEL_KEY)
    ap.add_argument("--ner-model-key", default=settings.NER_MODEL_KEY)

    ap.add_argument("--elicit-temperature", type=float, default=0.7)
    ap.add_argument("--ner-temperature", type=float, default=0.3)
    ap.add_argument("--elicit-top-p", type=float, default=None)
    ap.add_argument("--ner-top-p", type=float, default=None)
    ap.add_argument("--elicit-top-k", type=int, default=None)
    ap.add_argument("--ner-top-k", type=int, default=None)
    ap.add_argument("--elicit-max-tokens", type=int, default=4096)
    ap.add_argument("--ner-max-tokens", type=int, default=4096)

    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--max-retries", type=int, default=3)

    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--progress-metrics", dest="progress_metrics", action="store_true", default=True)
    ap.add_argument("--no-progress-metrics", dest="progress_metrics", action="store_false")

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reset-working", action="store_true")

    # NEW: Self-RAG options (mutually exclusive pair)
    sr_group = ap.add_mutually_exclusive_group()
    sr_group.add_argument("--use-selfrag", dest="use_selfrag", action="store_true", default=True,
                          help="Enable Self-RAG grounding context.")
    sr_group.add_argument("--no-selfrag", dest="use_selfrag", action="store_false",
                          help="Disable Self-RAG grounding context.")
    ap.add_argument("--selfrag-model-key", default=None, help="Optional: different model for Self-RAG. Defaults to elicitation model.")
    ap.add_argument("--selfrag-max-tokens", type=int, default=512)
    ap.add_argument("--selfrag-temperature", type=float, default=0.1)
    ap.add_argument("--selfrag-top-p", type=float, default=None)
    ap.add_argument("--selfrag-top-k", type=int, default=None)

    # Missing confidence behavior
    ap.add_argument("--force-missing-conf-zero", action="store_true", help="If a triple lacks confidence, set it to 0.0.")

    # NEW: reasoning/verbosity overrides (only for supported Responses API models)
    ap.add_argument("--reasoning-effort", choices=["minimal","low","medium","high"], default=None)
    ap.add_argument("--text-verbosity", choices=["low","medium","high"], default=None)

    args = ap.parse_args()

    out_dir = _ensure_output_dir(args.output_dir)
    paths = _build_paths(out_dir)
    _dbg(f"[runner] output_dir: {out_dir}")

    qdb = open_queue_db(paths["queue_sqlite"])
    fdb = open_facts_db(paths["facts_sqlite"])
    procq_init_cache(qdb)

    # ---- build cfgs + apply stage params ----
    def _apply_stage(which, cfg):
        if getattr(cfg, "use_responses_api", False):
            cfg.temperature = None; cfg.top_p = None; cfg.top_k = None
            if cfg.extra_inputs is None:
                cfg.extra_inputs = {}
            cfg.extra_inputs.setdefault("reasoning", {})
            cfg.extra_inputs.setdefault("text", {})
        else:
            t  = getattr(args, f"{which}_temperature")
            tp = getattr(args, f"{which}_top_p")
            tk = getattr(args, f"{which}_top_k")
            if t  is not None: cfg.temperature = t
            if tp is not None: cfg.top_p = tp
            if tk is not None: cfg.top_k = tk
        mt = getattr(args, f"{which}_max_tokens")
        if mt is not None: cfg.max_tokens = mt
        if getattr(cfg,"max_tokens", None) is None: cfg.max_tokens = 2048
        if hasattr(cfg,"request_timeout"): cfg.request_timeout = args.timeout
        elif hasattr(cfg,"timeout"):       cfg.timeout = args.timeout
        _apply_reasoning_text_overrides(cfg, args.reasoning_effort, args.text_verbosity, which)

    def _copy_model_cfg(model_key: str, label: str):
        try:
            return settings.MODELS[model_key].model_copy(deep=True)  # pydantic models
        except Exception as e:
            msg = f"[error] model key '{model_key}' not found in settings.MODELS for {label}"
            _append_jsonl(paths["errors_log"], {"ts": datetime.datetime.utcnow().isoformat()+"Z", "error": msg, "exc": str(e)})
            raise

    el_cfg = _copy_model_cfg(args.elicit_model_key, "elicitation")
    ner_cfg = _copy_model_cfg(args.ner_model_key, "ner")

    _apply_stage("elicit", el_cfg)
    _apply_stage("ner", ner_cfg)

    # Self-RAG cfg (defaults to elicitation model)
    if args.selfrag_model_key:
        selfrag_cfg = _copy_model_cfg(args.selfrag_model_key, "selfrag")
    else:
        selfrag_cfg = el_cfg.model_copy(deep=True)  # copy elicit cfg

    def _apply_selfrag_stage(cfg):
        if getattr(cfg, "use_responses_api", False):
            cfg.temperature = None; cfg.top_p = None; cfg.top_k = None
            if cfg.extra_inputs is None:
                cfg.extra_inputs = {}
            cfg.extra_inputs.setdefault("reasoning", {})
            cfg.extra_inputs.setdefault("text", {})
        else:
            cfg.temperature = args.selfrag_temperature
            if args.selfrag_top_p is not None: cfg.top_p = args.selfrag_top_p
            if args.selfrag_top_k is not None: cfg.top_k = args.selfrag_top_k
        cfg.max_tokens = args.selfrag_max_tokens
        if hasattr(cfg,"request_timeout"): cfg.request_timeout = args.timeout
        elif hasattr(cfg,"timeout"):       cfg.timeout = args.timeout
        _apply_reasoning_text_overrides(cfg, args.reasoning_effort, args.text_verbosity, "selfrag")

    _apply_selfrag_stage(selfrag_cfg)

    # ---- instantiate LLMs ----
    el_llm = make_llm_from_config(el_cfg)
    ner_llm = make_llm_from_config(ner_cfg)
    selfrag_llm = make_llm_from_config(selfrag_cfg) if args.use_selfrag else None

    # ---------- SEED QUEUE ROBUSTLY ----------
    def _queue_counts():
        cur = qdb.cursor()
        cur.execute("SELECT COUNT(1) FROM queue WHERE status='pending'"); p = cur.fetchone()[0]
        cur.execute("SELECT COUNT(1) FROM queue WHERE status='working'"); w = cur.fetchone()[0]
        cur.execute("SELECT COUNT(1) FROM queue WHERE status='done'");    d = cur.fetchone()[0]
        return p,w,d

    pending0, working0, done0 = _queue_counts()
    if pending0 + working0 + done0 == 0:
        _dbg("[seed] queue empty → enqueuing seed")
        for s, kept_hop, outcome in procq_enqueue(paths["queue_sqlite"], [(args.seed, 0)], leading_articles=PROCQ_LEADING):
            if outcome in ("inserted","hop_reduced"):
                _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": kept_hop, "event": outcome})
    elif args.resume and args.reset_working:
        n = reset_working_to_pending(qdb)
        _dbg(f"[resume] reset {n} working→pending")

    p1,w1,d1 = _queue_counts()
    _dbg(f"[queue] after seed: pending={p1} working={w1} done={d1}")

    # progress timing
    last_progress_ts = 0.0

    # shared state
    subjects_elicited_total = 0
    lowconf_accum: List[dict] = []
    ner_lowconf_accum: List[dict] = []
    seen_facts: Set[Tuple[str,str,str,int]] = set()

    # ---- worker (concurrency path) ----
    def _elicitation_and_ner(subject: str, hop: int):
        _dbg(f"[subject] '{subject}' hop={hop} start")
        try:
            # (1) Self-RAG grounding
            selfrag_context = None
            sr_prompt_tokens = None
            sr_completion_tokens = None

            if args.use_selfrag and selfrag_llm is not None:
                sr_msgs = _build_selfrag_messages(subject, args.seed)
                _ensure_json_keyword_in_msgs(sr_msgs, shape_hint='{"summary":"...","aliases":["..."],"salient_facts":[{"predicate":"...","object":"...","confidence":0.0}]}')
                if args.debug: _print_messages(f"SELF-RAG for [{subject}]", sr_msgs)
                sr_resp = None
                try:
                    sr_resp = selfrag_llm(sr_msgs, json_schema=SELF_RAG_SCHEMA)
                except Exception:
                    sr_resp = selfrag_llm(sr_msgs)

                sr_txt = _unwrap_text(sr_resp)
                sr_obj = _extract_json_block(sr_txt) if sr_txt else (sr_resp if isinstance(sr_resp, dict) else {})
                if isinstance(sr_obj, dict):
                    selfrag_context = {
                        "summary": sr_obj.get("summary") or "",
                        "aliases": sr_obj.get("aliases") or [],
                        "salient_facts": sr_obj.get("salient_facts") or []
                    }
                else:
                    selfrag_context = None

                # token usage (if available)
                usage = None
                if isinstance(sr_resp, dict) and "usage" in sr_resp:
                    usage = sr_resp["usage"]
                if usage:
                    sr_prompt_tokens = usage.get("prompt_tokens")
                    sr_completion_tokens = usage.get("completion_tokens")

                # write detailed SR log always
                _append_jsonl(paths["selfrag_log_jsonl"], {
                    "ts": datetime.datetime.utcnow().isoformat()+"Z",
                    "subject": subject,
                    "hop": hop,
                    "selfrag_model": getattr(selfrag_cfg, "model", None),
                    "prompt_messages": sr_msgs if args.debug else "[hidden, run with --debug to see]",
                    "raw_text": sr_txt,
                    "parsed": selfrag_context,
                    "usage": {
                        "prompt_tokens": sr_prompt_tokens,
                        "completion_tokens": sr_completion_tokens
                    }
                })

            # (2) Elicitation (grounded by Self-RAG if available)
            attempt = 0
            facts: List[dict] = []
            last_text = ""
            el_schema = ELICIT_SCHEMA_CAL if (args.elicitation_strategy=="calibrate") else ELICIT_SCHEMA_BASE

            while attempt < max(1, args.max_retries):
                el_messages = _build_elicitation_messages(args, subject)
                if selfrag_context:
                    _inject_selfrag_context_into_elicitation(el_messages, subject, selfrag_context)

                _ensure_json_keyword_in_msgs(el_messages, shape_hint='{"facts":[{"subject":"...","predicate":"...","object":"...","confidence":0.0}]}')
                if args.debug: _print_messages(f"ELICIT for [{subject}] (try {attempt+1})", el_messages)

                try:
                    resp = el_llm(el_messages, json_schema=el_schema)
                except Exception:
                    resp = el_llm(el_messages)

                facts, last_text = _extract_facts_from_resp(resp, debug=args.debug)
                if not facts and last_text:
                    salv = _salvage_facts_from_text(last_text, debug=args.debug)
                    if salv:
                        facts = salv

                if args.debug:
                    _append_jsonl(paths["errors_log"], {
                        "ts": datetime.datetime.utcnow().isoformat()+"Z",
                        "stage": "elicitation_debug",
                        "subject": subject,
                        "hop": hop,
                        "attempt": attempt+1,
                        "raw_text": last_text
                    })

                if facts:
                    break
                attempt += 1

            if not facts:
                write_triples_sink(get_thread_facts_conn(paths["facts_sqlite"]),
                    [(subject,"__empty__","__empty__",hop, getattr(el_cfg,"model",None),args.elicitation_strategy,None,"empty_or_unparseable_output")]
                )
                _dbg(f"[elicitation] no facts for '{subject}'")

            acc, lowconf, _ = _route_facts(args, facts, hop, getattr(el_cfg,"model",None))
            if acc:
                write_triples_accepted(get_thread_facts_conn(paths["facts_sqlite"]), acc)
                with _seen_facts_lock:
                    for s,p,o,_,m,st,c in acc:
                        key = (s,p,o,hop)
                        if key not in seen_facts:
                            seen_facts.add(key)
                            _append_jsonl(paths["facts_jsonl"], {
                                "subject": s, "predicate": p, "object": o,
                                "hop": hop, "model": m, "strategy": st, "confidence": c,
                                # NEW: Self-RAG fields echoed with each fact for auditability
                                "selfrag_model": getattr(selfrag_cfg,"model",None) if args.use_selfrag else None,
                                "selfrag_prompt_tokens": sr_prompt_tokens if args.use_selfrag else None,
                                "selfrag_completion_tokens": sr_completion_tokens if args.use_selfrag else None
                            })
            if lowconf:
                for item in lowconf: _append_jsonl(paths["lowconf_jsonl"], item)
                with _lowconf_lock: lowconf_accum.extend(lowconf)

            # (3) NER → enqueue
            cand = _filter_ner_candidates([t.get("object") for t in facts if isinstance(t, dict)], subject)
            if args.debug:
                _dbg(f"[ner] candidates={len(cand)}")
            next_subjects: List[str] = []
            i = 0
            while i < len(cand):
                chunk = cand[i: i + args.ner_batch_size]
                ner_messages = get_prompt_messages(args.ner_strategy, "ner",
                    domain=args.domain,
                    variables=dict(phrases_block="\n".join(chunk), root_subject=args.seed, subject_name=subject))
                ner_schema = NER_SCHEMA_CAL if (args.ner_strategy=="calibrate") else NER_SCHEMA_BASE
                if args.debug: _print_messages(f"NER for [{subject}] chunk[{i}:{i+args.ner_batch_size}]", ner_messages)
                try:
                    out = ner_llm(ner_messages, json_schema=ner_schema)
                except Exception:
                    out = ner_llm(ner_messages)
                norm = _parse_obj(out)
                decisions = norm.get("phrases", []) if isinstance(norm.get("phrases"), list) else []
                if not decisions:
                    decisions = [{"phrase": ph, "is_ne": _maybe_is_ne_heuristic(ph), "confidence": None} for ph in chunk]

                if args.ner_strategy == "calibrate":
                    for d in decisions:
                        if not isinstance(d.get("confidence"), (int, float)):
                            d["confidence"] = 0.90

                use_thr = (args.ner_strategy=="calibrate")
                for d in decisions:
                    phrase = d.get("phrase"); is_ne = bool(d.get("is_ne"))
                    conf = d.get("confidence")
                    try: conf = float(conf)
                    except Exception: conf = None
                    is_variant = _is_subject_variant(phrase, subject)
                    if is_variant:
                        is_ne = False; conf = 0.0 if conf is None else min(conf, 0.0)
                    conf_ok = (isinstance(conf,(int,float)) and conf >= args.ner_conf_threshold) if use_thr else True
                    record = {
                        "current_entity": subject, "hop": hop, "phrase": phrase,
                        "is_ne": is_ne, "is_variant": is_variant,
                        "confidence": (float(conf) if isinstance(conf,(int,float)) else None),
                        "ner_conf_threshold": float(args.ner_conf_threshold),
                        "passed_threshold": bool(conf_ok if use_thr else True),
                        "ner_model": getattr(ner_cfg,"model",None), "ner_strategy": args.ner_strategy,
                        "domain": args.domain, "root_subject": args.seed, "source": "model_or_fallback"
                    }
                    _append_jsonl(paths["ner_jsonl"], record)
                    if use_thr and not conf_ok:
                        low_item = {**record, "reason":"below_threshold"}
                        _append_jsonl(paths["ner_lowconf_jsonl"], low_item)
                        with _ner_lowconf_lock: ner_lowconf_accum.append(low_item)
                    if is_ne and conf_ok and not is_variant and isinstance(phrase,str):
                        next_subjects.append(phrase)
                i += args.ner_batch_size

            if next_subjects:
                results = procq_enqueue(
                    paths["queue_sqlite"],
                    [(s, hop+1) for s in next_subjects if (args.max_depth==0 or hop+1<=args.max_depth)],
                    leading_articles=PROCQ_LEADING
                )
                for s, kept_hop, outcome in results:
                    if outcome in ("inserted","hop_reduced"):
                        _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": kept_hop, "event": outcome})
                if args.debug:
                    _print_enqueue_summary(results)
            else:
                if args.debug:
                    _dbg(f"[ner] no next subjects from '{subject}'")

            mark_done_threadsafe(paths["queue_sqlite"], subject, hop)
            _dbg(f"[subject] '{subject}' hop={hop} done")
            return (subject, hop, None)
        except Exception:
            with open(paths["errors_log"], "a", encoding="utf-8") as ef:
                ef.write(f"[{datetime.datetime.now().isoformat()}] subject={subject}\n{traceback.format_exc()}\n")
            mark_pending_on_error(paths["queue_sqlite"], subject, hop)
            return (subject, hop, "error")

    # ------------- loop -------------
    while True:
        if args.progress_metrics:
            now = time.perf_counter()
            if now - last_progress_ts >= 2.0:
                cur = qdb.cursor()
                cur.execute("SELECT COUNT(1) FROM queue WHERE status='done'"); d = cur.fetchone()[0]
                cur.execute("SELECT COUNT(1) FROM queue WHERE status='working'"); w = cur.fetchone()[0]
                cur.execute("SELECT COUNT(1) FROM queue WHERE status='pending'"); p = cur.fetchone()[0]
                _dbg(f"[progress] done={d} working={w} pending={p}")
                last_progress_ts = now

        if args.max_subjects and subjects_elicited_total >= args.max_subjects:
            _dbg(f"[stop] max-subjects reached ({subjects_elicited_total})")
            break

        batch = _fetch_many_pending(qdb, max(1, args.concurrency))
        if not batch:
            # One-shot safety: if queue is completely empty, try seeding once more
            cur = qdb.cursor()
            cur.execute("SELECT COUNT(1) FROM queue"); total_rows = cur.fetchone()[0]
            if total_rows == 0:
                _dbg("[idle] queue empty — re-seeding with seed")
                for s, kept_hop, outcome in procq_enqueue(paths["queue_sqlite"], [(args.seed, 0)], leading_articles=PROCQ_LEADING):
                    if outcome in ("inserted","hop_reduced"):
                        _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": kept_hop, "event": outcome})
                # fetch again
                batch = _fetch_many_pending(qdb, max(1, args.concurrency))

            if not batch:
                _dbg("[idle] queue drained or empty.")
                break

        _dbg(f"[path=concurrency] subjects={len(batch)} workers={min(args.concurrency, len(batch))}")
        results = []
        with ThreadPoolExecutor(max_workers=min(args.concurrency, len(batch))) as pool:
            futs = [pool.submit(_elicitation_and_ner, s, h) for (s,h) in batch]
            for fut in as_completed(futs):
                results.append(fut.result())
        for _s,_h,err in results:
            if err is None:
                subjects_elicited_total += 1
                if args.max_subjects and subjects_elicited_total >= args.max_subjects:
                    _dbg(f"[stop] max-subjects reached ({subjects_elicited_total})")
                    break

    # ----- final snapshot + meta -----
    with open(paths["run_meta_json"], "w", encoding="utf-8") as f:
        json.dump({
            "timestamp_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "seed": args.seed, "domain": args.domain,
            "elicitation_strategy": args.elicitation_strategy, "ner_strategy": args.ner_strategy,
            "use_selfrag": bool(args.use_selfrag),
            "models": {
                "elicitation": {"model": getattr(el_cfg,"model",None), "provider": getattr(el_cfg,"provider","openai")},
                "ner": {"model": getattr(ner_cfg,"model",None), "provider": getattr(ner_cfg,"provider","openai")},
                "selfrag": {"model": getattr(selfrag_cfg,"model",None), "provider": getattr(selfrag_cfg,"provider","openai")} if args.use_selfrag else None,
            },
            "args_raw": vars(args),
        }, f, ensure_ascii=False, indent=2)

    print(f"[done] finished → {out_dir}")
    for k in ("queue_jsonl","facts_jsonl","lowconf_jsonl","ner_jsonl","ner_lowconf_jsonl","selfrag_log_jsonl","run_meta_json","errors_log"):
        print(f"[out] {k:18}: {paths[k]}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[interrupt] bye")
