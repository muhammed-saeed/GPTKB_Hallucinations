# Auto-generated GPTKB project dump
# Contains the directory tree and contents of all Python source files (.py)
# Excludes folders: old/, test/, and any starting with 'runs'

project_dump = '''
GPTKB_Hallucinations/
├── bench_runner_concurrent.py
│   --- File Content Start ---
│   #!/usr/bin/env python3
│   # bench_runner_concurrent.py
│   from __future__ import annotations
│   import argparse
│   import csv
│   import datetime as dt
│   import json
│   import os
│   import shlex
│   import subprocess
│   import sys
│   from concurrent.futures import ThreadPoolExecutor, as_completed
│   from itertools import product
│   from pathlib import Path
│   from typing import Dict, List, Tuple
│   
│   # ===================== small utils =====================
│   
│   def ts_for_dir() -> str:
│       return dt.datetime.now().strftime("%Y%m%d_%H%M%S")
│   
│   def sanitize_slug(s: str) -> str:
│       bad = '/\\?%*:|"<>'
│       out = s.strip().replace(" ", "_")
│       for ch in bad:
│           out = out.replace(ch, "")
│       return out
│   
│   def ensure_dir(p: str) -> None:
│       Path(p).mkdir(parents=True, exist_ok=True)
│   
│   def write_json(path: str, obj: dict) -> None:
│       ensure_dir(str(Path(path).parent))
│       with open(path, "w", encoding="utf-8") as f:
│           json.dump(obj, f, ensure_ascii=False, indent=2)
│   
│   def append_bench_log(root_out: str, line: str) -> None:
│       ensure_dir(root_out)
│       with open(os.path.join(root_out, "bench.log"), "a", encoding="utf-8") as f:
│           f.write(f"[{dt.datetime.now().isoformat()}] {line}\n")
│   
│   def expand_csv_header_safely(csv_path: str, new_row: Dict[str, object]) -> None:
│       """
│       Append a row to CSV while allowing new columns to appear later.
│       If the header needs to grow, rewrite file with expanded header.
│       """
│       ensure_dir(str(Path(csv_path).parent))
│       rows: List[Dict[str, object]] = []
│       existing_header: List[str] = []
│       if os.path.exists(csv_path):
│           with open(csv_path, "r", encoding="utf-8", newline="") as f:
│               r = csv.DictReader(f)
│               existing_header = r.fieldnames or []
│               for row in r:
│                   rows.append(row)
│   
│       all_keys = list(dict.fromkeys([*(existing_header or []), *list(new_row.keys())]))
│       rows.append(new_row)
│   
│       with open(csv_path, "w", encoding="utf-8", newline="") as f:
│           w = csv.DictWriter(f, fieldnames=all_keys)
│           w.writeheader()
│           for r in rows:
│               out = {k: r.get(k, "") for k in all_keys}
│               w.writerow(out)
│   
│   # ===================== profiles =====================
│   
│   PROFILE_KNOBS = {
│       "det":    {"temperature": 0.0, "top_p": 1.0,  "top_k": None, "max_tokens": 4096},
│       "medium": {"temperature": 0.7, "top_p": 0.95, "top_k": 50,   "max_tokens": 4096},
│       "wild":   {"temperature": 2.0, "top_p": 1.0,  "top_k": 100,  "max_tokens": 4096},
│   }
│   
│   # ===================== args =====================
│   
│   def build_arg_parser() -> argparse.ArgumentParser:
│       ap = argparse.ArgumentParser(
│           description="Concurrent benchmark runner for GPT-KB crawler (outer parallelism + per-run routing)."
│       )
│       ap.add_argument("--root-out", required=True,
│                       help="Root folder for all benchmark outputs (subfolders will be created).")
│       ap.add_argument("--crawler", default="crawler_batch_concurrency_topic.py",
│                       help="Crawler script to run (default: crawler_batch_concurrency_topic.py).")
│   
│       # grids
│       ap.add_argument("--domains", default="topic,general",
│                       help="Comma list: topic,general")
│       ap.add_argument("--seeds", default="Game of Thrones,Lionel Messi,World War II",
│                       help="Comma list of starting subjects.")
│       ap.add_argument("--models", default="deepseek,granite8b,gpt4o-mini",
│                       help="Comma list of model keys (must exist in settings.MODELS).")
│       ap.add_argument("--strategies", default="baseline,calibrate,icl,dont_know",
│                       help="Comma list of elicitation strategies.")
│       ap.add_argument("--profiles", default="det,medium,wild",
│                       help="Comma list of sampling profiles (det|medium|wild).")
│   
│       # crawler knobs (shared)
│       ap.add_argument("--max-depth", type=int, default=2)
│       ap.add_argument("--max-subjects", type=int, default=3,
│                       help="Hard cap of subjects per run; 0 means 'no cap' (crawler drains by hop).")
│       ap.add_argument("--max-facts-hint", type=int, default=100)
│       ap.add_argument("--ner-batch-size", type=int, default=50)
│       ap.add_argument("--concurrency", type=int, default=10,
│                       help="(legacy fallback) Per-run thread concurrency if default-concurrency not given.")
│       ap.add_argument("--ner-strategy", default="calibrate",
│                       help="NER strategy passed to crawler (often 'calibrate').")
│   
│       # OpenAI batch vs concurrency routing
│       ap.add_argument("--openai-batch-size", type=int, default=None,
│                       help="If set and model is OpenAI, pass --openai-batch and this size to the crawler.")
│       ap.add_argument("--default-concurrency", type=int, default=10,
│                       help="Per-run concurrency for non-OpenAI (and OpenAI without batch).")
│   
│       # NETWORK ROBUSTNESS (new)
│       ap.add_argument("--net-timeout", type=float, default=60.0,
│                       help="HTTP connect/read timeout in seconds (forwarded to crawler as --http-timeout and NET_TIMEOUT).")
│       ap.add_argument("--net-retries", type=int, default=6,
│                       help="HTTP retry attempts on transient errors (forwarded as --http-retries and NET_RETRIES).")
│       ap.add_argument("--net-backoff", type=float, default=0.5,
│                       help="Exponential backoff factor between retries (forwarded as --http-backoff and NET_BACKOFF).")
│   
│       # outer parallelism
│       ap.add_argument("--max-procs", type=int, default=1,
│                       help="How many crawler runs to execute in parallel (outer level).")
│   
│       # control / safety
│       ap.add_argument("--list", action="store_true",
│                       help="Only list planned runs then exit (no writes).")
│       ap.add_argument("--dry-run", action="store_true",
│                       help="Plan and write meta/CSV, but do NOT execute the crawler.")
│       ap.add_argument("--verbose", action="store_true", help="Verbose planning output.")
│       ap.add_argument("--skip-existing", action="store_true",
│                       help="If out_dir already exists, skip planning/execution for that run.")
│   
│       return ap
│   
│   # ===================== planning =====================
│   
│   def is_openai_model(model_key: str) -> bool:
│       key = (model_key or "").lower()
│       # Adjust as needed to match your settings.MODELS keys for OpenAI
│       return key in ("gpt4o-mini", "gpt-4o-mini", "gpt4o", "gpt-4o", "o3-mini", "o4-mini")
│   
│   def build_plan(args) -> List[Dict]:
│       # normalize grids
│       domains    = [s.strip() for s in args.domains.split(",") if s.strip()]
│       seeds      = [s.strip() for s in args.seeds.split(",") if s.strip()]
│       models     = [s.strip() for s in args.models.split(",") if s.strip()]
│       strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
│       profiles   = [s.strip() for s in args.profiles.split(",") if s.strip()]
│   
│       # sanity: profiles exist
│       for p in profiles:
│           if p not in PROFILE_KNOBS:
│               raise SystemExit(f"Unknown profile '{p}'. Use one of: {', '.join(PROFILE_KNOBS)}")
│   
│       plan: List[Dict] = []
│       seen = set()  # prevent duplicates
│   
│       for domain, model, strat, prof, seed in product(domains, models, strategies, profiles, seeds):
│           k = (domain, model, strat, prof, seed)
│           if k in seen:
│               continue
│           seen.add(k)
│   
│           seed_slug = sanitize_slug(seed)
│           out_dir = os.path.join(
│               args.root_out,
│               domain,
│               model,
│               strat,
│               prof,
│               seed_slug,
│               ts_for_dir(),
│           )
│   
│           if args.skip_existing and os.path.exists(out_dir):
│               if args.verbose:
│                   print(f"[bench] SKIP (exists): {out_dir}")
│               continue
│   
│           # base crawler command
│           cmd: List[str] = [
│               sys.executable, args.crawler,
│               "--seed", seed,
│               "--output-dir", out_dir,
│               "--domain", domain,
│               "--elicitation-strategy", strat,
│               "--ner-strategy", args.ner_strategy,
│               "--elicit-model-key", model,
│               "--ner-model-key", model,
│               "--max-depth", str(args.max_depth),
│               "--max-facts-hint", str(args.max_facts_hint),
│               "--max-subjects", str(args.max_subjects),
│               "--ner-batch-size", str(args.ner_batch_size),
│           ]
│   
│           # decide concurrency vs openai-batch passthrough
│           batch_mode = False
│           effective_conc = None
│           if is_openai_model(model) and args.openai_batch_size:
│               cmd += ["--openai-batch", "--openai-batch-size", str(args.openai_batch_size)]
│               batch_mode = True
│           else:
│               effective_conc = args.default_concurrency or args.concurrency or 10
│               cmd += ["--concurrency", str(effective_conc)]
│   
│           # sampling knobs from profile
│           knobs = PROFILE_KNOBS[prof]
│           if knobs.get("temperature") is not None:
│               cmd += ["--temperature", str(knobs["temperature"])]
│           if knobs.get("top_p") is not None:
│               cmd += ["--top-p", str(knobs["top_p"])]
│           if knobs.get("top_k") is not None:
│               cmd += ["--top-k", str(knobs["top_k"])]
│           if knobs.get("max_tokens") is not None:
│               cmd += ["--max-tokens", str(knobs["max_tokens"])]
│   
│           # NEW: pass network robustness knobs as flags too
│           cmd += [
│               "--http-timeout", str(args.net_timeout),
│               "--http-retries", str(args.net_retries),
│               "--http-backoff", str(args.net_backoff),
│           ]
│   
│           meta = {
│               "seed": seed,
│               "seed_slug": seed_slug,
│               "domain": domain,
│               "elicitation_strategy": strat,
│               "ner_strategy": args.ner_strategy,
│               "model": model,
│               "out_dir": out_dir,
│               "profile": prof,
│               "profile_knobs": knobs,
│               "max_depth": args.max_depth,
│               "max_subjects": args.max_subjects,
│               "max_facts_hint": args.max_facts_hint,
│               "ner_batch_size": args.ner_batch_size,
│               "crawler": args.crawler,
│               "python": sys.executable,
│               "command": " ".join(shlex.quote(c) for c in cmd),
│               "timestamp": dt.datetime.now().isoformat(),
│               "batch_mode": batch_mode,
│               "effective_concurrency": effective_conc,
│               # expose net knobs in meta (also used for env passing)
│               "net_timeout": args.net_timeout,
│               "net_retries": args.net_retries,
│               "net_backoff": args.net_backoff,
│           }
│   
│           plan.append({"cmd": cmd, "out_dir": out_dir, "meta": meta})
│   
│       return plan
│   
│   # ===================== execution helpers =====================
│   
│   def run_one(job: Dict, csv_path: str) -> Tuple[str, int]:
│       """
│       Execute a single crawler job (subprocess). Returns (out_dir, returncode).
│       Also appends a CSV row with status (OK/RC_x).
│       """
│       cmd = job["cmd"]
│       out_dir = job["out_dir"]
│       meta = job["meta"]
│   
│       # write per-run meta.json before executing
│       write_json(os.path.join(out_dir, "meta.json"), meta)
│   
│       rc = 0
│       try:
│           # Pass network knobs via env as a fallback for crawlers that read env vars
│           env = os.environ.copy()
│           env["NET_TIMEOUT"] = str(meta.get("net_timeout", 60))
│           env["NET_RETRIES"] = str(meta.get("net_retries", 6))
│           env["NET_BACKOFF"] = str(meta.get("net_backoff", 0.5))
│           rc = subprocess.run(cmd, check=False, env=env).returncode
│       except Exception:
│           rc = -1
│   
│       # append CSV row with outcome
│       csv_row = {
│           "status": "OK" if rc == 0 else f"RC_{rc}",
│           **{k: v for k, v in meta.items() if not isinstance(v, dict)}
│       }
│       expand_csv_header_safely(csv_path, csv_row)
│   
│       # tiny done marker
│       write_json(os.path.join(out_dir, "done.json"), {"returncode": rc})
│   
│       return out_dir, rc
│   
│   # ===================== main =====================
│   
│   def main():
│       args = build_arg_parser().parse_args()
│   
│       print("[bench] START", flush=True)
│       print(f"[bench] root_out={args.root_out}", flush=True)
│       print(f"[bench] crawler={args.crawler}", flush=True)
│   
│       if not os.path.exists(args.crawler):
│           print(f"[bench][ERROR] crawler not found: {args.crawler}", flush=True)
│           sys.exit(2)
│   
│       plan = build_plan(args)
│       print(f"[bench] total_planned={len(plan)}", flush=True)
│   
│       if args.verbose:
│           for i, job in enumerate(plan[:min(12, len(plan))]):
│               m = job["meta"]
│               print(f"  plan[{i}] domain={m['domain']} model={m['model']} seed={m['seed']} "
│                     f"strategy={m['elicitation_strategy']} profile={m['profile']} "
│                     f"batch={m['batch_mode']} conc={m['effective_concurrency']} → {m['out_dir']}", flush=True)
│   
│       append_bench_log(args.root_out, f"planned={len(plan)}")
│   
│       if not plan:
│           print("[bench][FATAL] No runs planned. Check your grids (--domains/--seeds/--models/--strategies/--profiles).", flush=True)
│           sys.exit(1)
│   
│       if args.list:
│           print("[bench] --list set; not executing.", flush=True)
│           return
│   
│       csv_path = os.path.join(args.root_out, "runs.csv")
│   
│       if args.dry_run:
│           # Write meta + CSV rows without executing the crawler
│           for job in plan:
│               out_dir = job["out_dir"]
│               meta = job["meta"]
│               if args.skip_existing and os.path.exists(out_dir):
│                   print(f"[bench][DRY] SKIP (exists): {out_dir}", flush=True)
│                   continue
│               write_json(os.path.join(out_dir, "meta.json"), meta)
│               csv_row = {"status": "DRY_RUN", **{k: v for k, v in meta.items() if not isinstance(v, dict)}}
│               expand_csv_header_safely(csv_path, csv_row)
│               write_json(os.path.join(out_dir, "done.json"), {"returncode": None, "dry_run": True})
│           print("[bench] DRY-RUN complete.", flush=True)
│           return
│   
│       # Execute with outer parallelism
│       max_procs = max(1, int(args.max_procs))
│       print(f"[bench] executing with max_procs={max_procs}", flush=True)
│   
│       futures = {}
│       ok = 0
│       failed = 0
│       skipped = 0
│   
│       with ThreadPoolExecutor(max_workers=max_procs) as pool:
│           for idx, job in enumerate(plan, start=1):
│               out_dir = job["out_dir"]
│   
│               if args.skip_existing and os.path.exists(out_dir):
│                   print(f"[RUN {idx}] SKIP (exists): {out_dir}", flush=True)
│                   skipped += 1
│                   continue
│   
│               print(f"\n[RUN {idx}/{len(plan)}] {job['meta']['command']}", flush=True)
│               append_bench_log(args.root_out, f"RUN {idx}/{len(plan)} {job['meta']['command']}")
│   
│               futures[pool.submit(run_one, job, csv_path)] = out_dir
│   
│           for fut in as_completed(futures):
│               out_dir, rc = fut.result()
│               if rc == 0:
│                   ok += 1
│                   print(f"[bench] OK: {out_dir}", flush=True)
│               else:
│                   failed += 1
│                   print(f"[bench] FAIL rc={rc}: {out_dir}", flush=True)
│   
│       print(f"\n[bench] DONE  ok={ok}  failed={failed}  skipped={skipped}", flush=True)
│   
│   if __name__ == "__main__":
│       main()
│   --- File Content End ---

├── prompter_parser.py
│   --- File Content Start ---
│   # # prompter_parser.py  (minimal edit)
│   # from __future__ import annotations
│   # import json
│   # from pathlib import Path
│   # from typing import Dict
│   
│   # # Only replace known {placeholder} keys; never interpret other braces.
│   # _ALLOWED_KEYS = {"subject_name", "phrases_block", "root_subject"}  # removed max_facts_hint
│   
│   # def _prompt_path(domain: str, strategy: str, ptype: str) -> Path:
│   #     # prompts/<domain>/<strategy>/<ptype>.json
│   #     return Path("prompts") / domain / strategy / f"{ptype}.json"
│   
│   # def _safe_render(template: str, vars: Dict[str, str] | None) -> str:
│   #     if not template:
│   #         return ""
│   #     if not vars:
│   #         return template
│   #     out = template
│   #     for k, v in vars.items():
│   #         if k in _ALLOWED_KEYS:
│   #             out = out.replace("{" + k + "}", str(v))
│   #     # leave ALL other { ... } untouched (JSON braces, examples, etc.)
│   #     return out
│   
│   # def get_prompt_messages(
│   #     strategy: str,
│   #     ptype: str,
│   #     *,
│   #     domain: str = "general",
│   #     vars: Dict[str, str] | None = None,
│   # ) -> list[dict]:
│   #     """
│   #     Load a prompt JSON with keys: {"system": "...", "user": "..."} and render
│   #     only whitelisted placeholders. Returns OpenAI-style messages list.
│   #     """
│   #     path = _prompt_path(domain, strategy, ptype)
│   #     if not path.exists():
│   #         raise FileNotFoundError(f"Prompt file not found: {path}")
│   
│   #     with path.open("r", encoding="utf-8") as f:
│   #         obj = json.load(f)
│   
│   #     if "system" not in obj or "user" not in obj:
│   #         raise ValueError(f"Prompt JSON must contain 'system' and 'user' keys: {path}")
│   
│   #     system_tmpl = obj.get("system", "") or ""
│   #     user_tmpl   = obj.get("user", "") or ""
│   
│   #     system_txt = _safe_render(system_tmpl, vars).strip()
│   #     user_txt   = _safe_render(user_tmpl, vars).strip()
│   
│   #     return [
│   #         {"role": "system", "content": system_txt},
│   #         {"role": "user",   "content": user_txt},
│   #     ]
│   
│   
│   # prompter_parser.py
│   from __future__ import annotations
│   import json
│   from pathlib import Path
│   from typing import Dict, List
│   
│   # Only replace known {placeholder} keys; never interpret other braces.
│   _ALLOWED_KEYS = {"subject_name", "phrases_block", "root_subject"}
│   
│   # Canonical footer we want in every elicitation *system* message
│   _ELICITATION_SYSTEM_FOOTER = (
│       "\n\nImportant:\n"
│       "- If you don’t know the subject, return an empty list.\n"
│       "- If the subject is not a named entity, return an empty list.\n"
│       "- If the subject is a named entity, include at least one triple where predicate is \"instanceOf\".\n"
│       "- Do not get too wordy.\n"
│       "- Separate several objects into multiple triples with one object."
│   )
│   
│   def _prompt_path(domain: str, strategy: str, ptype: str) -> Path:
│       # prompts/<domain>/<strategy>/<ptype>.json
│       return Path("prompts") / domain / strategy / f"{ptype}.json"
│   
│   def _safe_render(template: str, variables: Dict[str, str] | None) -> str:
│       if not template:
│           return ""
│       if not variables:
│           return template
│       out = template
│       for k, v in variables.items():
│           if k in _ALLOWED_KEYS:
│               out = out.replace("{" + k + "}", str(v))
│       # leave ALL other { ... } untouched (JSON braces, examples, etc.)
│       return out
│   
│   def _ensure_footer(system_txt: str, ptype: str) -> str:
│       """
│       Append the canonical elicitation footer to system text iff:
│         - ptype == 'elicitation', and
│         - the distinctive line isn't already present.
│       """
│       if ptype != "elicitation":
│           return system_txt or ""
│       marker = "include at least one triple where predicate is \"instanceOf\""
│       st = (system_txt or "")
│       if marker.lower() in st.lower():
│           # Assume the Important block (or equivalent) is already there.
│           return st
│       return (st.rstrip() + _ELICITATION_SYSTEM_FOOTER)
│   
│   def get_prompt_messages(
│       strategy: str,
│       ptype: str,
│       *,
│       domain: str = "general",
│       variables: Dict[str, str] | None = None,
│   ) -> List[dict]:
│       """
│       Load a prompt JSON with keys: {"system": "...", "user": "..."} and render
│       only whitelisted placeholders. Returns OpenAI-style messages list.
│   
│       Additionally, for ptype == 'elicitation', we append a canonical "Important:"
│       footer to the system message if it's not already present.
│       """
│       path = _prompt_path(domain, strategy, ptype)
│       if not path.exists():
│           raise FileNotFoundError(f"Prompt file not found: {path}")
│   
│       with path.open("r", encoding="utf-8") as f:
│           obj = json.load(f)
│   
│       system_tmpl = obj.get("system", "") or ""
│       user_tmpl   = obj.get("user", "") or ""
│   
│       system_txt = _safe_render(system_tmpl, variables).strip()
│       user_txt   = _safe_render(user_tmpl, variables).strip()
│   
│       # Ensure footer for elicitation system messages
│       system_txt = _ensure_footer(system_txt, ptype).strip()
│   
│       return [
│           {"role": "system", "content": system_txt},
│           {"role": "user",   "content": user_txt},
│       ]
│   --- File Content End ---

├── gptkb_project_dump.py
│   --- File Content Start ---
│   # Auto-generated GPTKB project dump
│   # Contains the directory tree and contents of all Python source files (.py)
│   # Excludes folders: old/, test/, and any starting with 'runs'
│   
│   project_dump = ''' + "'''" + '''
│   GPTKB_Hallucinations/
│   ├── bench_runner_concurrent.py
│   │   --- File Content Start ---
│   │   #!/usr/bin/env python3
│   │   # bench_runner_concurrent.py
│   │   from __future__ import annotations
│   │   import argparse
│   │   import csv
│   │   import datetime as dt
│   │   import json
│   │   import os
│   │   import shlex
│   │   import subprocess
│   │   import sys
│   │   from concurrent.futures import ThreadPoolExecutor, as_completed
│   │   from itertools import product
│   │   from pathlib import Path
│   │   from typing import Dict, List, Tuple
│   │   
│   │   # ===================== small utils =====================
│   │   
│   │   def ts_for_dir() -> str:
│   │       return dt.datetime.now().strftime("%Y%m%d_%H%M%S")
│   │   
│   │   def sanitize_slug(s: str) -> str:
│   │       bad = '/\\?%*:|"<>'
│   │       out = s.strip().replace(" ", "_")
│   │       for ch in bad:
│   │           out = out.replace(ch, "")
│   │       return out
│   │   
│   │   def ensure_dir(p: str) -> None:
│   │       Path(p).mkdir(parents=True, exist_ok=True)
│   │   
│   │   def write_json(path: str, obj: dict) -> None:
│   │       ensure_dir(str(Path(path).parent))
│   │       with open(path, "w", encoding="utf-8") as f:
│   │           json.dump(obj, f, ensure_ascii=False, indent=2)
│   │   
│   │   def append_bench_log(root_out: str, line: str) -> None:
│   │       ensure_dir(root_out)
│   │       with open(os.path.join(root_out, "bench.log"), "a", encoding="utf-8") as f:
│   │           f.write(f"[{dt.datetime.now().isoformat()}] {line}\n")
│   │   
│   │   def expand_csv_header_safely(csv_path: str, new_row: Dict[str, object]) -> None:
│   │       """
│   │       Append a row to CSV while allowing new columns to appear later.
│   │       If the header needs to grow, rewrite file with expanded header.
│   │       """
│   │       ensure_dir(str(Path(csv_path).parent))
│   │       rows: List[Dict[str, object]] = []
│   │       existing_header: List[str] = []
│   │       if os.path.exists(csv_path):
│   │           with open(csv_path, "r", encoding="utf-8", newline="") as f:
│   │               r = csv.DictReader(f)
│   │               existing_header = r.fieldnames or []
│   │               for row in r:
│   │                   rows.append(row)
│   │   
│   │       all_keys = list(dict.fromkeys([*(existing_header or []), *list(new_row.keys())]))
│   │       rows.append(new_row)
│   │   
│   │       with open(csv_path, "w", encoding="utf-8", newline="") as f:
│   │           w = csv.DictWriter(f, fieldnames=all_keys)
│   │           w.writeheader()
│   │           for r in rows:
│   │               out = {k: r.get(k, "") for k in all_keys}
│   │               w.writerow(out)
│   │   
│   │   # ===================== profiles =====================
│   │   
│   │   PROFILE_KNOBS = {
│   │       "det":    {"temperature": 0.0, "top_p": 1.0,  "top_k": None, "max_tokens": 4096},
│   │       "medium": {"temperature": 0.7, "top_p": 0.95, "top_k": 50,   "max_tokens": 4096},
│   │       "wild":   {"temperature": 2.0, "top_p": 1.0,  "top_k": 100,  "max_tokens": 4096},
│   │   }
│   │   
│   │   # ===================== args =====================
│   │   
│   │   def build_arg_parser() -> argparse.ArgumentParser:
│   │       ap = argparse.ArgumentParser(
│   │           description="Concurrent benchmark runner for GPT-KB crawler (outer parallelism + per-run routing)."
│   │       )
│   │       ap.add_argument("--root-out", required=True,
│   │                       help="Root folder for all benchmark outputs (subfolders will be created).")
│   │       ap.add_argument("--crawler", default="crawler_batch_concurrency_topic.py",
│   │                       help="Crawler script to run (default: crawler_batch_concurrency_topic.py).")
│   │   
│   │       # grids
│   │       ap.add_argument("--domains", default="topic,general",
│   │                       help="Comma list: topic,general")
│   │       ap.add_argument("--seeds", default="Game of Thrones,Lionel Messi,World War II",
│   │                       help="Comma list of starting subjects.")
│   │       ap.add_argument("--models", default="deepseek,granite8b,gpt4o-mini",
│   │                       help="Comma list of model keys (must exist in settings.MODELS).")
│   │       ap.add_argument("--strategies", default="baseline,calibrate,icl,dont_know",
│   │                       help="Comma list of elicitation strategies.")
│   │       ap.add_argument("--profiles", default="det,medium,wild",
│   │                       help="Comma list of sampling profiles (det|medium|wild).")
│   │   
│   │       # crawler knobs (shared)
│   │       ap.add_argument("--max-depth", type=int, default=2)
│   │       ap.add_argument("--max-subjects", type=int, default=3,
│   │                       help="Hard cap of subjects per run; 0 means 'no cap' (crawler drains by hop).")
│   │       ap.add_argument("--max-facts-hint", type=int, default=100)
│   │       ap.add_argument("--ner-batch-size", type=int, default=50)
│   │       ap.add_argument("--concurrency", type=int, default=10,
│   │                       help="(legacy fallback) Per-run thread concurrency if default-concurrency not given.")
│   │       ap.add_argument("--ner-strategy", default="calibrate",
│   │                       help="NER strategy passed to crawler (often 'calibrate').")
│   │   
│   │       # OpenAI batch vs concurrency routing
│   │       ap.add_argument("--openai-batch-size", type=int, default=None,
│   │                       help="If set and model is OpenAI, pass --openai-batch and this size to the crawler.")
│   │       ap.add_argument("--default-concurrency", type=int, default=10,
│   │                       help="Per-run concurrency for non-OpenAI (and OpenAI without batch).")
│   │   
│   │       # NETWORK ROBUSTNESS (new)
│   │       ap.add_argument("--net-timeout", type=float, default=60.0,
│   │                       help="HTTP connect/read timeout in seconds (forwarded to crawler as --http-timeout and NET_TIMEOUT).")
│   │       ap.add_argument("--net-retries", type=int, default=6,
│   │                       help="HTTP retry attempts on transient errors (forwarded as --http-retries and NET_RETRIES).")
│   │       ap.add_argument("--net-backoff", type=float, default=0.5,
│   │                       help="Exponential backoff factor between retries (forwarded as --http-backoff and NET_BACKOFF).")
│   │   
│   │       # outer parallelism
│   │       ap.add_argument("--max-procs", type=int, default=1,
│   │                       help="How many crawler runs to execute in parallel (outer level).")
│   │   
│   │       # control / safety
│   │       ap.add_argument("--list", action="store_true",
│   │                       help="Only list planned runs then exit (no writes).")
│   │       ap.add_argument("--dry-run", action="store_true",
│   │                       help="Plan and write meta/CSV, but do NOT execute the crawler.")
│   │       ap.add_argument("--verbose", action="store_true", help="Verbose planning output.")
│   │       ap.add_argument("--skip-existing", action="store_true",
│   │                       help="If out_dir already exists, skip planning/execution for that run.")
│   │   
│   │       return ap
│   │   
│   │   # ===================== planning =====================
│   │   
│   │   def is_openai_model(model_key: str) -> bool:
│   │       key = (model_key or "").lower()
│   │       # Adjust as needed to match your settings.MODELS keys for OpenAI
│   │       return key in ("gpt4o-mini", "gpt-4o-mini", "gpt4o", "gpt-4o", "o3-mini", "o4-mini")
│   │   
│   │   def build_plan(args) -> List[Dict]:
│   │       # normalize grids
│   │       domains    = [s.strip() for s in args.domains.split(",") if s.strip()]
│   │       seeds      = [s.strip() for s in args.seeds.split(",") if s.strip()]
│   │       models     = [s.strip() for s in args.models.split(",") if s.strip()]
│   │       strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
│   │       profiles   = [s.strip() for s in args.profiles.split(",") if s.strip()]
│   │   
│   │       # sanity: profiles exist
│   │       for p in profiles:
│   │           if p not in PROFILE_KNOBS:
│   │               raise SystemExit(f"Unknown profile '{p}'. Use one of: {', '.join(PROFILE_KNOBS)}")
│   │   
│   │       plan: List[Dict] = []
│   │       seen = set()  # prevent duplicates
│   │   
│   │       for domain, model, strat, prof, seed in product(domains, models, strategies, profiles, seeds):
│   │           k = (domain, model, strat, prof, seed)
│   │           if k in seen:
│   │               continue
│   │           seen.add(k)
│   │   
│   │           seed_slug = sanitize_slug(seed)
│   │           out_dir = os.path.join(
│   │               args.root_out,
│   │               domain,
│   │               model,
│   │               strat,
│   │               prof,
│   │               seed_slug,
│   │               ts_for_dir(),
│   │           )
│   │   
│   │           if args.skip_existing and os.path.exists(out_dir):
│   │               if args.verbose:
│   │                   print(f"[bench] SKIP (exists): {out_dir}")
│   │               continue
│   │   
│   │           # base crawler command
│   │           cmd: List[str] = [
│   │               sys.executable, args.crawler,
│   │               "--seed", seed,
│   │               "--output-dir", out_dir,
│   │               "--domain", domain,
│   │               "--elicitation-strategy", strat,
│   │               "--ner-strategy", args.ner_strategy,
│   │               "--elicit-model-key", model,
│   │               "--ner-model-key", model,
│   │               "--max-depth", str(args.max_depth),
│   │               "--max-facts-hint", str(args.max_facts_hint),
│   │               "--max-subjects", str(args.max_subjects),
│   │               "--ner-batch-size", str(args.ner_batch_size),
│   │           ]
│   │   
│   │           # decide concurrency vs openai-batch passthrough
│   │           batch_mode = False
│   │           effective_conc = None
│   │           if is_openai_model(model) and args.openai_batch_size:
│   │               cmd += ["--openai-batch", "--openai-batch-size", str(args.openai_batch_size)]
│   │               batch_mode = True
│   │           else:
│   │               effective_conc = args.default_concurrency or args.concurrency or 10
│   │               cmd += ["--concurrency", str(effective_conc)]
│   │   
│   │           # sampling knobs from profile
│   │           knobs = PROFILE_KNOBS[prof]
│   │           if knobs.get("temperature") is not None:
│   │               cmd += ["--temperature", str(knobs["temperature"])]
│   │           if knobs.get("top_p") is not None:
│   │               cmd += ["--top-p", str(knobs["top_p"])]
│   │           if knobs.get("top_k") is not None:
│   │               cmd += ["--top-k", str(knobs["top_k"])]
│   │           if knobs.get("max_tokens") is not None:
│   │               cmd += ["--max-tokens", str(knobs["max_tokens"])]
│   │   
│   │           # NEW: pass network robustness knobs as flags too
│   │           cmd += [
│   │               "--http-timeout", str(args.net_timeout),
│   │               "--http-retries", str(args.net_retries),
│   │               "--http-backoff", str(args.net_backoff),
│   │           ]
│   │   
│   │           meta = {
│   │               "seed": seed,
│   │               "seed_slug": seed_slug,
│   │               "domain": domain,
│   │               "elicitation_strategy": strat,
│   │               "ner_strategy": args.ner_strategy,
│   │               "model": model,
│   │               "out_dir": out_dir,
│   │               "profile": prof,
│   │               "profile_knobs": knobs,
│   │               "max_depth": args.max_depth,
│   │               "max_subjects": args.max_subjects,
│   │               "max_facts_hint": args.max_facts_hint,
│   │               "ner_batch_size": args.ner_batch_size,
│   │               "crawler": args.crawler,
│   │               "python": sys.executable,
│   │               "command": " ".join(shlex.quote(c) for c in cmd),
│   │               "timestamp": dt.datetime.now().isoformat(),
│   │               "batch_mode": batch_mode,
│   │               "effective_concurrency": effective_conc,
│   │               # expose net knobs in meta (also used for env passing)
│   │               "net_timeout": args.net_timeout,
│   │               "net_retries": args.net_retries,
│   │               "net_backoff": args.net_backoff,
│   │           }
│   │   
│   │           plan.append({"cmd": cmd, "out_dir": out_dir, "meta": meta})
│   │   
│   │       return plan
│   │   
│   │   # ===================== execution helpers =====================
│   │   
│   │   def run_one(job: Dict, csv_path: str) -> Tuple[str, int]:
│   │       """
│   │       Execute a single crawler job (subprocess). Returns (out_dir, returncode).
│   │       Also appends a CSV row with status (OK/RC_x).
│   │       """
│   │       cmd = job["cmd"]
│   │       out_dir = job["out_dir"]
│   │       meta = job["meta"]
│   │   
│   │       # write per-run meta.json before executing
│   │       write_json(os.path.join(out_dir, "meta.json"), meta)
│   │   
│   │       rc = 0
│   │       try:
│   │           # Pass network knobs via env as a fallback for crawlers that read env vars
│   │           env = os.environ.copy()
│   │           env["NET_TIMEOUT"] = str(meta.get("net_timeout", 60))
│   │           env["NET_RETRIES"] = str(meta.get("net_retries", 6))
│   │           env["NET_BACKOFF"] = str(meta.get("net_backoff", 0.5))
│   │           rc = subprocess.run(cmd, check=False, env=env).returncode
│   │       except Exception:
│   │           rc = -1
│   │   
│   │       # append CSV row with outcome
│   │       csv_row = {
│   │           "status": "OK" if rc == 0 else f"RC_{rc}",
│   │           **{k: v for k, v in meta.items() if not isinstance(v, dict)}
│   │       }
│   │       expand_csv_header_safely(csv_path, csv_row)
│   │   
│   │       # tiny done marker
│   │       write_json(os.path.join(out_dir, "done.json"), {"returncode": rc})
│   │   
│   │       return out_dir, rc
│   │   
│   │   # ===================== main =====================
│   │   
│   │   def main():
│   │       args = build_arg_parser().parse_args()
│   │   
│   │       print("[bench] START", flush=True)
│   │       print(f"[bench] root_out={args.root_out}", flush=True)
│   │       print(f"[bench] crawler={args.crawler}", flush=True)
│   │   
│   │       if not os.path.exists(args.crawler):
│   │           print(f"[bench][ERROR] crawler not found: {args.crawler}", flush=True)
│   │           sys.exit(2)
│   │   
│   │       plan = build_plan(args)
│   │       print(f"[bench] total_planned={len(plan)}", flush=True)
│   │   
│   │       if args.verbose:
│   │           for i, job in enumerate(plan[:min(12, len(plan))]):
│   │               m = job["meta"]
│   │               print(f"  plan[{i}] domain={m['domain']} model={m['model']} seed={m['seed']} "
│   │                     f"strategy={m['elicitation_strategy']} profile={m['profile']} "
│   │                     f"batch={m['batch_mode']} conc={m['effective_concurrency']} → {m['out_dir']}", flush=True)
│   │   
│   │       append_bench_log(args.root_out, f"planned={len(plan)}")
│   │   
│   │       if not plan:
│   │           print("[bench][FATAL] No runs planned. Check your grids (--domains/--seeds/--models/--strategies/--profiles).", flush=True)
│   │           sys.exit(1)
│   │   
│   │       if args.list:
│   │           print("[bench] --list set; not executing.", flush=True)
│   │           return
│   │   
│   │       csv_path = os.path.join(args.root_out, "runs.csv")
│   │   
│   │       if args.dry_run:
│   │           # Write meta + CSV rows without executing the crawler
│   │           for job in plan:
│   │               out_dir = job["out_dir"]
│   │               meta = job["meta"]
│   │               if args.skip_existing and os.path.exists(out_dir):
│   │                   print(f"[bench][DRY] SKIP (exists): {out_dir}", flush=True)
│   │                   continue
│   │               write_json(os.path.join(out_dir, "meta.json"), meta)
│   │               csv_row = {"status": "DRY_RUN", **{k: v for k, v in meta.items() if not isinstance(v, dict)}}
│   │               expand_csv_header_safely(csv_path, csv_row)
│   │               write_json(os.path.join(out_dir, "done.json"), {"returncode": None, "dry_run": True})
│   │           print("[bench] DRY-RUN complete.", flush=True)
│   │           return
│   │   
│   │       # Execute with outer parallelism
│   │       max_procs = max(1, int(args.max_procs))
│   │       print(f"[bench] executing with max_procs={max_procs}", flush=True)
│   │   
│   │       futures = {}
│   │       ok = 0
│   │       failed = 0
│   │       skipped = 0
│   │   
│   │       with ThreadPoolExecutor(max_workers=max_procs) as pool:
│   --- File Content End ---

├── __init__.py
│   --- File Content Start ---
│   ---
│   
│   ## `__init__.py`
│   
│   ```python
│   # package marker
│   --- File Content End ---

├── llm_wrapper.py
│   --- File Content Start ---
│   from llm.factory import make_llm_from_config
│   from llm.config import ModelConfig
│   
│   __all__ = ["make_llm_from_config", "ModelConfig"]
│   --- File Content End ---

├── db_models.py
│   --- File Content Start ---
│   import sqlite3
│   from typing import Iterable, Tuple
│   from settings import QUEUE_DDL, FACTS_DDL
│   
│   def _open_sqlite(path: str) -> sqlite3.Connection:
│       conn = sqlite3.connect(path, check_same_thread=False)
│       conn.execute("PRAGMA journal_mode=WAL;")
│       conn.execute("PRAGMA synchronous=NORMAL;")
│       conn.execute("PRAGMA temp_store=MEMORY;")
│       conn.execute("PRAGMA mmap_size=30000000000;")
│       conn.commit()
│       return conn
│   
│   def open_queue_db(path: str) -> sqlite3.Connection:
│       conn = _open_sqlite(path); conn.executescript(QUEUE_DDL); return conn
│   
│   def open_facts_db(path: str) -> sqlite3.Connection:
│       conn = _open_sqlite(path); conn.executescript(FACTS_DDL); return conn
│   
│   def enqueue_subjects(db: sqlite3.Connection, items: Iterable[Tuple[str, int]]):
│       cur = db.cursor()
│       for subject, hop in items:
│           cur.execute("""INSERT OR IGNORE INTO queue(subject, hop, status, retries)
│                          VALUES (?, ?, 'pending', 0)""", (subject, hop))
│       db.commit()
│   
│   def reset_working_to_pending(conn: sqlite3.Connection) -> int:
│       cur = conn.cursor()
│       cur.execute("UPDATE queue SET status='pending' WHERE status='working'")
│       conn.commit()
│       return cur.rowcount
│   
│   def queue_has_rows(conn: sqlite3.Connection) -> bool:
│       cur = conn.cursor(); cur.execute("SELECT 1 FROM queue LIMIT 1"); return cur.fetchone() is not None
│   
│   def count_queue(conn: sqlite3.Connection):
│       cur = conn.cursor()
│       cur.execute("SELECT COUNT(1) FROM queue WHERE status='pending'"); pending = cur.fetchone()[0]
│       cur.execute("SELECT COUNT(1) FROM queue WHERE status='working'"); working = cur.fetchone()[0]
│       cur.execute("SELECT COUNT(1) FROM queue WHERE status='done'");    done    = cur.fetchone()[0]
│       return done, working, pending, done + working + pending
│   
│   def write_triples_accepted(db: sqlite3.Connection, rows: Iterable[Tuple[str, str, str, int, str, str, float | None]]):
│       if not rows: return
│       cur = db.cursor()
│       cur.executemany("""INSERT OR IGNORE INTO triples_accepted
│                          (subject, predicate, object, hop, model_name, strategy, confidence)
│                          VALUES (?, ?, ?, ?, ?, ?, ?)""", rows)
│       db.commit()
│   
│   def write_triples_sink(db: sqlite3.Connection, rows: Iterable[Tuple[str, str, str, int, str, str, float | None, str]]):
│       if not rows: return
│       cur = db.cursor()
│       cur.executemany("""INSERT INTO triples_sink
│                          (subject, predicate, object, hop, model_name, strategy, confidence, reason)
│                          VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", rows)
│       db.commit()
│   --- File Content End ---

├── crawler_simple.py
│   --- File Content Start ---
│   # crawler_simple.py
│   from __future__ import annotations
│   import argparse
│   import datetime
│   import json
│   import os
│   import sqlite3
│   import time
│   import traceback
│   from typing import Dict, List, Tuple
│   
│   from dotenv import load_dotenv
│   
│   from settings import (
│       settings,
│       ELICIT_SCHEMA_BASE,
│       ELICIT_SCHEMA_CAL,
│       NER_SCHEMA_BASE,
│       NER_SCHEMA_CAL,
│   )
│   from prompter_parser import get_prompt_messages
│   from llm.factory import make_llm_from_config
│   from db_models import (
│       open_queue_db,
│       open_facts_db,
│       enqueue_subjects,
│       write_triples_accepted,
│       write_triples_sink,
│       queue_has_rows,
│       reset_working_to_pending,
│   )
│   
│   load_dotenv()
│   
│   
│   def _dbg(msg: str):
│       print(msg, flush=True)
│   
│   
│   def _ensure_output_dir(base_dir: str | None) -> str:
│       out = base_dir or os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
│       os.makedirs(out, exist_ok=True)
│       return out
│   
│   
│   def _build_paths(out_dir: str) -> dict:
│       return {
│           "queue_sqlite": os.path.join(out_dir, "queue.sqlite"),
│           "facts_sqlite": os.path.join(out_dir, "facts.sqlite"),
│           "queue_jsonl": os.path.join(out_dir, "queue.jsonl"),
│           "facts_jsonl": os.path.join(out_dir, "facts.jsonl"),
│           "queue_json": os.path.join(out_dir, "queue.json"),
│           "facts_json": os.path.join(out_dir, "facts.json"),
│           "errors_log": os.path.join(out_dir, "errors.log"),
│           "ner_jsonl": os.path.join(out_dir, "ner_decisions.jsonl"),
│           "lowconf_json": os.path.join(out_dir, "facts_lowconf.json"),
│           "lowconf_jsonl": os.path.join(out_dir, "facts_lowconf.jsonl"),
│       }
│   
│   
│   def _append_jsonl(path: str, obj: dict):
│       with open(path, "a", encoding="utf-8") as f:
│           f.write(json.dumps(obj, ensure_ascii=False) + "\n")
│   
│   
│   # -------------------- DB helpers --------------------
│   
│   def _fetch_one_pending(conn: sqlite3.Connection, max_depth: int) -> Tuple[str, int] | None:
│       cur = conn.cursor()
│       cur.execute("SELECT subject, hop FROM queue WHERE status='pending' AND hop<=? LIMIT 1", (max_depth,))
│       row = cur.fetchone()
│       if not row:
│           return None
│       s, h = row
│       cur.execute("UPDATE queue SET status='working' WHERE subject=?", (s,))
│       conn.commit()
│       return s, h
│   
│   
│   def _mark_done(conn: sqlite3.Connection, subject: str):
│       conn.execute("UPDATE queue SET status='done' WHERE subject=?", (subject,))
│       conn.commit()
│   
│   
│   def _counts(conn: sqlite3.Connection, max_depth: int):
│       cur = conn.cursor()
│       cur.execute("SELECT COUNT(1) FROM queue WHERE status='done' AND hop<=?", (max_depth,))
│       done = cur.fetchone()[0]
│       cur.execute("SELECT COUNT(1) FROM queue WHERE status='working' AND hop<=?", (max_depth,))
│       working = cur.fetchone()[0]
│       cur.execute("SELECT COUNT(1) FROM queue WHERE status='pending' AND hop<=?", (max_depth,))
│       pending = cur.fetchone()[0]
│       return done, working, pending, done + working + pending
│   
│   
│   # -------------------- Output normalization --------------------
│   
│   def _parse_obj(maybe_json) -> dict:
│       if isinstance(maybe_json, dict):
│           return maybe_json
│       if isinstance(maybe_json, str):
│           try:
│               return json.loads(maybe_json)
│           except Exception:
│               return {}
│       return {}
│   
│   def _normalize_elicitation_output(out) -> Dict[str, list]:
│       obj = _parse_obj(out)
│       facts = obj.get("facts")
│       if isinstance(facts, list):
│           return {"facts": [t for t in facts if isinstance(t, dict)]}
│       triples = obj.get("triples")
│       if isinstance(triples, list):
│           return {"facts": [t for t in triples if isinstance(t, dict)]}
│       return {"facts": []}
│   
│   def _normalize_ner_output(out) -> Dict[str, list]:
│       obj = _parse_obj(out)
│       if isinstance(obj.get("phrases"), list):
│           got = []
│           for ph in obj["phrases"]:
│               phrase = ph.get("phrase")
│               is_ne = bool(ph.get("is_ne"))
│               if isinstance(phrase, str):
│                   got.append({"phrase": phrase, "is_ne": is_ne})
│           return {"phrases": got}
│       ents = obj.get("entities")
│       if isinstance(ents, list):
│           mapped = []
│           for e in ents:
│               name = e.get("name") or e.get("phrase")
│               etype = (e.get("type") or "").strip().lower()
│               keep = e.get("keep")
│               is_ne = (etype == "ne") or (keep is True)
│               if isinstance(name, str):
│                   mapped.append({"phrase": name, "is_ne": bool(is_ne)})
│           return {"phrases": mapped}
│       return {"phrases": []}
│   
│   def _route_facts(args, facts: List[dict], hop: int, model_name: str):
│       acc, lowconf, objs = [], [], []
│       use_threshold = (args.elicitation_strategy == "calibrate")
│       thr = float(args.conf_threshold)
│   
│       for f in facts:
│           s, p, o = f.get("subject"), f.get("predicate"), f.get("object")
│           if not (isinstance(s, str) and isinstance(p, str) and isinstance(o, str)):
│               continue
│           conf = f.get("confidence")
│   
│           if use_threshold and isinstance(conf, (int, float)):
│               if conf < thr:
│                   lowconf.append({
│                       "subject": s, "predicate": p, "object": o,
│                       "hop": hop, "model": model_name,
│                       "strategy": args.elicitation_strategy,
│                       "confidence": float(conf),
│                       "threshold": thr
│                   })
│                   continue
│   
│           acc.append((s, p, o, hop, model_name, args.elicitation_strategy,
│                       conf if isinstance(conf, (int, float)) else None))
│           objs.append(o)
│   
│       return acc, lowconf, objs
│   
│   def _filter_ner_candidates(objs: List[str]) -> List[str]:
│       return sorted({o for o in objs if isinstance(o, str) and 1 <= len(o.split()) <= 6})
│   
│   def _enqueue_next(qdb, paths, phrases: List[str], hop: int, max_depth: int):
│       if not phrases:
│           return
│       next_hop = hop + 1
│       if next_hop > max_depth:
│           return
│       enqueue_subjects(qdb, ((s, next_hop) for s in phrases))
│       for s in phrases:
│           _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": next_hop})
│   
│   
│   # -------------------- Main --------------------
│   
│   def main():
│       ap = argparse.ArgumentParser(description="Simple crawler (system+user prompts from JSON).")
│       ap.add_argument("--seed", required=True)
│       ap.add_argument("--output-dir", default=None)
│   
│       # Strategies / domain
│       ap.add_argument("--elicitation-strategy", default="baseline", choices=["baseline","icl","dont_know","calibrate"])
│       ap.add_argument("--ner-strategy", default="baseline", choices=["baseline","icl","dont_know","calibrate"])
│       ap.add_argument("--domain", default="general", choices=["general","topic"])
│   
│       # Depth / batching
│       ap.add_argument("--max-depth", type=int, default=settings.MAX_DEPTH)
│       ap.add_argument("--ner-batch-size", type=int, default=settings.NER_BATCH_SIZE)
│       ap.add_argument("--max-facts-hint", default=str(settings.MAX_FACTS_HINT))
│       ap.add_argument("--conf-threshold", type=float, default=0.7)
│   
│       # Models
│       ap.add_argument("--elicit-model-key", default=settings.ELICIT_MODEL_KEY)
│       ap.add_argument("--ner-model-key", default=settings.NER_MODEL_KEY)
│   
│       # Sampler knobs (for non-Responses models)
│       ap.add_argument("--temperature", type=float, default=None)
│       ap.add_argument("--top-p", type=float, default=None)
│       ap.add_argument("--top-k", type=int, default=None)
│       ap.add_argument("--max-tokens", type=int, default=None)
│   
│       # Hard cap
│       ap.add_argument("--max-subjects", type=int, default=0)
│   
│       # Responses API extras
│       ap.add_argument("--reasoning-effort", choices=["minimal","low","medium","high"], default=None)
│       ap.add_argument("--verbosity", choices=["low","medium","high"], default=None)
│   
│       # Resume
│       ap.add_argument("--resume", action="store_true")
│       ap.add_argument("--reset-working", action="store_true")
│   
│       # Debug
│       ap.add_argument("--debug", action="store_true")
│   
│       args = ap.parse_args()
│   
│       out_dir = _ensure_output_dir(args.output_dir)
│       paths = _build_paths(out_dir)
│       _dbg(f"[simple] output_dir: {out_dir}")
│   
│       # DBs
│       qdb = open_queue_db(paths["queue_sqlite"])
│       fdb = open_facts_db(paths["facts_sqlite"])
│   
│       # Seed or resume
│       if args.resume and queue_has_rows(qdb):
│           if args.reset_working:
│               n = reset_working_to_pending(qdb)
│               _dbg(f"[simple] resume: reset {n} 'working' → 'pending'")
│           d0, w0, p0, t0 = _counts(qdb, args.max_depth)
│           _dbg(f"[simple] resume: queue found: done={d0} working={w0} pending={p0} total={t0}")
│       else:
│           enqueue_subjects(qdb, [(args.seed, 0)])
│           _dbg(f"[simple] seeded: {args.seed}")
│   
│       # Build LLMs
│       el_cfg = settings.MODELS[args.elicit_model_key].model_copy(deep=True)
│       ner_cfg = settings.MODELS[args.ner_model_key].model_copy(deep=True)
│   
│       # Respect per-provider rules
│       for cfg in (el_cfg, ner_cfg):
│           if getattr(cfg, "use_responses_api", False):
│               cfg.temperature = None
│               cfg.top_p = None
│               cfg.top_k = None
│               if cfg.extra_inputs is None:
│                   cfg.extra_inputs = {}
│               cfg.extra_inputs.setdefault("reasoning", {})
│               cfg.extra_inputs.setdefault("text", {})
│               if args.reasoning_effort:
│                   cfg.extra_inputs["reasoning"]["effort"] = args.reasoning_effort
│               if args.verbosity:
│                   cfg.extra_inputs["text"]["verbosity"] = args.verbosity
│           else:
│               if args.temperature is not None: cfg.temperature = args.temperature
│               if args.top_p is not None: cfg.top_p = args.top_p
│               if args.top_k is not None: cfg.top_k = args.top_k
│           if args.max_tokens is not None:
│               cfg.max_tokens = args.max_tokens
│           if cfg.max_tokens is None:
│               cfg.max_tokens = 2048
│   
│       el_llm = make_llm_from_config(el_cfg)
│       ner_llm = make_llm_from_config(ner_cfg)
│   
│       start = time.time()
│       subjects_elicited_total = 0
│       lowconf_accum: List[dict] = []
│   
│       while True:
│           if args.max_subjects and subjects_elicited_total >= args.max_subjects:
│               _dbg(f"[simple] max-subjects reached ({subjects_elicited_total}); stopping.")
│               break
│   
│           nxt = _fetch_one_pending(qdb, args.max_depth)
│           if not nxt:
│               d, w, p, t = _counts(qdb, args.max_depth)
│               if t == 0:
│                   _dbg("[simple] nothing to do.")
│               else:
│                   _dbg(f"[simple] queue drained: done={d} working={w} pending={p} total={t}")
│               break
│   
│           subject, hop = nxt
│           _dbg(f"[simple] eliciting '{subject}' (hop={hop})")
│   
│           try:
│               # ---------- ELICITATION ----------
│               el_messages = get_prompt_messages(
│                   args.elicitation_strategy, "elicitation",
│                   domain=args.domain,
│                   variables=dict(
│                       subject_name=subject,
│                       root_subject=args.seed,          # use seed as the topic anchor when domain == "topic"
│                       max_facts_hint=args.max_facts_hint,
│                   ),
│               )
│               if args.debug:
│                   print("\n--- ELICITATION MESSAGES ---")
│                   for m in el_messages: print(m["role"].upper()+":", m["content"][:4000])
│                   print("----------------------------\n")
│   
│               el_schema = ELICIT_SCHEMA_CAL if (args.elicitation_strategy == "calibrate") else ELICIT_SCHEMA_BASE
│               resp = el_llm(el_messages, json_schema=el_schema)
│   
│               # Normalize/salvage
│               obj = resp if isinstance(resp, dict) else _parse_obj(resp)
│               facts = []
│               if isinstance(obj.get("facts"), list):
│                   facts = [t for t in obj["facts"] if isinstance(t, dict)]
│               elif isinstance(obj.get("triples"), list):
│                   facts = [t for t in obj["triples"] if isinstance(t, dict)]
│   
│               acc, lowconf, objs = _route_facts(args, facts, hop, el_cfg.model)
│               write_triples_accepted(fdb, acc)
│   
│               for s, p, o, _, m, strat, c in acc:
│                   _append_jsonl(paths["facts_jsonl"], {
│                       "subject": s, "predicate": p, "object": o,
│                       "hop": hop, "model": m, "strategy": strat, "confidence": c
│                   })
│   
│               if lowconf:
│                   for item in lowconf:
│                       _append_jsonl(paths["lowconf_jsonl"], item)
│                   lowconf_accum.extend(lowconf)
│   
│               write_triples_sink(fdb, [])
│   
│               # ---------- NER ----------
│               cand = _filter_ner_candidates([t.get("object") for t in facts if isinstance(t, dict)])
│               next_subjects: List[str] = []
│               i = 0
│               while i < len(cand):
│                   chunk = cand[i: i + args.ner_batch_size]
│                   ner_messages = get_prompt_messages(
│                       args.ner_strategy, "ner",
│                       domain=args.domain,
│                       variables=dict(
│                           phrases_block="\n".join(chunk),
│                           root_subject=args.seed,
│                       ),
│                   )
│                   if args.debug:
│                       print("\n--- NER MESSAGES ---")
│                       for m in ner_messages: print(m["role"].upper()+":", m["content"][:4000])
│                       print("---------------------\n")
│   
│                   ner_schema = NER_SCHEMA_CAL if (args.ner_strategy == "calibrate") else NER_SCHEMA_BASE
│                   out = ner_llm(ner_messages, json_schema=ner_schema)
│   
│                   norm_ner = _normalize_ner_output(out)
│                   for ph in norm_ner.get("phrases", []):
│                       phrase = ph.get("phrase")
│                       is_ne = bool(ph.get("is_ne"))
│                       _append_jsonl(paths["ner_jsonl"], {
│                           "parent_subject": subject, "hop": hop,
│                           "phrase": phrase, "is_ne": is_ne,
│                           "ner_model": ner_cfg.model, "ner_strategy": args.ner_strategy,
│                           "domain": args.domain, "root_subject": args.seed if (args.domain == "topic") else None,
│                       })
│                       if is_ne and isinstance(phrase, str):
│                           next_subjects.append(phrase)
│   
│                   i += args.ner_batch_size
│   
│               _enqueue_next(qdb, paths, next_subjects, hop, args.max_depth)
│               _mark_done(qdb, subject)
│               subjects_elicited_total += 1
│   
│               if args.max_subjects and subjects_elicited_total >= args.max_subjects:
│                   _dbg(f"[simple] max-subjects reached ({subjects_elicited_total}); stopping.")
│                   break
│   
│           except KeyboardInterrupt:
│               n = reset_working_to_pending(qdb)
│               print(f"\n[simple] Interrupted. reset {n} 'working' → 'pending' for resume.")
│               break
│           except Exception:
│               with open(paths["errors_log"], "a", encoding="utf-8") as ef:
│                   ef.write(f"[{datetime.datetime.now().isoformat()}] subject={subject}\n{traceback.format_exc()}\n")
│               qdb.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=?", (subject,))
│               qdb.commit()
│   
│       # ----- Final snapshots -----
│       conn = sqlite3.connect(paths["queue_sqlite"])
│       cur = conn.cursor()
│       cur.execute("SELECT subject, hop, status, retries, created_at FROM queue ORDER BY hop, subject")
│       rows = cur.fetchall()
│       with open(paths["queue_json"], "w", encoding="utf-8") as f:
│           json.dump(
│               [{"subject": s, "hop": h, "status": st, "retries": r, "created_at": ts} for (s, h, st, r, ts) in rows],
│               f, ensure_ascii=False, indent=2
│           )
│       conn.close()
│   
│       conn = sqlite3.connect(paths["facts_sqlite"])
│       cur = conn.cursor()
│       cur.execute(
│           "SELECT subject, predicate, object, hop, model_name, strategy, confidence "
│           "FROM triples_accepted ORDER BY subject, predicate, object"
│       )
│       rows_acc = cur.fetchall()
│       cur.execute(
│           "SELECT subject, predicate, object, hop, model_name, strategy, confidence, reason "
│           "FROM triples_sink ORDER BY subject, predicate, object"
│       )
│       rows_sink = cur.fetchall()
│       with open(paths["facts_json"], "w", encoding="utf-8") as f:
│           json.dump(
│               {
│                   "accepted": [
│                       {"subject": s, "predicate": p, "object": o, "hop": h,
│                        "model": m, "strategy": st, "confidence": c}
│                       for (s, p, o, h, m, st, c) in rows_acc
│                   ],
│                   "sink": [
│                       {"subject": s, "predicate": p, "object": o, "hop": h,
│                        "model": m, "strategy": st, "confidence": c, "reason": r}
│                       for (s, p, o, h, m, st, c, r) in rows_sink
│                   ],
│               },
│               f, ensure_ascii=False, indent=2
│           )
│       conn.close()
│   
│       with open(paths["lowconf_json"], "w", encoding="utf-8") as f:
│           json.dump({"below_threshold": lowconf_accum}, f, ensure_ascii=False, indent=2)
│   
│       dur = time.time() - start
│       print(f"[simple] finished in {dur:.1f}s → outputs in {out_dir}")
│       print(f"[simple] queue.json        : {paths['queue_json']}")
│       print(f"[simple] facts.json        : {paths['facts_json']}")
│       print(f"[simple] facts.jsonl       : {paths['facts_jsonl']}")
│       print(f"[simple] lowconf.json      : {paths['lowconf_json']}")
│       print(f"[simple] lowconf.jsonl     : {paths['lowconf_jsonl']}")
│       print(f"[simple] ner log           : {paths['ner_jsonl']}")
│       print(f"[simple] errors.log        : {paths['errors_log']}")
│   
│   if __name__ == "__main__":
│       main()
│   --- File Content End ---

├── settings.py
│   --- File Content Start ---
│   # settings.py
│   from __future__ import annotations
│   from typing import Dict
│   from pydantic import BaseModel
│   from llm.config import ModelConfig
│   
│   # ---------- JSON Schemas ----------
│   
│   ELICIT_SCHEMA_BASE = {
│     "type": "object",
│     "properties": {
│       "facts": {"type": "array", "items": {
│         "type": "object",
│         "properties": {
│           "subject": {"type": "string"},
│           "predicate": {"type": "string"},
│           "object": {"type": "string"}
│         },
│         "required": ["subject", "predicate", "object"]
│       }}
│     },
│     "required": ["facts"]
│   }
│   
│   ELICIT_SCHEMA_CAL = {
│     "type": "object",
│     "properties": {
│       "facts": {"type": "array", "items": {
│         "type": "object",
│         "properties": {
│           "subject": {"type": "string"},
│           "predicate": {"type": "string"},
│           "object": {"type": "string"},
│           "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
│         },
│         "required": ["subject", "predicate", "object", "confidence"]
│       }}
│     },
│     "required": ["facts"]
│   }
│   
│   NER_SCHEMA_BASE = {
│     "type": "object",
│     "properties": {
│       "phrases": {"type": "array", "items": {
│         "type": "object",
│         "properties": {
│           "phrase": {"type": "string"},
│           "is_ne": {"type": "boolean"}
│         },
│         "required": ["phrase", "is_ne"]
│       }}
│     },
│     "required": ["phrases"]
│   }
│   
│   NER_SCHEMA_CAL = {
│     "type": "object",
│     "properties": {
│       "phrases": {"type": "array", "items": {
│         "type": "object",
│         "properties": {
│           "phrase": {"type": "string"},
│           "is_ne": {"type": "boolean"},
│           "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
│         },
│         "required": ["phrase", "is_ne", "confidence"]
│       }}
│     },
│     "required": ["phrases"]
│   }
│   
│   # ---------- SQLite DDL ----------
│   
│   QUEUE_DDL = """
│   CREATE TABLE IF NOT EXISTS queue(
│     subject TEXT PRIMARY KEY,
│     hop INT DEFAULT 0,
│     status TEXT DEFAULT 'pending',
│     retries INT DEFAULT 0,
│     created_at DATETIME DEFAULT CURRENT_TIMESTAMP
│   );
│   """
│   
│   FACTS_DDL = """
│   CREATE TABLE IF NOT EXISTS triples_accepted(
│     subject TEXT, predicate TEXT, object TEXT,
│     hop INT, model_name TEXT, strategy TEXT, confidence REAL,
│     PRIMARY KEY(subject,predicate,object)
│   );
│   CREATE TABLE IF NOT EXISTS triples_sink(
│     subject TEXT, predicate TEXT, object TEXT,
│     hop INT, model_name TEXT, strategy TEXT, confidence REAL, reason TEXT
│   );
│   """
│   
│   # ---------- Settings ----------
│   
│   class Settings(BaseModel):
│       CONCURRENCY: int = 8
│       MAX_DEPTH: int = 2
│       NER_BATCH_SIZE: int = 50
│       MAX_FACTS_HINT: int = 50
│   
│       MODELS: Dict[str, ModelConfig] = {
│           # -------- OpenAI (Chat Completions) --------
│           "gpt4o": ModelConfig(
│               provider="openai", model="gpt-4o",
│               api_key_env="OPENAI_API_KEY",
│               temperature=0.0, top_p=1.0, max_tokens=2000,
│               use_responses_api=False
│           ),
│           "gpt4o-mini": ModelConfig(
│               provider="openai", model="gpt-4o-mini",
│               api_key_env="OPENAI_API_KEY",
│               temperature=0.0, top_p=1.0, max_tokens=2000,
│               use_responses_api=False
│           ),
│           "gpt4-turbo": ModelConfig(
│               provider="openai", model="gpt-4-turbo",
│               api_key_env="OPENAI_API_KEY",
│               temperature=0.0, top_p=1.0, max_tokens=2000,
│               use_responses_api=False
│           ),
│   
│           # -------- OpenAI (Responses API) — GPT-5 family --------
│           "gpt-5": ModelConfig(
│               provider="openai",
│               model="gpt-5",
│               api_key_env="OPENAI_API_KEY",
│               temperature=None, top_p=None, max_tokens=2000,
│               use_responses_api=True,
│               extra_inputs={
│                   "reasoning": {"effort": "medium"},
│                   "text": {"verbosity": "medium"},
│               },
│           ),
│           "gpt-5-mini": ModelConfig(
│               provider="openai",
│               model="gpt-5-mini",
│               api_key_env="OPENAI_API_KEY",
│               temperature=None, top_p=None, max_tokens=2000,
│               use_responses_api=True,
│               extra_inputs={
│                   "reasoning": {"effort": "low"},
│                   "text": {"verbosity": "low"},
│               },
│           ),
│           "gpt-5-nano": ModelConfig(
│               provider="openai",
│               model="gpt-5-nano",
│               api_key_env="OPENAI_API_KEY",
│               use_responses_api=True,
│               extra_inputs={
│                   "reasoning": {"effort": "minimal"},
│                   "text": {"verbosity": "low"},
│               },
│               max_tokens=2000,
│           ),
│   
│           # -------- DeepSeek --------
│           "deepseek": ModelConfig(
│               provider="deepseek", model="deepseek-chat",
│               api_key_env="DEEPSEEK_API_KEY",
│               base_url="https://api.deepseek.com",
│               temperature=0.0, top_p=0.95, max_tokens=2000
│           ),
│           "deepseek-reasoner": ModelConfig(
│               provider="deepseek", model="deepseek-reasoner",
│               api_key_env="DEEPSEEK_API_KEY",
│               base_url="https://api.deepseek.com",
│               temperature=0.0, top_p=0.95, max_tokens=2000
│           ),
│   
│           # -------- Replicate core LLMs --------
│           "llama8b": ModelConfig(
│               provider="replicate", model="meta/meta-llama-3.1-8b-instruct",
│               api_key_env="REPLICATE_API_TOKEN",
│               temperature=0.6, top_p=0.9, top_k=50, max_tokens=1024,
│               extra_inputs={"system_prompt": "You are a helpful assistant.", "prompt_template": ""}
│           ),
│           "llama70b": ModelConfig(
│               provider="replicate", model="meta/meta-llama-3.1-70b-instruct",
│               api_key_env="REPLICATE_API_TOKEN",
│               temperature=0.6, top_p=0.9, top_k=50, max_tokens=1024,
│               extra_inputs={"system_prompt": "You are a helpful assistant.", "prompt_template": ""}
│           ),
│           "llama405b": ModelConfig(
│               provider="replicate", model="meta/meta-llama-3.1-405b-instruct",
│               api_key_env="REPLICATE_API_TOKEN",
│               temperature=0.6, top_p=0.9, top_k=50, max_tokens=1024,
│               extra_inputs={"system_prompt": "You are a helpful assistant.", "prompt_template": ""}
│           ),
│           "mistral7b": ModelConfig(
│               provider="replicate", model="mistralai/mistral-7b-instruct",
│               api_key_env="REPLICATE_API_TOKEN",
│               temperature=0.6, top_p=0.95, top_k=50, max_tokens=1024,
│               extra_inputs={"system_prompt": "You are a helpful assistant.", "prompt_template": ""}
│           ),
│           "mixtral8x7b": ModelConfig(
│               provider="replicate", model="mistralai/mixtral-8x7b-instruct",
│               api_key_env="REPLICATE_API_TOKEN",
│               temperature=0.6, top_p=0.95, top_k=50, max_tokens=1024,
│               extra_inputs={"system_prompt": "You are a helpful assistant.", "prompt_template": ""}
│           ),
│   
│           # -------- Replicate (Gemini / Grok / Claude) --------
│           "gemini-flash": ModelConfig(
│               provider="replicate",
│               model="google/gemini-2.5-flash",
│               api_key_env="REPLICATE_API_TOKEN",
│               temperature=0.2,
│               top_p=0.9,
│               max_tokens=1024,
│               extra_inputs={
│                   "prefer": "prompt",
│                   "dynamic_thinking": False
│               },
│           ),
│           "grok4": ModelConfig(
│               provider="replicate",
│               model="xai/grok-4",
│               api_key_env="REPLICATE_API_TOKEN",
│               temperature=0.1,
│               top_p=1.0,
│               max_tokens=2048,
│               extra_inputs={
│                   "presence_penalty": 0,
│                   "frequency_penalty": 0,
│                   "system_prompt": "You are a helpful assistant.",
│                   "prompt_template": "",
│               },
│           ),
│           "claude35h": ModelConfig(
│               provider="replicate",
│               model="anthropic/claude-3.5-haiku",
│               api_key_env="REPLICATE_API_TOKEN",
│               temperature=0.3,
│               top_p=0.9,
│               max_tokens=8192,
│               extra_inputs={
│                   "system_prompt": "You are a concise and creative assistant.",
│                   "prompt_template": "",
│               },
│           ),
│           "claude37s": ModelConfig(
│               provider="replicate",
│               model="anthropic/claude-3.7-sonnet",
│               api_key_env="REPLICATE_API_TOKEN",
│               temperature=0.2,
│               top_p=0.9,
│               max_tokens=8192,
│               extra_inputs={
│                   "extended_thinking": False,
│                   "max_image_resolution": 0.5,
│                   "thinking_budget_tokens": 1024,
│                   "system_prompt": "Return ONLY strict JSON; no prose; no fences.",
│               },
│           ),
│   
│           # -------- Replicate (others) --------
│           "gemma2b": ModelConfig(
│               provider="replicate", model="google-deepmind/gemma-2b-it",
│               api_key_env="REPLICATE_API_TOKEN",
│               temperature=0.7, top_p=0.95, top_k=50, max_tokens=200,
│               extra_inputs={"system_prompt": "You are a helpful assistant.", "prompt_template": ""}
│           ),
│           "qwen2-7b": ModelConfig(
│               provider="replicate", model="alibaba-nlp/qwen2-7b-instruct",
│               api_key_env="REPLICATE_API_TOKEN",
│               temperature=0.6, top_p=0.95, top_k=50, max_tokens=1024,
│               extra_inputs={"system_prompt": "You are a helpful assistant.", "prompt_template": ""}
│           ),
│           "falcon180b": ModelConfig(
│               provider="replicate", model="tiiuae/falcon-180b-instruct",
│               api_key_env="REPLICATE_API_TOKEN",
│               temperature=0.6, top_p=0.95, top_k=50, max_tokens=1024,
│               extra_inputs={"system_prompt": "You are a helpful assistant.", "prompt_template": ""}
│           ),
│   
│           # ------- Replicate (IBM Granite 3.3 8B Instruct) -------
│           "granite8b": ModelConfig(
│               provider="replicate",
│               model="ibm-granite/granite-3.3-8b-instruct",
│               api_key_env="REPLICATE_API_TOKEN",
│               temperature=0.6,
│               top_p=0.9,
│               top_k=50,
│               max_tokens=1024,
│               extra_inputs={
│                   "presence_penalty": 0,
│                   "frequency_penalty": 0,
│                   "add_generation_prompt": True,
│                   "stop": [],
│                   "tools": [],
│                   "chat_template_kwargs": {},
│                   "documents": [],
│                   "min_tokens": 0,
│                   "system_prompt": "Return ONLY strict JSON that validates against the provided schema.",
│               },
│           ),
│   
│           # ------- Replicate (OpenAI gpt-oss-20b) -------
│           "gpt-oss-20b": ModelConfig(
│               provider="replicate",
│               model="openai/gpt-oss-20b",
│               api_key_env="REPLICATE_API_TOKEN",
│               temperature=0.1,
│               top_p=1.0,
│               max_tokens=1024,
│               extra_inputs={
│                   "presence_penalty": 0,
│                   "frequency_penalty": 0,
│               },
│           ),
│   
│           # ------- Replicate (Qwen 3-235B) -------
│           "qwen3-235b": ModelConfig(
│               provider="replicate",
│               model="qwen/qwen3-235b-a22b-instruct-2507",
│               api_key_env="REPLICATE_API_TOKEN",
│               temperature=0.3,
│               top_p=0.9,
│               max_tokens=1536,
│               extra_inputs={
│                   "system_prompt": "Return ONLY strict JSON per schema; no prose; no fences."
│               },
│           ),
│   
│           # -------- Local via Unsloth (optional) --------
│           "smollm2-1.7b": ModelConfig(
│               provider="unsloth",
│               model="unsloth/SmolLM2-1.7B-Instruct-bnb-4bit",
│               api_key_env=None,
│               temperature=0.2, top_p=0.95, top_k=40, max_tokens=800,
│               extra_inputs={
│                   "max_seq_length": 2048,
│                   "load_in_4bit": False,
│                   "dtype": "float16",
│                   "device": "mps",
│               },
│           ),
│           "smollm2-360m": ModelConfig(
│               provider="unsloth",
│               model="unsloth/SmolLM2-360M-Instruct-bnb-4bit",
│               api_key_env=None,
│               temperature=0.2, top_p=0.95, top_k=40, max_tokens=512,
│               extra_inputs={
│                   "max_seq_length": 2048,
│                   "load_in_4bit": True,
│               },
│           ),
│       }
│   
│       # defaults; override via CLI
│       ELICIT_MODEL_KEY: str = "gpt4o-mini"
│       NER_MODEL_KEY: str = "gpt4o-mini"
│   
│   settings = Settings()
│   --- File Content End ---

├── crawler_concurrent.py
├── llm/
│   ├── config.py
│   │   --- File Content Start ---
│   │   from __future__ import annotations
│   │   from typing import Optional, Dict, Any
│   │   from pydantic import BaseModel
│   │   
│   │   class ModelConfig(BaseModel):
│   │       provider: str
│   │       model: str
│   │       api_key_env: Optional[str] = None
│   │       base_url: Optional[str] = None
│   │       temperature: Optional[float] = None
│   │       top_p: Optional[float] = None
│   │       top_k: Optional[int] = None
│   │       max_tokens: Optional[int] = None
│   │       extra_inputs: Optional[Dict[str, Any]] = None
│   │       seed: Optional[int] = None
│   │       use_responses_api: bool = False
│   │   --- File Content End ---

│   ├── unsloth_client.py
│   │   --- File Content Start ---
│   │   # llm/unsloth_client.py
│   │   from __future__ import annotations
│   │   import json
│   │   import os
│   │   import re
│   │   from typing import Any, Dict, List, Optional
│   │   
│   │   # --- Dependency gate with helpful error message --------------------------------
│   │   try:
│   │       import torch
│   │       from unsloth import FastLanguageModel
│   │       from transformers import AutoTokenizer  # noqa: imported for side-effects / tokenizer consistency
│   │   except Exception as e:
│   │       raise ImportError(
│   │           "Unsloth backend not available.\n"
│   │           "Install:\n"
│   │           "  pip install -U unsloth unsloth_zoo transformers accelerate safetensors\n"
│   │           "If you have an NVIDIA GPU (CUDA):\n"
│   │           "  pip install bitsandbytes  &&  install a CUDA build of torch\n"
│   │           f"\nOriginal import error: {e}"
│   │       )
│   │   
│   │   # --- Helpers -------------------------------------------------------------------
│   │   
│   │   JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
│   │   
│   │   
│   │   def _pick_device() -> str:
│   │       """
│   │       Choose device with environment override:
│   │         export UNSLOTH_DEVICE={cuda|mps|cpu}
│   │       """
│   │       env = (os.getenv("UNSLOTH_DEVICE") or "").strip().lower()
│   │       if env in {"cuda", "mps", "cpu"}:
│   │           return env
│   │       if torch.cuda.is_available():
│   │           return "cuda"
│   │       if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
│   │           return "mps"
│   │       return "cpu"
│   │   
│   │   
│   │   def _to_torch_dtype(name: Optional[str]) -> Optional[torch.dtype]:
│   │       """
│   │       Map string dtype names to torch dtypes. None -> auto.
│   │       Accepts: "float16", "bfloat16", "float32"
│   │       """
│   │       if name is None:
│   │           return None
│   │       name = str(name).lower()
│   │       if name in {"float16", "fp16", "f16"}:
│   │           return torch.float16
│   │       if name in {"bfloat16", "bf16"}:
│   │           return torch.bfloat16
│   │       if name in {"float32", "fp32", "f32"}:
│   │           return torch.float32
│   │       # fallback: ignore unknown and let Unsloth decide
│   │       return None
│   │   
│   │   
│   │   def _chat_to_prompt(messages: List[Dict[str, str]]) -> str:
│   │       """
│   │       Convert OpenAI-like chat messages into a single instruction prompt
│   │       that works well with *-Instruct local models.
│   │       """
│   │       sys_parts = [m["content"] for m in messages if m.get("role") == "system" and m.get("content")]
│   │       user_parts = [m["content"] for m in messages if m.get("role") == "user" and m.get("content")]
│   │       sys_txt = ("\n".join(sys_parts)).strip()
│   │       usr_txt = ("\n\n".join(user_parts)).strip()
│   │   
│   │       if sys_txt:
│   │           return (
│   │               "Below is a system rule and an instruction. Follow the system rule strictly.\n\n"
│   │               f"### System:\n{sys_txt}\n\n"
│   │               f"### Instruction:\n{usr_txt}\n\n"
│   │               "### Response:\n"
│   │           )
│   │       else:
│   │           return (
│   │               "Below is an instruction. Follow it strictly.\n\n"
│   │               f"### Instruction:\n{usr_txt}\n\n"
│   │               "### Response:\n"
│   │           )
│   │   
│   │   
│   │   def _extract_json(text: str) -> Optional[Dict[str, Any]]:
│   │       """
│   │       Best-effort JSON extractor for local model outputs:
│   │       1) Prefer fenced ```json blocks
│   │       2) Otherwise, try first balanced {...} region
│   │       """
│   │       m = JSON_BLOCK_RE.search(text)
│   │       if m:
│   │           try:
│   │               return json.loads(m.group(1))
│   │           except Exception:
│   │               pass
│   │   
│   │       # Try first balanced { ... }
│   │       start = text.find("{")
│   │       if start == -1:
│   │           return None
│   │   
│   │       depth = 0
│   │       for i in range(start, len(text)):
│   │           ch = text[i]
│   │           if ch == "{":
│   │               depth += 1
│   │           elif ch == "}":
│   │               depth -= 1
│   │               if depth == 0:
│   │                   candidate = text[start : i + 1]
│   │                   try:
│   │                       return json.loads(candidate)
│   │                   except Exception:
│   │                       break
│   │       return None
│   │   
│   │   
│   │   # --- Main client ----------------------------------------------------------------
│   │   
│   │   class UnslothLLM:
│   │       """
│   │       Minimal local LLM wrapper using Unsloth + HF Transformers.
│   │   
│   │       Usage parity with your other backends:
│   │         out = client.generate(messages, json_schema=..., temperature=..., top_p=..., top_k=..., max_tokens=..., seed=...)
│   │   
│   │       Notes for Apple Silicon (MPS):
│   │         - Set load_in_4bit=False (bitsandbytes is CUDA-only)
│   │         - Use dtype="float16" and device="mps" for best speed
│   │       """
│   │   
│   │       def __init__(
│   │           self,
│   │           model_name: str,
│   │           max_seq_length: int = 2048,
│   │           dtype: Optional[str] = None,        # "float16" | "bfloat16" | "float32" | None (auto)
│   │           load_in_4bit: bool = True,          # CUDA only; set False on Mac/CPU
│   │           device: Optional[str] = None,       # "cuda" | "mps" | "cpu" | None (auto)
│   │           trust_remote_code: bool = True,
│   │           extra: Optional[Dict[str, Any]] = None,
│   │       ):
│   │           self.model_name = model_name
│   │           self.device = device or _pick_device()
│   │           self.max_seq_length = max_seq_length
│   │           self.load_in_4bit = bool(load_in_4bit)
│   │           self.dtype = _to_torch_dtype(dtype)
│   │           self.trust_remote_code = trust_remote_code
│   │           self.extra = extra or {}
│   │   
│   │           # If not on CUDA, disable 4-bit to avoid bitsandbytes requirement.
│   │           if self.device != "cuda" and self.load_in_4bit:
│   │               self.load_in_4bit = False
│   │   
│   │           # Load model + tokenizer via Unsloth
│   │           self.model, self.tokenizer = FastLanguageModel.from_pretrained(
│   │               model_name=self.model_name,
│   │               max_seq_length=self.max_seq_length,
│   │               dtype=self.dtype,             # None => auto
│   │               load_in_4bit=self.load_in_4bit,
│   │               trust_remote_code=self.trust_remote_code,
│   │           )
│   │           FastLanguageModel.for_inference(self.model)  # enable fused kernels where available
│   │   
│   │           # Place model on device
│   │           if self.device == "cuda":
│   │               self.model = self.model.to("cuda")
│   │           elif self.device == "mps":
│   │               self.model = self.model.to("mps")
│   │           else:
│   │               self.model = self.model.to("cpu")
│   │   
│   │       def generate(
│   │           self,
│   │           messages: List[Dict[str, str]],
│   │           json_schema: Optional[Dict[str, Any]] = None,
│   │           temperature: float = 0.0,
│   │           top_p: float = 1.0,
│   │           top_k: Optional[int] = None,
│   │           max_tokens: int = 512,
│   │           seed: Optional[int] = None,
│   │           extra: Optional[Dict[str, Any]] = None,
│   │       ) -> Dict[str, Any]:
│   │           """
│   │           Returns:
│   │             - If json_schema is provided: a parsed dict (or {"_raw": "..."} if parsing failed)
│   │             - Otherwise: {"text": "..."} with raw string
│   │           """
│   │           cfg_extra = extra or self.extra or {}
│   │           gen_kwargs: Dict[str, Any] = dict(
│   │               do_sample=(temperature and temperature > 0.0) or (top_p is not None and top_p < 1.0) or (top_k is not None),
│   │               temperature=temperature if temperature is not None else 0.0,
│   │               top_p=top_p if top_p is not None else 1.0,
│   │               max_new_tokens=max_tokens if max_tokens is not None else 512,
│   │               repetition_penalty=cfg_extra.get("repetition_penalty", 1.0),
│   │           )
│   │           if top_k is not None:
│   │               gen_kwargs["top_k"] = int(top_k)
│   │           if seed is not None:
│   │               try:
│   │                   torch.manual_seed(int(seed))
│   │               except Exception:
│   │                   pass
│   │   
│   │           prompt = _chat_to_prompt(messages)
│   │   
│   │           # Strong nudge for JSON when schema requested
│   │           if json_schema is not None:
│   │               prompt += "\nReturn ONLY valid JSON. No prose, no code fences.\n"
│   │   
│   │           inputs = self.tokenizer([prompt], return_tensors="pt")
│   │           if self.device == "cuda":
│   │               inputs = {k: v.to("cuda") for k, v in inputs.items()}
│   │           elif self.device == "mps":
│   │               # MPS: tensors need to be moved individually
│   │               for k in inputs:
│   │                   inputs[k] = inputs[k].to("mps")
│   │   
│   │           # Generate (no streaming to keep API consistent with cloud backends)
│   │           outputs = self.model.generate(**inputs, **gen_kwargs)
│   │           text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
│   │   
│   │           # Keep only the assistant segment after the response marker, if present
│   │           if "### Response:" in text:
│   │               text = text.split("### Response:", 1)[-1].strip()
│   │   
│   │           if json_schema is not None:
│   │               parsed = _extract_json(text)
│   │               if parsed is None:
│   │                   # Return raw output for debugging; caller can decide how to handle
│   │                   return {"_raw": text}
│   │               return parsed
│   │   
│   │           return {"text": text}
│   │   --- File Content End ---

│   ├── factory.py
│   │   --- File Content Start ---
│   │   # llm/factory.py
│   │   from __future__ import annotations
│   │   import os
│   │   from typing import Any, List, Optional
│   │   from dotenv import load_dotenv
│   │   
│   │   from .config import ModelConfig
│   │   from .openai_client import OpenAIClient
│   │   from .replicate_client import ReplicateLLM
│   │   from .deepseek_client import DeepSeekLLM
│   │   
│   │   try:
│   │       from .unsloth_client import UnslothLLM
│   │       _HAS_UNSLOTH = True
│   │   except Exception:
│   │       _HAS_UNSLOTH = False
│   │   
│   │   load_dotenv()
│   │   
│   │   
│   │   def _get_key(env_name: Optional[str], fallbacks: Optional[List[str]] = None) -> Optional[str]:
│   │       if env_name:
│   │           v = os.getenv(env_name)
│   │           if v:
│   │               return v
│   │       if fallbacks:
│   │           for f in fallbacks:
│   │               v = os.getenv(f)
│   │               if v:
│   │                   return v
│   │       return None
│   │   
│   │   
│   │   def _is_gpt5_model(model_name: Optional[str]) -> bool:
│   │       """Heuristic: OpenAI GPT-5 family (Responses API)."""
│   │       if not model_name:
│   │           return False
│   │       return str(model_name).lower().startswith("gpt-5")
│   │   
│   │   
│   │   def make_llm_from_config(cfg: ModelConfig):
│   │       """
│   │       Returns a callable:
│   │           out = llm(messages, json_schema)
│   │       Out shape:
│   │         - with json_schema: parsed dict matching your schema (never raw string)
│   │         - without schema  : {"text": "..."}
│   │       """
│   │       provider = (cfg.provider or "").lower()
│   │   
│   │       # -------- OpenAI / compatible (single-call client) --------
│   │       if provider in ("openai", "openai_compatible"):
│   │           key = _get_key(cfg.api_key_env, ["OPENAI_API_KEY"])
│   │           if not key:
│   │               raise RuntimeError("OPENAI_API_KEY not set.")
│   │   
│   │           # Auto-select Responses API for GPT-5* models (e.g., gpt-5-nano) or when explicitly requested
│   │           use_responses_api = bool(cfg.use_responses_api or _is_gpt5_model(cfg.model))
│   │   
│   │           # Prefer cfg.base_url; otherwise OPENAI_BASE_URL; default official
│   │           base_url = cfg.base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
│   │   
│   │           # NOTE: OpenAIClient internally handles both Chat Completions and Responses API,
│   │           # controlled by use_responses_api flag; it also passes through extra_inputs
│   │           client = OpenAIClient(
│   │               model=cfg.model,
│   │               max_tokens=cfg.max_tokens or 1024,
│   │               temperature=cfg.temperature if cfg.temperature is not None else 0.0,
│   │               top_p=cfg.top_p if cfg.top_p is not None else 1.0,
│   │               api_key=key,
│   │               base_url=base_url,
│   │               extra_inputs=cfg.extra_inputs,   # for GPT-5: e.g. {"reasoning":{"effort":"minimal"}, "text":{"verbosity":"low"}}
│   │               use_responses_api=use_responses_api,
│   │           )
│   │   
│   │           def _gen(messages, json_schema=None):
│   │               return client(messages, json_schema)
│   │   
│   │           return _gen
│   │   
│   │       # -------- DeepSeek (OpenAI-compatible via base_url) --------
│   │       # if provider == "deepseek":
│   │       #     api_key = _get_key(cfg.api_key_env, ["DEEPSEEK_API_KEY"])
│   │       #     if not api_key:
│   │       #         raise RuntimeError("DEEPSEEK_API_KEY not set.")
│   │       #     client = DeepSeekLLM(
│   │       #         model=cfg.model,
│   │       #         api_key=api_key,
│   │       #         base_url=cfg.base_url or "https://api.deepseek.com",
│   │       #     )
│   │   
│   │       #     def _gen(messages, json_schema=None):
│   │       #         return client.generate(
│   │       #             messages,
│   │       #             json_schema=json_schema,
│   │       #             temperature=cfg.temperature if cfg.temperature is not None else 0.0,
│   │       #             top_p=cfg.top_p if cfg.top_p is not None else 1.0,
│   │       #             max_tokens=cfg.max_tokens or 1024,
│   │       #             seed=getattr(cfg, "seed", None),
│   │       #             extra=cfg.extra_inputs,
│   │       #         )
│   │   
│   │       #     return _gen
│   │   
│   │   
│   │       # llm/factory.py (DeepSeek section — keep this)
│   │       if provider == "deepseek":
│   │           api_key = _get_key(cfg.api_key_env, ["DEEPSEEK_API_KEY"])
│   │           if not api_key:
│   │               raise RuntimeError("DEEPSEEK_API_KEY not set.")
│   │           client = DeepSeekLLM(
│   │               model=cfg.model,
│   │               api_key=api_key,
│   │               base_url=cfg.base_url or "https://api.deepseek.com",
│   │           )
│   │   
│   │           def _gen(messages, json_schema=None):
│   │               return client.generate(
│   │                   messages,
│   │                   json_schema=json_schema,
│   │                   temperature=cfg.temperature if cfg.temperature is not None else 0.2,
│   │                   top_p=cfg.top_p if cfg.top_p is not None else 1.0,
│   │                   max_tokens=cfg.max_tokens or 2048,
│   │                   seed=getattr(cfg, "seed", None),
│   │                   extra=cfg.extra_inputs,
│   │               )
│   │   
│   │           return _gen
│   │   
│   │   
│   │       # -------- Replicate --------
│   │       # llm/factory.py (Replicate section)
│   │       # -------- Replicate --------
│   │       if provider == "replicate":
│   │           if not os.getenv("REPLICATE_API_TOKEN"):
│   │               raise RuntimeError("REPLICATE_API_TOKEN not set.")
│   │           client = ReplicateLLM(model=cfg.model)
│   │   
│   │           def _gen(messages, json_schema=None):
│   │               return client.generate(
│   │                   messages,
│   │                   json_schema=json_schema,
│   │                   temperature=cfg.temperature if cfg.temperature is not None else None,
│   │                   top_p=cfg.top_p if cfg.top_p is not None else None,
│   │                   top_k=cfg.top_k if cfg.top_k is not None else None,
│   │                   max_tokens=cfg.max_tokens if cfg.max_tokens is not None else None,
│   │                   seed=getattr(cfg, "seed", None),
│   │                   extra=cfg.extra_inputs,
│   │               )
│   │   
│   │           return _gen
│   │   
│   │   
│   │       # -------- Local via Unsloth --------
│   │       if provider == "unsloth":
│   │           if not _HAS_UNSLOTH:
│   │               raise RuntimeError("Unsloth backend not available. Install unsloth & deps or remove 'unsloth' models.")
│   │           extra = cfg.extra_inputs or {}
│   │           client = UnslothLLM(
│   │               model_name=cfg.model,
│   │               max_seq_length=int(extra.get("max_seq_length", 2048)),
│   │               dtype=extra.get("dtype"),
│   │               load_in_4bit=bool(extra.get("load_in_4bit", False)),
│   │               device=extra.get("device"),
│   │               trust_remote_code=True,
│   │               extra=extra,
│   │           )
│   │   
│   │           def _gen(messages, json_schema=None):
│   │               return client.generate(
│   │                   messages,
│   │                   json_schema=json_schema,
│   │                   temperature=cfg.temperature if cfg.temperature is not None else 0.0,
│   │                   top_p=cfg.top_p if cfg.top_p is not None else 1.0,
│   │                   top_k=cfg.top_k,
│   │                   max_tokens=cfg.max_tokens if cfg.max_tokens is not None else 512,
│   │                   seed=getattr(cfg, "seed", None),
│   │                   extra=cfg.extra_inputs,
│   │               )
│   │   
│   │           return _gen
│   │   
│   │       raise ValueError(f"Unknown provider: {cfg.provider!r}")
│   │   --- File Content End ---

│   ├── deepseek_client.py
│   │   --- File Content Start ---
│   │   # llm/deepseek_client.py
│   │   """
│   │   DeepSeek client with debug logging to identify JSON parsing issues.
│   │   """
│   │   
│   │   from __future__ import annotations
│   │   from typing import Any, Dict, List, Optional
│   │   import json
│   │   import os
│   │   import requests
│   │   from dotenv import load_dotenv
│   │   
│   │   
│   │   class DeepSeekClient:
│   │       """
│   │       DeepSeek client with detailed logging.
│   │       """
│   │   
│   │       def __init__(
│   │           self,
│   │           model: str,
│   │           api_key: str,
│   │           base_url: str = "https://api.deepseek.com",
│   │           max_tokens: int = 2048,
│   │           temperature: float = 0.2,
│   │           top_p: float = 1.0,
│   │       ):
│   │           load_dotenv()
│   │           self.model = model
│   │           self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
│   │           self.base_url = base_url.rstrip("/")
│   │           self.max_tokens = max_tokens
│   │           self.temperature = temperature
│   │           self.top_p = top_p
│   │   
│   │       def __call__(
│   │           self,
│   │           messages: List[Dict[str, str]],
│   │           json_schema: Optional[Dict[str, Any]] = None,
│   │       ) -> Dict[str, Any]:
│   │           """Direct callable interface"""
│   │           return self.generate(messages, json_schema=json_schema)
│   │   
│   │       def generate(
│   │           self,
│   │           messages: List[Dict[str, str]],
│   │           *,
│   │           json_schema: Optional[Dict[str, Any]] = None,
│   │           temperature: Optional[float] = None,
│   │           top_p: Optional[float] = None,
│   │           max_tokens: Optional[int] = None,
│   │           **kwargs,
│   │       ) -> Dict[str, Any]:
│   │           """Generate response with detailed debug logging"""
│   │           
│   │           # Use provided params or fall back to defaults
│   │           temp = temperature if temperature is not None else self.temperature
│   │           tp = top_p if top_p is not None else self.top_p
│   │           mt = max_tokens if max_tokens is not None else self.max_tokens
│   │           
│   │           # Make the API request
│   │           url = f"{self.base_url}/v1/chat/completions"
│   │           headers = {
│   │               "Authorization": f"Bearer {self.api_key}",
│   │               "Content-Type": "application/json",
│   │           }
│   │   
│   │           body = {
│   │               "model": self.model,
│   │               "messages": messages,
│   │               "temperature": temp,
│   │               "top_p": tp,
│   │               "max_tokens": mt,
│   │           }
│   │   
│   │           # Tell DeepSeek to return JSON when we request it
│   │           if json_schema:
│   │               body["response_format"] = {"type": "json_object"}
│   │   
│   │           # POST request
│   │           response = requests.post(url, headers=headers, json=body, timeout=90.0)
│   │           
│   │           if response.status_code != 200:
│   │               raise RuntimeError(f"DeepSeek API error: {response.status_code} {response.text[:200]}")
│   │   
│   │           # Extract the text response
│   │           data = response.json()
│   │           text = (data["choices"][0]["message"]["content"] or "").strip()
│   │   
│   │           # If no schema requested, just return text
│   │           if not json_schema:
│   │               return {"text": text}
│   │   
│   │           # DEBUG: Print what we're trying to parse
│   │           # print(f"[DeepSeekClient] text length: {len(text)}")
│   │           # print(f"[DeepSeekClient] text starts with: {text[:100]}")
│   │           # print(f"[DeepSeekClient] text ends with: {text[-100:]}")
│   │           # print(f"[DeepSeekClient] text type: {type(text)}")
│   │           
│   │           # Check if it's wrapped in quotes (string representation of JSON)
│   │           if text.startswith('"') and text.endswith('"'):
│   │               # print("[DeepSeekClient] WARNING: Text is quoted! Unquoting...")
│   │               text = text[1:-1]
│   │               # print(f"[DeepSeekClient] After unquote: {text[:100]}")
│   │   
│   │           # If schema requested, try to parse as JSON
│   │           try:
│   │               result = json.loads(text)
│   │               # print(f"[DeepSeekClient] ✓ json.loads() succeeded!")
│   │               # print(f"[DeepSeekClient] result type: {type(result)}")
│   │               # print(f"[DeepSeekClient] result keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
│   │               
│   │               if isinstance(result, dict):
│   │                   return result
│   │               else:
│   │                   # print(f"[DeepSeekClient] ✗ Result is not dict, got: {type(result)}")
│   │                   return {"_raw": text}
│   │                   
│   │           except json.JSONDecodeError as e:
│   │               # print(f"[DeepSeekClient] ✗ json.loads() FAILED!")
│   │               # print(f"[DeepSeekClient] Error: {e}")
│   │               # print(f"[DeepSeekClient] Error position: {e.pos}")
│   │               if e.pos is not None and e.pos < len(text):
│   │                   # print(f"[DeepSeekClient] Text around error: {text[max(0,e.pos-50):e.pos+50]}")
│   │                   a = "ok"
│   │               return {"_raw": text}
│   │   
│   │   
│   │   # For compatibility with code that expects DeepSeekLLM
│   │   DeepSeekLLM = DeepSeekClient
│   │   --- File Content End ---

│   ├── openai_client.py
│   │   --- File Content Start ---
│   │   # llm/openai_client.py
│   │   from __future__ import annotations
│   │   from typing import Any, Dict, List, Optional, Union
│   │   import json
│   │   from openai import OpenAI
│   │   
│   │   
│   │   class OpenAIClient:
│   │       """
│   │       Unified OpenAI client that can call either:
│   │         • Chat Completions API (gpt-4o, gpt-4o-mini, etc.)
│   │         • Responses API (gpt-5 family, e.g. gpt-5-nano)
│   │   
│   │       Usage:
│   │           client = OpenAIClient(
│   │               model="gpt-4o-mini",
│   │               api_key="sk-...",
│   │               base_url=None,               # or custom compatible base
│   │               max_tokens=1024,
│   │               temperature=0.0,
│   │               top_p=1.0,
│   │               use_responses_api=False,     # True for gpt-5 family
│   │               extra_inputs={
│   │                   # only used by Responses API:
│   │                   # "reasoning": {"effort": "low|medium|high|minimal"},
│   │                   # "text": {"verbosity": "low|medium|high"},
│   │               },
│   │           )
│   │           out = client(messages, json_schema=SCHEMA_OR_None)
│   │       """
│   │   
│   │       def __init__(
│   │           self,
│   │           model: str,
│   │           api_key: str,
│   │           base_url: Optional[str] = None,
│   │           max_tokens: Optional[int] = 1024,
│   │           temperature: Optional[float] = 0.0,
│   │           top_p: Optional[float] = 1.0,
│   │           use_responses_api: bool = False,
│   │           extra_inputs: Optional[Dict[str, Any]] = None,
│   │       ):
│   │           self.model = model
│   │           self.max_tokens = max_tokens
│   │           self.temperature = temperature
│   │           self.top_p = top_p
│   │           self.use_responses_api = bool(use_responses_api or (model or "").startswith("gpt-5"))
│   │           self.extra_inputs = extra_inputs or {}
│   │   
│   │           # Construct OpenAI SDK client
│   │           if base_url:
│   │               self.client = OpenAI(api_key=api_key, base_url=base_url)
│   │           else:
│   │               self.client = OpenAI(api_key=api_key)
│   │   
│   │       # ----- Public callable -----
│   │       def __call__(self, messages: List[Dict[str, str]], json_schema: Optional[Dict[str, Any]] = None):
│   │           if self.use_responses_api:
│   │               return self._call_responses(messages, json_schema)
│   │           return self._call_chat(messages, json_schema)
│   │   
│   │       # ----- Internal: Chat Completions API -----
│   │       def _call_chat(self, messages: List[Dict[str, str]], json_schema: Optional[Dict[str, Any]]):
│   │           kwargs: Dict[str, Any] = dict(
│   │               model=self.model,
│   │               messages=messages,
│   │               temperature=self.temperature,
│   │               top_p=self.top_p,
│   │               max_tokens=self.max_tokens,
│   │           )
│   │   
│   │           if json_schema:
│   │               # Chat Completions requires schema name
│   │               kwargs["response_format"] = {
│   │                   "type": "json_schema",
│   │                   "json_schema": {
│   │                       "name": "schema",
│   │                       "schema": json_schema,
│   │                   },
│   │               }
│   │   
│   │           resp = self.client.chat.completions.create(**kwargs)
│   │           text = (resp.choices[0].message.content or "").strip()
│   │   
│   │           if json_schema:
│   │               # try to parse JSON; fall back to a dict with _raw
│   │               try:
│   │                   return json.loads(text)
│   │               except Exception:
│   │                   return {"_raw": text}
│   │           else:
│   │               return {"text": text}
│   │   
│   │       # --- inside llm/openai_client.py ---
│   │   
│   │       def _call_responses(self, messages, json_schema):
│   │           """
│   │           Responses API (gpt-5 family). 
│   │           Handles both modern SDKs (with or without response_format) 
│   │           and automatically omits unsupported parameters.
│   │           """
│   │           reasoning = self.extra_inputs.get("reasoning")
│   │           text_opts = self.extra_inputs.get("text")
│   │   
│   │           # Base kwargs: omit temperature/top_p since GPT-5 disallows them
│   │           base_kwargs = {
│   │               "model": self.model,
│   │               "input": messages,
│   │               "max_output_tokens": self.max_tokens,
│   │           }
│   │   
│   │           if reasoning:
│   │               base_kwargs["reasoning"] = reasoning
│   │           if text_opts:
│   │               base_kwargs["text"] = text_opts
│   │   
│   │           # Try to include schema (new SDKs only)
│   │           if json_schema:
│   │               with_schema_kwargs = dict(base_kwargs)
│   │               with_schema_kwargs["response_format"] = {
│   │                   "type": "json_schema",
│   │                   "json_schema": {"name": "schema", "schema": json_schema},
│   │               }
│   │           else:
│   │               with_schema_kwargs = dict(base_kwargs)
│   │               with_schema_kwargs["response_format"] = {"type": "text"}
│   │   
│   │           try:
│   │               # Newer SDK (supports response_format)
│   │               resp = self.client.responses.create(**with_schema_kwargs)
│   │           except TypeError:
│   │               # Older SDK, retry without response_format
│   │               resp = self.client.responses.create(**base_kwargs)
│   │           except Exception as e:
│   │               # Some versions reject unsupported args; print and retry minimal
│   │               if "Unsupported parameter" in str(e):
│   │                   resp = self.client.responses.create(**base_kwargs)
│   │               else:
│   │                   raise
│   │   
│   │           # Extract text from output
│   │           output_text = getattr(resp, "output_text", None)
│   │           if not output_text:
│   │               try:
│   │                   parts = []
│   │                   for block in getattr(resp, "output", []) or []:
│   │                       for c in getattr(block, "content", []) or []:
│   │                           if getattr(c, "type", "") == "output_text":
│   │                               parts.append(getattr(c, "text", ""))
│   │                   output_text = "".join(parts).strip()
│   │               except Exception:
│   │                   output_text = ""
│   │   
│   │           # Return parsed JSON or raw text
│   │           if json_schema:
│   │               try:
│   │                   return json.loads(output_text)
│   │               except Exception:
│   │                   return {"_raw": output_text}
│   │           else:
│   │               return {"text": output_text}
│   │   
│   │   
│   │   __all__ = ["OpenAIClient"]
│   │   --- File Content End ---

│   ├── replicate_client.py
│   │   --- File Content Start ---
│   │   # llm/replicate_client.py
│   │   from __future__ import annotations
│   │   import os
│   │   import json
│   │   from typing import Any, Dict, List, Optional, Generator, Tuple
│   │   
│   │   from dotenv import load_dotenv
│   │   import replicate
│   │   
│   │   
│   │   # -------------------------- small helpers --------------------------
│   │   
│   │   def _minify_schema(schema: Dict[str, Any]) -> str:
│   │       try:
│   │           return json.dumps(schema, separators=(",", ":"), ensure_ascii=False)
│   │       except Exception:
│   │           return "{}"
│   │   
│   │   
│   │   def _collapse_messages(messages: List[Dict[str, str]]) -> str:
│   │       parts = []
│   │       for m in messages:
│   │           role = (m.get("role") or "user").upper()
│   │           content = (m.get("content") or "").strip()
│   │           parts.append(f"{role}: {content}")
│   │       parts.append("ASSISTANT:")
│   │       return "\n\n".join(parts)
│   │   
│   │   
│   │   def _strip_fences(text: str) -> str:
│   │       t = (text or "").strip()
│   │       if t.startswith("```"):
│   │           nl = t.find("\n")
│   │           if nl != -1:
│   │               t = t[nl + 1:].strip()
│   │           if t.endswith("```"):
│   │               t = t[:-3].strip()
│   │       return t
│   │   
│   │   
│   │   def _parse_json_best_effort(text: str) -> Dict[str, Any]:
│   │       if not text:
│   │           return {}
│   │       # 1) direct
│   │       try:
│   │           return json.loads(text)
│   │       except Exception:
│   │           pass
│   │       # 2) strip code fences
│   │       t = _strip_fences(text)
│   │       try:
│   │           return json.loads(t)
│   │       except Exception:
│   │           pass
│   │       # 3) first balanced {...}
│   │       s = t.find("{")
│   │       if s != -1:
│   │           depth = 0
│   │           in_str = False
│   │           esc = False
│   │           for i in range(s, len(t)):
│   │               ch = t[i]
│   │               if in_str:
│   │                   if esc:
│   │                       esc = False
│   │                   elif ch == "\\":
│   │                       esc = True
│   │                   elif ch == '"':
│   │                       in_str = False
│   │                   continue
│   │               if ch == '"':
│   │                   in_str = True
│   │                   continue
│   │               if ch == "{":
│   │                   depth += 1
│   │               elif ch == "}":
│   │                   depth -= 1
│   │                   if depth == 0:
│   │                       cand = t[s:i + 1]
│   │                       try:
│   │                           return json.loads(cand)
│   │                       except Exception:
│   │                           break
│   │       return {}
│   │   
│   │   
│   │   def _salvage_block(text: str, key: str) -> Dict[str, Any]:
│   │       """
│   │       Best-effort salvage when output contains the key but json.loads failed.
│   │       Try to extract balanced object or the array for that key.
│   │       """
│   │       if not text or key not in (text or ""):
│   │           return {}
│   │       t = _strip_fences(text)
│   │   
│   │       # Try a balanced object
│   │       s = t.find("{")
│   │       if s != -1:
│   │           depth = 0; in_str = False; esc = False
│   │           for i in range(s, len(t)):
│   │               ch = t[i]
│   │               if in_str:
│   │                   if esc: esc = False
│   │                   elif ch == "\\": esc = True
│   │                   elif ch == '"': in_str = False
│   │                   continue
│   │               if ch == '"': in_str = True; continue
│   │               if ch == "{": depth += 1
│   │               elif ch == "}":
│   │                   depth -= 1
│   │                   if depth == 0:
│   │                       cand = t[s:i+1]
│   │                       try:
│   │                           obj = json.loads(cand)
│   │                           if isinstance(obj, dict) and key in obj:
│   │                               return obj
│   │                       except Exception:
│   │                           break
│   │   
│   │       # Try to salvage the array value directly
│   │       for key_quoted in (f'"{key}"', f"'{key}'"):
│   │           kpos = t.find(key_quoted)
│   │           if kpos != -1:
│   │               arr_start = t.find("[", kpos)
│   │               if arr_start != -1:
│   │                   depth = 0; in_str = False; esc = False
│   │                   for i in range(arr_start, len(t)):
│   │                       ch = t[i]
│   │                       if in_str:
│   │                           if esc: esc = False
│   │                           elif ch == "\\": esc = True
│   │                           elif ch == '"': in_str = False
│   │                           continue
│   │                       if ch == '"': in_str = True; continue
│   │                       if ch == "[": depth += 1
│   │                       elif ch == "]":
│   │                           depth -= 1
│   │                           if depth == 0:
│   │                               arr_cand = t[arr_start:i+1]
│   │                               try:
│   │                                   arr = json.loads(arr_cand)
│   │                                   if isinstance(arr, list):
│   │                                       return {key: arr}
│   │                               except Exception:
│   │                                   break
│   │       return {}
│   │   
│   │   
│   │   def _parse_or_salvage(text: str, expect_key: Optional[str]) -> Dict[str, Any]:
│   │       obj = _parse_json_best_effort(text)
│   │       if obj:
│   │           return obj
│   │       if expect_key:
│   │           salv = _salvage_block(text, expect_key)
│   │           if salv:
│   │               return salv
│   │       return {}
│   │   
│   │   
│   │   def _clip01(x: Any, default: float = 0.9) -> float:
│   │       try:
│   │           v = float(x)
│   │       except Exception:
│   │           return default
│   │       if v < 0.0: return 0.0
│   │       if v > 1.0: return 1.0
│   │       return v
│   │   
│   │   
│   │   def _coerce_elicit(obj: Dict[str, Any], *, calibrated: bool) -> Dict[str, Any]:
│   │       facts = obj.get("facts")
│   │       if not isinstance(facts, list):
│   │           return {"facts": []}
│   │       out = []
│   │       for it in facts:
│   │           if not isinstance(it, dict):
│   │               continue
│   │           s = it.get("subject")
│   │           p = it.get("predicate")
│   │           o = it.get("object")
│   │           if not (isinstance(s, str) and isinstance(p, str) and (isinstance(o, str) or isinstance(o, (int, float, bool)))):
│   │               continue
│   │           if not isinstance(o, str):
│   │               o = str(o)
│   │           conf = it.get("confidence")
│   │           if calibrated:
│   │               conf = _clip01(conf, 0.9) if conf is not None else 0.9
│   │               out.append({"subject": s, "predicate": p, "object": o, "confidence": conf})
│   │           else:
│   │               out.append({"subject": s, "predicate": p, "object": o})
│   │       return {"facts": out}
│   │   
│   │   
│   │   def _coerce_ner(obj: Dict[str, Any], *, calibrated: bool) -> Dict[str, Any]:
│   │       phs = obj.get("phrases")
│   │       if not isinstance(phs, list):
│   │           return {"phrases": []}
│   │       out = []
│   │       for it in phs:
│   │           if not isinstance(it, dict):
│   │               continue
│   │           phrase = it.get("phrase")
│   │           is_ne = it.get("is_ne")
│   │           if not isinstance(phrase, str):
│   │               continue
│   │           is_ne = bool(is_ne)
│   │           if calibrated:
│   │               conf = _clip01(it.get("confidence"), 0.9)
│   │               out.append({"phrase": phrase, "is_ne": is_ne, "confidence": conf})
│   │           else:
│   │               out.append({"phrase": phrase, "is_ne": is_ne})
│   │       return {"phrases": out}
│   │   
│   │   
│   │   # -------------------------- client --------------------------
│   │   
│   │   class ReplicateLLM:
│   │       """
│   │       Replicate wrapper with:
│   │         - per-model builders (Gemini / Grok / Qwen / default)
│   │         - generate() -> JSON/text with robust parsing + single fallback to stream for Gemini
│   │         - stream_text() -> text chunks
│   │         - stream_json() -> buffers chunks and returns one final coerced JSON dict
│   │         - .env auto-load; keeps `_raw` in outputs for debugging
│   │       """
│   │   
│   │       def __init__(self, model: str, *, api_token: Optional[str] = None):
│   │           load_dotenv()
│   │           self.model = model
│   │           token = api_token or os.getenv("REPLICATE_API_TOKEN")
│   │           if not token:
│   │               raise RuntimeError("Missing REPLICATE_API_TOKEN in environment (or pass api_token=...).")
│   │           self._client = replicate.Client(api_token=token)
│   │           self._debug = os.getenv("REPLICATE_DEBUG", "") == "1"
│   │   
│   │       # --------- builders ---------
│   │   
│   │       def _inputs_common(
│   │           self,
│   │           *,
│   │           temperature: Optional[float],
│   │           top_p: Optional[float],
│   │           top_k: Optional[int],
│   │           max_tokens: Optional[int],
│   │           seed: Optional[int],
│   │           extra: Dict[str, Any],
│   │       ) -> Dict[str, Any]:
│   │           inp: Dict[str, Any] = {}
│   │           if temperature is not None: inp["temperature"] = temperature
│   │           if top_p is not None: inp["top_p"] = top_p
│   │           if top_k is not None: inp["top_k"] = top_k
│   │           if max_tokens is not None:
│   │               inp["max_tokens"] = max_tokens
│   │               inp["max_output_tokens"] = max_tokens
│   │           if seed is not None: inp["seed"] = seed
│   │           for k, v in (extra or {}).items():
│   │               inp[k] = v
│   │           return inp
│   │   
│   │       def _build_for_gemini(self, messages, json_schema, knobs) -> Dict[str, Any]:
│   │           schema_min = _minify_schema(json_schema)
│   │           system_prompt = (
│   │               "Return ONLY a single valid JSON object that matches this JSON Schema exactly. "
│   │               "No prose, no markdown, no code fences.\n"
│   │               f"SCHEMA: {schema_min}\n"
│   │               "If you truly don't know, return an empty but valid object per schema."
│   │           )
│   │           fewshot = (
│   │               "EXAMPLE:\n"
│   │               'USER: Subject: Ping\n'
│   │               'ASSISTANT: {"facts":[{"subject":"Ping","predicate":"instanceOf","object":"entity","confidence":1.0}]}\n\n'
│   │           )
│   │           prompt = fewshot + _collapse_messages(messages)
│   │           knobs.setdefault("temperature", 0.2)
│   │           knobs.setdefault("top_p", 0.9)
│   │           return {"prompt": prompt, "system_prompt": system_prompt, **knobs}
│   │   
│   │       def _build_for_grok_messages(self, messages, json_schema, knobs) -> Dict[str, Any]:
│   │           schema_min = _minify_schema(json_schema)
│   │           sys_msg = {
│   │               "role": "system",
│   │               "content": (
│   │                   "You are a JSON function. Return ONLY one JSON object validating this schema. "
│   │                   "No prose/markdown/code fences. If unsure, return an empty—but valid—object.\n"
│   │                   f"SCHEMA: {schema_min}"
│   │               ),
│   │           }
│   │           usr_msg = {"role": "user", "content": _collapse_messages(messages)}
│   │           inputs = {"messages": [sys_msg, usr_msg]}
│   │           for k in ("temperature", "top_p", "top_k", "max_tokens", "max_output_tokens", "seed"):
│   │               if k in knobs:
│   │                   inputs[k] = knobs[k]
│   │           return inputs
│   │   
│   │       def _build_for_qwen_prompt(self, messages, json_schema, knobs) -> Dict[str, Any]:
│   │           schema_min = _minify_schema(json_schema)
│   │           fewshot = (
│   │               "You must output ONE JSON object that VALIDATES this JSON Schema.\n"
│   │               "NO prose, NO markdown, NO code fences.\n"
│   │               f"SCHEMA: {schema_min}\n\n"
│   │               "EXAMPLE:\n"
│   │               'USER: Subject: Ping\n'
│   │               'ASSISTANT: {"facts":[{"subject":"Ping","predicate":"instanceOf","object":"entity","confidence":0.99}]}\n\n'
│   │           )
│   │           task = _collapse_messages(messages)
│   │           contract = (
│   │               "If you know the subject, produce 12–40 concise triples (no duplicates). "
│   │               'Always include at least one triple with predicate "instanceOf". '
│   │               'If uncertain overall, return {"facts":[]}.'
│   │           )
│   │           prompt = f"{fewshot}{task}\n\n{contract}"
│   │           knobs.setdefault("temperature", 0.3)
│   │           knobs.setdefault("top_p", 0.9)
│   │           knobs.setdefault("max_tokens", knobs.get("max_output_tokens", 1536))
│   │           return {"prompt": prompt, **knobs}
│   │   
│   │       def _build_inputs(self, messages, json_schema, knobs) -> Dict[str, Any]:
│   │           is_gemini = self.model.startswith("google/gemini")
│   │           is_grok = self.model.startswith("xai/grok-4") or "grok-4" in self.model
│   │           is_qwen = self.model.startswith("qwen/")
│   │   
│   │           if json_schema:
│   │               if is_gemini:
│   │                   return self._build_for_gemini(messages, json_schema, knobs)
│   │               if is_grok:
│   │                   return self._build_for_grok_messages(messages, json_schema, knobs)
│   │               if is_qwen:
│   │                   return self._build_for_qwen_prompt(messages, json_schema, knobs)
│   │               # default contract in system_prompt
│   │               schema_min = _minify_schema(json_schema)
│   │               system_prompt = (
│   │                   "Return ONLY a single valid JSON object matching this schema. "
│   │                   "No prose, no markdown, no code fences.\n"
│   │                   f"SCHEMA: {schema_min}"
│   │               )
│   │               prompt = _collapse_messages(messages)
│   │               return {"prompt": prompt, "system_prompt": system_prompt, **knobs}
│   │           # text mode
│   │           return {"prompt": _collapse_messages(messages), **knobs}
│   │   
│   │       # --------- internal single-call wrappers ---------
│   │   
│   │       def _blocking_once(self, inputs: Dict[str, Any]) -> str:
│   │           pred = self._client.predictions.create(model=self.model, input=inputs)
│   │           pred.wait()
│   │           return "".join(pred.output) if isinstance(pred.output, list) else (pred.output or "")
│   │   
│   │       def _stream_once(self, inputs: Dict[str, Any]) -> str:
│   │           chunks: List[str] = []
│   │           for event in replicate.stream(self.model, input=inputs):
│   │               chunks.append(str(event))
│   │           return "".join(chunks)
│   │   
│   │       # --------- schema-based coercion ---------
│   │   
│   │       def _coerce_by_schema(self, obj: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
│   │           props = (schema.get("properties") or {})
│   │           if "facts" in props:
│   │               calibrated = "confidence" in (props["facts"]["items"]["properties"] or {})
│   │               return _coerce_elicit(obj, calibrated=calibrated)
│   │           if "phrases" in props:
│   │               calibrated = "confidence" in (props["phrases"]["items"]["properties"] or {})
│   │               return _coerce_ner(obj, calibrated=calibrated)
│   │           # unknown schema → return original
│   │           return obj if isinstance(obj, dict) else {}
│   │   
│   │       # --------- public blocking API ---------
│   │   
│   │       def ping(self) -> Dict[str, Any]:
│   │           inp = {"prompt": 'Return ONLY this exact JSON: {"message":"PONG"}', "max_tokens": 32, "temperature": 0}
│   │           txt = self._blocking_once(inp)
│   │           obj = _parse_or_salvage(txt, expect_key=None)
│   │           return obj if obj else {"message": "PONG"}
│   │   
│   │       def generate(
│   │           self,
│   │           messages: List[Dict[str, str]],
│   │           *,
│   │           json_schema: Optional[Dict[str, Any]] = None,
│   │           temperature: Optional[float] = None,
│   │           top_p: Optional[float] = None,
│   │           top_k: Optional[int] = None,
│   │           max_tokens: Optional[float] = None,
│   │           seed: Optional[int] = None,
│   │           extra: Optional[Dict[str, Any]] = None,
│   │       ) -> Dict[str, Any]:
│   │           knobs = self._inputs_common(
│   │               temperature=temperature, top_p=top_p, top_k=top_k,
│   │               max_tokens=max_tokens, seed=seed, extra=extra or {},
│   │           )
│   │           inputs = self._build_inputs(messages, json_schema, knobs)
│   │   
│   │           # Text mode
│   │           if not json_schema:
│   │               text = self._blocking_once(inputs)
│   │               if self._debug:
│   │                   print("\n[replicate][raw output]\n" + text[:4000] + ("\n" if len(text) else ""), flush=True)
│   │               return {"text": text, "_raw": text}
│   │   
│   │           # JSON mode
│   │           props = (json_schema.get("properties") or {})
│   │           expect = "facts" if "facts" in props else ("phrases" if "phrases" in props else None)
│   │   
│   │           is_gemini = self.model.startswith("google/gemini")
│   │           is_grok = self.model.startswith("xai/grok-4") or "grok-4" in self.model
│   │   
│   │           # For Grok: stream-first (more reliable)
│   │           if is_grok:
│   │               text = self._stream_once(inputs)
│   │               if self._debug:
│   │                   print("\n[replicate][raw stream (grok)]\n" + text[:4000] + ("\n" if len(text) else ""), flush=True)
│   │               parsed = _parse_or_salvage(text, expect_key=expect)
│   │               result = self._coerce_by_schema(parsed, json_schema)
│   │               result["_raw"] = text
│   │               return result
│   │   
│   │           # For others (incl. Gemini): try blocking once
│   │           text = self._blocking_once(inputs)
│   │           if self._debug:
│   │               print("\n[replicate][raw output]\n" + text[:4000] + ("\n" if len(text) else ""), flush=True)
│   │           parsed = _parse_or_salvage(text, expect_key=expect)
│   │           if parsed:
│   │               result = self._coerce_by_schema(parsed, json_schema)
│   │               result["_raw"] = text
│   │               return result
│   │   
│   │           # If blocking failed and it's Gemini, do exactly ONE stream fallback
│   │           if is_gemini:
│   │               text = self._stream_once(inputs)
│   │               if self._debug:
│   │                   print("\n[replicate][raw stream (fallback gemini)]\n" + text[:4000] + ("\n" if len(text) else ""), flush=True)
│   │               parsed = _parse_or_salvage(text, expect_key=expect)
│   │               result = self._coerce_by_schema(parsed, json_schema)
│   │               result["_raw"] = text
│   │               return result
│   │   
│   │           # Otherwise: return empty-but-valid by schema, with raw attached
│   │           result = self._coerce_by_schema({}, json_schema)
│   │           result["_raw"] = text
│   │           return result
│   │   
│   │       # --------- streaming API ---------
│   │   
│   │       def stream_text(
│   │           self,
│   │           messages: List[Dict[str, str]],
│   │           *,
│   │           temperature: Optional[float] = None,
│   │           top_p: Optional[float] = None,
│   │           top_k: Optional[int] = None,
│   │           max_tokens: Optional[int] = None,
│   │           seed: Optional[int] = None,
│   │           extra: Optional[Dict[str, Any]] = None,
│   │       ) -> Generator[str, None, None]:
│   │           """
│   │           Yields raw text chunks as they arrive. (No JSON parsing.)
│   │           """
│   │           knobs = self._inputs_common(
│   │               temperature=temperature, top_p=top_p, top_k=top_k,
│   │               max_tokens=max_tokens, seed=seed, extra=extra or {},
│   │           )
│   │           inputs = self._build_inputs(messages, json_schema=None, knobs=knobs)
│   │   
│   │           for event in replicate.stream(self.model, input=inputs):
│   │               yield str(event)
│   │   
│   │       def stream_json(
│   │           self,
│   │           messages: List[Dict[str, str]],
│   │           *,
│   │           json_schema: Dict[str, Any],
│   │           temperature: Optional[float] = None,
│   │           top_p: Optional[float] = None,
│   │           top_k: Optional[int] = None,
│   │           max_tokens: Optional[int] = None,
│   │           seed: Optional[int] = None,
│   │           extra: Optional[Dict[str, Any]] = None,
│   │       ) -> Generator[Dict[str, Any], None, None]:
│   │           """
│   │           Streams text chunks, buffers them, and yields ONE final JSON dict coerced to schema.
│   │           """
│   │           buffer: List[str] = []
│   │           knobs = self._inputs_common(
│   │               temperature=temperature, top_p=top_p, top_k=top_k,
│   │               max_tokens=max_tokens, seed=seed, extra=extra or {},
│   │           )
│   │           inputs = self._build_inputs(messages, json_schema=json_schema, knobs=knobs)
│   │   
│   │           for event in replicate.stream(self.model, input=inputs):
│   │               buffer.append(str(event))
│   │   
│   │           text = "".join(buffer)
│   │           if self._debug:
│   │               print("\n[replicate][raw stream combined]\n" + text[:4000] + ("\n" if len(text) else ""), flush=True)
│   │   
│   │           props = (json_schema.get("properties") or {})
│   │           expect = "facts" if "facts" in props else ("phrases" if "phrases" in props else None)
│   │   
│   │           parsed = _parse_or_salvage(text, expect_key=expect)
│   │           result = self._coerce_by_schema(parsed, json_schema)
│   │           result["_raw"] = text
│   │           yield result
│   │   --- File Content End ---

│   ├── __pycache__/
├── core/
│   ├── pipeline_elicit.py
│   │   --- File Content Start ---
│   │   # core/pipeline_elicit.py
│   │   from __future__ import annotations
│   │   import json, re
│   │   from typing import Dict, Any, List
│   │   from pathlib import Path
│   │   
│   │   from core.prompt_loader import load_messages_from_prompt_json
│   │   from llm.factory import make_llm_from_config
│   │   from llm.config import ModelConfig
│   │   
│   │   TRIPLES_SCHEMA: Dict[str, Any] = {
│   │       "type": "object",
│   │       "properties": {
│   │           "facts": {
│   │               "type": "array",
│   │               "items": {
│   │                   "type": "object",
│   │                   "properties": {
│   │                       "subject": {"type": "string"},
│   │                       "predicate": {"type": "string"},
│   │                       "object": {"type": "string"}
│   │                   },
│   │                   "required": ["subject", "predicate", "object"],
│   │                   "additionalProperties": False
│   │               }
│   │           }
│   │       },
│   │       "required": ["facts"],
│   │       "additionalProperties": False
│   │   }
│   │   
│   │   # Best-effort cleaner for common LLM quirks
│   │   CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
│   │   
│   │   def _best_effort_parse(text: str) -> Dict[str, Any]:
│   │       if not text:
│   │           return {}
│   │       # 1) fenced block
│   │       m = CODE_FENCE_RE.search(text)
│   │       if m:
│   │           text = m.group(1)
│   │       # 2) direct JSON
│   │       try:
│   │           obj = json.loads(text)
│   │           if isinstance(obj, dict):
│   │               return obj
│   │           if isinstance(obj, str):  # JSON string containing JSON
│   │               return json.loads(obj)
│   │       except Exception:
│   │           pass
│   │       # 3) first balanced {...}
│   │       start = text.find("{")
│   │       if start != -1:
│   │           depth = 0
│   │           for i, ch in enumerate(text[start:], start):
│   │               if ch == "{": depth += 1
│   │               elif ch == "}":
│   │                   depth -= 1
│   │                   if depth == 0:
│   │                       try:
│   │                           return json.loads(text[start:i+1])
│   │                       except Exception:
│   │                           break
│   │       return {}
│   │   
│   │   def _normalize_facts_key(obj: Dict[str, Any]) -> Dict[str, Any]:
│   │       # Sometimes weird keys like '"facts"' appear; normalize them.
│   │       if "facts" in obj and isinstance(obj["facts"], list):
│   │           return obj
│   │       if '"facts"' in obj and isinstance(obj['"facts"'], list):
│   │           obj["facts"] = obj.pop('"facts"')
│   │           return obj
│   │       # Also accept 'triples' synonym if present
│   │       if "triples" in obj and isinstance(obj["triples"], list) and "facts" not in obj:
│   │           obj["facts"] = obj["triples"]
│   │           return obj
│   │       return obj
│   │   
│   │   def run_elicitation(
│   │       cfg: ModelConfig,
│   │       prompt_path: str,
│   │       subject_name: str,
│   │   ) -> Dict[str, Any]:
│   │       # Load system+user from your single JSON prompt file
│   │       messages = load_messages_from_prompt_json(prompt_path, subject_name=subject_name)
│   │   
│   │       # Build LLM for the provider/model in settings
│   │       llm = make_llm_from_config(cfg)
│   │   
│   │       # Ask for strict JSON if possible (OpenAI/DeepSeek/Replicate all supported in your codebase)
│   │       resp = llm(messages, json_schema=TRIPLES_SCHEMA)
│   │   
│   │       # Case A: schema succeeded and we got a dict with facts
│   │       if isinstance(resp, dict) and "facts" in resp:
│   │           return {"facts": resp["facts"]}
│   │   
│   │       # Case B: schema failed -> many clients return {"_raw": "..."} or {"text": "..."}
│   │       raw = ""
│   │       if isinstance(resp, dict):
│   │           raw = resp.get("_raw") or resp.get("text") or ""
│   │       elif isinstance(resp, str):
│   │           raw = resp
│   │   
│   │       parsed = _best_effort_parse(raw)
│   │       parsed = _normalize_facts_key(parsed)
│   │   
│   │       if isinstance(parsed, dict) and "facts" in parsed and isinstance(parsed["facts"], list):
│   │           return {"facts": parsed["facts"]}
│   │   
│   │       # Graceful empty result so the runner can keep going
│   │       return {"facts": []}
│   │   --- File Content End ---

│   ├── pipeline_ner.py
│   │   --- File Content Start ---
│   │   from __future__ import annotations
│   │   from typing import Dict, Any
│   │   from llm.factory import make_llm_from_config
│   │   from .prompt_loader import load_messages_from_prompt_json
│   │   from prompts.schemas import NER_SCHEMA
│   │   
│   │   def run_ner(
│   │       cfg,
│   │       prompt_path: str,
│   │       phrases_block: str,
│   │       *,
│   │       temperature: float | None = None,
│   │       top_p: float | None = None,
│   │       top_k: int | None = None,
│   │       max_tokens: int | None = None,
│   │       extra_inputs: Dict[str, Any] | None = None,
│   │   ) -> Dict[str, Any]:
│   │       """
│   │       Loads the prompt JSON, formats system+user, and calls the LLM with a strict JSON schema.
│   │       """
│   │       llm = make_llm_from_config(cfg)
│   │   
│   │       messages = load_messages_from_prompt_json(
│   │           prompt_path,
│   │           phrases_block=phrases_block
│   │       )
│   │   
│   │       out = llm(
│   │           messages,
│   │           json_schema=NER_SCHEMA
│   │       )
│   │       return out
│   │   --- File Content End ---

│   ├── settings.py
│   │   --- File Content Start ---
│   │   from __future__ import annotations
│   │   from llm.config import ModelConfig
│   │   
│   │   # Choose your default provider/model here. You can switch per script/run.
│   │   # OpenAI example (Responses or Chat Completions handled internally by your clients):
│   │   OPENAI_GENERAL = ModelConfig(
│   │       provider="openai",
│   │       model="gpt-4o-mini",          # or "gpt-5-nano" if you want Responses API automatically
│   │       api_key_env="OPENAI_API_KEY",
│   │       base_url=None,                 # or a compatible gateway
│   │       temperature=0.0,
│   │       top_p=1.0,
│   │       max_tokens=4096,
│   │       use_responses_api=False,       # True auto for gpt-5* via your OpenAIClient anyway
│   │       extra_inputs=None
│   │   )
│   │   
│   │   # DeepSeek example:
│   │   DEEPSEEK_GENERAL = ModelConfig(
│   │       provider="deepseek",
│   │       model="deepseek-chat",
│   │       api_key_env="DEEPSEEK_API_KEY",
│   │       base_url="https://api.deepseek.com",
│   │       temperature=0.2,
│   │       top_p=1.0,
│   │       max_tokens=4096
│   │   )
│   │   
│   │   # Replicate example (adjust model slug as needed)
│   │   REPLICATE_GENERAL = ModelConfig(
│   │       provider="replicate",
│   │       model="meta/meta-llama-3-8b-instruct",
│   │       api_key_env=None,
│   │       temperature=0.2,
│   │       top_p=0.9,
│   │       max_tokens=2048
│   │   )
│   │   
│   │   # Unsloth (local) example
│   │   UNSLOTH_LOCAL = ModelConfig(
│   │       provider="unsloth",
│   │       model="unsloth/Meta-Llama-3-8B-Instruct",
│   │       temperature=0.0,
│   │       top_p=1.0,
│   │       max_tokens=1024,
│   │       extra_inputs={
│   │           "max_seq_length": 4096,
│   │           "dtype": "float16",      # or "bfloat16"
│   │           "load_in_4bit": False,   # set True if CUDA + bitsandbytes
│   │           "device": None
│   │       }
│   │   )
│   │   --- File Content End ---

│   ├── prompt_loader.py
│   │   --- File Content Start ---
│   │   # core/prompt_loader.py
│   │   from __future__ import annotations
│   │   import json
│   │   from pathlib import Path
│   │   from typing import List, Dict
│   │   
│   │   def _resolve(path: str | Path) -> Path:
│   │       p = Path(path)
│   │       if p.exists():
│   │           return p
│   │       here = Path(__file__).resolve().parents[1]  # project root (.. from core/)
│   │       p2 = (here / p).resolve()
│   │       if p2.exists():
│   │           return p2
│   │       p3 = Path.cwd() / p
│   │       if p3.exists():
│   │           return p3
│   │       raise FileNotFoundError(f"Prompt not found. Tried: {p}, {p2}, {p3}")
│   │   
│   │   def load_messages_from_prompt_json(path: str | Path, **vars) -> List[Dict[str, str]]:
│   │       obj = json.loads(_resolve(path).read_text(encoding="utf-8"))
│   │       system = (obj.get("system") or "").format(**vars)
│   │       user   = (obj.get("user") or "").format(**vars)
│   │       return [
│   │           {"role": "system", "content": system.strip()},
│   │           {"role": "user",   "content": user.strip()},
│   │       ]
│   │   --- File Content End ---

│   ├── __pycache__/
├── consolidate/
├── __pycache__/
├── Evaluate/
│   ├── evaluate_kb.py
│   │   --- File Content Start ---
│   │   #!/usr/bin/env python3
│   │   # evaluate_kb.py
│   │   from __future__ import annotations
│   │   import argparse, csv, json, os, random, time, re
│   │   from pathlib import Path
│   │   from typing import Dict, List, Tuple, Iterable, Optional
│   │   
│   │   # -----------------------------
│   │   # I/O helpers
│   │   # -----------------------------
│   │   def load_triples(path: str, limit: Optional[int]=None) -> List[Dict[str,str]]:
│   │       p = Path(path)
│   │       rows: List[Dict[str,str]] = []
│   │       if p.suffix.lower() == ".jsonl":
│   │           with open(p, "r", encoding="utf-8") as f:
│   │               for i, line in enumerate(f):
│   │                   if limit and i >= limit: break
│   │                   if not line.strip(): continue
│   │                   obj = json.loads(line)
│   │                   rows.append({
│   │                       "subject": str(obj.get("subject","")).strip(),
│   │                       "predicate": str(obj.get("predicate","")).strip(),
│   │                       "object": str(obj.get("object","")).strip(),
│   │                       "class": str(obj.get("class","")).strip() if "class" in obj else ""
│   │                   })
│   │       else:
│   │           with open(p, "r", encoding="utf-8", newline="") as f:
│   │               r = csv.DictReader(f)
│   │               for i, row in enumerate(r):
│   │                   if limit and i >= limit: break
│   │                   rows.append({
│   │                       "subject": (row.get("subject") or "").strip(),
│   │                       "predicate": (row.get("predicate") or "").strip(),
│   │                       "object": (row.get("object") or "").strip(),
│   │                       "class": (row.get("class") or "").strip()
│   │                   })
│   │       # basic cleanup
│   │       rows = [t for t in rows if t["subject"] and t["predicate"] and t["object"]]
│   │       return rows
│   │   
│   │   def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
│   │       Path(path).parent.mkdir(parents=True, exist_ok=True)
│   │       with open(path, "w", encoding="utf-8") as f:
│   │           for r in rows:
│   │               f.write(json.dumps(r, ensure_ascii=False) + "\n")
│   │   
│   │   # -----------------------------
│   │   # Sampling
│   │   # -----------------------------
│   │   def sample_entities(triples: List[Dict[str,str]], n: int) -> List[str]:
│   │       subjects = list({t["subject"] for t in triples})
│   │       random.shuffle(subjects)
│   │       return subjects[:min(n, len(subjects))]
│   │   
│   │   def sample_triples(triples: List[Dict[str,str]], n: int) -> List[Dict[str,str]]:
│   │       n = min(n, len(triples))
│   │       return random.sample(triples, n) if n < len(triples) else triples
│   │   
│   │   # -----------------------------
│   │   # Web search adapter (implement one)
│   │   # -----------------------------
│   │   class SearchResult(Dict[str,str]): pass
│   │   
│   │   def search_snippets(query: str, k: int = 5) -> List[SearchResult]:
│   │       """
│   │       Implement ONE of the following and leave the others commented.
│   │   
│   │       Option A: Bing Web Search API (recommended)
│   │           - Set env BING_API_KEY
│   │           - pip install requests
│   │           - Endpoint: https://api.bing.microsoft.com/v7.0/search?q=<query>
│   │           - Return top k snippets
│   │   
│   │       Option B: SerpAPI (Google wrapper)
│   │           - Set env SERPAPI_KEY
│   │           - Endpoint: https://serpapi.com/search.json?q=<query>&engine=google
│   │   
│   │       Option C: Local/offline fallback
│   │           - Return [] to mark as unverifiable (dry runs)
│   │       """
│   │       BING_KEY = os.getenv("BING_API_KEY")
│   │       SERP_KEY = os.getenv("SERPAPI_KEY")
│   │   
│   │       if BING_KEY:
│   │           import requests
│   │           url = "https://api.bing.microsoft.com/v7.0/search"
│   │           headers = {"Ocp-Apim-Subscription-Key": BING_KEY}
│   │           params = {"q": query, "mkt": "en-US", "count": k}
│   │           r = requests.get(url, headers=headers, params=params, timeout=30)
│   │           r.raise_for_status()
│   │           web = r.json().get("webPages", {}).get("value", []) if isinstance(r.json(), dict) else []
│   │           out = []
│   │           for w in web[:k]:
│   │               out.append({"title": w.get("name",""), "snippet": w.get("snippet",""), "url": w.get("url","")})
│   │           return out
│   │   
│   │       if SERP_KEY:
│   │           import requests
│   │           url = "https://serpapi.com/search.json"
│   │           params = {"q": query, "engine": "google", "api_key": SERP_KEY, "num": k}
│   │           r = requests.get(url, params=params, timeout=30)
│   │           r.raise_for_status()
│   │           results = r.json().get("organic_results", [])
│   │           out = []
│   │           for w in results[:k]:
│   │               out.append({"title": w.get("title",""), "snippet": w.get("snippet",""), "url": w.get("link","")})
│   │           return out
│   │   
│   │       # Dry/offline: return nothing -> counts as unverifiable unless judged otherwise
│   │       return []
│   │   
│   │   # -----------------------------
│   │   # LLM judge adapter (implement one)
│   │   # -----------------------------
│   │   def llm_judge(prompt: str, system: Optional[str]=None) -> str:
│   │       """
│   │       Return ONE token string label from allowed set, given the prompt context.
│   │   
│   │       Implement one of:
│   │       - OpenAI Chat Completions via OPENAI_API_KEY (gpt-4o, gpt-4o-mini, etc.)
│   │       - Ollama (local) calling e.g., llama3.1
│   │       - Any HTTP LLM you have
│   │   
│   │       For simplicity here we implement OpenAI if OPENAI_API_KEY is set; else a dummy.
│   │       """
│   │       OPENAI_KEY = os.getenv("OPENAI_API_KEY")
│   │       if OPENAI_KEY:
│   │           import requests
│   │           url = "https://api.openai.com/v1/chat/completions"
│   │           headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
│   │           model = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
│   │           messages = []
│   │           if system:
│   │               messages.append({"role":"system","content":system})
│   │           messages.append({"role":"user","content":prompt})
│   │           data = {
│   │               "model": model,
│   │               "messages": messages,
│   │               "temperature": 0.0,
│   │               "max_tokens": 4  # we want a single-word label
│   │           }
│   │           r = requests.post(url, headers=headers, json=data, timeout=60)
│   │           r.raise_for_status()
│   │           out = r.json()["choices"][0]["message"]["content"].strip()
│   │           return out
│   │       # Fallback: deterministic 'plausible' so the pipeline runs
│   │       return "plausible"
│   │   
│   │   # -----------------------------
│   │   # Prompts (NLI-style judging)
│   │   # -----------------------------
│   │   ENTITY_PROMPT = """You are an expert verifier.
│   │   Given an entity label and {k} web snippets, decide one label:
│   │   - "verifiable" (snippets clearly support the entity exists as labeled),
│   │   - "plausible" (likely exists but evidence is indirect/weak),
│   │   - "unverifiable" (no support found in snippets).
│   │   
│   │   Respond with exactly one word: verifiable | plausible | unverifiable.
│   │   
│   │   Entity: {entity}
│   │   
│   │   Snippets:
│   │   {snips}
│   │   """
│   │   
│   │   TRIPLE_PROMPT = """You are an expert verifier.
│   │   Given a triple (subject, predicate, object) and {k} web snippets (retrieved with subject and object terms),
│   │   decide one label:
│   │   - "entailed" (snippets clearly support the triple),
│   │   - "plausible" (consistent but not explicitly stated),
│   │   - "implausible" (unlikely given snippets),
│   │   - "false" (contradicted by snippets).
│   │   
│   │   Respond with exactly one word: entailed | plausible | implausible | false.
│   │   
│   │   Triple:
│   │   subject = {subj}
│   │   predicate = {pred}
│   │   object = {obj}
│   │   
│   │   Snippets:
│   │   {snips}
│   │   """
│   │   
│   │   def format_snippets(snips: List[SearchResult]) -> str:
│   │       out = []
│   │       for i, s in enumerate(snips, 1):
│   │           out.append(f"[{i}] {s.get('title','')}\n{s.get('snippet','')}\n{ s.get('url','') }")
│   │       return "\n\n".join(out) if out else "(no snippets)"
│   │   
│   │   # -----------------------------
│   │   # Evaluations
│   │   # -----------------------------
│   │   def eval_entities(entities: List[str], k_snips: int, sleep: float) -> Dict[str,int]:
│   │       counts = {"verifiable":0, "plausible":0, "unverifiable":0}
│   │       per = []
│   │       for e in entities:
│   │           snips = search_snippets(e, k=k_snips)
│   │           prompt = ENTITY_PROMPT.format(entity=e, k=len(snips), snips=format_snippets(snips))
│   │           label = llm_judge(prompt).strip().lower()
│   │           label = {"verifiable":"verifiable","plausible":"plausible","unverifiable":"unverifiable"}.get(label,"unverifiable")
│   │           counts[label] += 1
│   │           per.append({"entity": e, "label": label})
│   │           if sleep: time.sleep(sleep)
│   │       return {"counts": counts, "details": per}
│   │   
│   │   def eval_triples(tris: List[Dict[str,str]], k_snips: int, sleep: float) -> Dict[str,int]:
│   │       counts = {"entailed":0, "plausible":0, "implausible":0, "false":0}
│   │       per = []
│   │       for t in tris:
│   │           # Following the paper, query with subject + object (keeps it cheap & general)
│   │           q = f"{t['subject']} {t['object']}"
│   │           snips = search_snippets(q, k=k_snips)
│   │           prompt = TRIPLE_PROMPT.format(
│   │               subj=t["subject"], pred=t["predicate"], obj=t["object"],
│   │               k=len(snips), snips=format_snippets(snips)
│   │           )
│   │           label = llm_judge(prompt).strip().lower()
│   │           label = {"entailed":"entailed","plausible":"plausible","implausible":"implausible","false":"false"}.get(label,"plausible")
│   │           counts[label] += 1
│   │           per.append({**t, "label": label})
│   │           if sleep: time.sleep(sleep)
│   │       return {"counts": counts, "details": per}
│   │   
│   │   # -----------------------------
│   │   # Simple structural checks (optional but useful)
│   │   # -----------------------------
│   │   def check_symmetry(triples: List[Dict[str,str]],
│   │                      symm_predicates: List[str] = ["spouse"]) -> Dict[str, float]:
│   │       # % of symmetric edges that are mirrored
│   │       idx = {}
│   │       for t in triples:
│   │           idx.setdefault((t["predicate"].lower(), t["subject"].lower(), t["object"].lower()), True)
│   │       out = {}
│   │       for p in symm_predicates:
│   │           p_low = p.lower()
│   │           pairs = [(t["subject"].lower(), t["object"].lower())
│   │                    for t in triples if t["predicate"].lower() == p_low]
│   │           if not pairs: 
│   │               out[p] = 0.0
│   │               continue
│   │           mirrored = 0
│   │           for s,o in pairs:
│   │               if (p_low, o, s) in idx:
│   │                   mirrored += 1
│   │           out[p] = mirrored / len(pairs)
│   │       return out
│   │   
│   │   def check_inverse(triples: List[Dict[str,str]],
│   │                     inv_map: Dict[str,str] = {"parent_company": "subsidiary", "subsidiary":"parent_company"}) -> Dict[str, float]:
│   │       idx = {}
│   │       for t in triples:
│   │           idx.setdefault((t["predicate"].lower(), t["subject"].lower(), t["object"].lower()), True)
│   │       out = {}
│   │       for p, q in inv_map.items():
│   │           p_low, q_low = p.lower(), q.lower()
│   │           pairs = [(t["subject"].lower(), t["object"].lower())
│   │                    for t in triples if t["predicate"].lower() == p_low]
│   │           if not pairs:
│   │               out[p] = 0.0
│   │               continue
│   │           mirrored = 0
│   │           for s,o in pairs:
│   │               if (q_low, o, s) in idx:
│   │                   mirrored += 1
│   │           out[p] = mirrored / len(pairs)
│   │       return out
│   │   
│   │   # -----------------------------
│   │   # Main
│   │   # -----------------------------
│   │   def main():
│   │       ap = argparse.ArgumentParser("Evaluate a KB (entity+triple verifiability) with web snippets + LLM judge.")
│   │       ap.add_argument("--kb", required=True, help="Path to triples file (.jsonl or .csv) with subject,predicate,object[,class].")
│   │       ap.add_argument("--seed", type=int, default=0, help="Random seed.")
│   │       ap.add_argument("--sample-entities", type=int, default=1000)
│   │       ap.add_argument("--sample-triples", type=int, default=1000)
│   │       ap.add_argument("--snippets", type=int, default=5, help="#web snippets per query")
│   │       ap.add_argument("--sleep", type=float, default=0.2, help="Politeness delay between API calls (seconds).")
│   │       ap.add_argument("--out-dir", default="runs/Eval", help="Directory to write JSONL outputs and a summary.json.")
│   │       ap.add_argument("--skip-entities", action="store_true")
│   │       ap.add_argument("--skip-triples", action="store_true")
│   │       ap.add_argument("--no-structure", action="store_true")
│   │       args = ap.parse_args()
│   │   
│   │       random.seed(args.seed)
│   │       triples = load_triples(args.kb)
│   │       Path(args.out_dir).mkdir(parents=True, exist_ok=True)
│   │   
│   │       summary = {
│   │           "kb": args.kb,
│   │           "n_triples_loaded": len(triples),
│   │           "judge_model": os.getenv("JUDGE_MODEL", "gpt-4o-mini (or fallback)"),
│   │           "search": "bing" if os.getenv("BING_API_KEY") else ("serpapi" if os.getenv("SERPAPI_KEY") else "none")
│   │       }
│   │   
│   │       if not args.skip_entities:
│   │           entities = sample_entities(triples, args.sample_entities)
│   │           e_res = eval_entities(entities, k_snips=args.snippets, sleep=args.sleep)
│   │           write_jsonl(os.path.join(args.out_dir, "entities_labeled.jsonl"), e_res["details"])
│   │           ce = e_res["counts"]
│   │           total_e = sum(ce.values()) or 1
│   │           summary["entities"] = {
│   │               **ce,
│   │               "verifiable_pct": round(100*ce["verifiable"]/total_e,1),
│   │               "plausible_pct": round(100*ce["plausible"]/total_e,1),
│   │               "unverifiable_pct": round(100*ce["unverifiable"]/total_e,1),
│   │               "n": total_e
│   │           }
│   │   
│   │       if not args.skip_triples:
│   │           sample = sample_triples(triples, args.sample_triples)
│   │           t_res = eval_triples(sample, k_snips=args.snippets, sleep=args.sleep)
│   │           write_jsonl(os.path.join(args.out_dir, "triples_labeled.jsonl"), t_res["details"])
│   │           ct = t_res["counts"]
│   │           total_t = sum(ct.values()) or 1
│   │           summary["triples"] = {
│   │               **ct,
│   │               "entailed_pct": round(100*ct["entailed"]/total_t,1),
│   │               "plausible_pct": round(100*ct["plausible"]/total_t,1),
│   │               "implausible_pct": round(100*ct["implausible"]/total_t,1),
│   │               "false_pct": round(100*ct["false"]/total_t,1),
│   │               "n": total_t
│   │           }
│   │   
│   │       if not args.no_structure:
│   │           summary["structure"] = {
│   │               "symmetry_spouse": check_symmetry(triples).get("spouse", 0.0),
│   │               "inverse_parent_company": check_inverse(triples).get("parent_company", 0.0)
│   │           }
│   │   
│   │       with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
│   │           json.dump(summary, f, ensure_ascii=False, indent=2)
│   │   
│   │       # console summary
│   │       print(json.dumps(summary, indent=2))
│   │   
│   │   if __name__ == "__main__":
│   │       main()
│   │   --- File Content End ---

├── prompts/
│   ├── _prompt_utils.py
│   │   --- File Content Start ---
│   │   from __future__ import annotations
│   │   import json
│   │   from pathlib import Path
│   │   from typing import List, Dict, Any
│   │   
│   │   def load_messages_from_prompt_json(path: str | Path, **vars) -> List[Dict[str, str]]:
│   │       """
│   │       Read a prompt JSON file with:
│   │         { "system": "...", "user": "..." }
│   │       and return OpenAI-like messages after Python .format(**vars).
│   │       """
│   │       obj = json.loads(Path(path).read_text(encoding="utf-8"))
│   │       system = (obj.get("system") or "").format(**vars)
│   │       user   = (obj.get("user") or "").format(**vars)
│   │       return [{"role":"system","content":system}, {"role":"user","content":user}]
│   │   --- File Content End ---

│   ├── schemas.py
│   │   --- File Content Start ---
│   │   # JSON Schemas you can pass as response_format for strict JSON.
│   │   ELICITATION_SCHEMA = {
│   │       "type": "object",
│   │       "properties": {
│   │           "facts": {
│   │               "type": "array",
│   │               "items": {
│   │                   "type": "object",
│   │                   "additionalProperties": False,
│   │                   "required": ["subject", "predicate", "object"],
│   │                   "properties": {
│   │                       "subject": {"type": "string"},
│   │                       "predicate": {"type": "string"},
│   │                       "object": {"type": "string"}
│   │                   }
│   │               }
│   │           }
│   │       },
│   │       "required": ["facts"],
│   │       "additionalProperties": False
│   │   }
│   │   
│   │   ELICITATION_WITH_CONFIDENCE_SCHEMA = {
│   │       "type": "object",
│   │       "properties": {
│   │           "facts": {
│   │               "type": "array",
│   │               "items": {
│   │                   "type": "object",
│   │                   "additionalProperties": False,
│   │                   "required": ["subject", "predicate", "object", "confidence"],
│   │                   "properties": {
│   │                       "subject": {"type": "string"},
│   │                       "predicate": {"type": "string"},
│   │                       "object": {"type": "string"},
│   │                       "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
│   │                   }
│   │               }
│   │           }
│   │       },
│   │       "required": ["facts"],
│   │       "additionalProperties": False
│   │   }
│   │   
│   │   NER_SCHEMA = {
│   │       "type": "object",
│   │       "properties": {
│   │           "entities": {
│   │               "type": "array",
│   │               "items": {
│   │                   "type": "object",
│   │                   "additionalProperties": False,
│   │                   "required": ["name", "type", "keep"],
│   │                   "properties": {
│   │                       "name": {"type": "string"},
│   │                       "type": {"type": "string", "enum": ["NE", "Literal", "Noise"]},
│   │                       "keep": {"type": "boolean"}
│   │                   }
│   │               }
│   │           }
│   │       },
│   │       "required": ["entities"],
│   │       "additionalProperties": False
│   │   }
│   │   --- File Content End ---

│   ├── general/
│   │   ├── calibration/
│   │   ├── ICL/
│   │   ├── baseline/
│   │   ├── dont_know/
│   ├── topicsnotTerminate/
│   │   ├── basline/
│   │   ├── calibration/
│   │   ├── ICL/
│   │   ├── dont_know/
│   ├── topic/
│   │   ├── ICL/
│   │   ├── baseline/
│   │   ├── calibrate/
│   │   ├── dont_know/
│   ├── __pycache__/
├── db/
│   ├── models.py
│   │   --- File Content Start ---
│   │   from datetime import datetime
│   │   from enum import Enum
│   │   
│   │   import sqlalchemy as sa
│   │   from sqlalchemy import Index
│   │   from sqlmodel import Field, SQLModel
│   │   
│   │   
│   │   class NodeType(Enum):
│   │       UNDEFINED = "undefined"
│   │       LITERAL = "literal"
│   │       INSTANCE = "instance"
│   │   
│   │   
│   │   class JobType(Enum):
│   │       ELICITATION = "elicitation"
│   │       NAMED_ENTITY_RECOGNITION = "ner"
│   │   
│   │   
│   │   class Node(SQLModel, table=True):
│   │       name: str = Field(primary_key=True)
│   │       type: str = Field(
│   │           default=NodeType.UNDEFINED.value,
│   │           index=True
│   │       )
│   │   
│   │       batch_id: str | None = Field(default=None, foreign_key="batch.id",
│   │                                    index=True)
│   │   
│   │       creating_batch_id: str | None = Field(default=None,  # seed subject
│   │                                             foreign_key="batch.id",
│   │                                             index=True)
│   │   
│   │       first_parent: str | None = Field(default=None,  # seed subject
│   │                                        index=True)
│   │   
│   │       bfs_level: int = Field(nullable=False, index=True)
│   │   
│   │       created_at: datetime | None = Field(
│   │           default=None,
│   │           sa_type=sa.DateTime(timezone=True),
│   │           sa_column_kwargs={"server_default": sa.func.now()},
│   │           nullable=False,
│   │       )
│   │   
│   │       def __repr__(self):
│   │           return f"< Node : {self.name} >"
│   │   
│   │   
│   │   class Batch(SQLModel, table=True):
│   │       id: str = Field(primary_key=True)
│   │       input_file_id: str
│   │       status: str = Field(index=True)
│   │       output_file_id: str | None = Field(default=None)
│   │   
│   │       job_type: str = Field(index=True, nullable=False)
│   │   
│   │       created_at: datetime | None = Field(
│   │           default=None,
│   │           sa_type=sa.DateTime(timezone=True),
│   │           sa_column_kwargs={"server_default": sa.func.now()},
│   │           nullable=False,
│   │       )
│   │   
│   │       def __repr__(self):
│   │           return f"Batch {self.id} ({self.status})"
│   │   
│   │   
│   │   class Predicate(SQLModel, table=True):
│   │       name: str = Field(primary_key=True)
│   │   
│   │       creating_batch_id: str = Field(nullable=False,
│   │                                      foreign_key="batch.id",
│   │                                      index=True)
│   │   
│   │       created_at: datetime | None = Field(
│   │           default=None,
│   │           sa_type=sa.DateTime(timezone=True),
│   │           sa_column_kwargs={"server_default": sa.func.now()},
│   │           nullable=False,
│   │       )
│   │   
│   │       def __repr__(self):
│   │           return f"< Predicate : {self.name} >"
│   │   
│   │   
│   │   class Triple(SQLModel, table=True):
│   │       id: int | None = Field(default=None, primary_key=True)
│   │   
│   │       subject: str = Field(index=True, nullable=False)
│   │       predicate: str = Field(index=True, nullable=False)
│   │       object: str = Field(index=True, nullable=False)
│   │   
│   │       creating_batch_id: str = Field(nullable=False,
│   │                                      foreign_key="batch.id",
│   │                                      index=True)
│   │   
│   │       created_at: datetime | None = Field(
│   │           default=None,
│   │           sa_type=sa.DateTime(timezone=True),
│   │           sa_column_kwargs={"server_default": sa.func.now()},
│   │           nullable=False,
│   │       )
│   │   
│   │       __table_args__ = (
│   │           Index(
│   │               "ix_triple_subject_predicate_object",
│   │               "subject", "predicate", "object",
│   │               unique=True
│   │           ),
│   │       )
│   │   
│   │   
│   │   class FailedSubject(SQLModel, table=True):
│   │       name: str = Field(primary_key=True)
│   │       error: str = Field(index=True, nullable=False)
│   │       batch_id: str = Field(index=True, nullable=False, foreign_key="batch.id")
│   │   
│   │       created_at: datetime | None = Field(
│   │           default=None,
│   │           sa_type=sa.DateTime(timezone=True),
│   │           sa_column_kwargs={"server_default": sa.func.now()},
│   │           nullable=False,
│   │       )
│   │   --- File Content End ---

│   ├── __init__.py
│   ├── __pycache__/
├── .git/
│   ├── objects/
│   │   ├── 61/
│   │   ├── 0d/
│   │   ├── 95/
│   │   ├── 59/
│   │   ├── 92/
│   │   ├── 0c/
│   │   ├── 66/
│   │   ├── 3e/
│   │   ├── 50/
│   │   ├── 68/
│   │   ├── 57/
│   │   ├── 3b/
│   │   ├── 6f/
│   │   ├── 9b/
│   │   ├── 9e/
│   │   ├── 04/
│   │   ├── 6a/
│   │   ├── 32/
│   │   ├── 35/
│   │   ├── 69/
│   │   ├── 3c/
│   │   ├── 56/
│   │   ├── 51/
│   │   ├── 3d/
│   │   ├── 58/
│   │   ├── 67/
│   │   ├── 0b/
│   │   ├── 94/
│   │   ├── 0e/
│   │   ├── 60/
│   │   ├── 34/
│   │   ├── 5a/
│   │   ├── 5f/
│   │   ├── 33/
│   │   ├── 05/
│   │   ├── 9d/
│   │   ├── 9c/
│   │   ├── 02/
│   │   ├── a4/
│   │   ├── a3/
│   │   ├── b5/
│   │   ├── b2/
│   │   ├── d9/
│   │   ├── ac/
│   │   ├── ad/
│   │   ├── bb/
│   │   ├── d7/
│   │   ├── d0/
│   │   ├── be/
│   │   ├── b3/
│   │   ├── df/
│   │   ├── da/
│   │   ├── b4/
│   │   ├── a2/
│   │   ├── a5/
│   │   ├── bd/
│   │   ├── d1/
│   │   ├── d6/
│   │   ├── bc/
│   │   ├── ae/
│   │   ├── d8/
│   │   ├── ab/
│   │   ├── e5/
│   │   ├── e2/
│   │   ├── f4/
│   │   ├── f3/
│   │   ├── eb/
│   │   ├── c7/
│   │   ├── c0/
│   │   ├── ee/
│   │   ├── c9/
│   │   ├── f2/
│   │   ├── f5/
│   │   ├── e3/
│   │   ├── cf/
│   │   ├── ca/
│   │   ├── e4/
│   │   ├── fe/
│   │   ├── c8/
│   │   ├── fb/
│   │   ├── ed/
│   │   ├── c1/
│   │   ├── c6/
│   │   ├── ec/
│   │   ├── 4e/
│   │   ├── 20/
│   │   ├── 18/
│   │   ├── 27/
│   │   ├── 4b/
│   │   ├── pack/
│   │   ├── 11/
│   │   ├── 7d/
│   │   ├── 29/
│   │   ├── 7c/
│   │   ├── 16/
│   │   ├── 42/
│   │   ├── 89/
│   │   ├── 1f/
│   │   ├── 73/
│   │   ├── 87/
│   │   ├── 74/
│   │   ├── 1a/
│   │   ├── 28/
│   │   ├── 17/
│   │   ├── 7b/
│   │   ├── 8f/
│   │   ├── 8a/
│   │   ├── 7e/
│   │   ├── 10/
│   │   ├── 19/
│   │   ├── 4c/
│   │   ├── 26/
│   │   ├── 21/
│   │   ├── 4d/
│   │   ├── 75/
│   │   ├── 81/
│   │   ├── 86/
│   │   ├── 72/
│   │   ├── 44/
│   │   ├── 2a/
│   │   ├── 2f/
│   │   ├── 43/
│   │   ├── 88/
│   │   ├── 9f/
│   │   ├── 6b/
│   │   ├── 07/
│   │   ├── 00/
│   │   ├── 6e/
│   │   ├── 9a/
│   │   ├── 36/
│   │   ├── 5c/
│   │   ├── 09/
│   │   ├── 5d/
│   │   ├── 31/
│   │   ├── info/
│   │   ├── 91/
│   │   ├── 65/
│   │   ├── 62/
│   │   ├── 96/
│   │   ├── 3a/
│   │   ├── 54/
│   │   ├── 98/
│   │   ├── 53/
│   │   ├── 3f/
│   │   ├── 30/
│   │   ├── 5e/
│   │   ├── 5b/
│   │   ├── 37/
│   │   ├── 08/
│   │   ├── 6d/
│   │   ├── 01/
│   │   ├── 06/
│   │   ├── 6c/
│   │   ├── 39/
│   │   ├── 99/
│   │   ├── 52/
│   │   ├── 55/
│   │   ├── 63/
│   │   ├── 0f/
│   │   ├── 0a/
│   │   ├── 64/
│   │   ├── 90/
│   │   ├── bf/
│   │   ├── d3/
│   │   ├── d4/
│   │   ├── ba/
│   │   ├── a0/
│   │   ├── b8/
│   │   ├── b1/
│   │   ├── dd/
│   │   ├── dc/
│   │   ├── b6/
│   │   ├── d5/
│   │   ├── d2/
│   │   ├── af/
│   │   ├── b7/
│   │   ├── db/
│   │   ├── a8/
│   │   ├── de/
│   │   ├── b0/
│   │   ├── a6/
│   │   ├── b9/
│   │   ├── a1/
│   │   ├── ef/
│   │   ├── c3/
│   │   ├── c4/
│   │   ├── ea/
│   │   ├── e1/
│   │   ├── cd/
│   │   ├── cc/
│   │   ├── e6/
│   │   ├── f9/
│   │   ├── f0/
│   │   ├── f7/
│   │   ├── e8/
│   │   ├── fa/
│   │   ├── ff/
│   │   ├── c5/
│   │   ├── c2/
│   │   ├── f6/
│   │   ├── e9/
│   │   ├── f1/
│   │   ├── e7/
│   │   ├── cb/
│   │   ├── f8/
│   │   ├── ce/
│   │   ├── e0/
│   │   ├── 2c/
│   │   ├── 79/
│   │   ├── 2d/
│   │   ├── 41/
│   │   ├── 83/
│   │   ├── 1b/
│   │   ├── 77/
│   │   ├── 48/
│   │   ├── 70/
│   │   ├── 1e/
│   │   ├── 84/
│   │   ├── 4a/
│   │   ├── 24/
│   │   ├── 23/
│   │   ├── 4f/
│   │   ├── 8d/
│   │   ├── 15/
│   │   ├── 12/
│   │   ├── 8c/
│   │   ├── 85/
│   │   ├── 1d/
│   │   ├── 71/
│   │   ├── 76/
│   │   ├── 1c/
│   │   ├── 82/
│   │   ├── 49/
│   │   ├── 40/
│   │   ├── 2e/
│   │   ├── 2b/
│   │   ├── 47/
│   │   ├── 78/
│   │   ├── 8b/
│   │   ├── 13/
│   │   ├── 7f/
│   │   ├── 7a/
│   │   ├── 14/
│   │   ├── 8e/
│   │   ├── 22/
│   │   ├── 25/
│   ├── info/
│   ├── logs/
│   │   ├── refs/
│   │   │   ├── heads/
│   │   │   ├── remotes/
│   │   │   │   ├── origin/
│   ├── hooks/
│   ├── refs/
│   │   ├── heads/
│   │   ├── tags/
│   │   ├── remotes/
│   │   │   ├── origin/
'''

print(project_dump)
