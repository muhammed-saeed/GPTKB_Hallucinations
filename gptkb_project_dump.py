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

├── gpt20p.py
│   --- File Content Start ---
│   from dotenv import load_dotenv
│   import os
│   from openai import OpenAI
│   
│   load_dotenv()
│   
│   client = OpenAI(api_key=os.getenv("ANTHROPIC_API_KEY"))
│   
│   from llm.anthropic_client import AnthropicLLM
│   
│   client = AnthropicLLM()  # loads ANTHROPIC_API_KEY from .env
│   
│   msgs = [
│       {"role": "system", "content": "You are a helpful assistant."},
│       {"role": "user", "content": "Give me 3 facts about Saturn."},
│   ]
│   
│   print("=== plain text call ===")
│   out = client(msgs, model="claude-sonnet-4-5-20250929", max_tokens=400)
│   print(out.get("text") if isinstance(out, dict) else out)
│   
│   print("\n=== reasoning/thinking call ===")
│   think = {"type": "enabled", "budget_tokens": 1024}
│   out2 = client(msgs, model="claude-sonnet-4-5-20250929", max_tokens=2000, reasoning=think)
│   print(out2.get("text") if isinstance(out2, dict) else out2)
│   --- File Content End ---

├── prompter_parser.py
│   --- File Content Start ---
│   # prompter_parser.py
│   from __future__ import annotations
│   import json
│   from pathlib import Path
│   from typing import Dict, List
│   
│   # Only replace known {placeholder} keys; never interpret other braces.
│   _ALLOWED_KEYS = {"subject_name", "phrases_block", "root_subject", "max_facts_hint"}
│   
│   # Canonical footer we want in every elicitation *system* message
│   _ELICITATION_SYSTEM_FOOTER = ( "" )
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
│       if ptype != "elicitation":
│           return system_txt or ""
│       marker = "include at least one triple where predicate is \"instanceOf\""
│       st = (system_txt or "")
│       if marker.lower() in st.lower():
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
│   # # db_models.py
│   # from __future__ import annotations
│   # import sqlite3
│   # import unicodedata, re
│   # from typing import Iterable, Tuple, Literal
│   
│   # from settings import QUEUE_DDL, FACTS_DDL
│   
│   # _WS = re.compile(r"\s+")
│   
│   # def normalize_subject(s: str) -> str:
│   #     if not isinstance(s, str):
│   #         return ""
│   #     s = unicodedata.normalize("NFKC", s)
│   #     s = _WS.sub(" ", s.strip())
│   #     return s.lower()
│   
│   # def _open_sqlite(path: str) -> sqlite3.Connection:
│   #     conn = sqlite3.connect(path, check_same_thread=False)
│   #     conn.execute("PRAGMA journal_mode=WAL;")
│   #     conn.execute("PRAGMA synchronous=NORMAL;")
│   #     conn.execute("PRAGMA temp_store=MEMORY;")
│   #     conn.execute("PRAGMA busy_timeout=5000;")
│   #     # mmap_size may fail on some platforms; you can keep or drop:
│   #     try:
│   #         conn.execute("PRAGMA mmap_size=30000000000;")
│   #     except sqlite3.OperationalError:
│   #         pass
│   #     conn.commit()
│   #     return conn
│   
│   # def _ensure_queue_indexes(conn: sqlite3.Connection):
│   #     cur = conn.cursor()
│   #     cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_queue_subject_hop ON queue(subject, hop)")
│   #     cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_queue_subject_norm ON queue(subject_norm)")
│   #     conn.commit()
│   
│   # def _ensure_facts_indexes(conn: sqlite3.Connection):
│   #     cur = conn.cursor()
│   #     cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_triples ON triples_accepted(subject, predicate, object, hop)")
│   #     conn.commit()
│   
│   # def open_queue_db(path: str) -> sqlite3.Connection:
│   #     conn = _open_sqlite(path)
│   #     conn.executescript(QUEUE_DDL)
│   #     _ensure_queue_indexes(conn)
│   #     return conn
│   
│   # def open_facts_db(path: str) -> sqlite3.Connection:
│   #     conn = _open_sqlite(path)
│   #     conn.executescript(FACTS_DDL)
│   #     _ensure_facts_indexes(conn)
│   #     return conn
│   
│   # EnqResult = Tuple[str, int, Literal["inserted", "hop_reduced", "ignored"]]
│   
│   # def enqueue_subjects(db: sqlite3.Connection, items: Iterable[Tuple[str, int]]) -> list[EnqResult]:
│   #     """
│   #     (Kept for backward-compat; not used by the processed queue.)
│   #     """
│   #     cur = db.cursor()
│   #     results: list[EnqResult] = []
│   
│   #     for subject, hop in items:
│   #         subj_norm = normalize_subject(subject)
│   
│   #         cur.execute("SELECT hop FROM queue WHERE subject_norm=?", (subj_norm,))
│   #         row = cur.fetchone()
│   #         before_hop = row[0] if row else None
│   
│   #         cur.execute(
│   #             """
│   #             INSERT INTO queue(subject, subject_norm, hop, status, retries)
│   #             VALUES (?, ?, ?, 'pending', 0)
│   #             ON CONFLICT(subject_norm) DO UPDATE SET
│   #               hop = CASE WHEN excluded.hop < hop THEN excluded.hop ELSE hop END
│   #             """,
│   #             (subject, subj_norm, hop),
│   #         )
│   
│   #         cur.execute("SELECT hop FROM queue WHERE subject_norm=?", (subj_norm,))
│   #         kept_hop = cur.fetchone()[0]
│   
│   #         if before_hop is None:
│   #             results.append((subject, kept_hop, "inserted"))
│   #         elif kept_hop < before_hop:
│   #             results.append((subject, kept_hop, "hop_reduced"))
│   #         else:
│   #             results.append((subject, kept_hop, "ignored"))
│   
│   #     db.commit()
│   #     return results
│   
│   # def reset_working_to_pending(conn: sqlite3.Connection) -> int:
│   #     cur = conn.cursor()
│   #     cur.execute("UPDATE queue SET status='pending' WHERE status='working'")
│   #     conn.commit()
│   #     return cur.rowcount
│   
│   # def queue_has_rows(conn: sqlite3.Connection) -> bool:
│   #     cur = conn.cursor()
│   #     cur.execute("SELECT 1 FROM queue LIMIT 1")
│   #     return cur.fetchone() is not None
│   
│   # def count_queue(conn: sqlite3.Connection):
│   #     cur = conn.cursor()
│   #     cur.execute("SELECT COUNT(1) FROM queue WHERE status='pending'"); pending = cur.fetchone()[0]
│   #     cur.execute("SELECT COUNT(1) FROM queue WHERE status='working'"); working = cur.fetchone()[0]
│   #     cur.execute("SELECT COUNT(1) FROM queue WHERE status='done'");    done    = cur.fetchone()[0]
│   #     return done, working, pending, done + working + pending
│   
│   # # -----------------------
│   # # Hardened triple writes
│   # # -----------------------
│   
│   # def _sanitize_row(row):
│   #     # row: (subject, predicate, object, hop, model_name, strategy, confidence)
│   #     s, p, o, h, m, st, c = row
│   
│   #     def as_str(x):
│   #         if x is None:
│   #             return ""
│   #         if isinstance(x, str):
│   #             return x
│   #         return str(x)
│   
│   #     s = as_str(s)
│   #     p = as_str(p)
│   #     o = as_str(o)
│   #     m = as_str(m)
│   #     st = as_str(st)
│   
│   #     try:
│   #         h = int(h)
│   #     except Exception:
│   #         h = 0
│   
│   #     try:
│   #         c = float(c) if c is not None else None
│   #     except Exception:
│   #         c = None
│   
│   #     return (s, p, o, h, m, st, c)
│   
│   # def write_triples_accepted(db: sqlite3.Connection, rows: Iterable[Tuple[str, str, str, int, str, str, float | None]]):
│   #     rows = [ _sanitize_row(r) for r in rows if r ]
│   #     if not rows:
│   #         return
│   #     cur = db.cursor()
│   #     cur.executemany(
│   #         """INSERT OR IGNORE INTO triples_accepted
│   #            (subject, predicate, object, hop, model_name, strategy, confidence)
│   #            VALUES (?, ?, ?, ?, ?, ?, ?)""",
│   #         rows,
│   #     )
│   #     db.commit()
│   
│   # def write_triples_sink(db: sqlite3.Connection, rows: Iterable[Tuple[str, str, str, int, str, str, float | None, str]]):
│   #     if not rows:
│   #         return
│   #     # sanitize + pad reason
│   #     clean_rows = []
│   #     for r in rows:
│   #         s, p, o, h, m, st, c, reason = r
│   #         s, p, o, h, m, st, c = _sanitize_row((s, p, o, h, m, st, c))
│   #         reason = "" if reason is None else (reason if isinstance(reason, str) else str(reason))
│   #         clean_rows.append((s, p, o, h, m, st, c, reason))
│   
│   #     cur = db.cursor()
│   #     cur.executemany(
│   #         """INSERT INTO triples_sink
│   #            (subject, predicate, object, hop, model_name, strategy, confidence, reason)
│   #            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
│   #         clean_rows,
│   #     )
│   #     db.commit()
│   # db_models.py
│   from __future__ import annotations
│   import sqlite3
│   import unicodedata, re
│   from typing import Iterable, Tuple, Literal
│   
│   from settings import QUEUE_DDL, FACTS_DDL
│   
│   _WS = re.compile(r"\s+")
│   
│   def normalize_subject(s: str) -> str:
│       if not isinstance(s, str):
│           return ""
│       s = unicodedata.normalize("NFKC", s)
│       s = _WS.sub(" ", s.strip())
│       return s.lower()
│   
│   def _open_sqlite(path: str) -> sqlite3.Connection:
│       conn = sqlite3.connect(path, check_same_thread=False)
│       conn.execute("PRAGMA journal_mode=WAL;")
│       conn.execute("PRAGMA synchronous=NORMAL;")
│       conn.execute("PRAGMA temp_store=MEMORY;")
│       conn.execute("PRAGMA busy_timeout=5000;")
│       try:
│           conn.execute("PRAGMA mmap_size=30000000000;")
│       except sqlite3.OperationalError:
│           pass
│       conn.commit()
│       return conn
│   
│   def _ensure_queue_indexes(conn: sqlite3.Connection):
│       cur = conn.cursor()
│       cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_queue_subject_norm  ON queue(subject_norm)")
│       cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_queue_subject_canon ON queue(subject_canon)")
│       cur.execute("CREATE INDEX IF NOT EXISTS ix_queue_status_hop ON queue(status, hop)")
│       conn.commit()
│   
│   def _ensure_facts_indexes(conn: sqlite3.Connection):
│       cur = conn.cursor()
│       cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_triples ON triples_accepted(subject, predicate, object, hop)")
│       conn.commit()
│   
│   def open_queue_db(path: str) -> sqlite3.Connection:
│       conn = _open_sqlite(path)
│       conn.executescript(QUEUE_DDL)
│       _ensure_queue_indexes(conn)
│       return conn
│   
│   def open_facts_db(path: str) -> sqlite3.Connection:
│       conn = _open_sqlite(path)
│       conn.executescript(FACTS_DDL)
│       _ensure_facts_indexes(conn)
│       return conn
│   
│   EnqResult = Tuple[str, int, Literal["inserted", "hop_reduced", "ignored"]]
│   
│   def reset_working_to_pending(conn: sqlite3.Connection) -> int:
│       cur = conn.cursor()
│       cur.execute("UPDATE queue SET status='pending' WHERE status='working'")
│       conn.commit()
│       return cur.rowcount
│   
│   def queue_has_rows(conn: sqlite3.Connection) -> bool:
│       cur = conn.cursor()
│       cur.execute("SELECT 1 FROM queue LIMIT 1")
│       return cur.fetchone() is not None
│   
│   def count_queue(conn: sqlite3.Connection):
│       cur = conn.cursor()
│       cur.execute("SELECT COUNT(1) FROM queue WHERE status='pending'"); pending = cur.fetchone()[0]
│       cur.execute("SELECT COUNT(1) FROM queue WHERE status='working'"); working = cur.fetchone()[0]
│       cur.execute("SELECT COUNT(1) FROM queue WHERE status='done'");    done    = cur.fetchone()[0]
│       return done, working, pending, done + working + pending
│   
│   # -------- triple writers --------
│   
│   def _sanitize_row(row):
│       s, p, o, h, m, st, c = row
│       def as_str(x):
│           if x is None: return ""
│           return x if isinstance(x, str) else str(x)
│       s, p, o, m, st = as_str(s), as_str(p), as_str(o), as_str(m), as_str(st)
│       try: h = int(h)
│       except Exception: h = 0
│       try: c = float(c) if c is not None else None
│       except Exception: c = None
│       return (s, p, o, h, m, st, c)
│   
│   def write_triples_accepted(db: sqlite3.Connection, rows: Iterable[Tuple[str, str, str, int, str, str, float | None]]):
│       rows = [ _sanitize_row(r) for r in rows if r ]
│       if not rows:
│           return
│       cur = db.cursor()
│       cur.executemany(
│           """INSERT OR IGNORE INTO triples_accepted
│              (subject, predicate, object, hop, model_name, strategy, confidence)
│              VALUES (?, ?, ?, ?, ?, ?, ?)""",
│           rows,
│       )
│       db.commit()
│   
│   def write_triples_sink(db: sqlite3.Connection, rows: Iterable[Tuple[str, str, str, int, str, str, float | None, str]]):
│       if not rows:
│           return
│       clean_rows = []
│       for r in rows:
│           s, p, o, h, m, st, c, reason = r
│           s, p, o, h, m, st, c = _sanitize_row((s, p, o, h, m, st, c))
│           reason = "" if reason is None else (reason if isinstance(reason, str) else str(reason))
│           clean_rows.append((s, p, o, h, m, st, c, reason))
│   
│       cur = db.cursor()
│       cur.executemany(
│           """INSERT INTO triples_sink
│              (subject, predicate, object, hop, model_name, strategy, confidence, reason)
│              VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
│           clean_rows,
│       )
│       db.commit()
│   --- File Content End ---

├── settings.py
│   --- File Content Start ---
│   from __future__ import annotations
│   from typing import Dict
│   from pydantic import BaseModel
│   from llm.config import ModelConfig
│   
│   # ---------- JSON Schemas ----------
│   
│   ELICIT_SCHEMA_BASE = {
│     "type": "object",
│     "additionalProperties": False,
│     "properties": {
│       "facts": {
│         "type": "array",
│         "items": {
│           "type": "object",
│           "additionalProperties": False,
│           "properties": {
│             "subject":   {"type": "string"},
│             "predicate": {"type": "string"},
│             "object":    {"type": "string"}
│           },
│           "required": ["subject", "predicate", "object"]
│         }
│       }
│     },
│     "required": ["facts"]
│   }
│   
│   ELICIT_SCHEMA_CAL = {
│     "type": "object",
│     "additionalProperties": False,
│     "properties": {
│       "facts": {
│         "type": "array",
│         "items": {
│           "type": "object",
│           "additionalProperties": False,
│           "properties": {
│             "subject":    {"type": "string"},
│             "predicate":  {"type": "string"},
│             "object":     {"type": "string"},
│             "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
│           },
│           "required": ["subject", "predicate", "object", "confidence"]
│         }
│       }
│     },
│     "required": ["facts"]
│   }
│   
│   NER_SCHEMA_BASE = {
│     "type": "object",
│     "additionalProperties": False,
│     "properties": {
│       "phrases": {
│         "type": "array",
│         "items": {
│           "type": "object",
│           "additionalProperties": False,
│           "properties": {
│             "phrase": {"type": "string"},
│             "is_ne":  {"type": "boolean"}
│           },
│           "required": ["phrase", "is_ne"]
│         }
│       }
│     },
│     "required": ["phrases"]
│   }
│   
│   NER_SCHEMA_CAL = {
│     "type": "object",
│     "additionalProperties": False,
│     "properties": {
│       "phrases": {
│         "type": "array",
│         "items": {
│           "type": "object",
│           "additionalProperties": False,
│           "properties": {
│             "phrase":     {"type": "string"},
│             "is_ne":      {"type": "boolean"},
│             "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
│           },
│           "required": ["phrase", "is_ne", "confidence"]
│         }
│       }
│     },
│     "required": ["phrases"]
│   }
│   
│   # ---------- SQLite DDL ----------
│   
│   QUEUE_DDL = """
│   CREATE TABLE IF NOT EXISTS queue(
│     subject        TEXT NOT NULL,
│     subject_norm   TEXT NOT NULL,
│     subject_canon  TEXT NOT NULL DEFAULT '',
│     hop            INT  NOT NULL DEFAULT 0,
│     status         TEXT NOT NULL DEFAULT 'pending',
│     retries        INT  NOT NULL DEFAULT 0,
│     created_at     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
│   );
│   """
│   
│   FACTS_DDL = """
│   CREATE TABLE IF NOT EXISTS triples_accepted(
│     subject     TEXT, 
│     predicate   TEXT, 
│     object      TEXT,
│     hop         INT, 
│     model_name  TEXT, 
│     strategy    TEXT, 
│     confidence  REAL,
│     PRIMARY KEY(subject, predicate, object, hop)
│   );
│   
│   CREATE TABLE IF NOT EXISTS triples_sink(
│     subject     TEXT, 
│     predicate   TEXT, 
│     object      TEXT,
│     hop         INT, 
│     model_name  TEXT, 
│     strategy    TEXT, 
│     confidence  REAL, 
│     reason      TEXT
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
│               temperature=0.0, top_p=1.0, max_tokens=4096,
│               use_responses_api=False
│           ),
│           "gpt4o-mini": ModelConfig(
│               provider="openai", model="gpt-4o-mini",
│               api_key_env="OPENAI_API_KEY",
│               temperature=0.0, top_p=1.0, max_tokens=4096,
│               use_responses_api=False
│           ),
│           "gpt4-turbo": ModelConfig(
│               provider="openai", model="gpt-4-turbo",
│               api_key_env="OPENAI_API_KEY",
│               temperature=0.0, top_p=1.0, max_tokens=4096,
│               use_responses_api=False
│           ),
│   
│           # -------- OpenAI (Responses API) — GPT-5 family --------
│           "gpt-5": ModelConfig(
│               provider="openai",
│               model="gpt-5",
│               api_key_env="OPENAI_API_KEY",
│               temperature=None, top_p=None, max_tokens=4096,
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
│               temperature=None, top_p=None, max_tokens=4096,
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
│               max_tokens=4096,
│           ),
│   
│           # -------- DeepSeek --------
│           "deepseek": ModelConfig(
│               provider="deepseek", model="deepseek-chat",
│               api_key_env="DEEPSEEK_API_KEY",
│               base_url="https://api.deepseek.com",
│               temperature=0.0, top_p=0.95, max_tokens=4096
│           ),
│           "deepseek-reasoner": ModelConfig(
│               provider="deepseek", model="deepseek-reasoner",
│               api_key_env="DEEPSEEK_API_KEY",
│               base_url="https://api.deepseek.com",
│               temperature=0.0, top_p=0.95, max_tokens=4096
│           ),
│   
│           # -------- Replicate (various) --------
│   
│           # Add these inside Settings.MODELS in settings.py
│   
│           # ------- Replicate (Meta Llama-3 70B Instruct) -------
│           "llama3-70b-instruct": ModelConfig(
│               provider="replicate",
│               model="meta/meta-llama-3-70b-instruct",
│               api_key_env="REPLICATE_API_TOKEN",
│               temperature=0.6,
│               top_p=0.9,
│               top_k=0,
│               max_tokens=4096,  # you can lower per-run; example snippet used 512
│               extra_inputs={
│                   # keep this EXACTLY — your pipeline will pass {system_prompt} and {prompt}
│                   "system_prompt": "You are a helpful assistant",
│                   "prompt_template": (
│                       "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
│                       "{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
│                       "{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
│                   ),
│                   "length_penalty": 1,
│                   "presence_penalty": 1.15,
│                   # Replicate runners typically prefer list form; your example showed a CSV string —
│                   # we include both-friendly variant; your factory should map to what the runner expects.
│                   "stop_sequences": ["<|end_of_text|>", "<|eot_id|>"],
│                   "log_performance_metrics": False,
│               },
│           ),
│   
│           # ------- Replicate (Meta Llama-3 8B Instruct) -------
│           "llama3-8b-instruct": ModelConfig(
│               provider="replicate",
│               model="meta/meta-llama-3-8b-instruct",
│               api_key_env="REPLICATE_API_TOKEN",
│               temperature=0.7,
│               top_p=0.95,
│               top_k=0,
│               max_tokens=4096,  # example used 512; keep 4096 default and cap per-run if needed
│               extra_inputs={
│                   # keep this EXACTLY — note this template uses {system_prompt} placeholder too
│                   "system_prompt": "You are a helpful assistant",
│                   "prompt_template": (
│                       "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
│                       "{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
│                       "{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
│                   ),
│                   "length_penalty": 1,
│                   "presence_penalty": 0,
│                   "max_new_tokens": 512,  # optional; some runners accept both max_tokens + max_new_tokens
│                   "stop_sequences": ["<|end_of_text|>", "<|eot_id|>"],
│                   "log_performance_metrics": False,
│               },
│           ),
│           # ------- Replicate (Meta Llama-3 8B — BASE, non-instruct) -------
│   "llama3-8b": ModelConfig(
│       provider="replicate",
│       model="meta/meta-llama-3-8b",           # BASE model (no -instruct)
│       api_key_env="REPLICATE_API_TOKEN",
│       temperature=0.6,
│       top_p=0.9,
│       top_k=0,
│       max_tokens=4096,
│       extra_inputs={
│           # Your factory should format: prompt_template.format(system_prompt=..., prompt=...)
│           # For BASE models we emulate chat by concatenating system + user.
│           "system_prompt": "You are a helpful assistant that returns STRICT JSON per schema.",
│           "prompt_template": "{system_prompt}\n\n{prompt}",
│           # Runners typically accept list or string for stop; list is safer:
│           "stop_sequences": ["<|end_of_text|>"],
│           "length_penalty": 1,
│           "presence_penalty": 0,
│           "log_performance_metrics": False,
│       },
│   ),
│   
│   # ------- Replicate (Meta Llama-3 70B — BASE, non-instruct) -------
│   "llama3-70b": ModelConfig(
│       provider="replicate",
│       model="meta/meta-llama-3-70b",          # BASE model (no -instruct)
│       api_key_env="REPLICATE_API_TOKEN",
│       temperature=0.6,
│       top_p=0.9,
│       top_k=0,
│       max_tokens=4096,
│       extra_inputs={
│           "system_prompt": "You are a helpful assistant that returns STRICT JSON per schema.",
│           "prompt_template": "{system_prompt}\n\n{prompt}",
│           "stop_sequences": ["<|end_of_text|>"],
│           "length_penalty": 1,
│           "presence_penalty": 0,
│           "log_performance_metrics": False,
│       },
│   ),
│   
│           "llama405b": ModelConfig(
│               provider="replicate", model="meta/meta-llama-3.1-405b-instruct",
│               api_key_env="REPLICATE_API_TOKEN",
│               temperature=0.6, top_p=0.9, top_k=50, max_tokens=4096,
│               extra_inputs={"system_prompt": "You are a helpful assistant.", "prompt_template": ""}
│           ),
│           "mistral7b": ModelConfig(
│               provider="replicate", model="mistralai/mistral-7b-instruct",
│               api_key_env="REPLICATE_API_TOKEN",
│               temperature=0.6, top_p=0.95, top_k=50, max_tokens=4096,
│               extra_inputs={"system_prompt": "You are a helpful assistant.", "prompt_template": ""}
│           ),
│           "mixtral8x7b": ModelConfig(
│               provider="replicate", model="mistralai/mixtral-8x7b-instruct",
│               api_key_env="REPLICATE_API_TOKEN",
│               temperature=0.6, top_p=0.95, top_k=50, max_tokens=4096,
│               extra_inputs={"system_prompt": "You are a helpful assistant.", "prompt_template": ""}
│           ),
│   
│           "gemini-flash": ModelConfig(
│               provider="replicate",
│               model="google/gemini-2.5-flash",
│               api_key_env="REPLICATE_API_TOKEN",
│               temperature=0.2,
│               top_p=0.9,
│               max_tokens=4096,
│               extra_inputs={"prefer": "prompt", "dynamic_thinking": False},
│           ),
│           "grok4": ModelConfig(
│               provider="replicate",
│               model="xai/grok-4",
│               api_key_env="REPLICATE_API_TOKEN",
│               temperature=0.1, top_p=1.0, max_tokens=2048,
│               extra_inputs={"system_prompt": "You are a helpful assistant.", "prompt_template": ""},
│           ),
│           "claude35h": ModelConfig(
│               provider="replicate",
│               model="anthropic/claude-3.5-haiku",
│               api_key_env="REPLICATE_API_TOKEN",
│               temperature=0.3, top_p=0.9, max_tokens=8192,
│               extra_inputs={"system_prompt": "You are a concise and creative assistant.", "prompt_template": ""},
│           ),
│           "claude37s": ModelConfig(
│               provider="replicate",
│               model="anthropic/claude-3.7-sonnet",
│               api_key_env="REPLICATE_API_TOKEN",
│               temperature=0.2, top_p=0.9, max_tokens=8192,
│               extra_inputs={
│                   "extended_thinking": False,
│                   "max_image_resolution": 0.5,
│                   "thinking_budget_tokens": 1024,
│                   "system_prompt": "Return ONLY strict JSON; no prose; no fences.",
│               },
│           ),
│   
│           "granite8b": ModelConfig(
│               provider="replicate",
│               model="ibm-granite/granite-3.3-8b-instruct",
│               api_key_env="REPLICATE_API_TOKEN",
│               temperature=0.6, top_p=0.9, top_k=50, max_tokens=4096,
│               extra_inputs={
│                   "system_prompt": "Return ONLY strict JSON that validates against the provided schema.",
│               },
│           ),
│           "gpt-oss-20b": ModelConfig(
│               provider="replicate",
│               model="openai/gpt-oss-20b",
│               api_key_env="REPLICATE_API_TOKEN",
│               temperature=0.1, top_p=1.0, max_tokens=4096,
│           ),
│           "gpt-oss-120b": ModelConfig(
│       provider="replicate",
│       model="openai/gpt-oss-120b",
│       api_key_env="REPLICATE_API_TOKEN",
│       temperature=0.1, top_p=1.0, max_tokens=4096,
│   ),
│   
│           "qwen3-235b": ModelConfig(
│               provider="replicate",
│               model="qwen/qwen3-235b-a22b-instruct-2507",
│               api_key_env="REPLICATE_API_TOKEN",
│               temperature=0.3, top_p=0.9, max_tokens=1536,
│               extra_inputs={"system_prompt": "Return ONLY strict JSON per schema; no prose; no fences."},
│           ),
│   
│           "granite20b": ModelConfig(
│               provider="replicate",
│               model="ibm-granite/granite-20b-code-instruct-8k",
│               api_key_env="REPLICATE_API_TOKEN",
│               temperature=0.6, top_p=0.9, top_k=50, max_tokens=512,
│               extra_inputs={"system_prompt": "", "prompt_template": ""},
│           ),
│   
│           # -------- Local via Unsloth (optional) --------
│           "smollm2-1.7b": ModelConfig(
│               provider="unsloth",
│               model="unsloth/SmolLM2-1.7B-Instruct-bnb-4bit",
│               api_key_env=None,
│               temperature=0.2, top_p=0.95, top_k=40, max_tokens=800,
│               extra_inputs={"max_seq_length": 2048, "load_in_4bit": False, "dtype": "float16", "device": "mps"},
│           ),
│           "smollm2-360m": ModelConfig(
│               provider="unsloth",
│               model="unsloth/SmolLM2-360M-Instruct-bnb-4bit",
│               api_key_env=None,
│               temperature=0.2, top_p=0.95, top_k=40, max_tokens=512,
│               extra_inputs={"max_seq_length": 2048, "load_in_4bit": True},
│           ),
│       }
│   
│       # defaults; override via CLI
│       ELICIT_MODEL_KEY: str = "gpt4o-mini"
│       NER_MODEL_KEY: str = "gpt4o-mini"
│   
│   settings = Settings()
│   --- File Content End ---

├── processing_queue.py
│   --- File Content Start ---
│   # processing_queue.py
│   from __future__ import annotations
│   
│   import re
│   import sqlite3
│   import unicodedata
│   from typing import Iterable, Tuple, List
│   
│   import threading
│   _thread_local = threading.local()
│   
│   DEFAULT_LEADING_ARTICLES = ("the", "a", "an")
│   
│   _ws = re.compile(r"\s+")
│   _nonword = re.compile(r"[^a-z0-9]")
│   
│   def get_thread_queue_conn(db_path: str) -> sqlite3.Connection:
│       key = f"queue_conn__{db_path}"
│       conn = getattr(_thread_local, key, None)
│       if conn is None:
│           conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
│           conn.execute("PRAGMA journal_mode=WAL;")
│           conn.execute("PRAGMA synchronous=NORMAL;")
│           conn.execute("PRAGMA busy_timeout=5000;")
│           conn.execute("PRAGMA temp_store=MEMORY;")
│           setattr(_thread_local, key, conn)
│       return conn
│   
│   def _canonical_key(s: str, leading_articles=DEFAULT_LEADING_ARTICLES) -> str:
│       """
│       Aggressive canonical form used to dedupe subject variants:
│       - Unicode NFKC, lowercased, collapse whitespace
│       - strip leading articles ("the", "a", "an")
│       - remove all non-alphanumerics
│       """
│       if not isinstance(s, str):
│           return ""
│       s = unicodedata.normalize("NFKC", s).strip().lower()
│       s = _ws.sub(" ", s)
│       for art in leading_articles:
│           if s.startswith(art + " "):
│               s = s[len(art) + 1:]
│               break
│       return _nonword.sub("", s)
│   
│   def _subject_norm(s: str) -> str:
│       """
│       Gentler normalization used for presentation:
│       - NFKC, lower, collapse spaces
│       """
│       if not isinstance(s, str):
│           return ""
│       s = unicodedata.normalize("NFKC", s).strip().lower()
│       return _ws.sub(" ", s)
│   
│   # ----- bootstrap / indices -----
│   
│   def ensure_processed_index(conn: sqlite3.Connection):
│       cur = conn.cursor()
│       cur.execute("""
│           CREATE TABLE IF NOT EXISTS processed_map (
│               canon_key TEXT PRIMARY KEY,
│               sample_original TEXT
│           )
│       """)
│       cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_queue_subject_norm   ON queue(subject_norm)")
│       cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_queue_subject_canon  ON queue(subject_canon)")
│       conn.commit()
│   
│   # ----- API -----
│   
│   EnqResult = Tuple[str, int, str]  # (subject original, kept_hop, outcome: inserted | hop_reduced | ignored)
│   
│   def init_cache(conn_or_path):
│       if isinstance(conn_or_path, str):
│           conn = get_thread_queue_conn(conn_or_path)
│       else:
│           conn = conn_or_path
│       ensure_processed_index(conn)
│   
│   def enqueue_subjects_processed(
│       db_path_or_conn,
│       items: Iterable[Tuple[str, int]],
│       leading_articles=DEFAULT_LEADING_ARTICLES
│   ) -> List[EnqResult]:
│       """
│       Enqueue with **canonical dedupe**:
│         - canonical key (subject_canon) ensures only *one* row ever exists per real-world subject
│         - if a new variant arrives with a *lower hop*, we *lower* the hop of the existing row (keep status as-is)
│         - if it’s a true duplicate with same-or-higher hop → outcome 'ignored'
│   
│       Returns: list of (subject, kept_hop, outcome)
│       """
│       conn = db_path_or_conn if not isinstance(db_path_or_conn, str) else get_thread_queue_conn(db_path_or_conn)
│       ensure_processed_index(conn)
│   
│       results: List[EnqResult] = []
│       cur = conn.cursor()
│   
│       with conn:
│           for subject, hop in items:
│               if not isinstance(subject, str) or not subject.strip():
│                   continue
│   
│               canon = _canonical_key(subject, leading_articles=leading_articles)
│               s_norm = _subject_norm(subject)
│   
│               # keep one sample per canonical key (for visibility)
│               cur.execute(
│                   "INSERT OR IGNORE INTO processed_map(canon_key, sample_original) VALUES (?, ?)",
│                   (canon, subject)
│               )
│   
│               # Read any existing row for this canonical key to determine outcome precisely
│               cur.execute("SELECT hop FROM queue WHERE subject_canon=?", (canon,))
│               row = cur.fetchone()
│               before_hop = row[0] if row else None
│   
│               # Upsert by canonical key; DO NOT touch status/retries if conflicting
│               cur.execute(
│                   """
│                   INSERT INTO queue(subject, subject_norm, subject_canon, hop, status, retries)
│                   VALUES (?, ?, ?, ?, 'pending', 0)
│                   ON CONFLICT(subject_canon) DO UPDATE SET
│                       hop = CASE WHEN excluded.hop < hop THEN excluded.hop ELSE hop END
│                   """,
│                   (subject, s_norm, canon, hop)
│               )
│   
│               # Read back the kept hop
│               cur.execute("SELECT hop FROM queue WHERE subject_canon=?", (canon,))
│               kept_hop = cur.fetchone()[0]
│   
│               if before_hop is None:
│                   outcome = "inserted"
│               elif kept_hop < before_hop:
│                   outcome = "hop_reduced"
│               else:
│                   outcome = "ignored"
│   
│               results.append((subject, kept_hop, outcome))
│   
│       return results
│   --- File Content End ---

├── crawler_runner_heuristics3.py
│   --- File Content Start ---
│   # crawler_runner_heuristics3.py
│   from __future__ import annotations
│   
│   import argparse, datetime, json, os, re, sqlite3, threading, time, traceback
│   from concurrent.futures import ThreadPoolExecutor, as_completed
│   from typing import Dict, List, Tuple, Set, Optional
│   
│   from dotenv import load_dotenv
│   load_dotenv()
│   
│   # ---------- locks & tiny utils ----------
│   _jsonl_lock = threading.Lock()
│   _seen_facts_lock = threading.Lock()
│   _lowconf_lock = threading.Lock()
│   _ner_lowconf_lock = threading.Lock()
│   
│   def _append_jsonl(path: str, obj: dict):
│       line = json.dumps(obj, ensure_ascii=False) + "\n"
│       with _jsonl_lock:
│           with open(path, "a", encoding="utf-8") as f:
│               f.write(line)
│   
│   def _dbg(msg: str): print(msg, flush=True)
│   
│   def _print_messages(tag: str, msgs: List[dict], limit: int | None = None):
│       print(f"\n--- {tag} MESSAGES ({len(msgs)}) ---")
│       for i, m in enumerate(msgs, 1):
│           role = (m.get("role") or "").upper()
│           content = m.get("content")
│           if isinstance(content, str) and limit:
│               content = (content[:limit] + "…") if len(content) > limit else content
│           print(f"[{i:02d}] {role}: {content if isinstance(content, str) else content}")
│       print(f"--- END {tag} ---\n")
│   
│   def _print_enqueue_summary(results: List[Tuple[str,int,str]]):
│       if not results:
│           print("[enqueue] (no results)")
│           return
│       ins = sum(1 for *_r, out in results if out == "inserted")
│       red = sum(1 for *_r, out in results if out == "hop_reduced")
│       ign = sum(1 for *_r, out in results if out == "ignored")
│       print(f"[enqueue] inserted={ins} hop_reduced={red} ignored={ign}")
│   
│   # ---------- repo imports ----------
│   from processing_queue import (
│       init_cache as procq_init_cache,
│       enqueue_subjects_processed as procq_enqueue,
│       DEFAULT_LEADING_ARTICLES as PROCQ_LEADING,
│       get_thread_queue_conn as procq_get_thread_conn,
│   )
│   from settings import (
│       settings,
│       ELICIT_SCHEMA_BASE, ELICIT_SCHEMA_CAL,
│       NER_SCHEMA_BASE,   NER_SCHEMA_CAL,
│   )
│   from prompter_parser import get_prompt_messages
│   from llm.factory import make_llm_from_config
│   from db_models import (
│       open_queue_db, open_facts_db,
│       write_triples_accepted, write_triples_sink,
│       queue_has_rows, reset_working_to_pending,
│   )
│   
│   # NEW: shared JSON extractor
│   from llm.json_utils import best_json
│   
│   # Optional OpenAI SDK (for Batch API)
│   try:
│       from openai import OpenAI as _OpenAI
│   except Exception:
│       _OpenAI = None
│   
│   # ---------- paths ----------
│   def _ensure_output_dir(base_dir: Optional[str]) -> str:
│       out = base_dir or os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
│       os.makedirs(out, exist_ok=True)
│       return out
│   
│   def _build_paths(out_dir: str) -> dict:
│       tmp = os.path.join(out_dir, "tmp")
│       os.makedirs(tmp, exist_ok=True)
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
│           "ner_lowconf_jsonl": os.path.join(out_dir, "ner_lowconf.jsonl"),
│           "ner_lowconf_json": os.path.join(out_dir, "ner_lowconf.json"),
│           "run_meta_json": os.path.join(out_dir, "run_meta.json"),
│           "tmp_dir": tmp,
│           "batch_req_jsonl": os.path.join(tmp, "batch_requests.jsonl"),
│           "batch_out_jsonl": os.path.join(tmp, "batch_results.jsonl"),
│       }
│   
│   def _write_queue_snapshot(qdb: sqlite3.Connection, snapshot_path: str, max_depth: int):
│       cur = qdb.cursor()
│       if max_depth == 0:
│           cur.execute("SELECT subject, hop, status, retries, created_at FROM queue ORDER BY hop, subject")
│       else:
│           cur.execute("SELECT subject, hop, status, retries, created_at FROM queue WHERE hop<=? ORDER BY hop, subject", (max_depth,))
│       rows = cur.fetchall()
│       with open(snapshot_path, "w", encoding="utf-8") as f:
│           json.dump(
│               [{"subject": s, "hop": h, "status": st, "retries": r, "created_at": ts} for (s, h, st, r, ts) in rows],
│               f, ensure_ascii=False, indent=2
│           )
│   
│   # ---------- per-thread sqlite ----------
│   _thread_local = threading.local()
│   
│   def get_thread_queue_conn(db_path: str) -> sqlite3.Connection:
│       return procq_get_thread_conn(db_path)
│   
│   def get_thread_facts_conn(db_path: str) -> sqlite3.Connection:
│       key = f"facts_conn__{db_path}"
│       conn = getattr(_thread_local, key, None)
│       if conn is None:
│           conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
│           conn.execute("PRAGMA journal_mode=WAL;")
│           conn.execute("PRAGMA synchronous=NORMAL;")
│           conn.execute("PRAGMA busy_timeout=5000;")
│           conn.execute("PRAGMA temp_store=MEMORY;")
│           setattr(_thread_local, key, conn)
│       return conn
│   
│   def mark_done_threadsafe(queue_db_path: str, subject: str, hop: int):
│       conn = get_thread_queue_conn(queue_db_path)
│       with conn:
│           conn.execute("UPDATE queue SET status='done' WHERE subject=? AND hop=? AND status='working'", (subject, hop))
│   
│   def mark_pending_on_error(queue_db_path: str, subject: str, hop: int):
│       conn = get_thread_queue_conn(queue_db_path)
│       with conn:
│           conn.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=? AND hop=? AND status='working'", (subject, hop))
│   
│   def _get_retries(queue_db_path: str, subject: str, hop: int) -> int:
│       conn = get_thread_queue_conn(queue_db_path)
│       cur = conn.cursor()
│       cur.execute("SELECT retries FROM queue WHERE subject=? AND hop=?", (subject, hop))
│       row = cur.fetchone()
│       return int(row[0]) if row else 0
│   
│   def _inc_retries_and_pending(queue_db_path: str, subject: str, hop: int):
│       conn = get_thread_queue_conn(queue_db_path)
│       with conn:
│           conn.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=? AND hop=?", (subject, hop))
│   
│   # ---------- claim helpers ----------
│   def _fetch_one_pending(conn: sqlite3.Connection, max_depth: int) -> Tuple[str,int] | None:
│       cur = conn.cursor()
│       try:
│           if max_depth == 0:
│               cur.execute("""
│                   UPDATE queue SET status='working'
│                   WHERE rowid = (SELECT rowid FROM queue WHERE status='pending' ORDER BY hop, created_at LIMIT 1)
│                   RETURNING subject, hop
│               """)
│           else:
│               cur.execute("""
│                   UPDATE queue SET status='working'
│                   WHERE rowid = (SELECT rowid FROM queue WHERE status='pending' AND hop<=?
│                                  ORDER BY hop, created_at LIMIT 1)
│                   RETURNING subject, hop
│               """, (max_depth,))
│           row = cur.fetchone()
│           conn.commit()
│           return (row[0], row[1]) if row else None
│       except sqlite3.OperationalError:
│           cur.execute("BEGIN IMMEDIATE")
│           if max_depth == 0:
│               cur.execute("SELECT rowid, subject, hop FROM queue WHERE status='pending' ORDER BY hop, created_at LIMIT 1")
│           else:
│               cur.execute("SELECT rowid, subject, hop FROM queue WHERE status='pending' AND hop<=? ORDER BY hop, created_at LIMIT 1", (max_depth,))
│           row = cur.fetchone()
│           if not row:
│               conn.commit(); return None
│           rowid, subject, hop = row
│           cur.execute("UPDATE queue SET status='working' WHERE rowid=? AND status='pending'", (rowid,))
│           changed = cur.rowcount
│           conn.commit()
│           return (subject, hop) if changed else None
│   
│   def _fetch_many_pending(conn: sqlite3.Connection, max_depth: int, limit: int) -> List[Tuple[str,int]]:
│       got = []
│       for _ in range(max(1,limit)):
│           one = _fetch_one_pending(conn, max_depth)
│           if not one: break
│           got.append(one)
│       return got
│   
│   def _counts(conn: sqlite3.Connection, max_depth: int):
│       cur = conn.cursor()
│       if max_depth == 0:
│           cur.execute("SELECT COUNT(1) FROM queue WHERE status='done'"); done = cur.fetchone()[0]
│           cur.execute("SELECT COUNT(1) FROM queue WHERE status='working'"); working = cur.fetchone()[0]
│           cur.execute("SELECT COUNT(1) FROM queue WHERE status='pending'"); pending = cur.fetchone()[0]
│       else:
│           cur.execute("SELECT COUNT(1) FROM queue WHERE status='done' AND hop<=?", (max_depth,)); done = cur.fetchone()[0]
│           cur.execute("SELECT COUNT(1) FROM queue WHERE status='working' AND hop<=?", (max_depth,)); working = cur.fetchone()[0]
│           cur.execute("SELECT COUNT(1) FROM queue WHERE status='pending' AND hop<=?", (max_depth,)); pending = cur.fetchone()[0]
│       return done, working, pending, done + working + pending
│   
│   # ---------- unwrap & salvage ----------
│   def _parse_obj(maybe_json) -> dict:
│       if isinstance(maybe_json, dict): return maybe_json
│       if isinstance(maybe_json, str):
│           try: return json.loads(maybe_json)
│           except Exception: return {}
│       return {}
│   
│   def _unwrap_text(resp):
│       if isinstance(resp, str): return resp
│       if isinstance(resp, dict):
│           for k in ("text","output_text","content","message","response"):
│               v = resp.get(k)
│               if isinstance(v, str): return v
│           ch = resp.get("choices")
│           if isinstance(ch, list) and ch:
│               c0 = ch[0] or {}
│               msg = c0.get("message") or {}
│               if isinstance(msg, dict) and isinstance(msg.get("content"), str):
│                   return msg["content"]
│               if isinstance(c0.get("text"), str): return c0["text"]
│           # NEW: handle our client wrappers
│           if isinstance(resp.get("_raw"), str): return resp["_raw"]
│           if isinstance(resp.get("raw"), str):  return resp["raw"]
│           if isinstance(resp.get("raw"), dict): return _unwrap_text(resp["raw"])
│       return ""
│   
│   def _extract_json_block(text: str):
│       obj = best_json(text)
│       return obj if isinstance(obj, (dict, list)) else {}
│   
│   def _normalize_fact_keys(d: dict) -> dict | None:
│       if not isinstance(d, dict): return None
│       key_map = {
│           "subject": ["subject","subj","s","head","h"],
│           "predicate": ["predicate","pred","p","relation","rel","r"],
│           "object": ["object","obj","o","tail","t","value","val"],
│           "confidence": ["confidence","conf","c","score","prob"]
│       }
│       out = {}
│       for std, alts in key_map.items():
│           for k in alts:
│               if k in d and isinstance(d[k], (str, float, int)):
│                   out[std] = d[k]
│                   break
│       s,p,o = out.get("subject"), out.get("predicate"), out.get("object")
│       if not (isinstance(s,str) and isinstance(p,str) and isinstance(o,str)):
│           return None
│       if "confidence" in out:
│           try: out["confidence"] = float(out["confidence"])
│           except Exception: out["confidence"] = None
│       else:
│           out["confidence"] = None
│       return out
│   
│   _TRIPLE_OBJ_RX = re.compile(r"\{[^{}]*?(\"subject\"|\"subj\"|\"s\"|\"head\")[^{}]*?\}", re.I)
│   _FLEX_TRIPLE_RX = re.compile(r"\{[^{}]*\}", re.S)
│   
│   def _salvage_facts_from_text(text: str, debug=False) -> List[dict]:
│       salvaged: List[dict] = []
│   
│       obj = _extract_json_block(text)
│       if obj:
│           if isinstance(obj, dict):
│               for key in ("facts","triples"):
│                   val = obj.get(key)
│                   if isinstance(val, list):
│                       for item in val:
│                           norm = _normalize_fact_keys(item)
│                           if norm: salvaged.append(norm)
│               if not salvaged:
│                   norm = _normalize_fact_keys(obj)
│                   if norm: salvaged.append(norm)
│           elif isinstance(obj, list):
│               for item in obj:
│                   norm = _normalize_fact_keys(item)
│                   if norm: salvaged.append(norm)
│   
│       if not salvaged:
│           for m in _TRIPLE_OBJ_RX.finditer(text or ""):
│               chunk = m.group(0)
│               try:
│                   d = json.loads(chunk)
│                   norm = _normalize_fact_keys(d)
│                   if norm: salvaged.append(norm)
│               except Exception:
│                   patched = chunk
│                   open_br = chunk.count("{")
│                   close_br = chunk.count("}")
│                   patched += "}" * max(0, open_br - close_br)
│                   try:
│                       d = json.loads(patched)
│                       norm = _normalize_fact_keys(d)
│                       if norm: salvaged.append(norm)
│                   except Exception:
│                       continue
│   
│       # extra flexible pass: any dicts
│       if not salvaged:
│           for m in _FLEX_TRIPLE_RX.finditer(text or ""):
│               try:
│                   d = json.loads(m.group(0))
│               except Exception:
│                   continue
│               norm = _normalize_fact_keys(d)
│               if norm:
│                   salvaged.append(norm)
│   
│       if debug and salvaged:
│           print(f"[salvage] recovered {len(salvaged)} triples from noisy output")
│   
│       facts = []
│       for t in salvaged:
│           facts.append({
│               "subject": t["subject"],
│               "predicate": t["predicate"],
│               "object": t["object"],
│               "confidence": t.get("confidence")
│           })
│       return facts
│   
│   def _extract_facts_from_resp(resp, debug=False) -> Tuple[List[dict], str]:
│       if isinstance(resp, list):
│           facts = [t for t in resp if isinstance(t, dict)]
│           return facts, ""
│       if isinstance(resp, dict):
│           for key in ("facts","triples"):
│               val = resp.get(key)
│               if isinstance(val, list):
│                   return [t for t in val if isinstance(t, dict)], ""
│       txt = _unwrap_text(resp)
│       obj = _extract_json_block(txt)
│       if isinstance(obj, dict):
│           for key in ("facts","triples"):
│               val = obj.get(key)
│               if isinstance(val, list):
│                   return [t for t in val if isinstance(t, dict)], txt
│       if isinstance(obj, list):
│           return [t for t in obj if isinstance(t, dict)], txt
│       return [], txt
│   
│   # ---------- NER heuristics ----------
│   _date_rx = re.compile(r"^\d{4}([-/]\d{2}){0,2}$|^(January|February|March|April|May|June|July|August|September|October|November|December)\b", re.I)
│   _url_rx  = re.compile(r"^https?://", re.I)
│   def _is_date_like(s:str)->bool: return bool(_date_rx.search(s or ""))
│   def _is_literal_like(s:str)->bool:
│       s = s or ""
│       if _url_rx.search(s): return True
│       if s.isdigit(): return True
│       if s.strip().lower() in {"human","engineer","inventor","person","male","female"}: return True
│       return False
│   def _titlecase_ratio(s:str)->float:
│       words = [w for w in re.split(r"\s+", (s or "").strip()) if w]
│       if not words: return 0.0
│       caps = sum(1 for w in words if w[:1].isupper())
│       return caps/len(words)
│   _variant_rx = re.compile(r"[\(\)\[\]\{\}:–—\-]")
│   def _norm(s:str)->str: return re.sub(r"\s+"," ",(s or "")).strip().lower()
│   def _is_subject_variant(phrase:str, subject:str)->bool:
│       ps, ss = _norm(phrase), _norm(subject)
│       if not ps or not ss: return False
│       if ps == ss: return True
│       if ps.startswith(ss+" (") or ps.startswith(ss+" -") or ps.startswith(ss+":"): return True
│       if _variant_rx.sub("", ps) == _variant_rx.sub("", ss): return True
│       if ps.startswith(ss) and any(ch in ps[len(ss):len(ss)+3] for ch in "():-—–[]{}"): return True
│       return False
│   def _maybe_is_ne_heuristic(phrase:str)->bool:
│       if not isinstance(phrase,str): return False
│       p = phrase.strip()
│       if not p: return False
│       if _is_date_like(p) or _is_literal_like(p): return False
│       if " " not in p and p.islower(): return False
│       if _titlecase_ratio(p) >= 0.6: return True
│       if " " in p and not p.islower(): return True
│       return False
│   def _filter_ner_candidates(objs: List[str], subject: Optional[str]=None)->List[str]:
│       uniq:Set[str] = set()
│       for o in objs:
│           if not isinstance(o,str): continue
│           o2 = o.strip()
│           if not o2: continue
│           if len(o2.split())>6: continue
│           if subject and _is_subject_variant(o2, subject): continue
│           if _is_date_like(o2) or _is_literal_like(o2): continue
│           uniq.add(o2)
│       return sorted(uniq)
│   
│   # ---------- prompts ----------
│   def _ensure_json_keyword_in_msgs(msgs: List[dict], shape_hint: str):
│       has_json = any(isinstance(m.get("content"),str) and "json" in (m.get("content") or "").lower() for m in msgs)
│       if not has_json:
│           # Prepend as system for maximum priority
│           msgs.insert(0, {"role":"system","content":f"Output ONLY JSON; shape: {shape_hint}"})
│   
│   def _build_elicitation_messages(args, subject:str)->List[dict]:
│       msgs = get_prompt_messages(
│           args.elicitation_strategy, "elicitation",
│           domain=args.domain,
│           variables=dict(subject_name=subject, root_subject=args.seed, max_facts_hint=args.max_facts_hint),
│       )
│       if getattr(args,"footer_mode",False):
│           footer = ("\n\nFinal important note:\n"
│                     "If the entity is famous, aim ~50 distinct triplets; else ~10 if any exist. "
│                     "Only concrete, verifiable info.")
│           for m in msgs:
│               if m.get("role")=="system":
│                   m["content"] = (m.get("content") or "") + footer
│                   break
│           else:
│               msgs.insert(0, {"role":"system","content":footer})
│       return msgs
│   
│   # ---------- provider helpers ----------
│   def _is_openai_model(cfg)->bool:
│       prov = (getattr(cfg,"provider","") or "").lower()
│       if "openai" in prov: return True
│       name = (getattr(cfg,"model","") or "").lower()
│       return "openai" in name or name.startswith("gpt-")
│   
│   def _route_facts(args, facts: List[dict], hop:int, model_name:str):
│       acc, lowconf, objs = [], [], []
│       use_thr = (args.elicitation_strategy == "calibrate")
│       thr = float(args.conf_threshold)
│       for f in facts:
│           s, p, o = f.get("subject"), f.get("predicate"), f.get("object")
│           if not (isinstance(s,str) and isinstance(p,str) and isinstance(o,str)): continue
│           conf = f.get("confidence")
│           if use_thr and isinstance(conf,(int,float)) and conf < thr:
│               lowconf.append({
│                   "subject": s, "predicate": p, "object": o,
│                   "hop": hop, "model": model_name, "strategy": args.elicitation_strategy,
│                   "confidence": float(conf), "threshold": thr
│               })
│               continue
│           acc.append((s,p,o,hop,model_name,args.elicitation_strategy, float(conf) if isinstance(conf,(int,float)) else None))
│           objs.append(o)
│       return acc, lowconf, objs
│   
│   # ---------- OpenAI Batch helpers ----------
│   def _make_openai_client_for_batch(el_cfg):
│       if _OpenAI is None:
│           raise RuntimeError("OpenAI SDK not installed. `pip install openai`")
│       api_key_env = getattr(el_cfg, "api_key_env", "OPENAI_API_KEY")
│       api_key = os.getenv(api_key_env or "OPENAI_API_KEY")
│       if not api_key: raise RuntimeError(f"Missing {api_key_env or 'OPENAI_API_KEY'} for Batch mode.")
│       base_url = getattr(el_cfg, "base_url", None)
│       return _OpenAI(api_key=api_key, base_url=base_url) if base_url else _OpenAI(api_key=api_key)
│   
│   def _write_batch_requests_jsonl(fp: str, subjects: List[str], el_cfg, messages_builder, args):
│       os.makedirs(os.path.dirname(fp), exist_ok=True)
│       schema = ELICIT_SCHEMA_CAL if (args.elicitation_strategy == "calibrate") else ELICIT_SCHEMA_BASE
│       with open(fp, "w", encoding="utf-8") as f:
│           for subject in subjects:
│               msgs = messages_builder(args, subject)
│               _ensure_json_keyword_in_msgs(msgs, shape_hint='{"facts":[{"subject":"...","predicate":"...","object":"..."}]}')
│               body = {
│                   "model": el_cfg.model,
│                   "messages": msgs,
│                   "temperature": getattr(el_cfg,"temperature", None),
│                   "top_p": getattr(el_cfg,"top_p", None),
│                   "max_tokens": getattr(el_cfg,"max_tokens", 2048),
│                   "response_format": {
│                       "type":"json_schema",
│                       "json_schema": {"name":"schema","schema": schema, "strict": True}
│                   }
│               }
│               line = {"custom_id": subject, "method":"POST", "url":"/v1/chat/completions", "body": body}
│               f.write(json.dumps(line, ensure_ascii=False) + "\n")
│   
│   def _parse_openai_batch_output_line(line: str, debug: bool=False) -> Tuple[str, List[dict], str]:
│       try:
│           obj = json.loads(line)
│       except Exception:
│           if debug: print(f"[batch-parse] not JSON line: {line[:200]} ...")
│           return "", [], ""
│   
│       subject = obj.get("custom_id") or ""
│       resp_body = ((obj.get("response") or {}).get("body")) or {}
│       choices = resp_body.get("choices") or []
│       content_text = ""
│       if choices:
│           msg = (choices[0] or {}).get("message") or {}
│           content_text = (msg.get("content") or "").strip()
│           if not content_text:
│               tool_calls = msg.get("tool_calls") or []
│               if tool_calls:
│                   try:
│                       arguments = ((tool_calls[0] or {}).get("function") or {}).get("arguments")
│                       if isinstance(arguments, str):
│                           content_text = arguments
│                       elif isinstance(arguments, dict):
│                           content_text = json.dumps(arguments)
│                   except Exception:
│                       pass
│   
│       parsed = {}
│       if content_text:
│           try:
│               parsed = json.loads(content_text)
│           except Exception:
│               parsed = best_json(content_text) or {}
│   
│       if not parsed and isinstance(resp_body, dict):
│           parsed = best_json(json.dumps(resp_body)) or {}
│   
│       facts: List[dict] = []
│       if isinstance(parsed, dict):
│           facts = parsed.get("facts") or parsed.get("triples") or []
│       elif isinstance(parsed, list):
│           facts = parsed
│   
│       facts = [t for t in facts if isinstance(t, dict)]
│       return subject, facts, content_text
│   
│   # ---------- main ----------
│   def main():
│       ap = argparse.ArgumentParser(description="Crawler v3: salvage & retries; max-inflight only for OpenAI Batch.")
│       ap.add_argument("--seed", required=True)
│       ap.add_argument("--output-dir", default=None)
│   
│       ap.add_argument("--elicitation-strategy", default="baseline", choices=["baseline","icl","dont_know","calibrate"])
│       ap.add_argument("--ner-strategy", default="baseline", choices=["baseline","icl","dont_know","calibrate"])
│       ap.add_argument("--domain", default="general", choices=["general","topic"])
│   
│       ap.add_argument("--max-depth", type=int, default=settings.MAX_DEPTH, help="0 = unlimited depth (stop when queue empty)")
│       ap.add_argument("--max-subjects", type=int, default=0, help="0 = unlimited subjects")
│       ap.add_argument("--ner-batch-size", type=int, default=settings.NER_BATCH_SIZE)
│       ap.add_argument("--max-facts-hint", default=str(settings.MAX_FACTS_HINT))
│       ap.add_argument("--conf-threshold", type=float, default=0.7)
│       ap.add_argument("--ner-conf-threshold", type=float, default=0.9)
│       ap.add_argument("--footer-mode", action="store_true")
│   
│       ap.add_argument("--elicit-model-key", default=settings.ELICIT_MODEL_KEY)
│       ap.add_argument("--ner-model-key", default=settings.NER_MODEL_KEY)
│   
│       ap.add_argument("--elicit-temperature", type=float, default=0.7)
│       ap.add_argument("--ner-temperature", type=float, default=0.3)
│       ap.add_argument("--elicit-top-p", type=float, default=None)
│       ap.add_argument("--ner-top-p", type=float, default=None)
│       ap.add_argument("--elicit-top-k", type=int, default=None)
│       ap.add_argument("--ner-top-k", type=int, default=None)
│       ap.add_argument("--elicit-max-tokens", type=int, default=4096)
│       ap.add_argument("--ner-max-tokens", type=int, default=4096)
│   
│       ap.add_argument("--batch-size", type=int, default=1, help="Subjects grouped per realtime .batch() call (if supported)")
│       ap.add_argument("--concurrency", type=int, default=8, help="Workers for providers without realtime batching")
│       ap.add_argument("--max-inflight", type=int, default=None, help="[OpenAI Batch ONLY] subjects to claim per batch")
│       ap.add_argument("--timeout", type=float, default=90.0)
│       ap.add_argument("--max-retries", type=int, default=3, help="Max attempts per subject (non-batch) or per subject line (batch).")
│   
│       ap.add_argument("--openai-batch-mode", action="store_true", help="Use OpenAI Batch API for elicitation (chat-completions only)")
│   
│       ap.add_argument("--debug", action="store_true")
│       ap.add_argument("--progress-metrics", dest="progress_metrics", action="store_true", default=True)
│       ap.add_argument("--no-progress-metrics", dest="progress_metrics", action="store_false")
│   
│       ap.add_argument("--resume", action="store_true")
│       ap.add_argument("--reset-working", action="store_true")
│   
│       args = ap.parse_args()
│   
│       out_dir = _ensure_output_dir(args.output_dir)
│       paths = _build_paths(out_dir)
│       _dbg(f"[runner] output_dir: {out_dir}")
│   
│       qdb = open_queue_db(paths["queue_sqlite"])
│       fdb = open_facts_db(paths["facts_sqlite"])
│       procq_init_cache(qdb)
│   
│       # seed/resume
│       if args.resume:
│           if not queue_has_rows(qdb):
│               for s, kept_hop, outcome in procq_enqueue(paths["queue_sqlite"], [(args.seed, 0)], leading_articles=PROCQ_LEADING):
│                   if outcome in ("inserted","hop_reduced"):
│                       _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": kept_hop, "event": outcome})
│               _write_queue_snapshot(qdb, paths["queue_json"], args.max_depth)
│           else:
│               if args.reset_working:
│                   n = reset_working_to_pending(qdb)
│                   _dbg(f"[resume] reset {n} working→pending")
│       else:
│           for s, kept_hop, outcome in procq_enqueue(paths["queue_sqlite"], [(args.seed, 0)], leading_articles=PROCQ_LEADING):
│               if outcome in ("inserted","hop_reduced"):
│                   _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": kept_hop, "event": outcome})
│           _write_queue_snapshot(qdb, paths["queue_json"], args.max_depth)
│   
│       # build cfgs + apply stage params
│       el_cfg = settings.MODELS[args.elicit_model_key].model_copy(deep=True)
│       ner_cfg = settings.MODELS[args.ner_model_key].model_copy(deep=True)
│   
│       def _apply_stage(which, cfg):
│           if getattr(cfg,"use_responses_api", False):
│               cfg.temperature = None; cfg.top_p = None; cfg.top_k = None
│               if cfg.extra_inputs is None: cfg.extra_inputs = {}
│               cfg.extra_inputs.setdefault("reasoning", {})
│               cfg.extra_inputs.setdefault("text", {})
│           else:
│               t  = getattr(args, f"{which}_temperature")
│               tp = getattr(args, f"{which}_top_p")
│               tk = getattr(args, f"{which}_top_k")
│               if t  is not None: cfg.temperature = t
│               if tp is not None: cfg.top_p = tp
│               if tk is not None: cfg.top_k = tk
│           mt = getattr(args, f"{which}_max_tokens")
│           if mt is not None: cfg.max_tokens = mt
│           if getattr(cfg,"max_tokens", None) is None:
│               cfg.max_tokens = 2048
│           if hasattr(cfg,"request_timeout"): cfg.request_timeout = args.timeout
│           elif hasattr(cfg,"timeout"):       cfg.timeout = args.timeout
│   
│       _apply_stage("elicit", el_cfg)
│       _apply_stage("ner", ner_cfg)
│   
│       el_llm = make_llm_from_config(el_cfg)
│       ner_llm = make_llm_from_config(ner_cfg)
│   
│       is_openai_el = _is_openai_model(el_cfg)
│       uses_responses = bool(getattr(el_cfg,"use_responses_api", False))
│       supports_realtime_batch = hasattr(el_llm, "batch")
│   
│       # ---- enforce policy for max-inflight ----
│       if args.openai_batch_mode:
│           if not is_openai_el:
│               raise SystemExit("--openai-batch-mode requires an OpenAI Chat Completions model.")
│           if uses_responses:
│               raise SystemExit("--openai-batch-mode incompatible with Responses (gpt-5*) models; use chat-completions.")
│           if args.concurrency and args.concurrency != 1:
│               _dbg("[note] --openai-batch-mode: ignoring --concurrency; Batch API is offline.")
│           if args.max_inflight is None:
│               args.max_inflight = max(1, args.batch_size)
│       else:
│           if args.max_inflight is not None:
│               _dbg("[note] ignoring --max-inflight (only honored with --openai-batch-mode).")
│           args.max_inflight = None
│   
│       # progress timing
│       last_progress_ts = 0.0
│   
│       # shared state
│       start = time.perf_counter()
│       subjects_elicited_total = 0
│       lowconf_accum: List[dict] = []
│       ner_lowconf_accum: List[dict] = []
│       seen_facts: Set[Tuple[str,str,str,int]] = set()
│   
│       # ---- worker for non-realtime-batch path (with retries + salvage) ----
│       def _elicitation_and_ner(subject: str, hop: int):
│           try:
│               attempt = 0
│               facts: List[dict] = []
│               last_text = ""
│               el_schema = ELICIT_SCHEMA_CAL if (args.elicitation_strategy=="calibrate") else ELICIT_SCHEMA_BASE
│   
│               while attempt < max(1, args.max_retries):
│                   el_messages = _build_elicitation_messages(args, subject)
│                   _ensure_json_keyword_in_msgs(el_messages, shape_hint='{"facts":[{"subject":"...","predicate":"...","object":"..."}]}')
│                   if args.debug: _print_messages(f"ELICIT for [{subject}] (try {attempt+1})", el_messages)
│   
│                   try:
│                       resp = el_llm(el_messages, json_schema=el_schema)
│                   except Exception:
│                       resp = el_llm(el_messages)
│   
│                   facts, last_text = _extract_facts_from_resp(resp, debug=args.debug)
│   
│                   if not facts and last_text:
│                       salv = _salvage_facts_from_text(last_text, debug=args.debug)
│                       if salv:
│                           facts = salv
│   
│                   if facts:
│                       break
│                   attempt += 1
│   
│               if not facts:
│                   write_triples_sink(get_thread_facts_conn(paths["facts_sqlite"]),
│                       [(subject,"__empty__","__empty__",hop, el_cfg.model,args.elicitation_strategy,None,"empty_or_unparseable_output")]
│                   )
│   
│               acc, lowconf, _ = _route_facts(args, facts, hop, el_cfg.model)
│               if acc:
│                   write_triples_accepted(get_thread_facts_conn(paths["facts_sqlite"]), acc)
│                   with _seen_facts_lock:
│                       for s,p,o,_,m,st,c in acc:
│                           key = (s,p,o,hop)
│                           if key not in seen_facts:
│                               seen_facts.add(key)
│                               _append_jsonl(paths["facts_jsonl"], {
│                                   "subject": s, "predicate": p, "object": o,
│                                   "hop": hop, "model": m, "strategy": st, "confidence": c
│                               })
│               if lowconf:
│                   for item in lowconf: _append_jsonl(paths["lowconf_jsonl"], item)
│                   with _lowconf_lock: lowconf_accum.extend(lowconf)
│   
│               # NER
│               cand = _filter_ner_candidates([t.get("object") for t in facts if isinstance(t, dict)], subject)
│               next_subjects: List[str] = []
│               i = 0
│               while i < len(cand):
│                   chunk = cand[i: i + args.ner_batch_size]
│                   ner_messages = get_prompt_messages(args.ner_strategy, "ner",
│                       domain=args.domain,
│                       variables=dict(phrases_block="\n".join(chunk), root_subject=args.seed, subject_name=subject))
│                   ner_schema = NER_SCHEMA_CAL if (args.ner_strategy=="calibrate") else NER_SCHEMA_BASE
│                   if args.debug: _print_messages(f"NER for [{subject}] chunk[{i}:{i+args.ner_batch_size}]", ner_messages)
│                   try:
│                       out = ner_llm(ner_messages, json_schema=ner_schema)
│                   except Exception:
│                       out = ner_llm(ner_messages)
│                   norm = _parse_obj(out)
│                   decisions = norm.get("phrases", []) if isinstance(norm.get("phrases"), list) else []
│                   if not decisions:
│                       decisions = [{"phrase": ph, "is_ne": _maybe_is_ne_heuristic(ph), "confidence": None} for ph in chunk]
│   
│                   # >>> force numeric confidence in calibrate, if missing <<<
│                   if args.ner_strategy == "calibrate":
│                       for d in decisions:
│                           if not isinstance(d.get("confidence"), (int, float)):
│                               d["confidence"] = 0.90
│   
│                   use_thr = (args.ner_strategy=="calibrate")
│                   for d in decisions:
│                       phrase = d.get("phrase"); is_ne = bool(d.get("is_ne"))
│                       conf = d.get("confidence")
│                       try: conf = float(conf)
│                       except Exception: conf = None
│                       is_variant = _is_subject_variant(phrase, subject)
│                       if is_variant:
│                           is_ne = False; conf = 0.0 if conf is None else min(conf, 0.0)
│                       conf_ok = (isinstance(conf,(int,float)) and conf >= args.ner_conf_threshold) if use_thr else True
│                       record = {
│                           "current_entity": subject, "hop": hop, "phrase": phrase,
│                           "is_ne": is_ne, "is_variant": is_variant,
│                           "confidence": (float(conf) if isinstance(conf,(int,float)) else None),
│                           "ner_conf_threshold": float(args.ner_conf_threshold),
│                           "passed_threshold": bool(conf_ok if use_thr else True),
│                           "ner_model": ner_cfg.model, "ner_strategy": args.ner_strategy,
│                           "domain": args.domain, "root_subject": args.seed, "source": "model_or_fallback"
│                       }
│                       _append_jsonl(paths["ner_jsonl"], record)
│                       if use_thr and not conf_ok:
│                           low_item = {**record, "reason":"below_threshold"}
│                           _append_jsonl(paths["ner_lowconf_jsonl"], low_item)
│                           with _ner_lowconf_lock: ner_lowconf_accum.append(low_item)
│                       if is_ne and conf_ok and not is_variant and isinstance(phrase,str):
│                           next_subjects.append(phrase)
│                   i += args.ner_batch_size
│   
│               if next_subjects:
│                   results = procq_enqueue(
│                       paths["queue_sqlite"],
│                       [(s, hop+1) for s in next_subjects if (args.max_depth==0 or hop+1<=args.max_depth)],
│                       leading_articles=PROCQ_LEADING
│                   )
│                   for s, kept_hop, outcome in results:
│                       if outcome in ("inserted","hop_reduced"):
│                           _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": kept_hop, "event": outcome})
│                   if args.debug:
│                       _print_enqueue_summary(results)
│                   _write_queue_snapshot(qdb, paths["queue_json"], args.max_depth)
│   
│               mark_done_threadsafe(paths["queue_sqlite"], subject, hop)
│               return (subject, hop, None)
│           except Exception:
│               with open(paths["errors_log"], "a", encoding="utf-8") as ef:
│                   ef.write(f"[{datetime.datetime.now().isoformat()}] subject={subject}\n{traceback.format_exc()}\n")
│               mark_pending_on_error(paths["queue_sqlite"], subject, hop)
│               return (subject, hop, "error")
│   
│       # ---- OpenAI Batch path (with salvage & queue-level retries) ----
│       def _elicitation_openai_batch(subjects_with_hops: List[Tuple[str,int]]):
│           if not subjects_with_hops: return 0
│           client = _make_openai_client_for_batch(el_cfg)
│           subjects = [s for s,_ in subjects_with_hops]
│           hops_map = {s:h for s,h in subjects_with_hops}
│           _write_batch_requests_jsonl(paths["batch_req_jsonl"], subjects, el_cfg, _build_elicitation_messages, args)
│           if args.debug:
│               print(f"[batch] wrote request JSONL: {paths['batch_req_jsonl']}")
│           with open(paths["batch_req_jsonl"], "rb") as f:
│               up = client.files.create(file=f, purpose="batch")
│           batch = client.batches.create(
│               input_file_id=up.id,
│               endpoint="/v1/chat/completions",
│               completion_window="24h",
│               metadata={"description":"crawler elicitation"}
│           )
│           _dbg(f"[batch] created {batch.id}")
│           st = batch.status
│           delay = 5
│           while st in ("created","validating","in_progress","finalizing"):
│               time.sleep(delay)
│               delay = min(int(delay * 1.5), 60)
│               batch = client.batches.retrieve(batch.id)
│               st = batch.status
│               _dbg(f"[batch] {batch.id} status={st}")
│           if st != "completed":
│               _dbg(f"[batch] status={st}; reverting claimed items")
│               with qdb:
│                   for s,h in subjects_with_hops:
│                       qdb.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=? AND hop=? AND status='working'", (s,h))
│               return 0
│           out_file_id = batch.output_file_id
│           content_bytes = client.files.content(out_file_id).content
│           with open(paths["batch_out_jsonl"], "wb") as f:
│               f.write(content_bytes)
│   
│           accepted_total = 0
│           seen_subjects: Set[str] = set()
│   
│           with open(paths["batch_out_jsonl"], "r", encoding="utf-8") as f:
│               for line in f:
│                   try:
│                       subject, facts, raw_text = _parse_openai_batch_output_line(line, debug=args.debug)
│                       if not subject:
│                           continue
│                       seen_subjects.add(subject)
│                       hop = hops_map.get(subject, 0)
│   
│                       if not facts and raw_text:
│                           salv = _salvage_facts_from_text(raw_text, debug=args.debug)
│                           if salv: facts = salv
│   
│                       if not facts:
│                           current_retries = _get_retries(paths["queue_sqlite"], subject, hop)
│                           if current_retries < args.max_retries:
│                               if args.debug:
│                                   print(f"[batch] empty/unparseable for '{subject}', re-queuing (retry {current_retries+1}/{args.max_retries})")
│                               _inc_retries_and_pending(paths["queue_sqlite"], subject, hop)
│                               continue
│                           else:
│                               if args.debug:
│                                   print(f"[batch] empty/unparseable for '{subject}', max retries reached → sink.")
│                               write_triples_sink(fdb, [(subject,"__empty__","__empty__",hop, el_cfg.model,args.elicitation_strategy,None,"empty_or_unparseable_output")])
│                               mark_done_threadsafe(paths["queue_sqlite"], subject, hop)
│                               continue
│   
│                       acc, lowconf, _ = _route_facts(args, facts, hop, el_cfg.model)
│                       if acc:
│                           write_triples_accepted(fdb, acc)
│                           with _seen_facts_lock:
│                               for s,p,o,_,m,stg,c in acc:
│                                   key = (s,p,o,hop)
│                                   if key not in seen_facts:
│                                       seen_facts.add(key)
│                                       _append_jsonl(paths["facts_jsonl"], {"subject": s,"predicate":p,"object":o,"hop":hop,"model":m,"strategy":stg,"confidence":c})
│                           accepted_total += len(acc)
│                       if lowconf:
│                           for item in lowconf: _append_jsonl(paths["lowconf_jsonl"], item)
│                           with _lowconf_lock: lowconf_accum.extend(lowconf)
│   
│                       # NER (real time)
│                       cand = _filter_ner_candidates([t.get("object") for t in facts if isinstance(t, dict)], subject)
│                       next_subjects: List[str] = []
│                       i = 0
│                       while i < len(cand):
│                           chunk = cand[i: i + args.ner_batch_size]
│                           ner_messages = get_prompt_messages(args.ner_strategy, "ner",
│                               domain=args.domain,
│                               variables=dict(phrases_block="\n".join(chunk), root_subject=args.seed, subject_name=subject))
│                           ner_schema = NER_SCHEMA_CAL if (args.ner_strategy=="calibrate") else NER_SCHEMA_BASE
│                           if args.debug: _print_messages(f"NER for [{subject}] chunk[{i}:{i+args.ner_batch_size}]", ner_messages)
│                           try: out = ner_llm(ner_messages, json_schema=ner_schema)
│                           except Exception: out = ner_llm(ner_messages)
│                           norm = _parse_obj(out)
│                           decisions = norm.get("phrases", []) if isinstance(norm.get("phrases"), list) else []
│                           if not decisions:
│                               decisions = [{"phrase": ph, "is_ne": _maybe_is_ne_heuristic(ph), "confidence": None} for ph in chunk]
│   
│                           # >>> force numeric confidence in calibrate, if missing <<<
│                           if args.ner_strategy == "calibrate":
│                               for d in decisions:
│                                   if not isinstance(d.get("confidence"), (int, float)):
│                                       d["confidence"] = 0.90
│   
│                           use_thr = (args.ner_strategy=="calibrate")
│                           for d in decisions:
│                               phrase = d.get("phrase")
│                               is_ne = bool(d.get("is_ne"))
│                               conf = d.get("confidence")
│                               try: conf = float(conf)
│                               except Exception: conf = None
│                               is_variant = _is_subject_variant(phrase, subject)
│                               if is_variant:
│                                   is_ne = False; conf = 0.0 if conf is None else min(conf, 0.0)
│                               conf_ok = (isinstance(conf,(int,float)) and conf >= args.ner_conf_threshold) if use_thr else True
│                               record = {
│                                   "current_entity": subject, "hop": hop, "phrase": phrase,
│                                   "is_ne": is_ne, "is_variant": is_variant,
│                                   "confidence": (float(conf) if isinstance(conf,(int,float)) else None),
│                                   "ner_conf_threshold": float(args.ner_conf_threshold),
│                                   "passed_threshold": bool(conf_ok if use_thr else True),
│                                   "ner_model": ner_cfg.model, "ner_strategy": args.ner_strategy,
│                                   "domain": args.domain, "root_subject": args.seed, "source": "model_or_fallback"
│                               }
│                               _append_jsonl(paths["ner_jsonl"], record)
│                               if use_thr and not conf_ok:
│                                   low_item = {**record, "reason":"below_threshold"}
│                                   _append_jsonl(paths["ner_lowconf_jsonl"], low_item)
│                                   with _ner_lowconf_lock: ner_lowconf_accum.append(low_item)
│                               if is_ne and conf_ok and not is_variant and isinstance(phrase,str):
│                                   next_subjects.append(phrase)
│                           i += args.ner_batch_size
│   
│                       if next_subjects:
│                           results = procq_enqueue(
│                               paths["queue_sqlite"],
│                               [(s, hop+1) for s in next_subjects if (args.max_depth==0 or hop+1<=args.max_depth)],
│                               leading_articles=PROCQ_LEADING
│                           )
│                           for s, kept_hop, outcome in results:
│                               if outcome in ("inserted","hop_reduced"):
│                                   _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": kept_hop, "event": outcome})
│                           if args.debug:
│                               _print_enqueue_summary(results)
│                           _write_queue_snapshot(qdb, paths["queue_json"], args.max_depth)
│   
│                       mark_done_threadsafe(paths["queue_sqlite"], subject, hop)
│   
│                   except Exception:
│                       with qdb:
│                           qdb.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=? AND hop=? AND status='working'", (subject, hop))
│                       with open(paths["errors_log"], "a", encoding="utf-8") as ef:
│                           ef.write(f"[{datetime.datetime.now().isoformat()}] batch_line_error\n{traceback.format_exc()}\n")
│   
│           # requeue any subject missing from output file entirely
│           for s in subjects:
│               if s not in seen_subjects:
│                   _inc_retries_and_pending(paths["queue_sqlite"], s, hops_map.get(s, 0))
│           return accepted_total
│   
│       # ------------- loop -------------
│       while True:
│           if args.progress_metrics:
│               now = time.perf_counter()
│               if now - last_progress_ts >= 2.0:
│                   d,w,p,t = _counts(qdb, args.max_depth)
│                   try:
│                       cur = qdb.cursor(); cur.execute("SELECT SUM(retries) FROM queue"); retry_sum = cur.fetchone()[0] or 0
│                   except Exception:
│                       retry_sum = 0
│                   _dbg(f"[progress] done={d} working={w} pending={p} total={t} retries={retry_sum}")
│                   last_progress_ts = now
│   
│           if args.max_subjects and subjects_elicited_total >= args.max_subjects:
│               _dbg(f"[stop] max-subjects reached ({subjects_elicited_total})")
│               break
│   
│           remaining_cap = (args.max_subjects - subjects_elicited_total) if args.max_subjects else None
│   
│           if args.openai_batch_mode:
│               claim_n = min(args.max_inflight or 1, args.batch_size)
│           elif supports_realtime_batch:
│               claim_n = args.batch_size
│           else:
│               claim_n = args.concurrency
│   
│           if remaining_cap is not None:
│               claim_n = max(1, min(claim_n, remaining_cap))
│   
│           batch = _fetch_many_pending(qdb, args.max_depth, max(1, claim_n))
│           if not batch:
│               d,w,p,t = _counts(qdb, args.max_depth)
│               if t == 0: _dbg("[idle] nothing to do.")
│               else: _dbg(f"[idle] queue drained: done={d} working={w} pending={p} total={t}")
│               break
│   
│           # --- OpenAI Batch (offline) ---
│           if args.openai_batch_mode:
│               _dbg(f"[path=batch] claim {len(batch)} subjects (max_inflight={args.max_inflight}, batch_size={args.batch_size})")
│               _ = _elicitation_openai_batch(batch)
│               subjects_elicited_total += len(batch)
│               continue
│   
│           # --- realtime .batch(...) path ---
│           if supports_realtime_batch:
│               subjects = [s for s,_ in batch]
│               _dbg(f"[path=realtime-batch] groupsize={len(subjects)} (batch_size={args.batch_size})")
│               messages_list = []
│               for s in subjects:
│                   msgs = _build_elicitation_messages(args, s)
│                   _ensure_json_keyword_in_msgs(msgs, shape_hint='{"facts":[{"subject":"...","predicate":"...","object":"..."}]}')
│                   messages_list.append(msgs)
│               if args.debug:
│                   for s,msgs in zip(subjects, messages_list):
│                       _print_messages(f"ELICIT (batch-call) for [{s}]", msgs)
│               el_schema = ELICIT_SCHEMA_CAL if (args.elicitation_strategy=="calibrate") else ELICIT_SCHEMA_BASE
│               try:
│                   try:
│                       resp_list = el_llm.batch(messages_list, json_schema=el_schema, timeout=args.timeout)  # type: ignore
│                   except TypeError:
│                       resp_list = el_llm.batch(messages_list, json_schema=el_schema)  # type: ignore
│               except Exception:
│                   with qdb:
│                       for subject, hop in batch:
│                           qdb.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=? AND hop=? AND status='working'", (subject, hop))
│                   _dbg("[warn] realtime batch call failed; reverted claims")
│                   continue
│               if len(resp_list) != len(batch):
│                   with qdb:
│                       for subject, hop in batch:
│                           qdb.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=? AND hop=? AND status='working'", (subject, hop))
│                   _dbg("[warn] batch size mismatch; reverted claims")
│                   continue
│   
│               for (subject, hop), resp in zip(batch, resp_list):
│                   try:
│                       facts, raw_txt = _extract_facts_from_resp(resp, debug=args.debug)
│                       if not facts and raw_txt:
│                           salv = _salvage_facts_from_text(raw_txt, debug=args.debug)
│                           if salv: facts = salv
│                       if not facts:
│                           write_triples_sink(fdb, [(subject,"__empty__","__empty__",hop, el_cfg.model,args.elicitation_strategy,None,"empty_or_unparseable_output")])
│   
│                       acc, lowconf, _ = _route_facts(args, facts, hop, el_cfg.model)
│                       if acc:
│                           write_triples_accepted(fdb, acc)
│                           with _seen_facts_lock:
│                               for s,p,o,_,m,st,c in acc:
│                                   key = (s,p,o,hop)
│                                   if key not in seen_facts:
│                                       seen_facts.add(key)
│                                       _append_jsonl(paths["facts_jsonl"], {"subject": s,"predicate":p,"object":o,"hop":hop,"model":m,"strategy":st,"confidence":c})
│                       if lowconf:
│                           for item in lowconf: _append_jsonl(paths["lowconf_jsonl"], item)
│                           with _lowconf_lock: lowconf_accum.extend(lowconf)
│   
│                       # NER
│                       cand = _filter_ner_candidates([t.get("object") for t in facts if isinstance(t, dict)], subject)
│                       next_subjects: List[str] = []
│                       i = 0
│                       while i < len(cand):
│                           chunk = cand[i: i + args.ner_batch_size]
│                           ner_messages = get_prompt_messages(args.ner_strategy, "ner",
│                               domain=args.domain,
│                               variables=dict(phrases_block="\n".join(chunk), root_subject=args.seed, subject_name=subject))
│                           ner_schema = NER_SCHEMA_CAL if (args.ner_strategy=="calibrate") else NER_SCHEMA_BASE
│                           if args.debug: _print_messages(f"NER for [{subject}] chunk[{i}:{i+args.ner_batch_size}]", ner_messages)
│                           try: out = ner_llm(ner_messages, json_schema=ner_schema)
│                           except Exception: out = ner_llm(ner_messages)
│                           norm = _parse_obj(out)
│                           decisions = norm.get("phrases", []) if isinstance(norm.get("phrases"), list) else []
│                           if not decisions:
│                               decisions = [{"phrase": ph, "is_ne": _maybe_is_ne_heuristic(ph), "confidence": None} for ph in chunk]
│   
│                           # >>> force numeric confidence in calibrate, if missing <<<
│                           if args.ner_strategy == "calibrate":
│                               for d in decisions:
│                                   if not isinstance(d.get("confidence"), (int, float)):
│                                       d["confidence"] = 0.90
│   
│                           use_thr = (args.ner_strategy=="calibrate")
│                           for d in decisions:
│                               phrase = d.get("phrase"); is_ne = bool(d.get("is_ne"))
│                               conf = d.get("confidence")
│                               try: conf = float(conf)
│                               except Exception: conf = None
│                               is_variant = _is_subject_variant(phrase, subject)
│                               if is_variant:
│                                   is_ne = False; conf = 0.0 if conf is None else min(conf, 0.0)
│                               conf_ok = (isinstance(conf,(int,float)) and conf >= args.ner_conf_threshold) if use_thr else True
│                               record = {
│                                   "current_entity": subject, "hop": hop, "phrase": phrase,
│                                   "is_ne": is_ne, "is_variant": is_variant,
│                                   "confidence": (float(conf) if isinstance(conf,(int,float)) else None),
│                                   "ner_conf_threshold": float(args.ner_conf_threshold),
│                                   "passed_threshold": bool(conf_ok if use_thr else True),
│                                   "ner_model": ner_cfg.model, "ner_strategy": args.ner_strategy,
│                                   "domain": args.domain, "root_subject": args.seed, "source": "model_or_fallback"
│                               }
│                               _append_jsonl(paths["ner_jsonl"], record)
│                               if use_thr and not conf_ok:
│                                   low_item = {**record, "reason":"below_threshold"}
│                                   _append_jsonl(paths["ner_lowconf_jsonl"], low_item)
│                                   with _ner_lowconf_lock: ner_lowconf_accum.append(low_item)
│                               if is_ne and conf_ok and not is_variant and isinstance(phrase,str):
│                                   next_subjects.append(phrase)
│                           i += args.ner_batch_size
│   
│                       if next_subjects:
│                           results = procq_enqueue(
│                               paths["queue_sqlite"],
│                               [(s, hop+1) for s in next_subjects if (args.max_depth==0 or hop+1<=args.max_depth)],
│                               leading_articles=PROCQ_LEADING
│                           )
│                           for s, kept_hop, outcome in results:
│                               if outcome in ("inserted","hop_reduced"):
│                                   _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": kept_hop, "event": outcome})
│                           if args.debug:
│                               _print_enqueue_summary(results)
│                           _write_queue_snapshot(qdb, paths["queue_json"], args.max_depth)
│   
│                       with qdb:
│                           qdb.execute("UPDATE queue SET status='done' WHERE subject=? AND hop=? AND status='working'", (subject, hop))
│                       subjects_elicited_total += 1
│                       if args.max_subjects and subjects_elicited_total >= args.max_subjects:
│                           _dbg(f"[stop] max-subjects reached ({subjects_elicited_total})")
│                           break
│   
│                   except Exception:
│                       with qdb:
│                           qdb.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=? AND hop=? AND status='working'", (subject, hop))
│                       with open(paths["errors_log"], "a", encoding="utf-8") as ef:
│                           ef.write(f"[{datetime.datetime.now().isoformat()}] subject={subject}\n{traceback.format_exc()}\n")
│   
│           # --- pure concurrency path ---
│           else:
│               _dbg(f"[path=concurrency] subjects={len(batch)} workers={min(args.concurrency, len(batch))}")
│               results = []
│               with ThreadPoolExecutor(max_workers=min(args.concurrency, len(batch))) as pool:
│                   futs = [pool.submit(_elicitation_and_ner, s, h) for (s,h) in batch]
│                   for fut in as_completed(futs):
│                       results.append(fut.result())
│               for _s,_h,err in results:
│                   if err is None:
│                       subjects_elicited_total += 1
│                       if args.max_subjects and subjects_elicited_total >= args.max_subjects:
│                           _dbg(f"[stop] max-subjects reached ({subjects_elicited_total})")
│                           break
│   
│       # ----- final snapshots -----
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
│       cur.execute("SELECT subject, predicate, object, hop, model_name, strategy, confidence FROM triples_accepted ORDER BY subject, predicate, object, hop")
│       rows_acc = cur.fetchall()
│       cur.execute("SELECT subject, predicate, object, hop, model_name, strategy, confidence, reason FROM triples_sink ORDER BY subject, predicate, object, hop")
│       rows_sink = cur.fetchall()
│       with open(paths["facts_json"], "w", encoding="utf-8") as f:
│           json.dump(
│               {
│                   "accepted": [
│                       {"subject": s, "predicate": p, "object": o, "hop": h, "model": m, "strategy": st, "confidence": c}
│                       for (s,p,o,h,m,st,c) in rows_acc
│                   ],
│                   "sink": [
│                       {"subject": s, "predicate": p, "object": o, "hop": h, "model": m, "strategy": st, "confidence": c, "reason": r}
│                       for (s,p,o,h,m,st,c,r) in rows_sink
│                   ],
│               },
│               f, ensure_ascii=False, indent=2
│           )
│       conn.close()
│   
│       with open(paths["lowconf_json"], "w", encoding="utf-8") as f:
│           json.dump(lowconf_accum, f, ensure_ascii=False, indent=2)
│       with open(paths["ner_lowconf_json"], "w", encoding="utf-8") as f:
│           json.dump(ner_lowconf_accum, f, ensure_ascii=False, indent=2)
│   
│       run_meta = {
│           "timestamp_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
│           "seed": args.seed, "domain": args.domain,
│           "elicitation_strategy": args.elicitation_strategy, "ner_strategy": args.ner_strategy,
│           "max_depth": args.max_depth, "max_subjects": args.max_subjects,
│           "concurrency": {
│               "batch_size": args.batch_size,
│               "concurrency": args.concurrency,
│               "max_inflight": (args.max_inflight if args.openai_batch_mode else None),
│               "timeout_s": args.timeout,
│               "openai_batch_mode": bool(args.openai_batch_mode),
│           },
│           "models": {
│               "elicitation": {
│                   "provider": getattr(el_cfg,"provider","openai"),
│                   "model": el_cfg.model,
│                   "use_responses_api": getattr(el_cfg,"use_responses_api", False),
│                   "temperature": getattr(el_cfg,"temperature", None),
│                   "top_p": getattr(el_cfg,"top_p", None),
│                   "top_k": getattr(el_cfg,"top_k", None),
│                   "max_tokens": getattr(el_cfg,"max_tokens", None),
│               },
│               "ner": {
│                   "provider": getattr(ner_cfg,"provider","openai"),
│                   "model": ner_cfg.model,
│                   "use_responses_api": getattr(ner_cfg,"use_responses_api", False),
│                   "temperature": getattr(ner_cfg,"temperature", None),
│                   "top_p": getattr(ner_cfg,"top_p", None),
│                   "top_k": getattr(ner_cfg,"top_k", None),
│                   "max_tokens": getattr(ner_cfg,"max_tokens", None),
│               },
│           },
│           "args_raw": vars(args),
│       }
│       with open(paths["run_meta_json"], "w", encoding="utf-8") as f:
│           json.dump(run_meta, f, ensure_ascii=False, indent=2)
│   
│       dur = time.perf_counter() - start
│       print(f"[done] finished in {dur:.1f}s → {out_dir}")
│       for k in ("queue_json","facts_json","facts_jsonl","lowconf_json","lowconf_jsonl","ner_jsonl","ner_lowconf_json","ner_lowconf_jsonl","run_meta_json","errors_log"):
│           print(f"[out] {k:18}: {paths[k]}")
│   
│   if __name__ == "__main__":
│       try:
│           main()
│       except KeyboardInterrupt:
│           print("\n[interrupt] bye")
│   --- File Content End ---

├── tracer.py
│   --- File Content Start ---
│   # tracer.py
│   from __future__ import annotations
│   import json, time, threading, datetime
│   from typing import Any, Dict, List, Optional
│   
│   _jsonl_lock = threading.Lock()
│   
│   def append_jsonl(path: str, obj: dict):
│       line = json.dumps(obj, ensure_ascii=False) + "\n"
│       with _jsonl_lock:
│           with open(path, "a", encoding="utf-8") as f:
│               f.write(line)
│   
│   def _knob(v):
│       try:
│           return float(v) if v is not None else None
│       except Exception:
│           return None
│   
│   def _now():
│       return datetime.datetime.utcnow().isoformat() + "Z"
│   
│   class TracedLLM:
│       """
│       Wraps an LLM client (e.g., from llm.factory.make_llm_from_config).
│       Logs request/response metadata to a JSONL file so you can verify
│       temperature/top_p/top_k/max_tokens, model, provider, durations, etc.
│       """
│       def __init__(self, llm, *, name: str, trace_path: str, echo: bool = False):
│           self._llm = llm
│           self._name = name
│           self._trace_path = trace_path
│           self._echo = echo  # also print a short line to stdout
│   
│       # --- helpers to read config fields if present ---
│       def _cfg_str(self, attr, default=None):
│           try:
│               return getattr(self._llm, attr)
│           except Exception:
│               return default
│   
│       def _cfg_num(self, attr):
│           return _knob(self._cfg_str(attr, None))
│   
│       def _provider(self):
│           # we try provider from config; fallback to class/module names
│           prov = self._cfg_str("provider", None)
│           if prov: return str(prov)
│           return f"{self._llm.__class__.__module__}.{self._llm.__class__.__name__}"
│   
│       def _model(self):
│           return self._cfg_str("model", None)
│   
│       def _max_tokens(self):
│           return self._cfg_num("max_tokens")
│   
│       def _knobs_snapshot(self) -> Dict[str, Any]:
│           return {
│               "temperature": self._cfg_num("temperature"),
│               "top_p": self._cfg_num("top_p"),
│               "top_k": self._cfg_num("top_k"),
│               "max_tokens": self._max_tokens(),
│           }
│   
│       def _messages_meta(self, messages) -> Dict[str, Any]:
│           try:
│               n = len(messages) if isinstance(messages, list) else None
│               total_chars = 0
│               if isinstance(messages, list):
│                   for m in messages:
│                       c = m.get("content")
│                       if isinstance(c, str):
│                           total_chars += len(c)
│               return {"count": n, "total_chars": total_chars}
│           except Exception:
│               return {"count": None, "total_chars": None}
│   
│       def _batch_meta(self, messages_list) -> Dict[str, Any]:
│           try:
│               n = len(messages_list) if isinstance(messages_list, list) else None
│               counts = []
│               chars = 0
│               if isinstance(messages_list, list):
│                   for msgs in messages_list:
│                       mm = self._messages_meta(msgs)
│                       counts.append(mm["count"])
│                       chars += (mm["total_chars"] or 0)
│               return {"batches": n, "per_batch_counts": counts[:10], "total_chars": chars}
│           except Exception:
│               return {"batches": None, "per_batch_counts": None, "total_chars": None}
│   
│       def _log(self, payload: dict):
│           payload.setdefault("ts", _now())
│           payload.setdefault("who", self._name)
│           append_jsonl(self._trace_path, payload)
│           if self._echo:
│               # single-line echo for quick eyes-on
│               kind = payload.get("event")
│               model = payload.get("model")
│               prov = payload.get("provider")
│               took = payload.get("took_ms")
│               knobs = payload.get("knobs", {})
│               print(f"[api-trace] {kind} {prov}:{model} took={took}ms "
│                     f"temp={knobs.get('temperature')} top_p={knobs.get('top_p')} top_k={knobs.get('top_k')} max_tokens={knobs.get('max_tokens')}",
│                     flush=True)
│   
│       # ---------------- public call wrappers ----------------
│       def __call__(self, messages: List[dict], **kwargs):
│           t0 = time.time()
│           req = {
│               "event": "request",
│               "provider": self._provider(),
│               "model": self._model(),
│               "api_method": "__call__",
│               "knobs": self._knobs_snapshot(),
│               "messages_meta": self._messages_meta(messages),
│               "kwargs": {
│                   # we only record presence / types for sensitive fields; avoid dumping prompts
│                   "json_schema": bool(kwargs.get("json_schema") is not None),
│                   "timeout": kwargs.get("timeout", None),
│               },
│           }
│           self._log(req)
│           try:
│               out = self._llm(messages, **kwargs)
│               took = int((time.time() - t0) * 1000)
│               resp = {
│                   "event": "response",
│                   "provider": self._provider(),
│                   "model": self._model(),
│                   "api_method": "__call__",
│                   "took_ms": took,
│                   # light footprint: record rough size/info, not full content
│                   "response_meta": _safe_shape(out),
│               }
│               self._log(resp)
│               return out
│           except Exception as e:
│               took = int((time.time() - t0) * 1000)
│               self._log({
│                   "event": "error",
│                   "provider": self._provider(),
│                   "model": self._model(),
│                   "api_method": "__call__",
│                   "took_ms": took,
│                   "error": repr(e),
│               })
│               raise
│   
│       def batch(self, messages_list: List[List[dict]], **kwargs):
│           t0 = time.time()
│           req = {
│               "event": "request",
│               "provider": self._provider(),
│               "model": self._model(),
│               "api_method": "batch",
│               "knobs": self._knobs_snapshot(),
│               "batch_meta": self._batch_meta(messages_list),
│               "kwargs": {
│                   "json_schema": bool(kwargs.get("json_schema") is not None),
│                   "timeout": kwargs.get("timeout", None),
│               },
│           }
│           self._log(req)
│           try:
│               out = self._llm.batch(messages_list, **kwargs)
│               took = int((time.time() - t0) * 1000)
│               resp = {
│                   "event": "response",
│                   "provider": self._provider(),
│                   "model": self._model(),
│                   "api_method": "batch",
│                   "took_ms": took,
│                   "response_meta": _safe_shape(out),
│               }
│               self._log(resp)
│               return out
│           except Exception as e:
│               took = int((time.time() - t0) * 1000)
│               self._log({
│                   "event": "error",
│                   "provider": self._provider(),
│                   "model": self._model(),
│                   "api_method": "batch",
│                   "took_ms": took,
│                   "error": repr(e),
│               })
│               raise
│   
│   def _safe_shape(obj):
│       """
│       Record a tiny ‘shape’ so you can see what came back without storing payloads.
│       """
│       try:
│           if isinstance(obj, list):
│               return {"type": "list", "len": len(obj)}
│           if isinstance(obj, dict):
│               keys = list(obj.keys())
│               return {"type": "dict", "keys": keys[:12], "nkeys": len(keys)}
│           if isinstance(obj, str):
│               return {"type": "str", "len": len(obj)}
│           return {"type": type(obj).__name__}
│       except Exception:
│           return {"type": "unknown"}
│   --- File Content End ---

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
│   │   from typing import Any, Dict
│   │   import os
│   │   
│   │   from llm.config import ModelConfig
│   │   from llm.openai_client import OpenAIClient
│   │   from llm.replicate_client import ReplicateLLM
│   │   from llm.deepseek_client import DeepSeekClient
│   │   
│   │   def make_llm_from_config(cfg: ModelConfig):
│   │       prov = (cfg.provider or "").lower()
│   │   
│   │       if prov == "openai":
│   │           api_key = os.getenv(cfg.api_key_env or "OPENAI_API_KEY")
│   │           if not api_key:
│   │               raise RuntimeError("Missing OPENAI_API_KEY.")
│   │           return OpenAIClient(
│   │               model=cfg.model,
│   │               api_key=api_key,
│   │               base_url=cfg.base_url,
│   │               max_tokens=cfg.max_tokens,
│   │               temperature=cfg.temperature,
│   │               top_p=cfg.top_p,
│   │               use_responses_api=bool(getattr(cfg, "use_responses_api", False)),
│   │               extra_inputs=getattr(cfg, "extra_inputs", None),
│   │           )
│   │   
│   │       if prov == "replicate":
│   │           token = os.getenv("REPLICATE_API_TOKEN")
│   │           if not token:
│   │               raise RuntimeError("Missing REPLICATE_API_TOKEN.")
│   │           # Pass model defaults (prompt_template, stop_sequences, etc.) through
│   │           return ReplicateLLM(
│   │               model=cfg.model,
│   │               api_token=token,
│   │               default_extra=getattr(cfg, "extra_inputs", None),
│   │           )
│   │   
│   │       if prov == "deepseek":
│   │           api_key = os.getenv(cfg.api_key_env or "DEEPSEEK_API_KEY")
│   │           if not api_key:
│   │               raise RuntimeError("Missing DEEPSEEK_API_KEY.")
│   │           return DeepSeekClient(
│   │               model=cfg.model,
│   │               api_key=api_key,
│   │               base_url=cfg.base_url,
│   │               max_tokens=cfg.max_tokens,
│   │               temperature=cfg.temperature,
│   │               top_p=cfg.top_p,
│   │               extra_inputs=getattr(cfg, "extra_inputs", None),
│   │           )
│   │   
│   │       if prov == "unsloth":
│   │           raise RuntimeError("Unsloth backend not available in this environment.")
│   │   
│   │       raise ValueError(f"Unknown provider: {prov}")
│   │   --- File Content End ---

│   ├── deepseek_client.py
│   │   --- File Content Start ---
│   │   # llm/deepseek_client.py
│   │   from __future__ import annotations
│   │   from typing import Any, Dict, List, Optional
│   │   import json
│   │   import time
│   │   import requests
│   │   
│   │   from llm.json_utils import best_json, strip_fences as _strip_fences  # unified utils
│   │   
│   │   def _schema_hint(schema: Dict[str, Any]) -> str:
│   │       return (
│   │           "Return ONLY one valid JSON object that matches this JSON Schema exactly. "
│   │           "No prose, no markdown, no code fences. "
│   │           "If unsure, return an empty but valid object per schema.\nSCHEMA:\n"
│   │           + json.dumps(schema, ensure_ascii=False)
│   │       )
│   │   
│   │   def _best_json(text: str) -> Dict[str, Any]:
│   │       obj = best_json(text)
│   │       return obj if isinstance(obj, dict) else {}
│   │   
│   │   class DeepSeekClient:
│   │       """
│   │       Minimal DeepSeek client (Chat-like).
│   │       We never use OpenAI 'response_format', since DeepSeek won't accept json_schema.
│   │       """
│   │   
│   │       def __init__(
│   │           self,
│   │           model: str,
│   │           api_key: str,
│   │           base_url: Optional[str] = "https://api.deepseek.com",
│   │           max_tokens: Optional[int] = 1024,
│   │           temperature: Optional[float] = 0.2,
│   │           top_p: Optional[float] = 1.0,
│   │           extra_inputs: Optional[Dict[str, Any]] = None,
│   │           request_timeout: float = 120.0,
│   │       ):
│   │           self.model = model
│   │           self.max_tokens = max_tokens
│   │           self.temperature = temperature
│   │           self.top_p = top_p
│   │           self.extra = extra_inputs or {}
│   │           self.url = f"{base_url.rstrip('/')}/chat/completions"
│   │           self.headers = {
│   │               "Authorization": f"Bearer {api_key}",
│   │               "Content-Type": "application/json",
│   │           }
│   │           self.request_timeout = request_timeout
│   │   
│   │       def __call__(self, messages: List[Dict[str, str]], json_schema: Optional[Dict[str, Any]] = None):
│   │           # Inject strict schema instructions in the system message instead of response_format
│   │           msgs = list(messages)
│   │           if json_schema:
│   │               if msgs and msgs[0].get("role") == "system":
│   │                   msgs[0] = {"role": "system", "content": msgs[0]["content"] + "\n\n" + _schema_hint(json_schema)}
│   │               else:
│   │                   msgs.insert(0, {"role": "system", "content": _schema_hint(json_schema)})
│   │   
│   │           payload: Dict[str, Any] = {
│   │               "model": self.model,
│   │               "messages": msgs,
│   │               "temperature": self.temperature,
│   │               "top_p": self.top_p,
│   │               "max_tokens": self.max_tokens,
│   │           }
│   │           # allow user extras (e.g., penalties) but remove Nones
│   │           for k, v in (self.extra or {}).items():
│   │               if v is not None:
│   │                   payload[k] = v
│   │   
│   │           # modest retry for transient HTTP errors
│   │           last_exc = None
│   │           for attempt in range(3):
│   │               try:
│   │                   r = requests.post(self.url, headers=self.headers, json=payload, timeout=self.request_timeout)
│   │                   r.raise_for_status()
│   │                   data = r.json()
│   │                   break
│   │               except requests.HTTPError as e:
│   │                   last_exc = e
│   │                   status = getattr(e.response, "status_code", None) if e.response else None
│   │                   if status in (429, 500, 502, 503, 504) and attempt < 2:
│   │                       time.sleep(2 ** attempt)
│   │                       continue
│   │                   raise
│   │               except Exception as e:
│   │                   last_exc = e
│   │                   if attempt < 2:
│   │                       time.sleep(2 ** attempt)
│   │                       continue
│   │                   raise last_exc  # re-raise after retries
│   │   
│   │           # content
│   │           try:
│   │               text = (data["choices"][0]["message"]["content"] or "").strip()
│   │           except Exception:
│   │               text = ""
│   │   
│   │           if not json_schema:
│   │               return {"text": text, "_raw": text}
│   │   
│   │           # parse/salvage
│   │           obj = _best_json(text)
│   │           if obj:
│   │               return obj
│   │           return {"_raw": text}
│   │   --- File Content End ---

│   ├── openai_client.py
│   │   --- File Content Start ---
│   │   # llm/openai_client.py
│   │   from __future__ import annotations
│   │   from typing import Any, Dict, List, Optional
│   │   import json
│   │   from openai import OpenAI
│   │   from openai import BadRequestError
│   │   
│   │   # ---------- helpers borrowed from DeepSeek client ----------
│   │   
│   │   def _schema_hint(schema: Dict[str, Any]) -> str:
│   │       return (
│   │           "Return ONLY one valid JSON object that matches this JSON Schema exactly. "
│   │           "No prose, no markdown, no code fences.\nSCHEMA:\n" +
│   │           json.dumps(schema, ensure_ascii=False)
│   │       )
│   │   
│   │   def _strip_fences(t: str) -> str:
│   │       s = (t or "").strip()
│   │       if s.startswith("```"):
│   │           nl = s.find("\n")
│   │           if nl != -1:
│   │               s = s[nl+1:].strip()
│   │           if s.endswith("```"):
│   │               s = s[:-3].strip()
│   │       return s
│   │   
│   │   def _best_json(text: str) -> Dict[str, Any]:
│   │       if not text:
│   │           return {}
│   │       # direct
│   │       try:
│   │           return json.loads(text)
│   │       except Exception:
│   │           pass
│   │       # strip fences
│   │       t = _strip_fences(text)
│   │       try:
│   │           return json.loads(t)
│   │       except Exception:
│   │           pass
│   │       # first balanced object
│   │       s = t.find("{")
│   │       if s != -1:
│   │           depth = 0
│   │           for i, ch in enumerate(t[s:], s):
│   │               if ch == "{":
│   │                   depth += 1
│   │               elif ch == "}":
│   │                   depth -= 1
│   │                   if depth == 0:
│   │                       try:
│   │                           return json.loads(t[s:i+1])
│   │                       except Exception:
│   │                           break
│   │       return {}
│   │   
│   │   def _lock_down_additional_props(schema: Any) -> Any:
│   │       """
│   │       Recursively enforce additionalProperties:false on all object nodes.
│   │       This prevents OpenAI's 'additionalProperties is required and must be false' error.
│   │       """
│   │       if isinstance(schema, dict):
│   │           t = schema.get("type")
│   │           if t == "object":
│   │               schema.setdefault("additionalProperties", False)
│   │               props = schema.get("properties")
│   │               if isinstance(props, dict):
│   │                   for k in list(props.keys()):
│   │                       props[k] = _lock_down_additional_props(props[k])
│   │           elif t == "array":
│   │               if "items" in schema:
│   │                   schema["items"] = _lock_down_additional_props(schema["items"])
│   │           else:
│   │               # primitives: nothing to do
│   │               pass
│   │       return schema
│   │   
│   │   def _inject_schema_hint_into_messages(messages: List[Dict[str, str]], json_schema: Dict[str, Any]) -> List[Dict[str, str]]:
│   │       """
│   │       DeepSeek-style: put the schema contract into the system message so that even
│   │       if response_format isn't honored, the model is still told to output strict JSON.
│   │       """
│   │       msgs = list(messages)
│   │       hint = _schema_hint(json_schema)
│   │       if msgs and (msgs[0].get("role") == "system"):
│   │           msgs[0] = {"role": "system", "content": (msgs[0].get("content","") + "\n\n" + hint)}
│   │       else:
│   │           msgs.insert(0, {"role": "system", "content": hint})
│   │       return msgs
│   │   
│   │   def _extract_text_from_chat(resp) -> str:
│   │       try:
│   │           return (resp.choices[0].message.content or "").strip()
│   │       except Exception:
│   │           return ""
│   │   
│   │   def _extract_text_from_responses_api(resp) -> str:
│   │       # Prefer convenience field when available
│   │       out = getattr(resp, "output_text", None)
│   │       if out:
│   │           return out.strip()
│   │       # Reconstruct from blocks if needed
│   │       try:
│   │           parts: List[str] = []
│   │           for block in getattr(resp, "output", []) or []:
│   │               for c in getattr(block, "content", []) or []:
│   │                   txt = getattr(c, "text", "")
│   │                   if txt:
│   │                       parts.append(txt)
│   │           return "".join(parts).strip()
│   │       except Exception:
│   │           return ""
│   │   
│   │   def _parse_with_salvage(text: str, want_schema: bool) -> Dict[str, Any]:
│   │       if not want_schema:
│   │           return {"text": text}
│   │       # strict first
│   │       try:
│   │           return json.loads(text)
│   │       except Exception:
│   │           pass
│   │       # salvage
│   │       obj = _best_json(text)
│   │       if obj:
│   │           return obj
│   │       return {"_raw": text}
│   │   
│   │   # ---------- client ----------
│   │   
│   │   class OpenAIClient:
│   │       """
│   │       Unified OpenAI client that can call either:
│   │         • Chat Completions API (gpt-4o, gpt-4o-mini, etc.)
│   │         • Responses API (gpt-5 family)
│   │   
│   │       DeepSeek-style hardening:
│   │         - Inject schema hint into system message
│   │         - Lock additionalProperties:false recursively
│   │         - Salvage JSON if strict parsing fails
│   │         - Retry without response_format if provider rejects schema
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
│   │           # Heuristic: Responses API for gpt-5* unless explicitly disabled
│   │           self.use_responses_api = bool(use_responses_api or (model or "").startswith("gpt-5"))
│   │           self.extra_inputs = extra_inputs or {}
│   │   
│   │           if base_url:
│   │               self.client = OpenAI(api_key=api_key, base_url=base_url)
│   │           else:
│   │               self.client = OpenAI(api_key=api_key)
│   │   
│   │       def __call__(self, messages: List[Dict[str, str]], json_schema: Optional[Dict[str, Any]] = None):
│   │           if self.use_responses_api:
│   │               return self._call_responses(messages, json_schema)
│   │           return self._call_chat(messages, json_schema)
│   │   
│   │       # ---------------- Chat Completions ----------------
│   │   
│   │       def _call_chat(self, messages: List[Dict[str, str]], json_schema: Optional[Dict[str, Any]]):
│   │           msgs = list(messages)
│   │           kwargs: Dict[str, Any] = dict(
│   │               model=self.model,
│   │               messages=msgs,
│   │               temperature=self.temperature,
│   │               top_p=self.top_p,
│   │               max_tokens=self.max_tokens,
│   │           )
│   │   
│   │           # If schema provided: lock schema, inject hint, try with response_format first
│   │           have_schema = json_schema is not None
│   │           if have_schema:
│   │               safe_schema = _lock_down_additional_props(json.loads(json.dumps(json_schema)))
│   │               msgs = _inject_schema_hint_into_messages(msgs, safe_schema)
│   │               kwargs["messages"] = msgs
│   │               kwargs["response_format"] = {
│   │                   "type": "json_schema",
│   │                   "json_schema": {
│   │                       "name": "schema",
│   │                       "schema": safe_schema,
│   │                       "strict": True,  # ask for validation
│   │                   },
│   │               }
│   │   
│   │           # 1st try: with response_format (when schema present)
│   │           try:
│   │               resp = self.client.chat.completions.create(**kwargs)
│   │               text = _extract_text_from_chat(resp)
│   │               if not have_schema:
│   │                   return {"text": text}
│   │               parsed = _parse_with_salvage(text, want_schema=True)
│   │               return parsed
│   │           except BadRequestError as e:
│   │               # Common case: JSON schema format complaints → retry without response_format
│   │               if have_schema:
│   │                   try:
│   │                       # Remove response_format, keep the DeepSeek-style system hint
│   │                       kwargs.pop("response_format", None)
│   │                       resp = self.client.chat.completions.create(**kwargs)
│   │                       text = _extract_text_from_chat(resp)
│   │                       parsed = _parse_with_salvage(text, want_schema=True)
│   │                       return parsed
│   │                   except Exception:
│   │                       raise
│   │               raise
│   │           except Exception:
│   │               # Last resort: retry without response_format if we had schema
│   │               if have_schema:
│   │                   kwargs.pop("response_format", None)
│   │                   resp = self.client.chat.completions.create(**kwargs)
│   │                   text = _extract_text_from_chat(resp)
│   │                   parsed = _parse_with_salvage(text, want_schema=True)
│   │                   return parsed
│   │               raise
│   │   
│   │       # ---------------- Responses API (gpt-5*) ----------------
│   │   
│   │       def _call_responses(self, messages: List[Dict[str, str]], json_schema: Optional[Dict[str, Any]]):
│   │           have_schema = json_schema is not None
│   │           msgs = list(messages)
│   │   
│   │           # Inject schema hint like DeepSeek even for Responses API
│   │           if have_schema:
│   │               safe_schema = _lock_down_additional_props(json.loads(json.dumps(json_schema)))
│   │               msgs = _inject_schema_hint_into_messages(msgs, safe_schema)
│   │           reasoning = self.extra_inputs.get("reasoning")
│   │           text_opts = self.extra_inputs.get("text")
│   │   
│   │           base_kwargs: Dict[str, Any] = {
│   │               "model": self.model,
│   │               "input": msgs,
│   │               "max_output_tokens": self.max_tokens,
│   │           }
│   │           if reasoning:
│   │               base_kwargs["reasoning"] = reasoning
│   │           if text_opts:
│   │               base_kwargs["text"] = text_opts
│   │   
│   │           with_schema_kwargs = dict(base_kwargs)
│   │           if have_schema:
│   │               with_schema_kwargs["response_format"] = {
│   │                   "type": "json_schema",
│   │                   "json_schema": {
│   │                       "name": "schema",
│   │                       "schema": safe_schema,
│   │                       "strict": True,
│   │                   },
│   │               }
│   │           else:
│   │               with_schema_kwargs["response_format"] = {"type": "text"}
│   │   
│   │           # 1st try with response_format (if schema)
│   │           try:
│   │               resp = self.client.responses.create(**with_schema_kwargs)
│   │               text = _extract_text_from_responses_api(resp)
│   │               if not have_schema:
│   │                   return {"text": text}
│   │               parsed = _parse_with_salvage(text, want_schema=True)
│   │               return parsed
│   │           except BadRequestError:
│   │               # Retry without response_format but keep hint
│   │               resp = self.client.responses.create(**base_kwargs)
│   │               text = _extract_text_from_responses_api(resp)
│   │               if not have_schema:
│   │                   return {"text": text}
│   │               parsed = _parse_with_salvage(text, want_schema=True)
│   │               return parsed
│   │           except TypeError:
│   │               # Older SDKs → missing response_format support; retry bare
│   │               resp = self.client.responses.create(**base_kwargs)
│   │               text = _extract_text_from_responses_api(resp)
│   │               if not have_schema:
│   │                   return {"text": text}
│   │               parsed = _parse_with_salvage(text, want_schema=True)
│   │               return parsed
│   │           except Exception:
│   │               # Final fallback
│   │               resp = self.client.responses.create(**base_kwargs)
│   │               text = _extract_text_from_responses_api(resp)
│   │               if not have_schema:
│   │                   return {"text": text}
│   │               parsed = _parse_with_salvage(text, want_schema=True)
│   │               return parsed
│   │   
│   │   
│   │   __all__ = ["OpenAIClient"]
│   │   --- File Content End ---

│   ├── json_utils.py
│   │   --- File Content Start ---
│   │   # llm/json_utils.py
│   │   from __future__ import annotations
│   │   import json
│   │   
│   │   def strip_fences(t: str) -> str:
│   │       s = (t or "").strip()
│   │       if s.startswith("```"):
│   │           nl = s.find("\n")
│   │           if nl != -1:
│   │               s = s[nl + 1:].strip()
│   │           if s.endswith("```"):
│   │               s = s[:-3].strip()
│   │       return s
│   │   
│   │   def best_json(text: str):
│   │       """
│   │       Robust, quote/escape-aware JSON extraction.
│   │       Returns a dict/list on success, or {} on failure.
│   │       """
│   │       if not text:
│   │           return {}
│   │       # direct attempt
│   │       try:
│   │           return json.loads(text)
│   │       except Exception:
│   │           pass
│   │   
│   │       t = strip_fences(text)
│   │       try:
│   │           return json.loads(t)
│   │       except Exception:
│   │           pass
│   │   
│   │       def scan_for(open_ch: str, close_ch: str):
│   │           s = -1
│   │           depth = 0
│   │           in_str = False
│   │           esc = False
│   │           for i, ch in enumerate(t):
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
│   │               if ch == open_ch:
│   │                   if depth == 0:
│   │                       s = i
│   │                   depth += 1
│   │               elif ch == close_ch and depth > 0:
│   │                   depth -= 1
│   │                   if depth == 0 and s != -1:
│   │                       chunk = t[s:i+1]
│   │                       try:
│   │                           return json.loads(chunk)
│   │                       except Exception:
│   │                           s = -1  # keep scanning
│   │           return {}
│   │   
│   │       return scan_for("{", "}") or scan_for("[", "]") or {}
│   │   --- File Content End ---

│   ├── anthropic_client.py
│   │   --- File Content Start ---
│   │   # llm/anthropic_client.py
│   │   from __future__ import annotations
│   │   
│   │   import os
│   │   import time
│   │   from typing import Any, Dict, List, Optional, Tuple
│   │   
│   │   from dotenv import load_dotenv
│   │   
│   │   # Load .env so ANTHROPIC_API_KEY is available
│   │   load_dotenv()
│   │   
│   │   try:
│   │       import anthropic
│   │   except Exception:
│   │       anthropic = None
│   │   
│   │   
│   │   class AnthropicLLM:
│   │       """
│   │       Minimal Anthropic wrapper with thinking constraints.
│   │   
│   │       - Reads ANTHROPIC_API_KEY from env (or pass api_key= explicitly).
│   │       - Accepts messages like [{"role":"system","content":"..."}, {"role":"user","content":"..."}].
│   │         We collect system messages into `system=` and pass the rest to `messages=`.
│   │       - Extended thinking via thinking={"type":"enabled","budget_tokens":...} (alias: reasoning=).
│   │         When thinking is enabled:
│   │           * temperature is FORCED to 1 (non-overridable).
│   │           * max_tokens is FORCED to be >= 1024.
│   │           * if budget_tokens is provided, max_tokens is FORCED to be > budget_tokens.
│   │       - Returns {"text": <string>, "_raw": <sdk_response>}.
│   │       """
│   │   
│   │       def __init__(self, api_key: Optional[str] = None, *, max_retries: int = 3, debug: bool = False):
│   │           if anthropic is None:
│   │               raise ImportError("anthropic SDK not installed. Run: pip install anthropic python-dotenv")
│   │   
│   │           self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
│   │           if not self.api_key:
│   │               raise ValueError("Missing ANTHROPIC_API_KEY. Set it in your .env or pass api_key=.")
│   │   
│   │           ClientClass = getattr(anthropic, "Anthropic", None) or getattr(anthropic, "Client", None)
│   │           if ClientClass is None:
│   │               raise RuntimeError("Anthropic SDK missing Anthropic/Client class.")
│   │   
│   │           self.client = ClientClass(api_key=self.api_key)
│   │           self.max_retries = max(1, int(max_retries))
│   │           self.debug = bool(debug or os.getenv("ANTHROPIC_DEBUG") == "1")
│   │   
│   │       def __call__(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
│   │           return self.generate(messages, **kwargs)
│   │   
│   │       # ---------- helpers ----------
│   │   
│   │       def _log(self, *a):
│   │           if self.debug:
│   │               print("[AnthropicLLM]", *a, flush=True)
│   │   
│   │       @staticmethod
│   │       def _split_system_and_dialog(messages: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
│   │           """
│   │           Returns (system_text, dialog_messages_without_system).
│   │           Concatenates multiple system messages with blank lines.
│   │           """
│   │           sys_parts: List[str] = []
│   │           dialog: List[Dict[str, str]] = []
│   │           for m in messages or []:
│   │               role = (m.get("role") or "").strip().lower()
│   │               content = (m.get("content") or "")
│   │               if role == "system":
│   │                   if content:
│   │                       sys_parts.append(str(content))
│   │               elif role in ("user", "assistant"):
│   │                   dialog.append({"role": role, "content": str(content)})
│   │               else:
│   │                   dialog.append({"role": "user", "content": str(content)})
│   │           return "\n\n".join(sys_parts).strip(), dialog
│   │   
│   │       @staticmethod
│   │       def _extract_text(resp: Any) -> str:
│   │           # Messages API: resp.content is list of blocks
│   │           try:
│   │               content = getattr(resp, "content", None)
│   │               if isinstance(content, list):
│   │                   parts: List[str] = []
│   │                   for blk in content:
│   │                       if hasattr(blk, "text"):
│   │                           parts.append(str(getattr(blk, "text") or ""))
│   │                       elif isinstance(blk, dict) and blk.get("type") == "text":
│   │                           parts.append(str(blk.get("text") or ""))
│   │                       elif isinstance(blk, str):
│   │                           parts.append(blk)
│   │                   return " ".join(p for p in parts if p).strip()
│   │               if isinstance(content, str):
│   │                   return content
│   │           except Exception:
│   │               pass
│   │   
│   │           # Legacy completion style
│   │           try:
│   │               comp = getattr(resp, "completion", None)
│   │               if isinstance(comp, str):
│   │                   return comp
│   │           except Exception:
│   │               pass
│   │   
│   │           # Dict-like fallback
│   │           if isinstance(resp, dict):
│   │               c = resp.get("content")
│   │               if isinstance(c, list):
│   │                   texts: List[str] = []
│   │                   for blk in c:
│   │                       if isinstance(blk, dict) and "text" in blk:
│   │                           texts.append(str(blk["text"] or ""))
│   │                       elif isinstance(blk, str):
│   │                           texts.append(blk)
│   │                   return " ".join(t for t in texts if t).strip()
│   │               if isinstance(c, str):
│   │                   return c
│   │               if isinstance(resp.get("completion"), str):
│   │                   return str(resp["completion"])
│   │   
│   │           try:
│   │               return str(resp)
│   │           except Exception:
│   │               return ""
│   │   
│   │       # ---------- main call ----------
│   │   
│   │       def generate(
│   │           self,
│   │           messages: List[Dict[str, str]],
│   │           *,
│   │           model: str = "claude-sonnet-4-5-20250929",
│   │           max_tokens: Optional[int] = 512,
│   │           temperature: Optional[float] = 0.0,
│   │           reasoning: Optional[Dict[str, Any]] = None,  # alias for thinking
│   │           thinking: Optional[Dict[str, Any]] = None,
│   │           **extra,
│   │       ) -> Dict[str, Any]:
│   │           """
│   │           Calls anthropic.messages.create() with enforced constraints when thinking is enabled.
│   │           """
│   │           system_text, dialog = self._split_system_and_dialog(messages)
│   │   
│   │           thinking_payload = thinking or reasoning
│   │   
│   │           # ----- Enforce MUSTs when thinking is enabled -----
│   │           if thinking_payload:
│   │               # 1) Force temperature=1, not changeable
│   │               if temperature != 1:
│   │                   self._log("forcing temperature=1 (thinking enabled)")
│   │               temperature = 1
│   │   
│   │               # 2) Force max_tokens >= 1024
│   │               if max_tokens is None or max_tokens < 1024:
│   │                   self._log(f"bumping max_tokens to >=1024 (was {max_tokens})")
│   │                   max_tokens = 1024
│   │   
│   │               # 3) Ensure max_tokens > thinking budget_tokens (if provided)
│   │               budget = None
│   │               if isinstance(thinking_payload, dict):
│   │                   budget = thinking_payload.get("budget_tokens")
│   │               if isinstance(budget, (int, float)):
│   │                   budget = int(budget)
│   │                   if max_tokens <= budget:
│   │                       new_max = budget + 1
│   │                       self._log(f"bumping max_tokens to > budget ({budget}); setting max_tokens={new_max}")
│   │                       max_tokens = new_max
│   │           # --------------------------------------------------
│   │   
│   │           # Build call kwargs
│   │           call_kwargs: Dict[str, Any] = {
│   │               "model": model,
│   │               "max_tokens": int(max_tokens if max_tokens is not None else 512),
│   │               "messages": dialog,
│   │           }
│   │           if system_text:
│   │               call_kwargs["system"] = system_text
│   │   
│   │           if thinking_payload:
│   │               call_kwargs["thinking"] = thinking_payload
│   │               call_kwargs["temperature"] = 1  # double-assert
│   │           else:
│   │               if temperature is not None:
│   │                   call_kwargs["temperature"] = temperature
│   │   
│   │           # Pass through any supported extras (avoid overriding our enforced keys)
│   │           for k, v in (extra or {}).items():
│   │               if k in ("model", "messages", "system", "max_tokens", "temperature", "thinking", "reasoning"):
│   │                   continue
│   │               if v is not None:
│   │                   call_kwargs[k] = v
│   │   
│   │           # Retry on transient errors
│   │           last_err: Optional[BaseException] = None
│   │           for attempt in range(1, self.max_retries + 1):
│   │               try:
│   │                   self._log(
│   │                       f"messages.create attempt={attempt} model={model} "
│   │                       f"thinking={'yes' if thinking_payload else 'no'} "
│   │                       f"temperature={call_kwargs.get('temperature')} max_tokens={call_kwargs.get('max_tokens')}"
│   │                   )
│   │                   resp = self.client.messages.create(**call_kwargs)
│   │                   text = self._extract_text(resp)
│   │                   return {"text": text, "_raw": resp}
│   │               except Exception as e:
│   │                   last_err = e
│   │                   self._log(f"error: {type(e).__name__}: {e}")
│   │                   if attempt == self.max_retries:
│   │                       break
│   │                   time.sleep(min(10.0, 0.6 * (2 ** (attempt - 1))))
│   │   
│   │           raise RuntimeError(f"Anthropic call failed after {self.max_retries} attempts: {last_err}")
│   │   --- File Content End ---

│   ├── replicate_client.py
│   │   --- File Content Start ---
│   │   # # llm/replicate_client.py
│   │   # from __future__ import annotations
│   │   
│   │   # import os
│   │   # import json
│   │   # import time
│   │   # import random
│   │   # from typing import Any, Dict, List, Optional, Generator
│   │   
│   │   # from dotenv import load_dotenv
│   │   # import replicate
│   │   
│   │   # # transient network exceptions
│   │   # import httpx
│   │   # import httpcore
│   │   
│   │   # # --- your shared util (unchanged import path) ---
│   │   # from llm.json_utils import best_json
│   │   
│   │   
│   │   # # -------------------------- small helpers --------------------------
│   │   
│   │   # def _minify_schema(schema: Dict[str, Any]) -> str:
│   │   #     try:
│   │   #         return json.dumps(schema, separators=(",", ":"), ensure_ascii=False)
│   │   #     except Exception:
│   │   #         return "{}"
│   │   
│   │   # def _collapse_messages(messages: List[Dict[str, str]]) -> str:
│   │   #     parts = []
│   │   #     for m in messages:
│   │   #         role = (m.get("role") or "user").upper()
│   │   #         content = (m.get("content") or "").strip()
│   │   #         parts.append(f"{role}: {content}")
│   │   #     parts.append("ASSISTANT:")
│   │   #     return "\n\n".join(parts)
│   │   
│   │   # def _strip_fences(text: str) -> str:
│   │   #     t = (text or "").strip()
│   │   #     if t.startswith("```"):
│   │   #         nl = t.find("\n")
│   │   #         if nl != -1:
│   │   #             t = t[nl + 1:].strip()
│   │   #         if t.endswith("```"):
│   │   #             t = t[:-3].strip()
│   │   #     return t
│   │   
│   │   # def _parse_json_best_effort(text: str) -> Dict[str, Any]:
│   │   #     obj = best_json(text)
│   │   #     return obj if isinstance(obj, dict) else {}
│   │   
│   │   # def _clip01(x: Any, default: float = 0.9) -> float:
│   │   #     try:
│   │   #         v = float(x)
│   │   #     except Exception:
│   │   #         return default
│   │   #     if v < 0.0: return 0.0
│   │   #     if v > 1.0: return 1.0
│   │   #     return v
│   │   
│   │   # def _coerce_elicit(obj: Dict[str, Any], *, calibrated: bool) -> Dict[str, Any]:
│   │   #     facts = obj.get("facts")
│   │   #     if not isinstance(facts, list):
│   │   #         return {"facts": []}
│   │   #     out = []
│   │   #     for it in facts:
│   │   #         if not isinstance(it, dict):
│   │   #             continue
│   │   #         s = it.get("subject"); p = it.get("predicate"); o = it.get("object")
│   │   #         if not (isinstance(s, str) and isinstance(p, str) and (isinstance(o, str) or isinstance(o, (int, float, bool)))):
│   │   #             continue
│   │   #         if not isinstance(o, str):
│   │   #             o = str(o)
│   │   #         if calibrated:
│   │   #             conf = _clip01(it.get("confidence"), 0.9)
│   │   #             out.append({"subject": s, "predicate": p, "object": o, "confidence": conf})
│   │   #         else:
│   │   #             out.append({"subject": s, "predicate": p, "object": o})
│   │   #     return {"facts": out}
│   │   
│   │   # def _coerce_ner(obj: Dict[str, Any], *, calibrated: bool) -> Dict[str, Any]:
│   │   #     phs = obj.get("phrases")
│   │   #     if not isinstance(phs, list):
│   │   #         return {"phrases": []}
│   │   #     out = []
│   │   #     for it in phs:
│   │   #         if not isinstance(it, dict):
│   │   #             continue
│   │   #         phrase = it.get("phrase"); is_ne = bool(it.get("is_ne"))
│   │   #         if not isinstance(phrase, str):
│   │   #             continue
│   │   #         if calibrated:
│   │   #             conf = _clip01(it.get("confidence"), 0.9)
│   │   #             out.append({"phrase": phrase, "is_ne": is_ne, "confidence": conf})
│   │   #         else:
│   │   #             out.append({"phrase": phrase, "is_ne": is_ne})
│   │   #     return {"phrases": out}
│   │   
│   │   # def _salvage_block(text: str, key: Optional[str]) -> Dict[str, Any]:
│   │   #     """
│   │   #     Try best_json first; if it returns an array and we expect a key (like 'facts'),
│   │   #     wrap it; else return {}.
│   │   #     (NOTE: parameter is 'key' to match calls; we also accept legacy 'expect_key' via wrapper below.)
│   │   #     """
│   │   #     obj = best_json(text)
│   │   #     if isinstance(obj, dict):
│   │   #         # already object — either conforms, or still usable downstream
│   │   #         return obj
│   │   #     if isinstance(obj, list) and key:
│   │   #         return {key: obj}
│   │   #     return {}
│   │   
│   │   # # Backward-compat wrapper in case other code calls with expect_key=
│   │   # def _salvage_block_expect_key(text: str, expect_key: Optional[str]) -> Dict[str, Any]:
│   │   #     return _salvage_block(text, expect_key)
│   │   
│   │   # # -------------------------- client --------------------------
│   │   
│   │   # class ReplicateLLM:
│   │   #     """
│   │   #     Replicate wrapper with model-specific prompt shaping and robust JSON salvage.
│   │   #     Also implements __call__(messages, json_schema=...) to match other clients.
│   │   #     Includes jittered exponential backoff for transient HTTP faults.
│   │   #     """
│   │   
│   │   #     def __init__(self, model: str, *, api_token: Optional[str] = None, default_extra: Optional[Dict[str, Any]] = None):
│   │   #         load_dotenv()
│   │   #         self.model = model
│   │   #         token = api_token or os.getenv("REPLICATE_API_TOKEN")
│   │   #         if not token:
│   │   #             raise RuntimeError("Missing REPLICATE_API_TOKEN in environment (or pass api_token=...).")
│   │   #         self._client = replicate.Client(api_token=token)
│   │   #         self._debug = os.getenv("REPLICATE_DEBUG", "") == "1"
│   │   #         self._default_extra = default_extra or {}
│   │   
│   │   #     # Let the object be called like other clients:
│   │   #     def __call__(self, messages: List[Dict[str, str]], *, json_schema: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any] | str:
│   │   #         return self.generate(messages, json_schema=json_schema, **kwargs)
│   │   
│   │   #     # --------- builders ---------
│   │   
│   │   #     def _inputs_common(
│   │   #         self,
│   │   #         *,
│   │   #         temperature: Optional[float],
│   │   #         top_p: Optional[float],
│   │   #         top_k: Optional[int],
│   │   #         max_tokens: Optional[int],
│   │   #         seed: Optional[int],
│   │   #         extra: Dict[str, Any],
│   │   #     ) -> Dict[str, Any]:
│   │   #         # merge defaults + per-call extras
│   │   #         merged_extra = {**(self._default_extra or {}), **(extra or {})}
│   │   
│   │   #         inp: Dict[str, Any] = {}
│   │   #         if temperature is not None: inp["temperature"] = temperature
│   │   #         if top_p is not None: inp["top_p"] = top_p
│   │   #         if top_k is not None: inp["top_k"] = top_k
│   │   #         if max_tokens is not None:
│   │   #             inp["max_tokens"] = max_tokens
│   │   #             inp["max_output_tokens"] = max_tokens
│   │   #         if seed is not None: inp["seed"] = seed
│   │   
│   │   #         # Replicate quirk: some runners expect scalar strings for stop / stop_sequences
│   │   #         if "stop_sequences" in merged_extra and isinstance(merged_extra["stop_sequences"], list):
│   │   #             merged_extra = {**merged_extra, "stop_sequences": merged_extra["stop_sequences"][0] if merged_extra["stop_sequences"] else ""}
│   │   #         if "stop" in merged_extra and isinstance(merged_extra["stop"], list):
│   │   #             merged_extra = {**merged_extra, "stop": merged_extra["stop"][0] if merged_extra["stop"] else ""}
│   │   
│   │   #         for k, v in (merged_extra or {}).items():
│   │   #             if v is not None:
│   │   #                 inp[k] = v
│   │   #         return inp
│   │   
│   │   #     def _build_for_gemini(self, messages, json_schema, knobs) -> Dict[str, Any]:
│   │   #         schema_min = _minify_schema(json_schema)
│   │   #         system_prompt = (
│   │   #             "Return ONLY a single valid JSON object that matches this JSON Schema exactly. "
│   │   #             "No prose, no markdown, no code fences.\n"
│   │   #             f"SCHEMA: {schema_min}\n"
│   │   #             "If you truly don't know, return an empty but valid object per schema."
│   │   #         )
│   │   #         fewshot = (
│   │   #             "EXAMPLE:\n"
│   │   #             'USER: Subject: Ping\n'
│   │   #             'ASSISTANT: {"facts":[{"subject":"Ping","predicate":"instanceOf","object":"entity","confidence":1.0}]}\n\n'
│   │   #         )
│   │   #         prompt = fewshot + _collapse_messages(messages)
│   │   #         knobs.setdefault("temperature", 0.2)
│   │   #         knobs.setdefault("top_p", 0.9)
│   │   #         return {"prompt": prompt, "system_prompt": system_prompt, **knobs}
│   │   
│   │   #     def _build_for_grok_messages(self, messages, json_schema, knobs) -> Dict[str, Any]:
│   │   #         schema_min = _minify_schema(json_schema)
│   │   #         sys_msg = {
│   │   #             "role": "system",
│   │   #             "content": (
│   │   #                 "You are a JSON function. Return ONLY one JSON object validating this schema. "
│   │   #                 "No prose/markdown/code fences. If unsure, return an empty—but valid—object.\n"
│   │   #                 f"SCHEMA: {schema_min}"
│   │   #             ),
│   │   #         }
│   │   #         usr_msg = {"role": "user", "content": _collapse_messages(messages)}
│   │   #         inputs = {"messages": [sys_msg, usr_msg]}
│   │   #         for k in ("temperature", "top_p", "top_k", "max_tokens", "max_output_tokens", "seed"):
│   │   #             if k in knobs:
│   │   #                 inputs[k] = knobs[k]
│   │   #         return inputs
│   │   
│   │   #     def _build_for_qwen_prompt(self, messages, json_schema, knobs) -> Dict[str, Any]:
│   │   #         schema_min = _minify_schema(json_schema)
│   │   #         fewshot = (
│   │   #             "You must output ONE JSON object that VALIDATES this JSON Schema.\n"
│   │   #             "NO prose, NO markdown, NO code fences.\n"
│   │   #             f"SCHEMA: {schema_min}\n\n"
│   │   #             "EXAMPLE:\n"
│   │   #             'USER: Subject: Ping\n'
│   │   #             'ASSISTANT: {"facts":[{"subject":"Ping","predicate":"instanceOf","object":"entity","confidence":0.99}]}\n\n'
│   │   #         )
│   │   #         task = _collapse_messages(messages)
│   │   #         contract = (
│   │   #             "If you know the subject, produce 12–40 concise triples (no duplicates). "
│   │   #             'Always include at least one triple with predicate "instanceOf". '
│   │   #             'If uncertain overall, return {"facts":[]}.'
│   │   #         )
│   │   #         prompt = f"{fewshot}{task}\n\n{contract}"
│   │   #         knobs.setdefault("temperature", 0.3)
│   │   #         knobs.setdefault("top_p", 0.9)
│   │   #         knobs.setdefault("max_tokens", knobs.get("max_output_tokens", 1536))
│   │   #         return {"prompt": prompt, **knobs}
│   │   
│   │   #     def _build_inputs(self, messages, json_schema, knobs) -> Dict[str, Any]:
│   │   #         is_gemini = self.model.startswith("google/gemini")
│   │   #         is_grok = self.model.startswith("xai/grok-4") or "grok-4" in self.model
│   │   #         is_qwen = self.model.startswith("qwen/")
│   │   
│   │   #         if json_schema:
│   │   #             if is_gemini:
│   │   #                 return self._build_for_gemini(messages, json_schema, knobs)
│   │   #             if is_grok:
│   │   #                 return self._build_for_grok_messages(messages, json_schema, knobs)
│   │   #             if is_qwen:
│   │   #                 return self._build_for_qwen_prompt(messages, json_schema, knobs)
│   │   #             schema_min = _minify_schema(json_schema)
│   │   #             system_prompt = (
│   │   #                 "Return ONLY a single valid JSON object matching this schema. "
│   │   #                 "No prose, no markdown, no code fences.\n"
│   │   #                 f"SCHEMA: {schema_min}"
│   │   #             )
│   │   #             prompt = _collapse_messages(messages)
│   │   #             return {"prompt": prompt, "system_prompt": system_prompt, **knobs}
│   │   #         return {"prompt": _collapse_messages(messages), **knobs}
│   │   
│   │   #     # --------- internal resilient wrappers ---------
│   │   
│   │   #     def _blocking_once(self, inputs: Dict[str, Any]) -> str:
│   │   #         transient = (
│   │   #             httpx.TimeoutException,
│   │   #             httpx.ConnectError,
│   │   #             httpx.ReadError,
│   │   #             httpx.RemoteProtocolError,
│   │   #             httpcore.RemoteProtocolError,
│   │   #             httpcore.WriteError,
│   │   #             httpcore.ReadTimeout,
│   │   #             httpcore.ConnectTimeout,
│   │   #         )
│   │   #         delay = 0.8
│   │   #         max_tries = 6
│   │   #         last_err: Optional[BaseException] = None
│   │   #         for attempt in range(1, max_tries + 1):
│   │   #             try:
│   │   #                 pred = self._client.predictions.create(model=self.model, input=inputs)
│   │   #                 pred.wait()
│   │   #                 out = pred.output
│   │   #                 return "".join(out) if isinstance(out, list) else (out or "")
│   │   #             except transient as e:
│   │   #                 last_err = e
│   │   #                 if self._debug:
│   │   #                     print(f"[replicate][retry {attempt}/{max_tries}] {type(e).__name__}: {e}", flush=True)
│   │   #                 if attempt == max_tries:
│   │   #                     raise
│   │   #                 time.sleep(delay + random.random() * 0.3)
│   │   #                 delay = min(delay * 1.8, 10.0)
│   │   #             except Exception:
│   │   #                 raise
│   │   #         raise last_err or RuntimeError("replicate _blocking_once failed without exception")
│   │   
│   │   #     def _stream_once(self, inputs: Dict[str, Any]) -> str:
│   │   #         transient = (
│   │   #             httpx.TimeoutException,
│   │   #             httpx.ConnectError,
│   │   #             httpx.ReadError,
│   │   #             httpx.RemoteProtocolError,
│   │   #             httpcore.RemoteProtocolError,
│   │   #             httpcore.WriteError,
│   │   #             httpcore.ReadTimeout,
│   │   #             httpcore.ConnectTimeout,
│   │   #         )
│   │   #         delay = 0.8
│   │   #         max_tries = 6
│   │   #         last_err: Optional[BaseException] = None
│   │   #         for attempt in range(1, max_tries + 1):
│   │   #             try:
│   │   #                 chunks: List[str] = []
│   │   #                 for event in replicate.stream(self.model, input=inputs):
│   │   #                     chunks.append(str(event))
│   │   #                 return "".join(chunks)
│   │   #             except transient as e:
│   │   #                 last_err = e
│   │   #                 if self._debug:
│   │   #                     print(f"[replicate][stream retry {attempt}/{max_tries}] {type(e).__name__}: {e}", flush=True)
│   │   #                 if attempt == max_tries:
│   │   #                     raise
│   │   #                 time.sleep(delay + random.random() * 0.3)
│   │   #                 delay = min(delay * 1.8, 10.0)
│   │   #             except Exception:
│   │   #                 raise
│   │   #         raise last_err or RuntimeError("replicate _stream_once failed without exception")
│   │   
│   │   #     # --------- schema-based coercion ---------
│   │   
│   │   #     def _coerce_by_schema(self, obj: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
│   │   #         props = (schema.get("properties") or {})
│   │   #         if "facts" in props:
│   │   #             calibrated = "confidence" in (props["facts"]["items"]["properties"] or {})
│   │   #             return _coerce_elicit(obj, calibrated=calibrated)
│   │   #         if "phrases" in props:
│   │   #             calibrated = "confidence" in (props["phrases"]["items"]["properties"] or {})
│   │   #             return _coerce_ner(obj, calibrated=calibrated)
│   │   #         return obj if isinstance(obj, dict) else {}
│   │   
│   │   #     # --------- public blocking API ---------
│   │   
│   │   #     def ping(self) -> Dict[str, Any]:
│   │   #         inp = {"prompt": 'Return ONLY this exact JSON: {"message":"PONG"}', "max_tokens": 32, "temperature": 0}
│   │   #         txt = self._blocking_once(inp)
│   │   #         obj = _parse_json_best_effort(txt)
│   │   #         return obj if obj else {"message": "PONG"}
│   │   
│   │   #     def generate(
│   │   #         self,
│   │   #         messages: List[Dict[str, str]],
│   │   #         *,
│   │   #         json_schema: Optional[Dict[str, Any]] = None,
│   │   #         temperature: Optional[float] = None,
│   │   #         top_p: Optional[float] = None,
│   │   #         top_k: Optional[int] = None,
│   │   #         max_tokens: Optional[int] = None,
│   │   #         seed: Optional[int] = None,
│   │   #         extra: Optional[Dict[str, Any]] = None,
│   │   #     ) -> Dict[str, Any]:
│   │   #         knobs = self._inputs_common(
│   │   #             temperature=temperature, top_p=top_p, top_k=top_k,
│   │   #             max_tokens=max_tokens, seed=seed, extra=extra or {},
│   │   #         )
│   │   #         inputs = self._build_inputs(messages, json_schema, knobs)
│   │   
│   │   #         if not json_schema:
│   │   #             text = self._blocking_once(inputs)
│   │   #             if self._debug:
│   │   #                 print("\n[replicate][raw output]\n" + text[:4000] + ("\n" if len(text) else ""), flush=True)
│   │   #             return {"text": text, "_raw": text}
│   │   
│   │   #         props = (json_schema.get("properties") or {})
│   │   #         expect = "facts" if "facts" in props else ("phrases" if "phrases" in props else None)
│   │   
│   │   #         is_grok = self.model.startswith("xai/grok-4") or "grok-4" in self.model
│   │   
│   │   #         if is_grok:
│   │   #             text = self._stream_once(inputs)
│   │   #             if self._debug:
│   │   #                 print("\n[replicate][raw stream (grok)]\n" + text[:4000] + ("\n" if len(text) else ""), flush=True)
│   │   #             # accept both 'key=' and legacy 'expect_key=' styles
│   │   #             parsed = _salvage_block(text, key=expect)
│   │   #             result = self._coerce_by_schema(parsed, json_schema)
│   │   #             result["_raw"] = text
│   │   #             return result
│   │   
│   │   #         text = self._blocking_once(inputs)
│   │   #         if self._debug:
│   │   #             print("\n[replicate][raw output]\n" + text[:4000] + ("\n" if len(text) else ""), flush=True)
│   │   
│   │   #         # accept both names to avoid mismatches from older call sites
│   │   #         parsed = _salvage_block(text, key=expect)
│   │   #         if not parsed:
│   │   #             parsed = _salvage_block_expect_key(text, expect_key=expect)
│   │   
│   │   #         if parsed:
│   │   #             result = self._coerce_by_schema(parsed, json_schema)
│   │   #             result["_raw"] = text
│   │   #             return result
│   │   
│   │   #         # final fallback: just coerce empty object so caller gets schema shape
│   │   #         result = self._coerce_by_schema({}, json_schema)
│   │   #         result["_raw"] = text
│   │   #         return result
│   │   
│   │   #     # --------- streaming API ---------
│   │   
│   │   #     def stream_text(
│   │   #         self,
│   │   #         messages: List[Dict[str, str]],
│   │   #         *,
│   │   #         temperature: Optional[float] = None,
│   │   #         top_p: Optional[float] = None,
│   │   #         top_k: Optional[int] = None,
│   │   #         max_tokens: Optional[int] = None,
│   │   #         seed: Optional[int] = None,
│   │   #         extra: Optional[Dict[str, Any]] = None,
│   │   #     ) -> Generator[str, None, None]:
│   │   #         knobs = self._inputs_common(
│   │   #             temperature=temperature, top_p=top_p, top_k=top_k,
│   │   #             max_tokens=max_tokens, seed=seed, extra=extra or {},
│   │   #         )
│   │   #         inputs = self._build_inputs(messages, json_schema=None, knobs=knobs)
│   │   #         # resilient streaming
│   │   #         transient = (
│   │   #             httpx.TimeoutException,
│   │   #             httpx.ConnectError,
│   │   #             httpx.ReadError,
│   │   #             httpx.RemoteProtocolError,
│   │   #             httpcore.RemoteProtocolError,
│   │   #             httpcore.WriteError,
│   │   #             httpcore.ReadTimeout,
│   │   #             httpcore.ConnectTimeout,
│   │   #         )
│   │   #         delay = 0.8
│   │   #         max_tries = 6
│   │   #         attempt = 1
│   │   #         while True:
│   │   #             try:
│   │   #                 for event in replicate.stream(self.model, input=inputs):
│   │   #                     yield str(event)
│   │   #                 break
│   │   #             except transient as e:
│   │   #                 if self._debug:
│   │   #                     print(f"[replicate][stream_text retry {attempt}/{max_tries}] {type(e).__name__}: {e}", flush=True)
│   │   #                 if attempt >= max_tries:
│   │   #                     raise
│   │   #                 time.sleep(delay + random.random() * 0.3)
│   │   #                 delay = min(delay * 1.8, 10.0)
│   │   #                 attempt += 1
│   │   
│   │   #     def stream_json(
│   │   #         self,
│   │   #         messages: List[Dict[str, str]],
│   │   #         *,
│   │   #         json_schema: Dict[str, Any],
│   │   #         temperature: Optional[float] = None,
│   │   #         top_p: Optional[float] = None,
│   │   #         top_k: Optional[int] = None,
│   │   #         max_tokens: Optional[int] = None,
│   │   #         seed: Optional[int] = None,
│   │   #         extra: Optional[Dict[str, Any]] = None,
│   │   #     ) -> Generator[Dict[str, Any], None, None]:
│   │   #         buffer: List[str] = []
│   │   #         knobs = self._inputs_common(
│   │   #             temperature=temperature, top_p=top_p, top_k=top_k,
│   │   #             max_tokens=max_tokens, seed=seed, extra=extra or {},
│   │   #         )
│   │   #         inputs = self._build_inputs(messages, json_schema=json_schema, knobs=knobs)
│   │   #         # resilient stream collect
│   │   #         text = ""
│   │   #         transient = (
│   │   #             httpx.TimeoutException,
│   │   #             httpx.ConnectError,
│   │   #             httpx.ReadError,
│   │   #             httpx.RemoteProtocolError,
│   │   #             httpcore.RemoteProtocolError,
│   │   #             httpcore.WriteError,
│   │   #             httpcore.ReadTimeout,
│   │   #             httpcore.ConnectTimeout,
│   │   #         )
│   │   #         delay = 0.8
│   │   #         max_tries = 6
│   │   #         for attempt in range(1, max_tries + 1):
│   │   #             try:
│   │   #                 buffer.clear()
│   │   #                 for event in replicate.stream(self.model, input=inputs):
│   │   #                     buffer.append(str(event))
│   │   #                 text = "".join(buffer)
│   │   #                 break
│   │   #             except transient as e:
│   │   #                 if self._debug:
│   │   #                     print(f"[replicate][stream_json retry {attempt}/{max_tries}] {type(e).__name__}: {e}", flush=True)
│   │   #                 if attempt == max_tries:
│   │   #                     raise
│   │   #                 time.sleep(delay + random.random() * 0.3)
│   │   #                 delay = min(delay * 1.8, 10.0)
│   │   
│   │   #         if self._debug:
│   │   #             print("\n[replicate][raw stream combined]\n" + text[:4000] + ("\n" if len(text) else ""), flush=True)
│   │   
│   │   #         props = (json_schema.get("properties") or {})
│   │   #         expect = "facts" if "facts" in props else ("phrases" if "phrases" in props else None)
│   │   #         parsed = _salvage_block(text, key=expect) or _salvage_block_expect_key(text, expect_key=expect)
│   │   #         result = self._coerce_by_schema(parsed, json_schema)
│   │   #         result["_raw"] = text
│   │   #         yield result
│   │   
│   │   # llm/replicate_client.py
│   │   from __future__ import annotations
│   │   
│   │   import os
│   │   import json
│   │   import time
│   │   import random
│   │   from typing import Any, Dict, List, Optional, Generator
│   │   
│   │   from dotenv import load_dotenv
│   │   import replicate
│   │   
│   │   # transient network exceptions
│   │   import httpx
│   │   import httpcore
│   │   
│   │   # --- shared util ---
│   │   from llm.json_utils import best_json
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
│   │   def _collapse_messages(messages: List[Dict[str, str]]) -> str:
│   │       """
│   │       Generic chat collapse used by most runners that accept a 'prompt' string,
│   │       while still preserving roles for readability.
│   │       """
│   │       parts = []
│   │       for m in messages:
│   │           role = (m.get("role") or "user").upper()
│   │           content = (m.get("content") or "").strip()
│   │           parts.append(f"{role}: {content}")
│   │       parts.append("ASSISTANT:")
│   │       return "\n\n".join(parts)
│   │   
│   │   def _collapse_single_prompt(messages: List[Dict[str, str]]) -> str:
│   │       """
│   │       Collapse chat messages into ONE prompt for single-prompt-only models
│   │       (e.g., openai/gpt-oss-*). We keep explicit headers and end with 'Assistant:'.
│   │       """
│   │       sys_parts: List[str] = []
│   │       convo_parts: List[str] = []
│   │   
│   │       for m in messages or []:
│   │           role = (m.get("role") or "user").strip().lower()
│   │           content = (m.get("content") or "").strip()
│   │           if not content:
│   │               continue
│   │           if role == "system":
│   │               sys_parts.append(content)
│   │           elif role == "assistant":
│   │               convo_parts.append(f"Assistant: {content}")
│   │           else:
│   │               # default to user
│   │               convo_parts.append(f"User: {content}")
│   │   
│   │       out: List[str] = []
│   │       if sys_parts:
│   │           out.append("\n\n".join(f"System: {p}" for p in sys_parts))
│   │       if convo_parts:
│   │           out.append("\n\n".join(convo_parts))
│   │       out.append("Assistant:")
│   │       return "\n\n".join(out).strip()
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
│   │   def _parse_json_best_effort(text: str) -> Dict[str, Any]:
│   │       obj = best_json(text)
│   │       return obj if isinstance(obj, dict) else {}
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
│   │   def _coerce_elicit(obj: Dict[str, Any], *, calibrated: bool) -> Dict[str, Any]:
│   │       facts = obj.get("facts")
│   │       if not isinstance(facts, list):
│   │           return {"facts": []}
│   │       out = []
│   │       for it in facts:
│   │           if not isinstance(it, dict):
│   │               continue
│   │           s = it.get("subject"); p = it.get("predicate"); o = it.get("object")
│   │           if not (isinstance(s, str) and isinstance(p, str) and (isinstance(o, str) or isinstance(o, (int, float, bool)))):
│   │               continue
│   │           if not isinstance(o, str):
│   │               o = str(o)
│   │           if calibrated:
│   │               conf = _clip01(it.get("confidence"), 0.9)
│   │               out.append({"subject": s, "predicate": p, "object": o, "confidence": conf})
│   │           else:
│   │               out.append({"subject": s, "predicate": p, "object": o})
│   │       return {"facts": out}
│   │   
│   │   def _coerce_ner(obj: Dict[str, Any], *, calibrated: bool) -> Dict[str, Any]:
│   │       phs = obj.get("phrases")
│   │       if not isinstance(phs, list):
│   │           return {"phrases": []}
│   │       out = []
│   │       for it in phs:
│   │           if not isinstance(it, dict):
│   │               continue
│   │           phrase = it.get("phrase"); is_ne = bool(it.get("is_ne"))
│   │           if not isinstance(phrase, str):
│   │               continue
│   │           if calibrated:
│   │               conf = _clip01(it.get("confidence"), 0.9)
│   │               out.append({"phrase": phrase, "is_ne": is_ne, "confidence": conf})
│   │           else:
│   │               out.append({"phrase": phrase, "is_ne": is_ne})
│   │       return {"phrases": out}
│   │   
│   │   def _salvage_block(text: str, key: Optional[str]) -> Dict[str, Any]:
│   │       """
│   │       Try best_json first; if it returns an array and we expect a key (like 'facts'),
│   │       wrap it; else return {}.
│   │       """
│   │       obj = best_json(text)
│   │       if isinstance(obj, dict):
│   │           return obj
│   │       if isinstance(obj, list) and key:
│   │           return {key: obj}
│   │       return {}
│   │   
│   │   # Back-compat shim for older call sites that used expect_key=
│   │   def _salvage_block_expect_key(text: str, expect_key: Optional[str]) -> Dict[str, Any]:
│   │       return _salvage_block(text, expect_key)
│   │   
│   │   def _is_single_prompt_only(model_name: str) -> bool:
│   │       """
│   │       True for Replicate models that take ONLY a single 'prompt' (no messages/system).
│   │       We scope this STRICTLY to openai/gpt-oss-* per request.
│   │       """
│   │       return (model_name or "").lower().startswith("openai/gpt-oss-")
│   │   
│   │   
│   │   # -------------------------- client --------------------------
│   │   
│   │   class ReplicateLLM:
│   │       """
│   │       Replicate wrapper with model-specific prompt shaping and robust JSON salvage.
│   │       Implements __call__(messages, json_schema=...) to match other clients.
│   │       Includes jittered exponential backoff for transient HTTP faults.
│   │       """
│   │   
│   │       def __init__(self, model: str, *, api_token: Optional[str] = None, default_extra: Optional[Dict[str, Any]] = None):
│   │           load_dotenv()
│   │           self.model = model
│   │           token = api_token or os.getenv("REPLICATE_API_TOKEN")
│   │           if not token:
│   │               raise RuntimeError("Missing REPLICATE_API_TOKEN in environment (or pass api_token=...).")
│   │           self._client = replicate.Client(api_token=token)
│   │           self._debug = os.getenv("REPLICATE_DEBUG", "") == "1"
│   │           self._default_extra = default_extra or {}
│   │   
│   │       # Allow call-style usage like other LLM clients
│   │       def __call__(self, messages: List[Dict[str, str]], *, json_schema: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any] | str:
│   │           return self.generate(messages, json_schema=json_schema, **kwargs)
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
│   │           # merge defaults + per-call extras
│   │           merged_extra = {**(self._default_extra or {}), **(extra or {})}
│   │   
│   │           inp: Dict[str, Any] = {}
│   │           if temperature is not None: inp["temperature"] = temperature
│   │           if top_p is not None: inp["top_p"] = top_p
│   │           if top_k is not None: inp["top_k"] = top_k
│   │           if max_tokens is not None:
│   │               inp["max_tokens"] = max_tokens
│   │               inp["max_output_tokens"] = max_tokens
│   │           if seed is not None: inp["seed"] = seed
│   │   
│   │           # Replicate quirk: some runners expect scalar strings for stop / stop_sequences
│   │           if "stop_sequences" in merged_extra and isinstance(merged_extra["stop_sequences"], list):
│   │               merged_extra = {**merged_extra, "stop_sequences": merged_extra["stop_sequences"][0] if merged_extra["stop_sequences"] else ""}
│   │           if "stop" in merged_extra and isinstance(merged_extra["stop"], list):
│   │               merged_extra = {**merged_extra, "stop": merged_extra["stop"][0] if merged_extra["stop"] else ""}
│   │   
│   │           for k, v in (merged_extra or {}).items():
│   │               if v is not None:
│   │                   inp[k] = v
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
│   │           """
│   │           Build Replicate payload, with a STRICT special-case only for openai/gpt-oss-* models
│   │           that accept a single 'prompt'. All other models behave as before.
│   │           """
│   │           is_gemini = self.model.startswith("google/gemini")
│   │           is_grok = self.model.startswith("xai/grok-4") or "grok-4" in self.model
│   │           is_qwen = self.model.startswith("qwen/")
│   │           single_prompt_only = _is_single_prompt_only(self.model)
│   │   
│   │           if json_schema:
│   │               schema_min = _minify_schema(json_schema)
│   │               schema_instr = (
│   │                   "You must return ONLY one valid JSON object that matches the JSON Schema below.\n"
│   │                   "No prose, no markdown, no code fences. If unsure, return an empty but valid object.\n"
│   │                   f"SCHEMA: {schema_min}\n\n"
│   │               )
│   │   
│   │               if single_prompt_only:
│   │                   combined = _collapse_single_prompt(messages)
│   │                   prompt = schema_instr + combined
│   │                   return {"prompt": prompt, **knobs}
│   │   
│   │               if is_gemini:
│   │                   fewshot = (
│   │                       "EXAMPLE:\n"
│   │                       'USER: Subject: Ping\n'
│   │                       'ASSISTANT: {"facts":[{"subject":"Ping","predicate":"instanceOf","object":"entity","confidence":1.0}]}\n\n'
│   │                   )
│   │                   prompt = fewshot + _collapse_messages(messages)
│   │                   system_prompt = (
│   │                       "Return ONLY a single valid JSON object that matches this JSON Schema exactly. "
│   │                       "No prose, no markdown, no code fences.\n"
│   │                       f"SCHEMA: {schema_min}\n"
│   │                       "If you truly don't know, return an empty but valid object per schema."
│   │                   )
│   │                   knobs.setdefault("temperature", 0.2)
│   │                   knobs.setdefault("top_p", 0.9)
│   │                   return {"prompt": prompt, "system_prompt": system_prompt, **knobs}
│   │   
│   │               if is_grok:
│   │                   return self._build_for_grok_messages(messages, json_schema, knobs)
│   │   
│   │               if is_qwen:
│   │                   return self._build_for_qwen_prompt(messages, json_schema, knobs)
│   │   
│   │               # generic (unchanged)
│   │               system_prompt = (
│   │                   "Return ONLY a single valid JSON object matching this schema. "
│   │                   "No prose, no markdown, no code fences.\n"
│   │                   f"SCHEMA: {schema_min}"
│   │               )
│   │               prompt = _collapse_messages(messages)
│   │               return {"prompt": prompt, "system_prompt": system_prompt, **knobs}
│   │   
│   │           # -------- no json_schema (plain text) --------
│   │           if single_prompt_only:
│   │               return {"prompt": _collapse_single_prompt(messages), **knobs}
│   │   
│   │           if is_gemini:
│   │               return {"prompt": _collapse_messages(messages), "system_prompt": "", **knobs}
│   │   
│   │           if is_grok:
│   │               sys_msg = {"role": "system", "content": "You are a helpful assistant."}
│   │               usr_msg = {"role": "user", "content": _collapse_messages(messages)}
│   │               inputs = {"messages": [sys_msg, usr_msg]}
│   │               for k in ("temperature", "top_p", "top_k", "max_tokens", "max_output_tokens", "seed"):
│   │                   if k in knobs:
│   │                       inputs[k] = knobs[k]
│   │               return inputs
│   │   
│   │           if is_qwen:
│   │               return {"prompt": _collapse_messages(messages), **knobs}
│   │   
│   │           # default
│   │           return {"prompt": _collapse_messages(messages), **knobs}
│   │   
│   │       # --------- internal resilient wrappers ---------
│   │   
│   │       def _blocking_once(self, inputs: Dict[str, Any]) -> str:
│   │           transient = (
│   │               httpx.TimeoutException,
│   │               httpx.ConnectError,
│   │               httpx.ReadError,
│   │               httpx.RemoteProtocolError,
│   │               httpcore.RemoteProtocolError,
│   │               httpcore.WriteError,
│   │               httpcore.ReadTimeout,
│   │               httpcore.ConnectTimeout,
│   │           )
│   │           delay = 0.8
│   │           max_tries = 6
│   │           last_err: Optional[BaseException] = None
│   │           for attempt in range(1, max_tries + 1):
│   │               try:
│   │                   pred = self._client.predictions.create(model=self.model, input=inputs)
│   │                   pred.wait()
│   │                   out = pred.output
│   │                   return "".join(out) if isinstance(out, list) else (out or "")
│   │               except transient as e:
│   │                   last_err = e
│   │                   if self._debug:
│   │                       print(f"[replicate][retry {attempt}/{max_tries}] {type(e).__name__}: {e}", flush=True)
│   │                   if attempt == max_tries:
│   │                       raise
│   │                   time.sleep(delay + random.random() * 0.3)
│   │                   delay = min(delay * 1.8, 10.0)
│   │               except Exception:
│   │                   raise
│   │           raise last_err or RuntimeError("replicate _blocking_once failed without exception")
│   │   
│   │       def _stream_once(self, inputs: Dict[str, Any]) -> str:
│   │           transient = (
│   │               httpx.TimeoutException,
│   │               httpx.ConnectError,
│   │               httpx.ReadError,
│   │               httpx.RemoteProtocolError,
│   │               httpcore.RemoteProtocolError,
│   │               httpcore.WriteError,
│   │               httpcore.ReadTimeout,
│   │               httpcore.ConnectTimeout,
│   │           )
│   │           delay = 0.8
│   │           max_tries = 6
│   │           last_err: Optional[BaseException] = None
│   │           for attempt in range(1, max_tries + 1):
│   │               try:
│   │                   chunks: List[str] = []
│   │                   for event in replicate.stream(self.model, input=inputs):
│   │                       chunks.append(str(event))
│   │                   return "".join(chunks)
│   │               except transient as e:
│   │                   last_err = e
│   │                   if self._debug:
│   │                       print(f"[replicate][stream retry {attempt}/{max_tries}] {type(e).__name__}: {e}", flush=True)
│   │                   if attempt == max_tries:
│   │                       raise
│   │                   time.sleep(delay + random.random() * 0.3)
│   │                   delay = min(delay * 1.8, 10.0)
│   │               except Exception:
│   │                   raise
│   │           raise last_err or RuntimeError("replicate _stream_once failed without exception")
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
│   │           return obj if isinstance(obj, dict) else {}
│   │   
│   │       # --------- public blocking API ---------
│   │   
│   │       def ping(self) -> Dict[str, Any]:
│   │           inp = {"prompt": 'Return ONLY this exact JSON: {"message":"PONG"}', "max_tokens": 32, "temperature": 0}
│   │           txt = self._blocking_once(inp)
│   │           obj = _parse_json_best_effort(txt)
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
│   │           max_tokens: Optional[int] = None,
│   │           seed: Optional[int] = None,
│   │           extra: Optional[Dict[str, Any]] = None,
│   │       ) -> Dict[str, Any]:
│   │           knobs = self._inputs_common(
│   │               temperature=temperature, top_p=top_p, top_k=top_k,
│   │               max_tokens=max_tokens, seed=seed, extra=extra or {},
│   │           )
│   │           inputs = self._build_inputs(messages, json_schema, knobs)
│   │   
│   │           if not json_schema:
│   │               text = self._blocking_once(inputs)
│   │               if self._debug:
│   │                   print("\n[replicate][raw output]\n" + text[:4000] + ("\n" if len(text) else ""), flush=True)
│   │               return {"text": text, "_raw": text}
│   │   
│   │           props = (json_schema.get("properties") or {})
│   │           expect = "facts" if "facts" in props else ("phrases" if "phrases" in props else None)
│   │   
│   │           is_grok = self.model.startswith("xai/grok-4") or "grok-4" in self.model
│   │   
│   │           if is_grok:
│   │               text = self._stream_once(inputs)
│   │               if self._debug:
│   │                   print("\n[replicate][raw stream (grok)]\n" + text[:4000] + ("\n" if len(text) else ""), flush=True)
│   │               parsed = _salvage_block(text, key=expect)
│   │               result = self._coerce_by_schema(parsed, json_schema)
│   │               result["_raw"] = text
│   │               return result
│   │   
│   │           text = self._blocking_once(inputs)
│   │           if self._debug:
│   │               print("\n[replicate][raw output]\n" + text[:4000] + ("\n" if len(text) else ""), flush=True)
│   │   
│   │           parsed = _salvage_block(text, key=expect) or _salvage_block_expect_key(text, expect_key=expect)
│   │   
│   │           if parsed:
│   │               result = self._coerce_by_schema(parsed, json_schema)
│   │               result["_raw"] = text
│   │               return result
│   │   
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
│   │           knobs = self._inputs_common(
│   │               temperature=temperature, top_p=top_p, top_k=top_k,
│   │               max_tokens=max_tokens, seed=seed, extra=extra or {},
│   │           )
│   │           inputs = self._build_inputs(messages, json_schema=None, knobs=knobs)
│   │           # resilient streaming
│   │           transient = (
│   │               httpx.TimeoutException,
│   │               httpx.ConnectError,
│   │               httpx.ReadError,
│   │               httpx.RemoteProtocolError,
│   │               httpcore.RemoteProtocolError,
│   │               httpcore.WriteError,
│   │               httpcore.ReadTimeout,
│   │               httpcore.ConnectTimeout,
│   │           )
│   │           delay = 0.8
│   │           max_tries = 6
│   │           attempt = 1
│   │           while True:
│   │               try:
│   │                   for event in replicate.stream(self.model, input=inputs):
│   │                       yield str(event)
│   │                   break
│   │               except transient as e:
│   │                   if self._debug:
│   │                       print(f"[replicate][stream_text retry {attempt}/{max_tries}] {type(e).__name__}: {e}", flush=True)
│   │                   if attempt >= max_tries:
│   │                       raise
│   │                   time.sleep(delay + random.random() * 0.3)
│   │                   delay = min(delay * 1.8, 10.0)
│   │                   attempt += 1
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
│   │           seed: Optional[float] = None,
│   │           extra: Optional[Dict[str, Any]] = None,
│   │       ) -> Generator[Dict[str, Any], None, None]:
│   │           buffer: List[str] = []
│   │           knobs = self._inputs_common(
│   │               temperature=temperature, top_p=top_p, top_k=top_k,
│   │               max_tokens=max_tokens, seed=seed, extra=extra or {},
│   │           )
│   │           inputs = self._build_inputs(messages, json_schema=json_schema, knobs=knobs)
│   │           # resilient stream collect
│   │           text = ""
│   │           transient = (
│   │               httpx.TimeoutException,
│   │               httpx.ConnectError,
│   │               httpx.ReadError,
│   │               httpx.RemoteProtocolError,
│   │               httpcore.RemoteProtocolError,
│   │               httpcore.WriteError,
│   │               httpcore.ReadTimeout,
│   │               httpcore.ConnectTimeout,
│   │           )
│   │           delay = 0.8
│   │           max_tries = 6
│   │           for attempt in range(1, max_tries + 1):
│   │               try:
│   │                   buffer.clear()
│   │                   for event in replicate.stream(self.model, input=inputs):
│   │                       buffer.append(str(event))
│   │                   text = "".join(buffer)
│   │                   break
│   │               except transient as e:
│   │                   if self._debug:
│   │                       print(f"[replicate][stream_json retry {attempt}/{max_tries}] {type(e).__name__}: {e}", flush=True)
│   │                   if attempt == max_tries:
│   │                       raise
│   │                   time.sleep(delay + random.random() * 0.3)
│   │                   delay = min(delay * 1.8, 10.0)
│   │   
│   │           if self._debug:
│   │               print("\n[replicate][raw stream combined]\n" + text[:4000] + ("\n" if len(text) else ""), flush=True)
│   │   
│   │           props = (json_schema.get("properties") or {})
│   │           expect = "facts" if "facts" in props else ("phrases" if "phrases" in props else None)
│   │           parsed = _salvage_block(text, key=expect) or _salvage_block_expect_key(text, expect_key=expect)
│   │           result = self._coerce_by_schema(parsed, json_schema)
│   │           result["_raw"] = text
│   │           yield result
│   │   --- File Content End ---

│   ├── __pycache__/
├── Termintationtest3Prompt2/
│   ├── deepseekTBBT/
│   │   ├── tmp/
│   ├── gpt4omini5/
│   │   ├── tmp/
│   ├── llama3-8b-instructTBBT/
│   │   ├── tmp/
│   ├── llama3-8b-instructDAX-40-Index/
│   │   ├── tmp/
│   ├── llama3-8b-instructAncientBabylon/
│   │   ├── tmp/
│   ├── llama3-8b-instructTBBT_40Conf/
│   │   ├── tmp/
│   ├── deepseekAncientCityofBabylon/
│   │   ├── tmp/
│   ├── gpt4ominiTBBT/
│   │   ├── tmp/
│   ├── gpt4ominiBabylon/
│   │   ├── tmp/
│   ├── deepseekDAX40Index/
│   │   ├── tmp/
│   ├── gpt5miniBabylon/
│   │   ├── tmp/
│   ├── deepseekAncientCityofBabylon2/
│   │   ├── tmp/
│   ├── llama3-70b-instructAncientBabylon/
│   │   ├── tmp/
│   ├── llama3-70b-instruct/
│   │   ├── tmp/
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
│   │   from typing import List, Dict, Any
│   │   
│   │   # Only these placeholders will be replaced; all other braces are left intact.
│   │   _ALLOWED_KEYS = {"subject_name", "phrases_block", "root_subject"}
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
│   │   def _safe_render(template: str, variables: Dict[str, Any] | None) -> str:
│   │       """
│   │       Replace ONLY whitelisted placeholders like {subject_name} or {phrases_block}.
│   │       Leave ALL other { ... } untouched (e.g., JSON braces, schema examples).
│   │       """
│   │       if not template:
│   │           return ""
│   │       if not variables:
│   │           return template
│   │       out = template
│   │       for k, v in variables.items():
│   │           if k in _ALLOWED_KEYS:
│   │               out = out.replace("{" + k + "}", str(v))
│   │       return out
│   │   
│   │   def load_messages_from_prompt_json(path: str | Path, **vars) -> List[Dict[str, str]]:
│   │       obj = json.loads(_resolve(path).read_text(encoding="utf-8"))
│   │       system = _safe_render(obj.get("system") or "", vars).strip()
│   │       user   = _safe_render(obj.get("user") or "", vars).strip()
│   │       return [
│   │           {"role": "system", "content": system},
│   │           {"role": "user",   "content": user},
│   │       ]
│   │   --- File Content End ---

│   ├── __pycache__/
├── Termintationtest3/
│   ├── gpt4omini5/
│   │   ├── tmp/
│   ├── Termintationtest3deepseek/
│   │   ├── tmp/
│   ├── deepseek/
│   │   ├── tmp/
│   ├── llama3-8b-instruct/
│   │   ├── tmp/
│   ├── llama3-70b-instruct/
│   │   ├── tmp/
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
│   │   ├── ICL/
│   │   ├── baseline/
│   │   ├── calibrate/
│   │   ├── dont_know/
│   ├── topic/
│   │   ├── ICL/
│   │   ├── baseline/
│   │   ├── calibrate/
│   │   ├── dont_know/
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
│   │   ├── 03/
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
│   │   ├── 93/
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
│   │   ├── fc/
│   │   ├── fd/
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
│   │   ├── 45/
│   │   ├── 1f/
│   │   ├── 73/
│   │   ├── 87/
│   │   ├── 80/
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
│   │   ├── 38/
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
│   │   ├── 97/
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
│   │   ├── a7/
│   │   ├── b8/
│   │   ├── b1/
│   │   ├── dd/
│   │   ├── dc/
│   │   ├── b6/
│   │   ├── a9/
│   │   ├── d5/
│   │   ├── d2/
│   │   ├── aa/
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
│   │   ├── 46/
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
├── replicateClientTest/
│   ├── llama3-8b-instructTBBT_40ConfFooter/
│   │   ├── tmp/
│   ├── gpt-oss-120b/
│   │   ├── tmp/
│   ├── gpt-oss-20b/
│   │   ├── tmp/
│   ├── llama3-8b-instructTBBT_40Conf/
│   │   ├── tmp/
│   ├── gpt-oss-20bBabylon/
│   │   ├── tmp/
'''

print(project_dump)
