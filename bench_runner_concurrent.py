#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import dataclasses
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

# ---------- small utils ----------

BERLIN_TZ = dt.timezone(dt.timedelta(hours=2))  # Europe/Berlin (DST not tracked here; fine for stamping)
NOW = lambda: dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def ts_for_dir() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def sanitize_slug(s: str) -> str:
    bad = '/\\?%*:|"<>'
    out = s.strip().replace(" ", "_")
    for ch in bad:
        out = out.replace(ch, "")
    return out

def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def write_json(path: str, obj: dict) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def append_bench_log(root_out: str, line: str) -> None:
    ensure_dir(root_out)
    with open(os.path.join(root_out, "bench.log"), "a", encoding="utf-8") as f:
        f.write(f"[{dt.datetime.now().isoformat()}] {line}\n")

def expand_csv_header_safely(csv_path: str, new_row: Dict[str, object]) -> None:
    """
    Append a row to CSV while allowing new columns to appear later.
    If the header needs to grow, we rewrite the file with the expanded header.
    """
    ensure_dir(str(Path(csv_path).parent))
    rows: List[Dict[str, object]] = []
    existing_header: List[str] = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            existing_header = r.fieldnames or []
            for row in r:
                rows.append(row)

    # union header
    all_keys = list(dict.fromkeys([*(existing_header or []), *list(new_row.keys())]))
    rows.append(new_row)

    # rewrite file with expanded header
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for r in rows:
            # fill missing keys with ""
            out = {k: r.get(k, "") for k in all_keys}
            w.writerow(out)

# ---------- profiles (deterministic / medium / wild) ----------

PROFILE_KNOBS = {
    # Deterministic
    "det":   {"temperature": 0.0, "top_p": 1.0,  "top_k": None, "max_tokens": 2000},
    # Medium (your day-to-day defaults)
    "medium":{"temperature": 0.7, "top_p": 0.95, "top_k": 50,   "max_tokens": 2000},
    # Wild (probe model breadth; expect more noise)
    "wild":  {"temperature": 2.0, "top_p": 1.0,  "top_k": 100,  "max_tokens": 2000},
}

# ---------- CLI ----------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Concurrent benchmark runner for GPT-KB crawler (no OpenAI Batch)."
    )
    ap.add_argument("--root-out", required=True,
                    help="Root folder for all benchmark outputs (will create subfolders).")
    ap.add_argument("--crawler", default="crawler_batch_concurrency_topic.py",
                    help="Crawler script to run (default: crawler_batch_concurrency_topic.py).")

    # grids
    ap.add_argument("--domains", default="topic,general",
                    help="Comma list: topic,general")
    ap.add_argument("--seeds", default="Game of Thrones,Lionel Messi,World War II",
                    help="Comma list of starting subjects.")
    ap.add_argument("--models", default="deepseek,granite8b,gpt4o-mini",
                    help="Comma list of model keys (must exist in settings.MODELS).")
    ap.add_argument("--strategies", default="baseline,calibrate,icl,dont_know",
                    help="Comma list of elicitation strategies.")
    ap.add_argument("--profiles", default="det,medium,wild",
                    help="Comma list of sampling profiles (det|medium|wild).")

    # crawler knobs (shared)
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--max-subjects", type=int, default=3)
    ap.add_argument("--max-facts-hint", type=int, default=100)
    ap.add_argument("--ner-batch-size", type=int, default=50)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--ner-strategy", default="calibrate",
                    help="NER strategy (typically 'calibrate').")

    # control
    ap.add_argument("--list", action="store_true", help="Only list planned runs then exit.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands, do not execute.")
    ap.add_argument("--verbose", action="store_true", help="Verbose planning output.")

    return ap

# ---------- plan builder ----------

def build_plan(args) -> List[Dict]:
    # normalize grids
    domains    = [s.strip() for s in args.domains.split(",") if s.strip()]
    seeds      = [s.strip() for s in args.seeds.split(",") if s.strip()]
    models     = [s.strip() for s in args.models.split(",") if s.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    profiles   = [s.strip() for s in args.profiles.split(",") if s.strip()]

    # sanity
    for p in profiles:
        if p not in PROFILE_KNOBS:
            raise SystemExit(f"Unknown profile '{p}'. Use one of: {', '.join(PROFILE_KNOBS)}")

    plan: List[Dict] = []
    for domain, seed, model, strat, prof in product(domains, seeds, models, strategies, profiles):
        knobs = PROFILE_KNOBS[prof]
        seed_slug = sanitize_slug(seed)
        out_dir = os.path.join(
            args.root_out,
            domain,
            model,
            seed_slug,
            f"{strat}__{prof}__{ts_for_dir()}",
        )
        ensure_dir(out_dir)

        # build command
        cmd: List[str] = [
            sys.executable, args.crawler,
            "--seed", seed,
            "--output-dir", out_dir,
            "--domain", domain,
            "--elicitation-strategy", strat,
            "--ner-strategy", args.ner_strategy,
            "--elicit-model-key", model,
            "--ner-model-key", model,
            "--max-depth", str(args.max_depth),
            "--max-facts-hint", str(args.max_facts_hint),
            "--max-subjects", str(args.max_subjects),
            "--ner-batch-size", str(args.ner_batch_size),
            "--concurrency", str(args.concurrency),
        ]
        # sampling from profile
        if knobs.get("temperature") is not None:
            cmd += ["--temperature", str(knobs["temperature"])]
        if knobs.get("top_p") is not None:
            cmd += ["--top-p", str(knobs["top_p"])]
        if knobs.get("top_k") is not None:
            cmd += ["--top-k", str(knobs["top_k"])]
        if knobs.get("max_tokens") is not None:
            cmd += ["--max-tokens", str(knobs["max_tokens"])]

        # meta for per-run file + CSV
        meta = {
            "seed": seed,
            "seed_slug": seed_slug,
            "domain": domain,
            "elicitation_strategy": strat,
            "ner_strategy": args.ner_strategy,
            "model": model,
            "out_dir": out_dir,
            "profile": prof,
            "profile_knobs": knobs,
            "max_depth": args.max_depth,
            "max_subjects": args.max_subjects,
            "max_facts_hint": args.max_facts_hint,
            "ner_batch_size": args.ner_batch_size,
            "concurrency": args.concurrency,
            "crawler": args.crawler,
            "python": sys.executable,
            "command": " ".join(shlex.quote(c) for c in cmd),
            "timestamp": dt.datetime.now().isoformat(),
        }

        plan.append({"cmd": cmd, "out_dir": out_dir, "meta": meta})
    return plan

# ---------- main ----------

def main():
    args = build_arg_parser().parse_args()

    print("[bench] START", flush=True)
    print(f"[bench] root_out={args.root_out}", flush=True)
    print(f"[bench] crawler={args.crawler}", flush=True)

    if not os.path.exists(args.crawler):
        print(f"[bench][ERROR] crawler not found: {args.crawler}", flush=True)
        sys.exit(2)

    plan = build_plan(args)
    print(f"[bench] total_planned={len(plan)}", flush=True)
    if args.verbose:
        for i, job in enumerate(plan[:10]):
            m = job["meta"]
            print(f"  plan[{i}] domain={m['domain']} seed={m['seed']} model={m['model']} "
                  f"strategy={m['elicitation_strategy']} profile={m['profile']} → {m['out_dir']}", flush=True)

    append_bench_log(args.root_out, f"planned={len(plan)}")

    if not plan:
        print("[bench][FATAL] No runs planned. Check your grids (--domains/--seeds/--models/--strategies/--profiles).", flush=True)
        sys.exit(1)

    if args.list:
        print("[bench] --list set; not executing.", flush=True)
        return

    # execute
    for idx, job in enumerate(plan, start=1):
        cmd = job["cmd"]
        out_dir = job["out_dir"]
        meta = job["meta"]

        # write per-run meta.json _before_ executing
        write_json(os.path.join(out_dir, "meta.json"), meta)

        print(f"\n[RUN {idx}/{len(plan)}]", " ".join(cmd), flush=True)
        append_bench_log(args.root_out, f"RUN {idx}/{len(plan)} {meta['command']}")

        if args.dry_run:
            # still log the row in CSV so you can review the matrix
            csv_row = {
                "status": "DRY_RUN",
                **{k: v for k, v in meta.items() if not isinstance(v, dict)}
            }
            expand_csv_header_safely(os.path.join(args.root_out, "runs.csv"), csv_row)
            continue

        rc = 0
        try:
            rc = subprocess.run(cmd, check=False).returncode
        except KeyboardInterrupt:
            print("\n[bench] Interrupted by user.", flush=True)
            append_bench_log(args.root_out, "INTERRUPTED")
            raise
        except Exception as e:
            rc = -1
            print(f"[bench][ERROR] exception while running: {e}", flush=True)

        # append CSV row with outcome
        csv_row = {
            "status": "OK" if rc == 0 else f"RC_{rc}",
            **{k: v for k, v in meta.items() if not isinstance(v, dict)}
        }
        expand_csv_header_safely(os.path.join(args.root_out, "runs.csv"), csv_row)

        # also drop a tiny done marker in each out_dir
        write_json(os.path.join(out_dir, "done.json"), {"returncode": rc})

    print("\n[bench] DONE", flush=True)

if __name__ == "__main__":
    main()
