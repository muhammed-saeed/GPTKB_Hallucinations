

```markdown

---

# KB_Crawler_GPT — Knowledge Graph Crawler (Hybrid, Multi-LLM)

A production-ready, **resumable BFS crawler** that elicits (subject, predicate, object) triples from LLMs, expands via NER, and persists to SQLite with JSONL streams and JSON snapshots.

It supports **multiple LLM backends** (OpenAI, GPT-5 via Responses API, DeepSeek, Replicate), **two execution modes** (OpenAI **Batch** or async **Concurrency**), **NER batching**, and **dual stop conditions** (max hops and/or max subjects).

---

## ✨ Highlights

* **Hybrid runner**:

  * **OpenAI Batch mode** for elicitation when using official OpenAI chat models (e.g., GPT-4o/mini).
  * **Concurrency mode** for DeepSeek / Replicate / OpenAI-compatible (with `base_url`).
* **GPT-5 support (Responses API)**: uses `reasoning.effort`, `text.verbosity`, `max_output_tokens`; no `response_format` or sampling knobs.
* **Prompt strategies** with Jinja2 templates for `elicitation` and `ner`.
* **Calibration routing**: keep/sink by confidence when using `--elicitation-strategy calibrate`.
* **Resumable**: queue state in SQLite (`pending` / `working` / `done`).
* **Streaming logs**: JSONL for queue, facts, and NER decisions; snapshots for quick inspection.

---

## 🗂 Project Layout

```
.
├─ crawlerTry.py                # main entry (Hybrid: OpenAI Batch OR Concurrency)
├─ diag.py                      # quick environment + smoke test
├─ db_models.py                 # queue + facts persistence
├─ prompter_parser.py           # loads Jinja2 templates
├─ prompts/
│  ├─ baseline/elicitation.j2
│  └─ baseline/ner.j2
├─ llm/
│  ├─ config.py                 # ModelConfig
│  ├─ factory.py                # selects backend
│  ├─ openai_client.py          # OpenAI Chat & Responses (GPT-5)
│  ├─ deepseek_like.py          # DeepSeek via OpenAI SDK (base_url)
│  └─ replicate_client.py       # Replicate wrapper
├─ settings.py                  # JSON schemas, SQLite DDL, model registry
├─ requirements.txt
└─ .env                         # your API keys
```

---

## ⚙️ Setup

### 1) Environment variables

Create `.env` and set any providers you’ll use:

```env
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=...
REPLICATE_API_TOKEN=...
# Optional: OPENAI_BASE_URL for openai-compatible endpoints (forces Concurrency mode)
```

### 2) Dependencies

```bash
pip install -r requirements.txt
```

### 3) Quick diagnostics

```bash
python diag.py
# Shows which keys are present and does a tiny JSON call.
```

---

## 🧠 How the crawl works

1. **Seed** a subject (hop=0).
2. **Elicit facts** (triples) for that subject.
3. **NER expansion**: we extract short object phrases and ask the NER prompt; entities marked `is_ne=true` are **enqueued** as next-hop subjects.
4. Stop when:

   * hop exceeds `--max-depth`, or
   * total elicited subjects reaches `--max-subjects` (if set).

**Calibration mode** (`--elicitation-strategy calibrate`): only accept facts with `confidence >= --conf-threshold`; others go to the sink table with a reason.

---

## 🏃 Modes of operation

### 1) OpenAI **Batch mode** (elicitation)

Used automatically when:

* provider is `openai` **and**
* no custom `base_url` is set.

NER is done synchronously in small batches per subject.

#### Example: GPT-4o-mini (classic Chat + Batch)

```bash
python crawlerTry.py \
  --seed "Alan Turing" \
  --output-dir runs/gpt4omini_batch \
  --elicitation-strategy baseline \
  --ner-strategy baseline \
  --elicit-model-key gpt4o-mini \
  --ner-model-key gpt4o-mini \
  --max-depth 2 \
  --batch-size 50 \
  --max-inflight 3 \
  --poll-interval 20 \
  --completion-window 24h \
  --openai-batch-min 5 \
  --openai-batch-timeout 15 \
  --openai-fastpath-max 10 \
  --max-tokens 2000 \
  --debug
```

**Batch tuning flags**

* `--batch-size`: subjects per OpenAI batch file.
* `--max-inflight`: how many batches to keep in flight.
* `--poll-interval`: seconds between polling batch status.
* `--completion-window`: e.g. `24h`.
* `--openai-batch-min`: minimum queued requests before we prefer submitting a batch.
* `--openai-batch-timeout`: how long we wait while aggregating before giving up on forming a batch.
* `--openai-fastpath-max`: if we timed out and have a handful queued, we send up to this many **direct** API calls (non-batch) to avoid getting stuck.

> **Note:** For GPT-4o/mini/turbo we use Chat Completions with JSON schema in batch & fast-path.

---

### 2) **Concurrency mode** (DeepSeek / Replicate / OpenAI-compatible)

Used automatically when:

* provider is not `openai`, or
* you set a custom `base_url` (OpenAI-compatible).

Async workers fetch distinct pending subjects up to capacity. Optionally limit global RPM.

#### Example: DeepSeek (concurrency)

```bash
python crawlerTry.py \
  --seed "Margaret Hamilton" \
  --output-dir runs/deepseek_conc \
  --elicitation-strategy baseline \
  --ner-strategy baseline \
  --elicit-model-key deepseek \
  --ner-model-key deepseek \
  --max-depth 2 \
  --concurrency 10 \
  --target-rpm 120 \
  --ner-batch-size 50 \
  --max-tokens 2000 \
  --debug
```

**Concurrency tuning flags**

* `--concurrency`: number of async workers.
* `--target-rpm`: global request cap (requests/min across all workers).

---

## 🤖 GPT-5 (Responses API) specifics

GPT-5 (including `gpt-5-nano`) **must** use the **Responses API**. These models **do not** accept `temperature`, `top_p`, or `response_format`. Instead, you control:

* `--gpt5-effort`: `minimal | low | medium | high` (reasoning depth)
* `--gpt5-verbosity`: `low | medium | high` (output verbosity)
* `--max-tokens`: mapped to `max_output_tokens` for GPT-5

We force JSON output by instruction and robust parsing.

> In **Batch mode**, we still submit elicitation via OpenAI’s `/batches`; for GPT-5 the crawler uses a JSON-only instruction and parses the `output_text` / `content.text`. In **fast-path** (direct calls) and in **Concurrency**, GPT-5 always goes through the Responses API without `response_format`.

#### Example: GPT-5-nano (Batch + fast-path as needed)

```bash
python crawlerTry.py \
  --seed "Grace Hopper" \
  --output-dir runs/gpt5nano_batch \
  --elicitation-strategy baseline \
  --ner-strategy baseline \
  --elicit-model-key gpt-5-nano \
  --ner-model-key gpt-5-nano \
  --max-depth 2 \
  --batch-size 50 \
  --max-inflight 3 \
  --poll-interval 20 \
  --completion-window 24h \
  --openai-batch-min 5 \
  --openai-batch-timeout 15 \
  --openai-fastpath-max 10 \
  --gpt5-effort minimal \
  --gpt5-verbosity low \
  --max-tokens 2048 \
  --debug
```

> If you see `status: incomplete` with `incomplete_details: max_output_tokens`, increase `--max-tokens` or lower verbosity/effort.

---

## 🧩 All important CLI flags (quick reference)

**General**

* `--seed <str>`: starting subject.
* `--output-dir <path>`: where to write DBs and logs.
* `--max-depth <int>`: hop limit (default from `settings.py`).
* `--max-subjects <int>`: total subjects cap (0 = unlimited).
* `--elicitation-strategy` / `--ner-strategy`: `baseline | icl | dont_know | calibrate`.
* `--conf-threshold <float>`: used when elicitation strategy is `calibrate`.
* `--ner-batch-size <int>`: phrases per NER call.

**Model selection**

* `--elicit-model-key <key>`
* `--ner-model-key <key>`

> Default keys defined in `settings.py` → `settings.MODELS`. Examples:
> `gpt4o-mini`, `gpt4o`, `gpt-5-nano`, `deepseek`, `deepseek-reasoner`, replicate variants.

**Sampling / length**

* `--max-tokens <int>`: Chat → `max_tokens`, GPT-5 → `max_output_tokens`.
* `--temperature`, `--top-p`, `--top-k`: **ignored by GPT-5** (do not send).

**GPT-5 only**

* `--gpt5-effort minimal|low|medium|high`
* `--gpt5-verbosity low|medium|high`

**OpenAI Batch (for official OpenAI chat models)**

* `--batch-size <int>`
* `--max-inflight <int>`
* `--poll-interval <sec>`
* `--completion-window <str>` (e.g., `24h`)
* `--openai-batch-min <int>`
* `--openai-batch-timeout <sec>`
* `--openai-fastpath-max <int>`

**Concurrency (non-OpenAI or `base_url` set)**

* `--concurrency <int>`
* `--target-rpm <int>`

**Resume**

* `--resume` (continue from existing DBs)
* `--reset-working` (move any `working` back to `pending`)

**Debug**

* `--debug` (verbose progress)

---

## 📦 Outputs

Inside each `--output-dir`:

```
queue.sqlite            # queue (pending/working/done), resumable
facts.sqlite            # triples_accepted + triples_sink
queue.jsonl             # enqueued subjects (stream)
facts.jsonl             # accepted/sink facts (stream)
queue.json              # snapshot
facts.json              # snapshot
ner_decisions.jsonl     # phrase-level NER outputs
batches/                # OpenAI batch request/results files
tmp/                    # scratch files
errors.log              # exceptions & parse issues
run_meta.json           # parameters & environment used
```

---

## 🔧 Examples you can copy-paste

### OpenAI — GPT-4o-mini with Batch

```bash
python crawlerTry.py \
  --seed "Tim Berners-Lee" \
  --output-dir runs/tbl_gpt4omini \
  --elicit-model-key gpt4o-mini \
  --ner-model-key gpt4o-mini \
  --max-depth 2 \
  --batch-size 40 \
  --max-inflight 2 \
  --poll-interval 15 \
  --completion-window 24h \
  --openai-batch-min 5 \
  --openai-batch-timeout 10 \
  --openai-fastpath-max 8 \
  --max-tokens 1800 \
  --debug
```

### OpenAI — GPT-5-nano (Responses API)

```bash
python crawlerTry.py \
  --seed "Grace Hopper" \
  --output-dir runs/hopper_gpt5nano \
  --elicit-model-key gpt-5-nano \
  --ner-model-key gpt-5-nano \
  --max-depth 2 \
  --batch-size 50 \
  --max-inflight 3 \
  --poll-interval 20 \
  --completion-window 24h \
  --openai-batch-min 5 \
  --openai-batch-timeout 15 \
  --openai-fastpath-max 10 \
  --gpt5-effort minimal \
  --gpt5-verbosity low \
  --max-tokens 2048 \
  --debug
```

### DeepSeek — Concurrency

```bash
python crawlerTry.py \
  --seed "Margaret Hamilton" \
  --output-dir runs/hamilton_deepseek \
  --elicit-model-key deepseek \
  --ner-model-key deepseek \
  --max-depth 2 \
  --concurrency 10 \
  --target-rpm 120 \
  --ner-batch-size 50 \
  --max-tokens 2000 \
  --debug
```


## Calibration Strategy

### GPT-4o mini

```bash

python crawlerTry.py \
  --seed "Grace Hopper" \
  --output-dir runs/gpt4omini_calib \
  --elicit-model-key gpt4o-mini \
  --ner-model-key gpt4o-mini \
  --elicitation-strategy calibrate \
  --ner-strategy baseline \
  --conf-threshold 0.75 \
  --max-depth 2 \
  --batch-size 50 \
  --max-inflight 3 \
  --poll-interval 20 \
  --completion-window 24h \
  --openai-batch-min 5 \
  --openai-batch-timeout 15 \
  --openai-fastpath-max 10 \
  --max-tokens 2000 \
  --debug

```



### Deepseek
```bash

python crawlerTry.py \
  --seed "Grace Hopper" \
  --output-dir runs/deepseek_calib \
  --elicit-model-key deepseek \
  --ner-model-key deepseek \
  -- domain general \
  --elicitation-strategy calibrate \
  --ner-strategy baseline \
  --conf-threshold 0.75 \
  --max-depth 2 \
  --max-subjects 10 \
  --concurrency 10 \
  --target-rpm 120 \
  --ner-batch-size 50 \
  --max-tokens 2000 \
  --debug
```

---

## 🧪 Troubleshooting

* **`Responses.create() got an unexpected keyword argument 'response_format'`**
  You’re calling GPT-5 via the Responses API. The client should **not** send `response_format`. This repo’s `llm/openai_client.py` already avoids it.

* **`status: incomplete` with `incomplete_details: max_output_tokens` (GPT-5)**
  Increase `--max-tokens`, or reduce `--gpt5-verbosity`, or lower `--gpt5-effort`.

* **Rate limits**
  Lower `--concurrency` / `--target-rpm` (Concurrency) or reduce `--max-inflight` (Batch).

* **Bad JSON**
  See `errors.log`. For Chat models, schema enforcement is strict; for GPT-5 we force JSON via instruction and robustly parse `output_text`/`content.text`.

* **Resume after crash**
  Use `--resume --reset-working` once to push in-flight items back to `pending`.

---

## 🔒 Notes & Safety

* Don’t commit your `.env`.
* OpenAI **Batch** requires the official OpenAI endpoint (do not set `OPENAI_BASE_URL`).
* If you set a custom `base_url`, the runner switches to **Concurrency** (no Batch).

---

## 🧩 Extend & Customize

* Register models in `settings.py → settings.MODELS` (set `provider`, `model`, `base_url`, `use_responses_api` for GPT-5).
* Adjust JSON output shapes in `settings.py` (`ELICIT_SCHEMA_*`, `NER_SCHEMA_*`).
* Edit prompt templates in `prompts/` and add new strategies.

---

## 📄 License

MIT — use, modify, and integrate freely.


```
