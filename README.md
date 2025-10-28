
---

```markdown
# 🧠 GPT-KB — Multi-LLM Knowledge Graph Crawler (Hybrid Batch + Concurrency)

**GPT-KB** is a production-ready, resumable BFS crawler that elicits `(subject, predicate, object)` triples from LLMs, expands via NER, and persists the evolving knowledge graph to SQLite with JSONL streams and snapshots.

It supports **multiple LLM backends** — OpenAI (incl. GPT-5), DeepSeek, Claude, and Replicate (Gemini-Flash, Grok) — and can run in **OpenAI Batch** or **Concurrent** mode.

---

## ✨ Key Features

- **Hybrid execution**  
  - OpenAI **Batch mode** for GPT-4o / GPT-5 models.  
  - **Concurrent mode** for DeepSeek, Claude, Replicate (Gemini, Grok), or any OpenAI-compatible API.

- **Multi-LLM backend**  
  DeepSeek, OpenAI (Chat + Responses), Claude, and Replicate are all plug-and-play through `llm_wrapper.py`.

- **Calibration routing**  
  In `calibrate` mode, only keeps facts above a configurable confidence threshold.

- **Resumable graph traversal**  
  Queue state persisted in SQLite (`pending` → `working` → `done`).

- **Structured outputs**  
  JSONL logs for streaming facts & entities, plus JSON snapshots for quick inspection.

- **Robust JSON recovery**  
  Auto-repairs malformed model output (via `_parse_json_best_effort`).

---

## 🗂 Project Structure

```

GPTKB_Hallucinations/
├── crawler_openai_batch_concurrency-deepseek_claude.py   # hybrid runner (Batch + Concurrency)
├── crawler_simple_openai_claude_claude.py                # simple concurrent runner
├── db_models.py                                          # SQLite ORM (queue + facts)
├── diag.py                                               # environment and API smoke tests
├── llm/
│   ├── **init**.py
│   ├── llm_wrapper.py                                   # unified model interface
│   └── replicate_client.py                              # Replicate Gemini / Grok client
├── prompter_parser.py                                   # loads & renders Jinja2 prompts
├── prompts/
│   ├── baseline/elicitation.j2
│   └── baseline/ner.j2
├── settings.py                                          # schemas, model registry, DDL
├── requirements.txt
├── runs*/                                               # output directories (Batch, Concurrency, etc.)
└── README.md

````

---

## ⚙️ Setup

### 1️⃣ Environment Variables

Create a `.env` file or export manually:

```bash
export OPENAI_API_KEY=sk-...
export DEEPSEEK_API_KEY=...
export REPLICATE_API_TOKEN=...
# Optional (forces concurrency mode)
export OPENAI_BASE_URL=https://api.deepseek.com/v1
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Verify Environment

```bash
python diag.py
```

---

## 🧩 How It Works

1. **Seed**: enqueue an initial subject (`hop=0`).
2. **Elicit facts**: model generates triples `(subject, predicate, object)`.
3. **NER expansion**: extract candidate entities (`object` phrases) and run a NER prompt.
4. **Enqueue**: any `is_ne=true` entities are added as next-hop subjects.
5. **Repeat** until:

   * hop exceeds `--max-depth`, or
   * total subjects hit `--max-subjects`.

> In `calibrate` mode, only facts with `confidence >= --conf-threshold` are accepted.

---

## 🚀 Runners

### 🅰️ `crawler_openai_batch_concurrency-deepseek_claude.py`

**Hybrid runner** combining:

* **OpenAI Batch** for official OpenAI models (GPT-4o, GPT-5, mini/turbo)
* **Concurrent workers** for DeepSeek, Claude, or Replicate models

#### Example: GPT-4o-mini (Batch)

```bash
python crawler_openai_batch_concurrency-deepseek_claude.py \
  --seed "Umar ibn al-Khattab" \
  --output-dir runsBatch/gpt4omini \
  --elicit-model-key gpt4o-mini \
  --ner-model-key gpt4o-mini \
  --max-depth 2 \
  --batch-size 50 \
  --max-subjects 10
```

#### Example: DeepSeek (Concurrency)

```bash
python crawler_openai_batch_concurrency-deepseek_claude.py \
  --seed "Umar ibn al-Khattab" \
  --output-dir runsConcc/deepseek \
  --elicit-model-key deepseek \
  --ner-model-key deepseek \
  --concurrency 15 \
  --max-subjects 10
```

#### Example: Claude

```bash
python crawler_openai_batch_concurrency-deepseek_claude.py \
  --seed "Umar ibn al-Khattab" \
  --output-dir runsConn/claude35h \
  --elicit-model-key claude35h \
  --ner-model-key claude35h \
  --concurrency 15 \
  --max-subjects 10
```

---

### 🅱️ `crawler_simple_openai_claude_claude.py`

Simpler concurrent crawler, used for **DeepSeek**, **Replicate**, or **Claude** without Batch logic.
All workers run in parallel using your chosen provider’s API.

#### Example: DeepSeek

```bash
python crawler_simple_openai_claude_claude.py \
  --seed "Khalid ibn al-Walid" \
  --output-dir runsSimple/deepseek \
  --elicit-model-key deepseek \
  --ner-model-key deepseek \
  --concurrency 10 \
  --max-subjects 10
```

#### Example: Replicate (Gemini-Flash)

```bash
python crawler_simple_openai_claude_claude.py \
  --seed "Khalid ibn al-Walid" \
  --output-dir runsSimple/gemini_flash \
  --elicit-model-key gemini-flash \
  --ner-model-key gemini-flash \
  --concurrency 10
```

#### Example: Replicate (Grok-4)

```bash
python crawler_simple_openai_claude_claude.py \
  --seed "Khalid ibn al-Walid" \
  --output-dir runsSimple/grok4 \
  --elicit-model-key grok-4 \
  --ner-model-key grok-4 \
  --concurrency 10
```

---

## 🧮 Calibration Mode

Add `--elicitation-strategy calibrate` and (optionally) `--conf-threshold 0.7` to filter by confidence.

```bash
python crawler_openai_batch_concurrency-deepseek_claude.py \
  --seed "Umar ibn al-Khattab" \
  --elicit-model-key gpt4o-mini \
  --ner-model-key gpt4o-mini \
  --output-dir runsBatch/gpt4omini_cal \
  --elicitation-strategy calibrate \
  --conf-threshold 0.7
```

---

## 🧠 GPT-5 (Responses API)

GPT-5 models (e.g., `gpt-5-nano`) automatically route through the **Responses API** — no `response_format`, `temperature`, or `top_p`.

Control reasoning via:

* `--gpt5-effort minimal|low|medium|high`
* `--gpt5-verbosity low|medium|high`

Example:

```bash
python crawler_openai_batch_concurrency-deepseek_claude.py \
  --seed "Grace Hopper" \
  --output-dir runsBatch/gpt5nano \
  --elicit-model-key gpt-5-nano \
  --ner-model-key gpt-5-nano \
  --gpt5-effort minimal \
  --gpt5-verbosity low \
  --max-tokens 2048
```

---

## 🧾 Outputs

Each run produces:

```
output-dir/
├── queue.sqlite          # BFS queue (pending, working, done)
├── facts.sqlite          # accepted and sinked triples
├── queue.jsonl           # enqueued subjects
├── facts.jsonl           # elicited triples
├── ner_decisions.jsonl   # entity decisions
├── errors.log            # parsing/runtime errors
├── run_meta.json         # run parameters & env
└── batches/ or tmp/      # OpenAI Batch artifacts / scratch
```

---

## 🧩 CLI Reference

| Flag                                                                    | Description                               |     |           |            |
| ----------------------------------------------------------------------- | ----------------------------------------- | --- | --------- | ---------- |
| `--seed`                                                                | starting subject                          |     |           |            |
| `--output-dir`                                                          | output folder                             |     |           |            |
| `--max-depth`                                                           | hop limit                                 |     |           |            |
| `--max-subjects`                                                        | total subjects limit                      |     |           |            |
| `--elicitation-strategy` / `--ner-strategy`                             | `baseline                                 | icl | dont_know | calibrate` |
| `--conf-threshold`                                                      | minimum confidence (calibrate)            |     |           |            |
| `--elicit-model-key` / `--ner-model-key`                                | registered model keys (see `settings.py`) |     |           |            |
| `--max-tokens`                                                          | output token cap                          |     |           |            |
| `--temperature`, `--top-p`, `--top-k`                                   | sampling controls (ignored by GPT-5)      |     |           |            |
| `--batch-size`, `--max-inflight`                                        | OpenAI Batch tuning                       |     |           |            |
| `--poll-interval`, `--completion-window`                                | Batch polling                             |     |           |            |
| `--openai-batch-min`, `--openai-batch-timeout`, `--openai-fastpath-max` | Batch submission thresholds               |     |           |            |
| `--concurrency`, `--target-rpm`                                         | concurrency & rate limits                 |     |           |            |
| `--resume`, `--reset-working`                                           | resume crashed runs                       |     |           |            |
| `--debug`                                                               | verbose mode                              |     |           |            |

---

## 🧩 Extending

* **Add new models**: `settings.MODELS` → specify `provider`, `model`, and `base_url` if OpenAI-compatible.
* **Edit prompts**: `prompts/baseline/*.j2`.
* **Adjust schemas**: `settings.py` → `ELICIT_SCHEMA_*`, `NER_SCHEMA_*`.
* **Plug new LLM backends**: implement a new client in `llm/`.

---

## 🧪 Troubleshooting

| Symptom                                                                | Cause / Fix                                         |
| ---------------------------------------------------------------------- | --------------------------------------------------- |
| `Responses.create() got unexpected keyword argument 'response_format'` | GPT-5 uses Responses API; remove `response_format`. |
| `status: incomplete / max_output_tokens`                               | Increase `--max-tokens` or reduce verbosity/effort. |
| Slow crawl                                                             | Lower `--concurrency` or `--max-inflight`.          |
| Bad JSON                                                               | See `errors.log`; JSON recovery runs automatically. |
| Resume after crash                                                     | `--resume --reset-working`.                         |

---

## 🔒 Notes

* Do **not** commit your `.env` file.
* OpenAI **Batch** requires the official endpoint (no `base_url`).
* Setting `OPENAI_BASE_URL` automatically switches to concurrency mode.
* Replicate models (Gemini, Grok) require `REPLICATE_API_TOKEN`.

---

## 📄 License

MIT — freely use, modify, and integrate.

```

---
