```markdown
# 🧠 GPT-KB — Multi-LLM Knowledge Graph Crawler (Batch + Concurrency)

**GPT-KB** builds a structured **knowledge graph** from large language models by crawling entities, eliciting factual triples `(subject, predicate, object)`, expanding through NER, and persisting everything in SQLite and JSONL logs.

It supports **multiple LLM providers** (OpenAI, DeepSeek, Claude, Replicate/Gemini, Replicate/Grok), and can run in either **Batch** (OpenAI official) or **Concurrent** mode.

---

## ✨ Highlights

- 🧩 **Hybrid Execution**
  - **Batch mode** for OpenAI GPT-4o / GPT-5 models.
  - **Concurrent mode** for DeepSeek, Claude, and Replicate models.
- ⚙️ **Multi-Backend Design**
  - Unified adapter layer through `llm_wrapper.py`.
- 🧮 **Calibration-Aware**
  - Filters facts by confidence threshold in “calibrate” mode.
- 🔁 **Resumable BFS Crawler**
  - Persistent queue: `pending → working → done`.
- 💾 **Structured Logging**
  - JSONL streams for incremental results; SQLite for durability.
- 🧱 **Robust JSON Repair**
  - Recovers malformed model outputs automatically.
- 🌍 **Prompt Architecture**
  - Domain-specific prompt sets: `general/` and `topics/`.

---

## 🧱 Prompt Architecture

GPT-KB uses **Jinja2 templates** for both *elicitation* and *NER* prompts, organized by **domain** and **strategy**.

```

prompts/
├── general/                      # 🌍 General knowledge (default)
│   ├── baseline/
│   ├── calibration/
│   ├── ICL/
│   └── ...
│
├── topics/                       # 🎯 Topic-anchored crawls
│   ├── baseline/
│   ├── calibration/
│   └── ...
│
├── calibration/
├── baseline/
└── selfRag/

```

### Domain Control
| Flag | Path Used | Description |
|------|------------|-------------|
| `--domain general` | `prompts/general/` | Default. Open-domain knowledge expansion. |
| `--domain topic` | `prompts/topics/` | Anchors all prompts to the seed topic (e.g., “Game of Thrones”). |

When `--domain topic` is used, the crawler injects the seed entity as `ROOT_SUBJECT` inside the prompt, so every subsequent generation stays **on-topic**.

### Template Resolution
Each step looks up:
```

prompts/{domain}/{strategy}/{task}.j2

```

Example:
```

prompts/general/calibration/elicitation.j2
prompts/topics/baseline/ner.j2

````

---

## ⚙️ Setup

### 1️⃣ Environment
Create a `.env` or export:
```bash
export OPENAI_API_KEY=sk-...
export DEEPSEEK_API_KEY=...
export REPLICATE_API_TOKEN=...
# Optional: to force concurrency for OpenAI-compatible providers
export OPENAI_BASE_URL=https://api.deepseek.com/v1
````

### 2️⃣ Install

```bash
pip install -r requirements.txt
```

### 3️⃣ Verify

```bash
python diag.py
```

---

## 🧩 Architecture Overview

```
GPTKB_Hallucinations/
├── crawler_openai_batch_concurrency-deepseek_claude.py   # Hybrid runner (Batch + Concurrency)
├── crawler_simple_both_general_topic_based.py             # Simple concurrent runner
├── db_models.py                                           # SQLite models (facts + queue)
├── llm/
│   ├── llm_wrapper.py                                    # Unified adapter for all LLMs
│   └── replicate_client.py                               # Replicate (Gemini/Grok)
├── prompts/                                              # Prompt templates
├── settings.py                                           # Model registry + schemas
├── diag.py                                               # Environment/API smoke tests
└── runs*/                                                # Run outputs
```

---

## 🚀 Running the Crawlers

GPT-KB includes two main crawler types:

| Script                                                | Mode               | Best For                             |
| ----------------------------------------------------- | ------------------ | ------------------------------------ |
| `crawler_simple_both_general_topic_based.py`          | Concurrent         | DeepSeek, Claude, Replicate          |
| `crawler_openai_batch_concurrency-deepseek_claude.py` | Batch + Concurrent | OpenAI GPT-4o/GPT-5 and mixed setups |

---

### 🅰️ Simple Concurrent Crawler

This is the easiest entry point — it runs workers in parallel for non-OpenAI providers.

#### 🔹 General Crawl Example

```bash
python crawler_simple_both_general_topic_based.py \
  --seed "Isaac Newton" \
  --domain general \
  --output-dir runsSimple/general_newton \
  --elicit-model-key deepseek \
  --ner-model-key deepseek \
  --elicitation-strategy calibrate \
  --ner-strategy calibrate \
  --concurrency 10 \
  --max-depth 2 \
  --max-subjects 20
```

🧭 **Behavior:**
Expands freely into open-domain knowledge — any related facts and entities can be followed.

---

#### 🔹 Topic-Anchored Crawl Example

```bash
python crawler_simple_both_general_topic_based.py \
  --seed "Game of Thrones" \
  --domain topic \
  --output-dir runsSimple/topic_got \
  --elicit-model-key gemini-flash \
  --ner-model-key gemini-flash \
  --elicitation-strategy calibrate \
  --ner-strategy calibrate \
  --concurrency 10 \
  --max-depth 2 \
  --max-subjects 15
```

🎯 **Behavior:**
All elicitation and NER remain within the context of the topic “Game of Thrones.”
It will not drift into unrelated entities.

---

### 🅱️ Hybrid Batch + Concurrency Crawler

This runner mixes **OpenAI Batch** (for GPT-4o/GPT-5) with **concurrent workers** for other providers.

#### 🔹 OpenAI (Batch Mode)

> ⚠️ With current testing, the most reliable **batch size is 10**.
> Larger values may increase latency or reduce completion rate.

```bash
python crawler_openai_batch_concurrency-deepseek_claude.py \
  --seed "Umar ibn al-Khattab" \
  --output-dir runsBatch/gpt4omini \
  --elicit-model-key gpt4o-mini \
  --ner-model-key gpt4o-mini \
  --openai-batch \
  --openai-batch-size 10 \
  --max-depth 2 \
  --max-subjects 10
```

#### 🔹 Topic-Scoped (Batch)

```bash
python crawler_openai_batch_concurrency-deepseek_claude.py \
  --seed "Game of Thrones" \
  --domain topic \
  --elicitation-strategy calibrate \
  --ner-strategy calibrate \
  --elicit-model-key gpt4o-mini \
  --ner-model-key gpt4o-mini \
  --openai-batch \
  --openai-batch-size 10 \
  --max-depth 2 \
  --max-subjects 10 \
  --output-dir runsBatch/topic_gpt4omini
```

#### 🔹 DeepSeek (Concurrency Mode within Hybrid)

```bash
python crawler_openai_batch_concurrency-deepseek_claude.py \
  --seed "Umar ibn al-Khattab" \
  --output-dir runsConcc/deepseek \
  --elicit-model-key deepseek \
  --ner-model-key deepseek \
  --concurrency 15 \
  --max-subjects 10
```

#### 🔹 Claude (Concurrency)

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

## 🧮 Calibration Mode

Add `--elicitation-strategy calibrate` and (optionally) `--conf-threshold 0.7` to filter facts by confidence.

```bash
python crawler_openai_batch_concurrency-deepseek_claude.py \
  --seed "Umar ibn al-Khattab" \
  --output-dir runsBatch/gpt4omini_cal \
  --elicit-model-key gpt4o-mini \
  --ner-model-key gpt4o-mini \
  --openai-batch \
  --openai-batch-size 10 \
  --elicitation-strategy calibrate \
  --conf-threshold 0.7
```

---

## 🧠 GPT-5 (Responses API)

GPT-5 models route automatically through the **Responses API** (no `response_format`, `temperature`, or `top_p`).

Control reasoning via:

* `--reasoning-effort minimal|low|medium|high`
* `--verbosity low|medium|high`

Example:

```bash
python crawler_openai_batch_concurrency-deepseek_claude.py \
  --seed "Grace Hopper" \
  --output-dir runsBatch/gpt5nano \
  --elicit-model-key gpt-5-nano \
  --ner-model-key gpt-5-nano \
  --reasoning-effort minimal \
  --verbosity low \
  --openai-batch-size 10 \
  --max-tokens 2048
```

---

## 📊 Outputs

Each run produces:

```
output-dir/
├── queue.sqlite          # BFS queue (pending/working/done)
├── facts.sqlite          # accepted triples
├── queue.jsonl           # enqueued subjects
├── facts.jsonl           # elicited triples
├── ner_decisions.jsonl   # NER extraction trace
├── errors.log            # parsing/runtime issues
├── run_meta.json         # run parameters and environment snapshot
└── batches/ or tmp/      # OpenAI Batch artifacts / scratch data
```

---

## 🧩 CLI Flags Reference

| Flag                                       | Description                                    |
| ------------------------------------------ | ---------------------------------------------- |
| `--seed`                                   | Starting subject for the crawl                 |
| `--domain`                                 | `general` or `topic` (prompt domain selection) |
| `--output-dir`                             | Folder for outputs                             |
| `--max-depth`, `--max-subjects`            | Crawl limits                                   |
| `--elicitation-strategy`, `--ner-strategy` | `baseline`, `icl`, `dont_know`, `calibrate`    |
| `--conf-threshold`                         | Confidence cutoff in calibrate mode            |
| `--batch-size`, `--openai-batch-size`      | OpenAI batch size (use 10)                     |
| `--concurrency`                            | Parallel requests for non-OpenAI models        |
| `--resume`, `--reset-working`              | Resume crashed runs                            |
| `--debug`                                  | Verbose logs                                   |

---

## ✅ Current Best Practices

* 🧠 **Batch size = 10** → most stable OpenAI Batch configuration.
* ⚙️ **Calibrate early** — improves factual precision.
* 🌍 **Use `--domain topic`** when working inside a single thematic space.
* 🧩 **Depth**: `1–2` for quick runs; `3+` for deep exploration.
* 💾 **Resume safely** using `--resume` after interruptions.

---

## 🧪 Troubleshooting

| Symptom                                                                | Likely Cause / Fix                              |
| ---------------------------------------------------------------------- | ----------------------------------------------- |
| `Responses.create() got unexpected keyword argument 'response_format'` | GPT-5 model — remove legacy parameters.         |
| `status: incomplete / max_output_tokens`                               | Increase `--max-tokens` or reduce verbosity.    |
| Crawl hangs or slows                                                   | Lower `--concurrency` or `--openai-batch-size`. |
| Bad JSON                                                               | Auto-repair enabled, see `errors.log`.          |
| Replicate auth errors                                                  | Ensure `REPLICATE_API_TOKEN` is set.            |

---

## 🔒 Notes

* Do **not** commit `.env`.
* OpenAI Batch requires the official endpoint.
* Setting `OPENAI_BASE_URL` switches to concurrency mode.
* Replicate (Gemini/Grok) requires `REPLICATE_API_TOKEN`.

---



## 📚 Model Catalog (from `settings.MODELS`)

> Tip: With your current experience, **OpenAI Batch** is most reliable at **batch size = 10**.

### 🔷 OpenAI — Chat Completions

| Key          | Provider | Model       | API Key Env      | Responses API | Batch-friendly | Default Temp | Top-p | Top-k | Max Tokens | Notes                                                            |
| ------------ | -------- | ----------- | ---------------- | ------------- | -------------- | ------------ | ----- | ----- | ---------- | ---------------------------------------------------------------- |
| `gpt4o`      | openai   | gpt-4o      | `OPENAI_API_KEY` | No            | ✅ Yes          | 0.0          | 1.0   | —     | 2000       | Strong all-rounder; JSON mode via `response_format=json_schema`. |
| `gpt4o-mini` | openai   | gpt-4o-mini | `OPENAI_API_KEY` | No            | ✅ Yes          | 0.0          | 1.0   | —     | 2000       | Great price/perf; **recommended for batch (size=10)**.           |
| `gpt4-turbo` | openai   | gpt-4-turbo | `OPENAI_API_KEY` | No            | ✅ Yes          | 0.0          | 1.0   | —     | 2000       | Legacy but stable; also supports JSON schema.                    |

### 🟣 OpenAI — GPT-5 family (Responses API)

| Key          | Provider | Model      | API Key Env      | Responses API | Batch-friendly | Max Tokens | Reasoning Effort | Text Verbosity | Notes                                             |
| ------------ | -------- | ---------- | ---------------- | ------------- | -------------- | ---------- | ---------------- | -------------- | ------------------------------------------------- |
| `gpt-5`      | openai   | gpt-5      | `OPENAI_API_KEY` | ✅ Yes         | ✅ Yes          | 2000       | `medium`         | `medium`       | Uses Responses API; don’t pass `response_format`. |
| `gpt-5-mini` | openai   | gpt-5-mini | `OPENAI_API_KEY` | ✅ Yes         | ✅ Yes          | 2000       | `low`            | `low`          | Low-cost GPT-5 variant.                           |
| `gpt-5-nano` | openai   | gpt-5-nano | `OPENAI_API_KEY` | ✅ Yes         | ✅ Yes          | 2000       | `minimal`        | `low`          | Ultra-cheap; great for breadth at shallow depth.  |

> **Notes (GPT-5):**
> • Use the **Responses API** knobs you already wired: `extra_inputs.reasoning.effort` and `extra_inputs.text.verbosity`.
> • Do **not** send `response_format` or classic sampling params; your code already strips them.

### 🟠 DeepSeek (OpenAI-compatible)

| Key                 | Provider | Model             | Base URL                   | API Key Env        | Batch-friendly    | Temp | Top-p | Top-k | Max Tokens | Notes                                    |
| ------------------- | -------- | ----------------- | -------------------------- | ------------------ | ----------------- | ---- | ----- | ----- | ---------- | ---------------------------------------- |
| `deepseek`          | deepseek | deepseek-chat     | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` | ❌ Use concurrency | 0.0  | 0.95  | —     | 2000       | Cheap & fast; good for expansion passes. |
| `deepseek-reasoner` | deepseek | deepseek-reasoner | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` | ❌ Use concurrency | 0.0  | 0.95  | —     | 2000       | Reasoning flavor; higher latency.        |

> **Run mode:** Use the **concurrent** crawler. You can also point OpenAI-compatible clients at `OPENAI_BASE_URL`.

### 🟡 Replicate — Llama / Mistral / Others

| Key           | Provider  | Model                             | API Key Env           | Temp | Top-p | Top-k | Max Tokens | Notes                          |
| ------------- | --------- | --------------------------------- | --------------------- | ---- | ----- | ----- | ---------- | ------------------------------ |
| `llama8b`     | replicate | meta/meta-llama-3.1-8b-instruct   | `REPLICATE_API_TOKEN` | 0.6  | 0.9   | 50    | 1024       | General baseline; low cost.    |
| `llama70b`    | replicate | meta/meta-llama-3.1-70b-instruct  | `REPLICATE_API_TOKEN` | 0.6  | 0.9   | 50    | 1024       | More capable; slower.          |
| `llama405b`   | replicate | meta/meta-llama-3.1-405b-instruct | `REPLICATE_API_TOKEN` | 0.6  | 0.9   | 50    | 1024       | High-end; $$$.                 |
| `mistral7b`   | replicate | mistralai/mistral-7b-instruct     | `REPLICATE_API_TOKEN` | 0.6  | 0.95  | 50    | 1024       | Efficient; good for breadth.   |
| `mixtral8x7b` | replicate | mistralai/mixtral-8x7b-instruct   | `REPLICATE_API_TOKEN` | 0.6  | 0.95  | 50    | 1024       | MoE; decent JSON compliance.   |
| `gemma2b`     | replicate | google-deepmind/gemma-2b-it       | `REPLICATE_API_TOKEN` | 0.7  | 0.95  | 50    | 200        | Tiny; consider for toy runs.   |
| `qwen2-7b`    | replicate | alibaba-nlp/qwen2-7b-instruct     | `REPLICATE_API_TOKEN` | 0.6  | 0.95  | 50    | 1024       | Solid budget model.            |
| `falcon180b`  | replicate | tiiuae/falcon-180b-instruct       | `REPLICATE_API_TOKEN` | 0.6  | 0.95  | 50    | 1024       | Heavy; OK JSON with prompting. |

> **Run mode:** Use the **simple concurrent** crawler.
> **Prompts:** Your wrapper already sanitizes prompts for Replicate (system/prompt contracts).

### 🟢 Replicate — Special (Gemini / Grok / Claude Haiku)

| Key            | Provider  | Model                      | API Key Env           | Temp | Top-p | Max Tokens | Extras / Notes                                                      |
| -------------- | --------- | -------------------------- | --------------------- | ---- | ----- | ---------- | ------------------------------------------------------------------- |
| `gemini-flash` | replicate | google/gemini-2.5-flash    | `REPLICATE_API_TOKEN` | 0.2  | 0.9   | 1024       | Set `prefer="prompt"`; your JSON builder for Gemini is included.    |
| `grok4`        | replicate | xai/grok-4                 | `REPLICATE_API_TOKEN` | 0.1  | 1.0   | 2048       | Uses messages API on Replicate; your Grok JSON builder is included. |
| `claude35h`    | replicate | anthropic/claude-3.5-haiku | `REPLICATE_API_TOKEN` | 0.3  | 0.9   | 8192       | Good cost/perf; watch JSON strictness.                              |

> **Gemini & Grok tip:** If a blocking `generate()` ever returns empty but `stream` yields JSON, your latest client already includes the per-model inputs and JSON salvage; keep `temperature` modest (0.1–0.3) and ensure `max_tokens` ≥ 1k for elicitation.

### 🧪 Local (Unsloth)

| Key            | Provider | Model                                  | API Key Env | Max Tokens | Notes                                                        |
| -------------- | -------- | -------------------------------------- | ----------- | ---------- | ------------------------------------------------------------ |
| `smollm2-1.7b` | unsloth  | unsloth/SmolLM2-1.7B-Instruct-bnb-4bit | —           | 800        | Local inference knobs in `extra_inputs` (MPS, dtype, 4-bit). |
| `smollm2-360m` | unsloth  | unsloth/SmolLM2-360M-Instruct-bnb-4bit | —           | 512        | Very small; use for debugging pipelines, not quality.        |

---

## 🎛 Defaults

* **Default elicitation model**: `gpt4o-mini`
* **Default NER model**: `gpt4o-mini`
* **Batch size recommendation**: **10** for OpenAI Batch (best reliability from your experiments)

---

## ✅ Quick Pointers

* **OpenAI Batch:** Prefer for OpenAI models; keep `--openai-batch-size 10`.
* **DeepSeek / Replicate / Claude:** Use the **concurrent** crawler.
* **Topic runs:** Pass `--domain topic` — the seed is injected as `ROOT_SUBJECT` to anchor all hops.
* **Calibrate mode:** Use `--elicitation-strategy calibrate --conf-threshold 0.7` to keep only high-confidence triples.
* **Replicate (Gemini/Grok):** Your `replicate_client.py` has per-model builders to avoid empty JSON in blocking mode; if needed, fallback to `stream_json()` is 

## 📄 License

MIT — use, modify, and extend freely.

```