# Correctness-Aware NL-to-SQL for Object-Centric ERP Event Logs

[![Project site](https://img.shields.io/badge/project-site-2b6cb0)](https://o-2wice.github.io/correctness-aware-nl-query-translation-ocel/)
![Tests](https://img.shields.io/badge/tests-87%20passed-brightgreen)
![Verifier](https://img.shields.io/badge/verifier-typed%20IR%20%2B%20join%20checks-blue)
![Engine](https://img.shields.io/badge/engine-DuckDB%20%2B%20Python-orange)
![Site](https://img.shields.io/badge/site-Quarto-purple)

Text-to-SQL over process data can fail quietly. A generated query may execute
without errors and still answer the wrong business question because it joined
through the wrong object path.

This repository explores a stricter pattern for SAP-style order-to-cash event
logs. The model does not write executable SQL directly. It proposes a typed
intermediate representation; deterministic code verifies schema references,
relation paths, SQL policy, and result traces before compiling the plan to
DuckDB SQL.

The output is therefore not treated as correct just because it runs. Accepted
answers carry an inspectable path from question, to typed IR, to compiled SQL,
to denotation hash.

## Highlights

- Typed intermediate representation instead of direct raw-SQL generation.
- Schema and enum checks for tables, columns, event types, object types, and
  relation types.
- Relation whitelist enforcement for object-centric joins.
- Deterministic compiler from accepted IR to DuckDB SQL.
- Read-only SQL policy checks before execution.
- Stable denotation hashing for result comparison.
- Regression tests for verifier, compiler, semantic templates, schema retrieval,
  and result hashing.
- Quarto site with diagrams, notebooks, and saved result artifacts.

## Architecture

```text
Natural-language question
  -> schema retrieval
  -> LLM-to-typed-IR translation
  -> IR verification and repair hints
  -> deterministic SQL compilation
  -> SQL policy check
  -> DuckDB execution
  -> result hash and provenance metadata
```

| Component | Role |
| --- | --- |
| `src/nl2ocel/schema_retriever.py` | Selects a compact schema slice for the prompt. |
| `src/nl2ocel/nl_to_ir.py` | Converts natural language into typed IR with repair attempts. |
| `src/nl2ocel/query_verifier.py` | Validates intents, tables, columns, enum values, aggregations, and relation paths. |
| `src/nl2ocel/ir_to_sql.py` | Compiles accepted IR into deterministic DuckDB SQL. |
| `src/nl2ocel/pipeline.py` | Orchestrates retrieval, translation, verification, compilation, execution, and grounding. |
| `src/nl2ocel/result_hash.py` | Computes stable value-only result hashes. |
| `src/nl2ocel/baseline_translator.py` | Implements B1 zero-shot, B2 few-shot, and B3 DIN-SQL-style baselines. |

## Verification Layers

Accepted SQL is constrained by code, not by prompt wording alone.

| Layer | File | Purpose |
| --- | --- | --- |
| Schema catalog | `configs/schema_catalog.json` | Defines valid tables, columns, and sample enum values. |
| Relation whitelist | `configs/relation_whitelist.json` | Defines permitted object-centric relation paths. |
| SQL policy | `configs/sql_policy.yaml` | Restricts SQL keywords, base tables, predicates, and aggregations. |
| IR verifier | `src/nl2ocel/query_verifier.py` | Rejects or repairs invalid structured query plans. |
| SQL policy checker | `src/nl2ocel/pipeline.py` | Rechecks compiled SQL before execution. |

The model may propose a query plan; deterministic code decides whether the plan
is valid and how it becomes executable SQL.

## Benchmark

The benchmark contains 120 natural-language questions over an object-centric
SAP order-to-cash event log.

| Asset | Description |
| --- | --- |
| `benchmark/nl2ocel_benchmark_v1.csv` | Canonical 120-question benchmark with split labels and gold result hashes. |
| `benchmark/nl2ocel_benchmark_dev.csv` | Development split, 74 questions. |
| `benchmark/nl2ocel_benchmark_test.csv` | Held-out test split, 46 questions. |
| `outputs/reports/gate_e_comparison.csv` | Summary comparison across methods. |
| `outputs/reports/per_class_phase2.csv` | Per-query-class denotation accuracy. |
| `outputs/reports/confidence_intervals.csv` | Bootstrap confidence intervals. |

Question classes include `count_filter`, `group_topk`, `temporal_trend`,
`path_relation`, `delay_analysis`, `anomaly_filter`, `conformance`,
`nested_agg`, and `window_agg`.

## Data Policy

The original SAP-style extracts are excluded from version control. They include
raw ERP table exports and customer-master-style fields, so Git LFS would solve
file size but not redistribution or disclosure risk.

To keep the runnable path open, `scripts/create_demo_ocel.py` generates a small
synthetic dataset with the same OCEL table contract. That dataset is enough to
exercise the pipeline mechanics, API, verifier, compiler, and SQL execution
path. The saved benchmark metrics in `outputs/reports/` remain the documented
result artifacts from the original project run.

## Results

Protocol notes:

- Main comparison backend: DeepSeek Chat for B1, B2, B3, and the constrained
  pipeline.
- B1 setup: Rajkumar-style zero-shot Create Table + Select 3 prompting, with no
  demonstration examples.
- Decoding: greedy generation via `temperature = 0.0` in
  `src/nl2ocel/llm_client.py`; B1 caps generation at `200` tokens, while
  DIN-SQL uses `600` for stages 1-3 and `350` for self-correction.
- Splits: 74 development questions and 46 held-out test questions.
- Scoring: execution rate, denotation accuracy by value-only result hash,
  relation-whitelist violations, and latency.

| Split | Method | ExecRate | DenAcc | JoinHall | Avg latency |
| --- | --- | ---: | ---: | ---: | ---: |
| Dev | B1 Zero-shot | 86.5% | 41.9% | 1.4% | 2.1s |
| Dev | B2 Few-shot | 100.0% | 12.5% | 0.0% | 43.9s |
| Dev | B3 DIN-SQL | 66.2% | 40.5% | 0.0% | 10.0s |
| Dev | Constrained pipeline | 86.5% | 64.9% | 0.0% | 3.3s |
| Test | B1 Zero-shot | 93.5% | 41.3% | 0.0% | 2.2s |
| Test | B2 Few-shot | 100.0% | 20.0% | 0.0% | 40.1s |
| Test | B3 DIN-SQL | 73.9% | 45.7% | 0.0% | 10.3s |
| Test | Constrained pipeline | 78.3% | 50.0% | 0.0% | 3.3s |

Metrics:

- `ExecRate`: generated SQL executes without error.
- `DenAcc`: result hash matches the gold answer.
- `JoinHall`: SQL uses a relation type outside the whitelist.
- `Latency`: end-to-end translation and execution time.

## Repository Layout

```text
benchmark/                  Benchmark questions and gold hashes
configs/                    Schema catalog, relation whitelist, SQL policy
src/nl2ocel/                Core package
tests/                      Unit and regression tests
scripts/                    Evaluation, ablation, hashing, and API helpers
app/                        Streamlit demo and Flask API
notebooks/                  Reproducible analysis notebooks
outputs/reports/            Saved evaluation summaries
_quarto.yml                 Quarto site configuration
index.qmd                   Quarto site source
styles.css                  Quarto site styling
.github/workflows/pages.yml GitHub Pages deployment workflow
```

Raw ERP extracts, generated figures, local credentials, and local-only working
materials are intentionally excluded from version control. The repository
includes `scripts/create_demo_ocel.py` to generate a tiny synthetic OCEL dataset
for local smoke tests without publishing the original SAP-style extracts.

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

For cloud LLM backends, copy `.env.example` to `.env` and set only the key for
the backend being used. Local `.env` files are ignored and should not be
committed.

Supported backend names are `ollama`, `deepseek`, `openai`, and `anthropic`.

## Run Tests

```powershell
.venv\Scripts\python -m pytest tests/ -v
```

Current local verification:

```text
87 passed
```

## End-to-End Demo

Generate the synthetic demo dataset:

```powershell
.\.venv\Scripts\python scripts\create_demo_ocel.py --overwrite
```

Run a deterministic pipeline smoke test. This question uses a semantic template,
so it does not require a live model call:

```powershell
$env:NL2OCEL_BACKEND = "ollama"
.\.venv\Scripts\python -m nl2ocel.pipeline
```

Expected shape:

```text
Status: accept
SQL: SELECT ...
```

To run the Flask API with a cloud model, set one provider key:

```powershell
$env:NL2OCEL_BACKEND = "deepseek"
$env:DEEPSEEK_API_KEY = Read-Host "DeepSeek API key"
.\scripts\start_api.ps1
```

Then test the API:

```powershell
curl http://localhost:8000/health
curl "http://localhost:8000/query/How%20many%20order%20items%20are%20linked%20to%20a%20customer%20that%20received%20a%20dunning%20notice%3F"
```

For OpenAI or Anthropic, set `NL2OCEL_BACKEND` to `openai` or `anthropic` and
set the matching `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.

## Run Evaluation

The saved benchmark metrics in `outputs/reports/` were produced from the
original project data. To recompute them, provide compatible OCEL parquet files
under `data/processed/ocel/` and run with your own model credentials.

```powershell
# Development split
.venv\Scripts\python scripts\run_full_eval.py --backend deepseek --split dev

# Held-out test split
.venv\Scripts\python scripts\run_full_eval.py --backend deepseek --split test

# Resume an interrupted run
.venv\Scripts\python scripts\run_full_eval.py --backend deepseek --split dev --resume

# Single method
.venv\Scripts\python scripts\run_full_eval.py --mode b1 --backend deepseek
```

Recompute derived artifacts:

```powershell
.venv\Scripts\python scripts\refresh_saved_hashes.py
.venv\Scripts\python scripts\run_ablation.py --backend deepseek --split dev
.venv\Scripts\python scripts\compute_confidence_intervals.py
```

## Run Demos

Streamlit:

```powershell
.venv\Scripts\streamlit run app\demo.py
```

Flask API:

```powershell
$env:DEEPSEEK_API_KEY = Read-Host "DeepSeek API key"
.\scripts\start_api.ps1

curl http://localhost:8000/health
curl http://localhost:8000/examples
```

Generic REST/no-code integration notes are in `app/SAP_BUILD_APPS_README.md`.

## Quarto Site

The project site renders with Quarto:

```powershell
quarto render
```

The GitHub Pages workflow renders the site and publishes `_site/` through
GitHub Actions. In repository settings, configure Pages to deploy from
GitHub Actions.
