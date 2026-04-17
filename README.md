# Correctness-Aware NL-to-SQL Translation for Object-Centric ERP Event Logs

## What this project is about

Business systems like SAP record every step of a business process — a customer places an order, a delivery is created, an invoice is raised, a payment clears. This data lives across many interconnected tables, not one flat spreadsheet.

This project lets an analyst type a plain English question — *"which customers had the most delayed payments last year?"* — and get a schema-checked SQL query plus an auditable result, without needing to know the database structure.

The hard part is **correctness**. A language model (LLM) can write SQL that runs without errors but still gives the wrong answer, because it connected the wrong tables together. In process data this is especially dangerous: a wrong join does not just return bad numbers, it misrepresents the business process itself.

**What we built:** a pipeline that checks the LLM's work before executing it. The LLM proposes a structured query plan (not raw SQL). A verifier checks whether that plan uses valid tables, valid columns, and only permitted connections between record types. Only after passing those checks is the plan compiled into SQL and executed. This guarantees zero hallucinated joins by construction, not by luck.

**What we evaluated:** we ran this pipeline and three baselines against a 120-question benchmark built from a real SAP Order-to-Cash event log, covering nine query classes derived from data-driven analyses (IsolationForest anomaly thresholds, Apriori-style event co-occurrence, Naive-Bayes class-conditional expectations, rolling-window analytics). The pipeline eliminates join hallucination on both splits by construction.

---

## Research questions

1. Does schema-constrained translation improve correctness over unconstrained NL-to-SQL baselines on OCEL data?
2. Do typed guardrails reduce non-executable, unsafe, and hallucinated query outputs?
3. Does grounded provenance make NL query answers more traceable and auditable?
4. What is the correctness-versus-latency trade-off of retrieval, verification, and constrained compilation?
5. How robust is the pipeline across query classes and difficulty levels?

## What is implemented (current expanded-benchmark state)

### Data

- OCEL 2.0 event log from SAP ECC Order-to-Cash process
- 3 parquet tables: `events` (157,338 rows), `objects` (158,761), `relations` (117,983)
- 9 event types, 6 object types, 7 relation types

### Benchmark

- 120-question verified benchmark (`benchmark/nl2ocel_benchmark_v1.csv`)
- Dev split: 74 questions (`nl2ocel_benchmark_dev.csv`)
- Test split: 46 questions (`nl2ocel_benchmark_test.csv`)
- 9 query classes (each with ≥ 12 items): `count_filter`, `group_topk`, `temporal_trend`, `path_relation`, `delay_analysis`, `anomaly_filter`, `conformance`, `nested_agg`, `window_agg`
- Gold result hashes computed from DuckDB (value-only SHA-256, order-independent, float aggregates rounded for stable reruns)

### How the code uses the dev and held-out questions

The dev and held-out test files do not represent two different project goals. They are two parts of the same benchmark, separated so the project has a fair evaluation process.

The code flow is:

1. **Canonical benchmark source**
   `benchmark/nl2ocel_benchmark_v1.csv` is the full 120-question benchmark. It contains the question text, query class, difficulty, split label, gold SQL, and gold result hash.

2. **Development split**
   `benchmark/nl2ocel_benchmark_dev.csv` has 74 questions. This is the build-and-debug set. We run it while developing prompts, verifier rules, repair hints, schema retrieval, and compiler behavior.

3. **Held-out test split**
   `benchmark/nl2ocel_benchmark_test.csv` has 46 questions. This is the final unseen set. We run it only after the method is fixed, to check whether the final pipeline generalizes.

4. **Evaluation loop for each question**
   For every question, `scripts/run_full_eval.py` loads the selected split, sends the question through each method, executes the generated SQL in DuckDB, hashes the output, and compares it with the gold result hash. If the hashes match, the answer is counted as correct.

5. **How dev results are used**
   Dev failures are allowed to influence the system. For example, if the dev set reveals wrong relation choices or missing filters, we can improve the schema retriever, prompt, verifier, repair hints, or IR-to-SQL compiler.

6. **How test results are used**
   Test failures are not used to tune the fixed method. They are reported as final evidence of what the already-locked pipeline can and cannot do.

Typical commands:

```powershell
# Build and diagnose on development questions
.venv\Scripts\python scripts\run_full_eval.py --backend deepseek --split dev

# Final unseen check on held-out questions
.venv\Scripts\python scripts\run_full_eval.py --backend deepseek --split test
```

In simple words: **the dev set is the workshop; the held-out test set is the exam.** Both use the same pipeline and metrics, but only the dev set is used to improve the method.

### Pipeline (Method M)

`src/nl2ocel/` implements the full constrained pipeline:

| Module | Role |
| ------ | ---- |
| `schema_retriever.py` | TF-IDF top-k schema slice for LLM prompt |
| `nl_to_ir.py` | NL → typed IR dict via LLM (up to 2 repair attempts) |
| `query_verifier.py` | IR verifier — accept / reject / repair |
| `ir_to_sql.py` | Deterministic IR → DuckDB SQL; join whitelist enforced |
| `pipeline.py` | End-to-end orchestrator |
| `baseline_translator.py` | B1 zero-shot / B2 Nan-style few-shot voting / B3 DIN-SQL |

### Evaluation results (current 120-question benchmark)

The repo now includes the completed expanded-benchmark rerun: 74 development questions and 46 held-out test questions across all nine query classes. The canonical machine-readable summary is `outputs/reports/gate_e_comparison.csv`.

#### Dev (n=74)

| Method | ExecRate | DenAcc | JoinHall | Avg latency |
| ------ | -------- | ------ | -------- | ----------- |
| B1 Zero-shot | 86.5% | 41.9% | 1.4% | 2.1s |
| B2 Few-shot | 94.6% | 23.0% | 0.0% | 65.0s |
| B3 DIN-SQL | 66.2% | 40.5% | 0.0% | 10.0s |
| **Method M (ours)** | **86.5%** | **64.9%** | **0.0%** | **3.3s** |

#### Test (n=46, held-out)

| Method | ExecRate | DenAcc | JoinHall | Avg latency |
| ------ | -------- | ------ | -------- | ----------- |
| B1 Zero-shot | 93.5% | 41.3% | 0.0% | 2.2s |
| B2 Few-shot | 97.8% | 21.7% | 0.0% | 66.8s |
| B3 DIN-SQL | 73.9% | 45.7% | 0.0% | 10.3s |
| **Method M (ours)** | **78.3%** | **50.0%** | **0.0%** | **3.3s** |

Current narrative: Method M is the top-accuracy system on both splits while preserving 0% join hallucination by construction. The verifier/repair path trades some execution success for structural safety, but still finishes ahead of the prompting baselines on denotation accuracy. The main remaining weaknesses are anomaly-style, conformance-style, and window-style queries rather than open-ended join invention.

### Notebooks

| Notebook | Purpose |
| -------- | ------- |
| `00_data_pipeline.ipynb` | Part 1: SAP table validation + linkage coverage; Part 2: OCEL construction + quality gates |
| `01_ocel_schema_exploration.ipynb` | Schema profiling, IsolationForest anomaly-threshold calibration, benchmark support |
| `02_evaluation.ipynb` | Current artifact-review notebook: inspects saved baseline outputs and walks through Method M without requiring fresh API calls |
| `03_phase2_results.ipynb` | Phase-2 results notebook: figures, repair-loop analysis, and failure taxonomy over the current result CSVs |

### Supporting artifacts

- `app/demo.py` — Streamlit interactive demo
- `app/api.py` — Flask REST API used by SAP Build Apps or any HTTP frontend
- `app/SAP_BUILD_APPS_README.md` — click-by-click SAP Build Apps setup and usage guide
- `app/examples.py` — benchmark-grounded demo questions shared between demo and API
- `scripts/run_full_eval.py` — full evaluation runner (supports `--split dev|test`)
- `scripts/run_ablation.py` — ablation runner (`M-norepair` vs full Method M)
- `scripts/compute_confidence_intervals.py` — bootstrapped 95% CIs for ExecRate and DenAcc
- `docs/benchmark_traceability.md` — maps all 120 benchmark questions to the notebook analysis that produced each one
- `docs/key_project_concepts.md` — concise map of the data science and process-mining concepts used in the project
- `docs/unit_regression_tests.md` — notes on code-level validation tests kept separate from benchmark metrics
- `docs/extension_distillation.md` — concrete next-step extensions (distillation, multi-schema, backbone comparison, etc.)
- `docs/motivation_ocel_vs_joule.md` — positioning vs enterprise NL tools
- `docs/ocel_data_contract.md` — column-level contract for the three OCEL parquet tables
- `docs/project_diagram_blueprints.md` — pipeline architecture diagram (Mermaid source)
- `archive/signavio/signavio_import_guide.md` — archived Signavio import guide and pack
- `outputs/figures/` — generated report figures (PDF + PNG)
- `manuscript/latex/images/` — tracked figure copies used by the LaTeX build
- `outputs/reports/gate_e_comparison.csv` — machine-readable summary

## Repository layout

```text
correctness-aware-nl-query-translation-ocel/
  benchmark/
    nl2ocel_benchmark_v1.csv         120 questions (verified, dev/test split annotated)
    nl2ocel_benchmark_dev.csv        74 questions — development split
    nl2ocel_benchmark_test.csv       46 questions — held-out test split
  configs/
    schema_catalog.json              TF-IDF-indexed schema catalog (S)
    relation_whitelist.json          7 whitelisted relation types (J)
    guardrail_policy.yaml            read-only + policy constraints (P)
    din_sql_demos/
      demos.json                     DIN-SQL few-shot demo bank (hand-annotated)
  data/                              git-ignored — ERP data is local-only
    raw/                             25 SAP ECC TSV exports (ABAP extraction)
    processed/
      ocel/                          events.parquet, objects.parquet, relations.parquet
      sap_process_glossary.csv       SAP internal column name -> semantic name map
  notebooks/
    00_data_pipeline.ipynb           raw SAP validation + OCEL table build
    01_ocel_schema_exploration.ipynb schema profiling, IsolationForest, benchmark seeds
    02_evaluation.ipynb              baseline artifact review + Method M walkthrough
    03_phase2_results.ipynb          Phase-2 figures, repair analysis, per-class breakdown
  outputs/
    reports/                         eval CSVs, confidence intervals, ablation summaries
    figures/                         git-ignored — generated PDF/PNG figures
  scripts/
    run_full_eval.py                 full eval runner: B1 / B2 / B3 / Method M
    run_ablation.py                  component ablation: no-retrieval, no-repair, no-verifier
    compute_confidence_intervals.py  bootstrap 95% CIs for ExecRate and DenAcc
    benchmark_dedup_analysis.py      surface vs template accuracy deduplication check
    generate_schema_catalog.py       one-time: builds configs/schema_catalog.json from OCEL
    refresh_saved_hashes.py          recompute benchmark result hashes with stable hashing
    start_api.ps1 / stop_api.ps1     PowerShell helpers to start / stop the Flask API
    build_overleaf_upload.ps1        packages LaTeX chapters for Overleaf upload
    temp/                            exploration scripts kept for reference, not part of core pipeline
      expand_benchmark.py            one-time: expanded benchmark from 48 to 120 questions
      refresh_phase2_results.py      one-time: patched manuscript tables from saved CSVs
      audit_phase2.py                pre-API offline audit (schema, verifier, hash checks)
      visualize_process_local.py     pm4py DFG / variant / duration diagrams
      grounding_coverage_report.py   offline provenance coverage report
      build_presentation.py          generated presentation slides
      generate_ocel_schema_summary_figure.py  schema summary figure for docs
  src/
    nl2ocel/                         core pipeline package (installed via pyproject.toml)
      pipeline.py                    top-level orchestrator: R -> T -> V -> C -> E
      nl_to_ir.py                    LLM-based NL -> typed IR translator (T)
      ir_to_sql.py                   deterministic IR -> DuckDB SQL compiler (C)
      query_verifier.py              schema + join-whitelist verifier (V)
      schema_retriever.py            TF-IDF schema slice retriever (R)
      llm_client.py                  provider-agnostic LLM HTTP client
      result_hash.py                 value-only SHA-256 denotation hasher
      grounding.py                   provenance metadata + auditable grounding queries (E)
      baseline_translator.py         B1 / B2 / B3 prompt builders
      din_sql_demos.py               DIN-SQL demo formatter (B3)
      nan_sampler.py                 Similarity-Diversity demo sampler (B2)
      natsql.py                      NatSQL IR helper for B3 non-nested class
      semantic_coverage.py           semantic template coverage analysis
    nb_utils.py                      shared notebook display utilities
  app/
    api.py                           Flask REST API for SAP Build Apps / external clients
    demo.py                          Streamlit interactive demo
    examples.py                      benchmark-grounded demo questions
    SAP_BUILD_APPS_README.md         SAP Build Apps integration guide
  docs/
    ocel_data_contract.md            column-level schema contract for OCEL parquets
    benchmark_traceability.md        maps all 120 questions to the analysis step that produced each
    key_project_concepts.md          concise map of OCEL + process-mining concepts used here
    pipeline_technical_reference.md  component-level technical reference for the pipeline
    unit_regression_tests.md         guide to the test suite and regression checks
    extension_distillation.md        next-step extensions (distillation, multi-schema, HANA)
    motivation_ocel_vs_joule.md      positioning vs enterprise NL tools (Joule, Copilot)
    project_diagram_blueprints.md    pipeline architecture diagram (Mermaid source)
    literature/
      literature_review_tracker.csv  per-paper relevance tracker
      references.bib                 BibTeX library
  tests/
    test_ir_to_sql.py                unit tests: IR -> SQL compiler
    test_query_verifier.py           unit tests: verifier accept / repair / reject logic
    test_schema_retriever.py         unit tests: TF-IDF schema retriever
    test_phase2_ir.py                integration tests: all 9 intent classes end-to-end
    test_nl_to_ir.py                 smoke tests: NL -> IR translation
    test_result_hash.py              unit tests: denotation hashing
    test_pipeline_semantic_templates.py  semantic template regression tests
    test_semantic_coverage.py        semantic coverage analysis tests
  manuscript/
    latex/                           LaTeX thesis source, tracked figures, compiled PDF
    drafts/                          git-ignored — working MD drafts and backup copies
  pyproject.toml                     package definition + dependency list (pip install -e .)
  requirements.txt                   pinned dependencies for reproducible installs
  WORKPLAN.md                        research workplan and phase tracker
  .gitignore
```

## Setup

### 1. Create virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Add raw data

Place SAP TSV exports in `data/raw/` (not tracked in git — ERP data is sensitive).

### 3. Optional: local LLM backend

```powershell
ollama pull qwen3:8b
ollama serve    # keep running in separate terminal
```

Or use a cloud backend by setting the matching key and passing `--backend deepseek`, `--backend openai`, or `--backend anthropic`. The reported benchmark uses DeepSeek (`DEEPSEEK_API_KEY`).

## Running the evaluation

```powershell
# Dev set (all methods)
.venv\Scripts\python scripts\run_full_eval.py --backend deepseek --split dev

# Test set (Gate E3)
.venv\Scripts\python scripts\run_full_eval.py --backend deepseek --split test

# Resume interrupted run
.venv\Scripts\python scripts\run_full_eval.py --backend deepseek --split dev --resume

# Single method
.venv\Scripts\python scripts\run_full_eval.py --mode b1 --backend deepseek
```

## Running the Streamlit demo

```powershell
.venv\Scripts\streamlit run app\demo.py
```

## Running the REST API demo

```powershell
$env:DEEPSEEK_API_KEY = Read-Host "DeepSeek API key"
.\scripts\start_api.ps1

# then in another terminal
curl http://localhost:8000/health
curl http://localhost:8000/examples
```

For the SAP Build Apps frontend, follow `app/SAP_BUILD_APPS_README.md`.

## Evaluation metrics

| Metric | Definition |
| ------ | ---------- |
| ExecRate | SQL executes without error |
| DenAcc | Result hash matches gold (value-only, order-independent) |
| JoinHall | SQL uses a relation_type not in whitelist |
| Latency | End-to-end translation + execution time |

## Rebuilding saved artifacts

```powershell
# Recompute benchmark/result hashes using stable float-aware denotation hashing
.venv\Scripts\python scripts\refresh_saved_hashes.py

# Recompute comparison tables and patch manuscript result tables
.venv\Scripts\python scripts\refresh_phase2_results.py

# Rebuild the offline RQ3 provenance coverage report
.venv\Scripts\python scripts\grounding_coverage_report.py

# Run the full offline pre-API audit
.venv\Scripts\python scripts\audit_phase2.py

# Run ablation (M-norepair vs full Method M) and recompute confidence intervals
.venv\Scripts\python scripts\run_ablation.py --backend deepseek --split dev
.venv\Scripts\python scripts\compute_confidence_intervals.py
```

## Running tests

```powershell
.venv\Scripts\python -m pytest tests/ -v
```

## Notes

- `data/` is git-ignored — ERP extracts are sensitive and large.
- Gold result hashes use value-only SHA-256 (ignores column aliases, sorts rows, rounds floating aggregates to 6 decimals) for fair comparison.
- `nl2ocel_benchmark_v1.csv` is the canonical 120-question source with `split` column. `dev.csv` and `test.csv` are derived splits used by eval scripts.
- The expanded benchmark covers 9 query classes with balanced support (12-15 questions per class), replacing the earlier sparse 48-question prototype setup.
- `outputs/reports/gate_e_comparison.csv`, `per_class_phase2.csv`, `confidence_intervals.csv`, `ablation_summary.csv`, and `backbone_ablation.csv` are the current report-facing artifacts. The fixed-control ablation confirms `M-norepair` runs with exactly one IR attempt per question.
- Current future-work scope is no longer benchmark expansion. The natural next extensions are multi-schema evaluation beyond SAP O2C, stronger local-model deployment, and enterprise deployment hardening.
