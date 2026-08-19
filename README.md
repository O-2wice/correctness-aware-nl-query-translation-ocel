# Correctness-Aware NL-to-SQL for Object-Centric Event Logs (OCEL)

[![Read the write-up](https://img.shields.io/badge/read-the%20write--up-2b6cb0)](https://o-2wice.github.io/correctness-aware-nl-query-translation-ocel/)
![Data](https://img.shields.io/badge/data-OCEL%202.0-2a9d8f)
![Engine](https://img.shields.io/badge/engine-DuckDB-orange)
![Language](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Put business questions to an order-to-cash event log in natural language,
without trusting the model to write the SQL.

The log is an **object-centric event log (OCEL 2.0)**: instead of one row per
case, it records events, business objects, and the typed relations between
those objects. That last part is what makes querying it hard. Answering a
question usually means walking the right path between objects, and picking the
wrong path still returns a number.

So the model never emits SQL. It proposes a typed intermediate representation,
deterministic code verifies that IR against a schema catalog, a relation
whitelist and a SQL policy, and only verified plans reach the compiler.

Against three published prompting baselines on the same 120 questions and the
same backend, this gains 23 points of denotation accuracy on the development
split and holds relation-path violations at zero.

**[Read the write-up](https://o-2wice.github.io/correctness-aware-nl-query-translation-ocel/)**
for the method, the figures and the full results.

| Development split | DenAcc    | JoinHall | Avg latency |
| ----------------- | --------: | -------: | ----------: |
| B1 Zero-shot      | 41.9%     | 1.4%     | 2.1s        |
| B2 Few-shot       | 12.5%     | 0.0%     | 43.9s       |
| B3 DIN-SQL        | 40.5%     | 0.0%     | 10.0s       |
| This pipeline     | **64.9%** | 0.0%     | 3.3s        |

## Quick Start

```bash
pip install -r requirements.txt
pip install -e .
```

The original ERP extracts are not committed (see [Data](#data)), so generate the
synthetic log first. It uses the same table contract:

```bash
python scripts/create_demo_ocel.py --overwrite
```

That writes `events`, `objects` and `relations` parquet files to
`data/processed/ocel/`. From there, neither of these needs an API key:

```bash
pytest tests/ -q
python -m nl2ocel.pipeline
```

The smoke test question resolves through a semantic template, so it compiles and
executes without a model call and prints an accepted status with the SQL. The
test suite covers the verifier, the compiler for each query class, semantic
template coverage, schema retrieval and result hashing.

For live translation, copy `.env.template` to `.env` and set one provider key.
`NL2OCEL_BACKEND` accepts `deepseek`, `openai`, `anthropic` or `ollama`.

## Method

```text
question
  → schema retrieval    compact schema slice for the prompt
  → translation         natural language → typed IR, never SQL
  → verification        accept | repair (≤2 attempts) | reject
  → compilation         IR → SQL
  → semantic coverage   does the query still ask what the question asked?
  → policy check        read-only, approved tables, bounded selects
  → execution           DuckDB over the OCEL views: events / objects / relations
  → result hash         stable value-only hash
```

`verify_ir` returns `accept`, `repair` or `reject`. A `repair` verdict sends the
specific violations back to the model as hints for up to two further attempts; a
`reject` abandons the question rather than executing a plan it cannot justify.

Compilation is where the SQL text finally gets written, and nothing generative
is involved. `ir_to_sql.py` reads the verified plan and assembles the query from
fixed clause builders and per-intent templates, so the same accepted plan
always produces byte-identical SQL. It refuses to emit a join for any relation type
outside the whitelist, so an illegal path cannot survive compilation even if it
somehow survived verification. The finished string is then checked against the
SQL policy and executed by DuckDB against three views over the OCEL parquet
files.

| Module | Role |
| --- | --- |
| [`schema_retriever.py`](src/nl2ocel/schema_retriever.py) | Selects the schema slice shown to the model. |
| [`nl_to_ir.py`](src/nl2ocel/nl_to_ir.py) | Translates a question into typed IR, with the repair loop. |
| [`query_verifier.py`](src/nl2ocel/query_verifier.py) | Validates intents, schema references, enums, aggregations, relation paths. |
| [`ir_to_sql.py`](src/nl2ocel/ir_to_sql.py) | Compiles accepted IR into SQL. |
| [`semantic_coverage.py`](src/nl2ocel/semantic_coverage.py) | Rejects valid SQL that lost a condition the question asked for. |
| [`pipeline.py`](src/nl2ocel/pipeline.py) | Orchestrates the stages and records provenance. |
| [`result_hash.py`](src/nl2ocel/result_hash.py) | Value-only hashing, insensitive to alias and row order. |
| [`baseline_translator.py`](src/nl2ocel/baseline_translator.py) | The three prompting baselines. |

What is enforceable lives in configuration: `configs/schema_catalog.json` for
valid tables, columns and enum values, `configs/relation_whitelist.json` for
legal object paths, `configs/sql_policy.yaml` for what compiled SQL may do.

## Demo and API

```bash
streamlit run app/demo.py
```

```bash
export NL2OCEL_BACKEND=deepseek
export DEEPSEEK_API_KEY=...
python app/api.py

curl http://localhost:8000/health
curl http://localhost:8000/examples
```

Responses carry the answer, the typed IR, the compiled SQL, the execution status
and provenance metadata, which is enough to drive a no-code frontend.
[`app/REST_INTEGRATION.md`](app/REST_INTEGRATION.md) has those notes.
`scripts/start_api.ps1` and `scripts/stop_api.ps1` wrap the same thing on
Windows.

## Reproducing the Evaluation

The published numbers came from the original ERP extract. Recomputing them needs
compatible OCEL parquet files under `data/processed/ocel/` and your own model
credentials. The synthetic log exercises the machinery but will not reproduce
the metrics.

```bash
python scripts/run_full_eval.py --backend deepseek --split dev
python scripts/run_full_eval.py --backend deepseek --split test
python scripts/run_full_eval.py --backend deepseek --split dev --resume
```

`--mode` accepts `all`, `b1`, `b2`, `b3` or `pipeline`. Derived artifacts:

```bash
python scripts/refresh_saved_hashes.py
python scripts/run_ablation.py --backend deepseek --split dev
python scripts/compute_confidence_intervals.py
```

Saved runs are in `outputs/reports/`: `method_comparison.csv` for the headline
table, `per_class_accuracy.csv` by query class, `confidence_intervals.csv` for
bootstrap intervals and significance tests, `ablation_summary.csv` and
`backbone_ablation.csv` for the two ablations, and
`baseline_b{1,2,3}_{dev,test}.csv` and `pipeline_{dev,test}.csv` per question.

## Data

The original extracts come from an SAP ECC training client, with fictional
company codes, fictional customers and no personal data. They are still not
redistributed here, because SAP-shipped sample data is not mine to republish.

`scripts/create_demo_ocel.py` generates a small synthetic log with the same table
contract, which is enough to run the pipeline, the API, the verifier, the
compiler and the tests. The metrics in `outputs/reports/` are the artifacts of
the original run.

## Layout

```text
index.qmd             # write-up, renders to _site/
src/nl2ocel/          # retrieval, translation, verification, compilation
configs/              # schema catalog, relation whitelist, SQL policy, demos
benchmark/            # 120 questions with splits, classes, gold hashes
tests/                # verifier, compiler, template, retrieval, hashing
scripts/              # evaluation, ablations, intervals, demo data generator
notebooks/            # data pipeline, schema exploration, evaluation, results
app/                  # Streamlit demo and Flask API
outputs/reports/      # saved evaluation artifacts
```

| Notebook | Contents |
| --- | --- |
| [`00_data_pipeline.ipynb`](notebooks/00_data_pipeline.ipynb) | Builds the object-centric views and validates the source contracts. |
| [`01_ocel_schema_exploration.ipynb`](notebooks/01_ocel_schema_exploration.ipynb) | Profiles events, objects and relation types. |
| [`02_evaluation.ipynb`](notebooks/02_evaluation.ipynb) | Runs the baselines and records comparison artifacts. |
| [`03_results.ipynb`](notebooks/03_results.ipynb) | Confidence intervals, per-class behaviour, failure taxonomy. |

## Report

`index.qmd` is the written version. Render it with:

```bash
quarto render
```

`.github/workflows/pages.yml` renders and publishes `_site/` through the manual
GitHub Actions dispatch while the project is under review. Repository Pages
settings need to be set to deploy from GitHub Actions.

## Notes

Seven of the 120 benchmark questions match hardcoded semantic templates in
`pipeline.py` and take a known-good IR shape without a model call: four in the
development split, three in the held-out split. They still pass through the
compiler and the relation whitelist, but they do not exercise translation.
Removing them from the development split leaves the pipeline near 62.9% against
the 41.9% baseline.

Removing the verifier scores *higher* on the development split. It stayed in
anyway; the write-up explains that trade.

Raw extracts, credentials, generated figures and local working files are not
committed. Everything needed to regenerate them is.

## License

MIT. See [LICENSE](LICENSE).
