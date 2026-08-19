# REST / No-Code Frontend Integration

The Flask API in `app/api.py` exposes the correctness-aware NL-to-SQL pipeline
through ordinary JSON endpoints. It can be used from SAP Build Apps, Retool,
Power Apps, a custom web frontend, or any client that can call REST APIs.

```text
frontend
  -> POST /query
  -> app/api.py
  -> typed IR + verifier + SQL compiler
  -> DuckDB OCEL views
  -> answer, SQL, status, chart payload, provenance
```

## Start The API

Create the demo OCEL files first if you are running without the original local
data:

```powershell
.\.venv\Scripts\python scripts\create_demo_ocel.py --overwrite
```

For a cloud backend, set one provider key:

```powershell
$env:NL2OCEL_BACKEND = "deepseek"
$env:DEEPSEEK_API_KEY = Read-Host "DeepSeek API key"
.\scripts\start_api.ps1
```

For a local Ollama backend:

```powershell
$env:NL2OCEL_BACKEND = "ollama"
ollama serve
.\scripts\start_api.ps1
```

## Endpoints

```text
GET  /health
GET  /examples
GET  /schema
POST /query
GET  /query?question=...
GET  /query/<question>
```

Example request:

```powershell
curl -X POST http://localhost:8000/query `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"How many order items are linked to a customer that received a dunning notice?\"}"
```

Example response shape:

```json
{
  "answer_text": "n_orders_with_dunning_customer: 2",
  "status": "accept",
  "sql": "SELECT ...",
  "intent": "path_relation",
  "columns": ["n_orders_with_dunning_customer"],
  "rows": [[2]],
  "chart_available": false,
  "latency_s": 0.12,
  "ir_attempts": 0,
  "verify_errors": [],
  "provenance": {
    "tables_used": ["objects", "relations", "events"],
    "joins_used": ["order_to_customer"]
  }
}
```

## Frontend Binding Notes

Use `answer_text` for a simple first UI. Then add tabs or panels for:

```text
status
sql
intent
verify_errors_text
tables_used_text
joins_used_text
error_explanation_text
```

For grouped or trend answers, the API also returns chart-friendly fields:

```text
chart_available
chart_type
chart_x
chart_y
chart_labels
chart_values
chart_points
chart_text
```

The frontend should treat any non-`accept` status as a visible rejection or
execution failure, not as an empty answer.
