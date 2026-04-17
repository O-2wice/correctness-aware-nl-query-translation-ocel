# SAP Build Apps Demo Setup

This README is the click-by-click guide for connecting the SAP Build Apps demo app to this project's backend.

The SAP app is business-facing. The recommended app name is:

```text
O2C Insight Assistant
```

The backend remains the project API in `app/api.py`.

```text
SAP Build Apps
    -> Universal REST API Integration
    -> HTTPS tunnel or deployed API URL
    -> app/api.py
    -> correctness-aware NL-to-SQL pipeline
    -> SAP O2C event-log data
```

## Connection Choice

Use REST for the main project demo.

OData is best for stable business entities such as sales orders, business partners, and products:

```text
GET SalesOrders
GET BusinessPartners
FILTER customers
UPDATE one order
```

This project is not a normal CRUD entity app. It is an analytics action:

```text
POST /query
{
  "question": "What is the total number of billing creation events?"
}
```

The response is dynamic and includes generated SQL, result rows, verifier state, latency, and provenance. That contract fits REST better than OData.

## Part 1 - Start The Backend API

Open VS Code in this repository.

Open a terminal and run:

```powershell
.\.venv\Scripts\Activate.ps1
```

If using DeepSeek, set the key:

```powershell
$env:DEEPSEEK_API_KEY = Read-Host "DeepSeek API key"
```

Start the API:

```powershell
.\scripts\start_api.ps1
```

Leave this terminal running. Press `CTRL+C` in this same terminal to stop it.

You should see:

```text
Starting NL-to-SQL API on http://0.0.0.0:8000
Running on http://127.0.0.1:8000
Running on http://192.168.0.196:8000
```

Those two visible URLs mean:

```text
http://127.0.0.1:8000      local test URL on this laptop
http://192.168.0.196:8000  local Wi-Fi/LAN URL for devices on the same network
```

SAP Build Apps uses the public HTTPS tunnel or deployed URL from Part 2.

In a second terminal, test it:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected:

```text
status   pipeline
------   --------
ok       correctness-aware NL-to-SQL v1.0
```

Browser checks:

```text
http://127.0.0.1:8000
```

Shows the API name and available endpoints.

```text
http://127.0.0.1:8000/health
```

Shows `status: ok`.

```text
http://127.0.0.1:8000/examples
```

Shows the demo question groups.

## Part 2 - Expose The Local API To SAP Build Apps

SAP Build Apps reaches the backend through a public HTTPS URL.

For a fast demo, use VS Code Port Forwarding.

1. Keep the API terminal running.
2. In VS Code, open the bottom panel.
3. Click the `PORTS` tab.
4. Click `Forward a Port` or the `+` button.
5. Enter:

```text
8000
```

6. If VS Code asks to sign in with GitHub for `Local Tunnel Port Forwarding`, click `Allow`.
7. Complete the GitHub sign-in in the browser.
8. Return to VS Code.
9. In the `PORTS` tab, find port `8000`.
10. Right-click the row for port `8000`.
11. Click `Port Visibility`.
12. Click `Public`.
13. Copy the `Forwarded Address`.

It should look like:

```text
https://something-8000.devtunnels.ms
```

Test it in a browser:

```text
https://something-8000.devtunnels.ms/health
```

If it returns `status: ok`, this is your temporary API base URL.

Optional label cleanup:

```text
Right-click port 8000 -> Set Port Label / Rename
```

Use:

```text
O2C Pipeline API
```

This label is only for your VS Code display. The generated tunnel URL will still look like `https://...devtunnels.ms`. In SAP Build Apps, use the data entity name `O2C Pipeline API`.

If `Port Visibility` is not visible, make the `PORTS` panel wider or right-click directly on the `8000` row, not the empty space. If `Public` is missing, sign in again through the VS Code prompt and retry.

Alternative with ngrok:

```powershell
winget install --id Ngrok.Ngrok -e
ngrok http 8000
```

Copy the HTTPS forwarding URL and test:

```text
https://your-ngrok-url.ngrok-free.app/health
```

For the final version, deploy `app/api.py` to a real HTTPS host instead of using a tunnel.

## Part 3 - Create The SAP Build Apps Project

Open:

```text
https://uccsandbox1-3mg3ufj8.eu10.build.cloud.sap/
```

If asked, choose:

```text
UCC Magdeburg Identity Management
```

From the lobby:

1. Click `Create`.
2. Choose `Web and Mobile Application`.
3. Project name:

```text
O2C Insight Assistant
```

4. Description:

```text
Ask trusted questions about orders, deliveries, billing, and payment clearing.
```

5. Click `Create`.

## Part 4 - Add The REST Integration In SAP Build Apps

Inside the new app:

1. Click `Integrations` in the top menu.
2. Scroll to `SAP Build Apps classic data entities`.
3. Click `Create Data Entity`.
4. Choose `Universal REST API Integration`.
5. Fill the `Base` tab exactly like this:

```text
Resource ID: O2CPipelineAPI
Short description: Backend API for trusted order-to-cash process questions.
Resource URL: https://something-8000.devtunnels.ms
```

Use your real public tunnel or deployed API URL. Do not add `/health` or `/query` to the `Resource URL`.

6. Leave these base-tab sections empty for now if they appear:

```text
Path variables
Common request headers
Additional inputs
```

The JSON header and request body are configured later on the `Create Record (POST)` method, not on this first base form.

7. Click `Save`.

Official SAP Build Apps direct REST reference:

- https://help.sap.com/docs/build-apps/service-guide/rest-api-direct-integration

## Part 5 - Configure REST Methods

Configure the methods on the `O2C Pipeline API` data entity.

### Method 1 - Health Check

Enable the GET-style method shown by your SAP Build Apps screen. Depending on the UI wording, it may be called `retrieve`, `list`, or `Get record`.

Use:

```text
Name: Pipeline Health
Method: GET
Relative path: /health
```

Test URL:

```text
https://something-8000.devtunnels.ms/health
```

Expected response fields:

```text
status: text
pipeline: text
```

### Method 2 - Example Questions

Use:

```text
Name: Example Questions
Method: GET
Relative path: /examples
```

Expected response field:

```text
examples: object
```

### Method 3 - Schema Catalog

This is optional but useful for the demo.

Use:

```text
Name: Schema Catalog
Method: GET
Relative path: /schema
```

Expected response fields:

```text
schema_catalog: object
relation_whitelist: object
```

### Method 4 - Query Answer

This is the main endpoint.

Use:

```text
Name: Query Answer
Method: POST
Relative path: /query
```

Request body schema:

```text
question: text
```

Response fields:

```text
question: text
status: text
answer_text: text
sql: text
intent: text
columns: list of text
rows: list
chart_available: true/false
chart_type: text
chart_text: text
chart_labels: list of text
chart_values: list of number
chart_points: list
n_rows: number
latency_s: number
ir_attempts: number
verify_errors: list
exec_error: text
provenance: object
error_explanation: text
```

If SAP Build Apps asks for sample data, use:

```json
{
  "question": "What is the total number of billing creation events?"
}
```

Expected successful response shape:

```json
{
  "question": "What is the total number of billing creation events?",
  "status": "accept",
  "sql": "SELECT COUNT(*) AS n FROM events WHERE event_type = 'billing_created'",
  "intent": "count_filter",
  "columns": ["n"],
  "rows": [[34990]],
  "n_rows": 1,
  "latency_s": 5.2,
  "ir_attempts": 1,
  "verify_errors": [],
  "exec_error": null,
  "provenance": {
    "tables_used": ["events"],
    "joins_used": [],
    "intent": "count_filter",
    "filters_count": 2,
    "result_rows": 1,
    "grounding": {
      "grounding_status": "events",
      "n_source_events": 34990,
      "n_source_objects": 0,
      "n_source_relations": 0,
      "provenance_query": "SELECT ..."
    }
  },
  "error_explanation": null
}
```

## Part 6 - Create App Variables

Click:

```text
Variables
```

Create app variables:

```text
question
Type: Text
Default: What is the total number of billing creation events?
```

```text
queryResult
Type: Object
Default: empty object
```

```text
isLoading
Type: True/false
Default: false
```

```text
errorMessage
Type: Text
Default: empty
```

## Part 7 - Build The User Interface

Click:

```text
User Interface
```

On the first page, build this layout.

### Header

Add a title text:

```text
O2C Insight Assistant
```

Add a subtitle text:

```text
Ask trusted questions about orders, deliveries, billing, and payment clearing.
```

### Question Input

Add an input field.

Label:

```text
Ask a question
```

Bind input value to:

```text
App variable: question
```

### Run Button

Add a button.

Text:

```text
Run Analysis
```

Button logic:

1. Set app variable `isLoading` to `true`.
2. Set app variable `errorMessage` to empty text.
3. Call the `create` operation on data entity `O2C Pipeline API`.
4. Use the create record input that SAP Build Apps exposes.

In the current Universal REST API setup, SAP may expose the create input as `id`.
That is okay. The backend accepts `id` and treats it as the natural-language
question.

Use:

```json
{
  "id": "appVars.question"
}
```

If your SAP Build Apps screen exposes a `question` input instead of `id`, use:

```json
{
  "question": "appVars.question"
}
```

5. Store response in app variable `queryResult`.
6. Set app variable `isLoading` to `false`.

If the call fails:

1. Set `isLoading` to `false`.
2. Set `errorMessage` to the API or network error.

### Result Section

Add text fields bound to:

```text
Answer: queryResult.answer_text
Status: queryResult.status
Intent: queryResult.intent
Rows returned: queryResult.n_rows
Latency: queryResult.latency_s
IR attempts: queryResult.ir_attempts
```

For trend or ranking questions, add a simple trend section first:

```text
Trend type: queryResult.chart_type
Trend values: queryResult.chart_text
```

The backend sets `queryResult.chart_available` to `true` when the result has a label/value shape that can be charted. Example trend questions include yearly billing creation counts, monthly billing counts, customer ranking, and running totals.

If you add a chart component later, bind it to:

```text
Labels: queryResult.chart_labels
Values: queryResult.chart_values
Series/list points: queryResult.chart_points
```

For a table-style fallback, add a text field bound to:

```text
queryResult.answer_text
```

Then refine the visual component once the API call is working.

### SQL Section

Add a collapsible/secondary panel titled:

```text
Generated SQL
```

Bind the text to:

```text
queryResult.sql
```

### Provenance Section

Add a collapsible/secondary panel titled:

```text
Audit Evidence
```

Bind the text to:

```text
queryResult.provenance
```

### Error Banner

Add a warning text/banner.

Show it only when:

```text
queryResult.status is not "accept"
```

Text:

```text
queryResult.error_explanation
```

## Part 8 - Save And Preview

Click:

```text
Save
```

Then click:

```text
Preview
```

In preview, ask:

```text
What is the total number of billing creation events?
```

Expected business result:

```text
34990
```

Expected technical result:

```text
status = accept
intent = count_filter
sql is visible
provenance is visible
```

## How To Use The Demo

Before presenting:

1. Start the backend:

```powershell
.\.venv\Scripts\Activate.ps1
.\scripts\start_api.ps1
```

2. Start or refresh the public tunnel for port `8000`.
3. Test:

```text
https://your-public-url/health
```

4. If the public URL changed, update the `Base API URL` in the `O2C Pipeline API` data entity.
5. Open SAP Build Apps.
6. Open `O2C Insight Assistant`.
7. Click `Preview`.
8. Ask one of the demo questions.

Good demo questions:

```text
Starter questions:
What is the total number of billing creation events?
Count invoice creation events.
How many payment clearing events are recorded?
What are the top 5 event types by frequency?
Which order items have delivery before billing?
Which cases have billing but no payment clearing?

Medium process questions:
Show yearly billing creation event counts.
How many billing documents are linked to AR items?
How many order items are linked to both delivery and billing documents?
How many order items are linked to a customer that received a dunning notice?
What is the average number of days from billing creation to payment clearing?
On average, how long does cash clearing take after billing?
How many dunning notices were raised per month in 2006?

Hard audit questions:
What is the 90th percentile of days between billing and payment clearing for cleared invoices?
What is the average time from order creation to payment clearing across the full order-to-cash chain?
What is the average billing-to-payment clearing delay for USD billing documents?
How many billing-to-payment cases take more than 873 days?
How many order items had no delivery event within 90 days of order creation?
How many billing documents have no corresponding payment clearing event?
How many order items were billed but never delivered?
How many order items have only an order creation event and no downstream activity?
How many customers have more than the average number of linked order items?
Which years had more billing creation events than 2004?
Which months in 2005 had billing creation counts above the monthly average?
Rank event types by total count in descending order.
For each customer, show the number of linked order items and rank customers by volume.
For each month in 2005, show billing creation count as a share of the yearly total.
```

When explaining the demo, say:

```text
This SAP Build Apps frontend lets a business user ask natural-language questions about the O2C process. The app sends the question to a REST API. The backend translates the question into a constrained query plan, verifies it against the allowed schema and relation rules, compiles it to SQL, executes it over the O2C event-log data, and returns the answer with SQL and provenance for auditability.
```

For the dunning-customer question, the backend uses a benchmark-backed semantic template. The generated SQL must include `EXISTS`, `order_to_customer`, and `dunning_raised`; otherwise the query is only counting order items linked to customers and is not strict enough.

## Troubleshooting

### `ngrok is not recognized`

Use VS Code `PORTS` forwarding, or install ngrok:

```powershell
winget install --id Ngrok.Ngrok -e
```

### SAP Build Apps cannot call the API

Check these in order:

1. `.\scripts\start_api.ps1` is still running.
2. The public URL opens `/health`.
3. The `O2C Pipeline API` data entity `Base API URL` matches the current tunnel or deployment URL.
4. The method relative paths are `/health`, `/examples`, `/schema`, and `/query`.
5. The `Content-Type` request header is `application/json`.

### Local browser still shows 404 or no change

Check whether an old Python server is still holding port `8000`:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue |
  Select-Object LocalAddress,LocalPort,OwningProcess,State
```

Stop the project API processes:

```powershell
.\scripts\stop_api.ps1
```

Start the API again:

```powershell
.\scripts\start_api.ps1
```

Then test:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/health
```

### `/query` returns a model or pipeline error

Check the model backend:

```powershell
$env:DEEPSEEK_API_KEY
```

If using a local backend, set the backend before starting the API:

```powershell
$env:NL2OCEL_BACKEND = "ollama"
.\scripts\start_api.ps1
```

For cloud backends, set `NL2OCEL_BACKEND` plus the matching API key:

```powershell
$env:NL2OCEL_BACKEND = "openai"      # or "anthropic" / "deepseek"
$env:OPENAI_API_KEY = Read-Host "OpenAI API key"
.\scripts\start_api.ps1
```

### The tunnel URL changed

This is normal for temporary tunnels.

Update:

```text
SAP Build Apps -> Integrations -> O2C Pipeline API -> Base API URL
```

Then save the data entity and run the `/health` test again.
