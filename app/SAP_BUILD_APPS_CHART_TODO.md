# SAP Build Apps Chart And Trend TODO

This note parks the chart work so the current SAP demo can be tested first.

The backend already returns chart-ready fields for trend, ranking, and grouped result questions. The SAP UI can use the text fallback first, then later upgrade to a visual chart component.

## Backend Fields Already Available

```text
queryResult.chart_available
queryResult.chart_type
queryResult.chart_x
queryResult.chart_y
queryResult.chart_labels
queryResult.chart_values
queryResult.chart_points
queryResult.chart_text
```

## Questions That Should Produce Chart Data

```text
Show yearly billing creation event counts.
How many dunning notices were raised per month in 2006?
Show the year-over-year change in payment clearing event counts.
Rank event types by total count in descending order.
For each customer, show the number of linked order items and rank customers by volume.
For each month in 2005, show billing creation count as a share of the yearly total.
```

## Phase 1 - Simple Trend Text

1. Restart the API.
2. In SAP Build Apps, ask a trend question such as:

```text
Show yearly billing creation event counts.
```

3. If SAP autodetect does not add them, manually add these properties under app variable `queryResult`:

```text
chart_available: true/false
chart_type: text
chart_text: text
chart_labels: list of texts
chart_values: list of numbers
chart_points: list
```

4. Add a UI section titled:

```text
Trend
```

5. Add a Text component and bind it to:

```text
queryResult.chart_text
```

Expected display shape:

```text
2004: 1234
2005: 5678
2006: 9012
```

## Phase 2 - Visual Chart

After the main query demo is stable, add a chart component and bind:

```text
Labels: queryResult.chart_labels
Values: queryResult.chart_values
```

Use `queryResult.chart_type` to decide whether to display a line chart or bar chart:

```text
line = trends over year/month/time
bar = rankings and category counts
```

## Current Decision

Do not block the demo on visual charts. First verify:

```text
Question input -> Run Analysis -> Answer -> Status -> SQL -> audit/provenance
```

Then come back to this chart TODO.
