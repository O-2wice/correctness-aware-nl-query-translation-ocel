"""Shared demo questions for the Streamlit and Flask demo surfaces.

Keep these examples benchmark-grounded so both demos exercise the current
120-question query surface instead of old or empty-year examples.
"""

EXAMPLES: dict[str, list[str]] = {
    "count_filter": [
        "What is the total number of billing creation events?",
        "Count invoice creation events.",
        "How many pricing condition events are in the log?",
        "How many payment clearing events are recorded?",
    ],
    "group_topk": [
        "What are the top 5 event types by frequency?",
        "How many objects do we have per object type?",
        "Which object types appear most often in the process data?",
    ],
    "temporal_trend": [
        "Show yearly billing creation event counts.",
        "How many dunning notices were raised per month in 2006?",
        "Show the year-over-year change in payment clearing event counts.",
    ],
    "path_relation": [
        "How many billing documents are linked to AR items?",
        "How many order items are linked to both delivery and billing documents?",
        "How many order items are linked to a customer that received a dunning notice?",
        "Which order items have delivery before billing?",
    ],
    "delay_analysis": [
        "What is the average number of days from billing creation to payment clearing?",
        "On average, how long does cash clearing take after billing?",
        "What is the 90th percentile of days between billing and payment clearing for cleared invoices?",
        "What is the average billing-to-payment clearing delay for USD billing documents?",
        "What is the average time from order creation to payment clearing across the full order-to-cash chain?",
    ],
    "anomaly_filter": [
        "How many billing-to-payment cases take more than 873 days?",
        "How many AR items were cleared within 7 days of their billing event?",
        "How many order items had no delivery event within 90 days of order creation?",
        "How many customers have more than 3 linked order items?",
    ],
    "conformance": [
        "How many order items never had a goods issue event?",
        "How many billing documents have no corresponding payment clearing event?",
        "Which cases have billing but no payment clearing?",
        "How many order items were billed but never delivered?",
        "How many order items have only an order creation event and no downstream activity?",
    ],
    "nested_agg": [
        "How many event types had at least as many events as the median event-type count?",
        "Which event types had above-average counts in 2005?",
        "How many customers have more than the average number of linked order items?",
        "Which months in 2005 had billing creation counts above the monthly average?",
        "Which years had more billing creation events than 2004?",
    ],
    "window_agg": [
        "Rank event types by total count in descending order.",
        "Show the running total of billing creation events per month in 2005.",
        "For each customer, show the number of linked order items and rank customers by volume.",
        "For each month in 2005, show billing creation count as a share of the yearly total.",
    ],
}


STREAMLIT_EXAMPLES: list[str] = [
    EXAMPLES["count_filter"][0],
    EXAMPLES["temporal_trend"][0],
    EXAMPLES["path_relation"][0],
    EXAMPLES["delay_analysis"][0],
    EXAMPLES["anomaly_filter"][0],
    EXAMPLES["conformance"][0],
    EXAMPLES["nested_agg"][0],
    EXAMPLES["window_agg"][0],
    EXAMPLES["group_topk"][0],
    EXAMPLES["temporal_trend"][1],
]
