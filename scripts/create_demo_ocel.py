"""Create a tiny synthetic OCEL parquet dataset for public demos.

The original SAP-style extracts are intentionally not committed. This script
creates a deterministic, non-private replacement with the same three OCEL views
used by the pipeline:

    data/processed/ocel/events.parquet
    data/processed/ocel/objects.parquet
    data/processed/ocel/relations.parquet

It is small enough for smoke tests and API demos. It is not used to reproduce
the saved benchmark metrics, which were computed from the original project
artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "processed" / "ocel"


def _ts(value: str):
    return pd.Timestamp(value)


def build_demo_frames() -> dict[str, pd.DataFrame]:
    """Return deterministic events, objects, and relations demo frames."""
    objects = pd.DataFrame([
        {
            "object_id": "O1001", "object_type": "order_item",
            "FKDAT": pd.NaT, "NETWR": 1200.0, "WAERK": "USD", "ZTERM": "NT30",
            "KUNAG": "C001", "BUKRS": "1000", "AUGDT": pd.NaT,
            "WRBTR": pd.NA, "WAERS": pd.NA, "KUNNR": "C001",
            "XBLNR": "REF-O1001", "cleared": pd.NA,
        },
        {
            "object_id": "O1002", "object_type": "order_item",
            "FKDAT": pd.NaT, "NETWR": 800.0, "WAERK": "USD", "ZTERM": "NT30",
            "KUNAG": "C001", "BUKRS": "1000", "AUGDT": pd.NaT,
            "WRBTR": pd.NA, "WAERS": pd.NA, "KUNNR": "C001",
            "XBLNR": "REF-O1002", "cleared": pd.NA,
        },
        {
            "object_id": "O1003", "object_type": "order_item",
            "FKDAT": pd.NaT, "NETWR": 450.0, "WAERK": "EUR", "ZTERM": "NT60",
            "KUNAG": "C002", "BUKRS": "2000", "AUGDT": pd.NaT,
            "WRBTR": pd.NA, "WAERS": pd.NA, "KUNNR": "C002",
            "XBLNR": "REF-O1003", "cleared": pd.NA,
        },
        {
            "object_id": "O1004", "object_type": "order_item",
            "FKDAT": pd.NaT, "NETWR": 300.0, "WAERK": "EUR", "ZTERM": pd.NA,
            "KUNAG": "C002", "BUKRS": "2000", "AUGDT": pd.NaT,
            "WRBTR": pd.NA, "WAERS": pd.NA, "KUNNR": "C002",
            "XBLNR": "REF-O1004", "cleared": pd.NA,
        },
        {
            "object_id": "C001", "object_type": "customer",
            "FKDAT": pd.NaT, "NETWR": pd.NA, "WAERK": pd.NA, "ZTERM": "NT30",
            "KUNAG": pd.NA, "BUKRS": "1000", "AUGDT": pd.NaT,
            "WRBTR": pd.NA, "WAERS": pd.NA, "KUNNR": "C001",
            "XBLNR": pd.NA, "cleared": pd.NA,
        },
        {
            "object_id": "C002", "object_type": "customer",
            "FKDAT": pd.NaT, "NETWR": pd.NA, "WAERK": pd.NA, "ZTERM": "NT60",
            "KUNAG": pd.NA, "BUKRS": "2000", "AUGDT": pd.NaT,
            "WRBTR": pd.NA, "WAERS": pd.NA, "KUNNR": "C002",
            "XBLNR": pd.NA, "cleared": pd.NA,
        },
        {
            "object_id": "D1001", "object_type": "delivery_item",
            "FKDAT": pd.NaT, "NETWR": pd.NA, "WAERK": pd.NA, "ZTERM": pd.NA,
            "KUNAG": pd.NA, "BUKRS": "1000", "AUGDT": pd.NaT,
            "WRBTR": pd.NA, "WAERS": pd.NA, "KUNNR": pd.NA,
            "XBLNR": "REF-D1001", "cleared": pd.NA,
        },
        {
            "object_id": "D1002", "object_type": "delivery_item",
            "FKDAT": pd.NaT, "NETWR": pd.NA, "WAERK": pd.NA, "ZTERM": pd.NA,
            "KUNAG": pd.NA, "BUKRS": "1000", "AUGDT": pd.NaT,
            "WRBTR": pd.NA, "WAERS": pd.NA, "KUNNR": pd.NA,
            "XBLNR": "REF-D1002", "cleared": pd.NA,
        },
        {
            "object_id": "B1001", "object_type": "billing_doc",
            "FKDAT": _ts("2010-05-10"), "NETWR": 1200.0, "WAERK": "USD",
            "ZTERM": "NT30", "KUNAG": "C001", "BUKRS": "1000",
            "AUGDT": pd.NaT, "WRBTR": pd.NA, "WAERS": "USD",
            "KUNNR": "C001", "XBLNR": "REF-B1001", "cleared": pd.NA,
        },
        {
            "object_id": "B1002", "object_type": "billing_doc",
            "FKDAT": _ts("2010-05-18"), "NETWR": 800.0, "WAERK": "USD",
            "ZTERM": "NT30", "KUNAG": "C001", "BUKRS": "1000",
            "AUGDT": pd.NaT, "WRBTR": pd.NA, "WAERS": "USD",
            "KUNNR": "C001", "XBLNR": "REF-B1002", "cleared": pd.NA,
        },
        {
            "object_id": "B1003", "object_type": "billing_doc",
            "FKDAT": _ts("2011-02-10"), "NETWR": 450.0, "WAERK": "EUR",
            "ZTERM": pd.NA, "KUNAG": "C002", "BUKRS": "2000",
            "AUGDT": pd.NaT, "WRBTR": pd.NA, "WAERS": "EUR",
            "KUNNR": "C002", "XBLNR": "REF-B1003", "cleared": pd.NA,
        },
        {
            "object_id": "AR1001", "object_type": "ar_item",
            "FKDAT": pd.NaT, "NETWR": pd.NA, "WAERK": pd.NA, "ZTERM": "NT30",
            "KUNAG": "C001", "BUKRS": "1000", "AUGDT": _ts("2010-06-02"),
            "WRBTR": 1200.0, "WAERS": "USD", "KUNNR": "C001",
            "XBLNR": "REF-AR1001", "cleared": True,
        },
        {
            "object_id": "AR1002", "object_type": "ar_item",
            "FKDAT": pd.NaT, "NETWR": pd.NA, "WAERK": pd.NA, "ZTERM": "NT30",
            "KUNAG": "C001", "BUKRS": "1000", "AUGDT": pd.NaT,
            "WRBTR": 800.0, "WAERS": "USD", "KUNNR": "C001",
            "XBLNR": "REF-AR1002", "cleared": False,
        },
        {
            "object_id": "AR1003", "object_type": "ar_item",
            "FKDAT": pd.NaT, "NETWR": pd.NA, "WAERK": pd.NA, "ZTERM": "NT60",
            "KUNAG": "C002", "BUKRS": "2000", "AUGDT": _ts("2011-04-20"),
            "WRBTR": 450.0, "WAERS": "EUR", "KUNNR": "C002",
            "XBLNR": "REF-AR1003", "cleared": True,
        },
    ])

    events = pd.DataFrame([
        ("E001", "order_created", "2010-05-01", "O1001", "order_item", "DEMO"),
        ("E002", "order_created", "2010-05-03", "O1002", "order_item", "DEMO"),
        ("E003", "order_created", "2011-02-01", "O1003", "order_item", "DEMO"),
        ("E004", "order_created", "2011-03-01", "O1004", "order_item", "DEMO"),
        ("E005", "delivery_created", "2010-05-05", "D1001", "delivery_item", "DEMO"),
        ("E006", "delivery_created", "2010-05-08", "D1002", "delivery_item", "DEMO"),
        ("E007", "goods_issue", "2010-05-06", "D1001", "delivery_item", "DEMO"),
        ("E008", "billing_created", "2010-05-10", "B1001", "billing_doc", "DEMO"),
        ("E009", "billing_created", "2010-05-18", "B1002", "billing_doc", "DEMO"),
        ("E010", "billing_created", "2011-02-10", "B1003", "billing_doc", "DEMO"),
        ("E011", "payment_clearing", "2010-06-02", "AR1001", "ar_item", "DEMO"),
        ("E012", "payment_clearing", "2011-04-20", "AR1003", "ar_item", "DEMO"),
        ("E013", "dunning_raised", "2010-06-15", "C001", "customer", "DEMO"),
        ("E014", "pricing_condition", "2010-05-02", "O1001", "order_item", "DEMO"),
    ], columns=["event_id", "event_type", "timestamp", "object_id", "object_type", "source_table"])
    events["timestamp"] = pd.to_datetime(events["timestamp"])

    relations = pd.DataFrame([
        ("O1001", "order_item", "C001", "customer", "order_to_customer", "DEMO"),
        ("O1002", "order_item", "C001", "customer", "order_to_customer", "DEMO"),
        ("O1003", "order_item", "C002", "customer", "order_to_customer", "DEMO"),
        ("O1004", "order_item", "C002", "customer", "order_to_customer", "DEMO"),
        ("O1001", "order_item", "D1001", "delivery_item", "order_to_delivery", "DEMO"),
        ("O1002", "order_item", "D1002", "delivery_item", "order_to_delivery", "DEMO"),
        ("O1001", "order_item", "B1001", "billing_doc", "order_to_billing", "DEMO"),
        ("O1002", "order_item", "B1002", "billing_doc", "order_to_billing", "DEMO"),
        ("O1003", "order_item", "B1003", "billing_doc", "order_to_billing", "DEMO"),
        ("D1001", "delivery_item", "B1001", "billing_doc", "delivery_to_billing", "DEMO"),
        ("D1002", "delivery_item", "B1002", "billing_doc", "delivery_to_billing", "DEMO"),
        ("B1001", "billing_doc", "AR1001", "ar_item", "billing_to_ar", "DEMO"),
        ("B1002", "billing_doc", "AR1002", "ar_item", "billing_to_ar", "DEMO"),
        ("B1003", "billing_doc", "AR1003", "ar_item", "billing_to_ar", "DEMO"),
        ("O1001", "order_item", "M001", "material", "order_to_material", "DEMO"),
        ("O1002", "order_item", "M002", "material", "order_to_material", "DEMO"),
        ("O1001", "order_item", "C001", "customer", "order_to_payer", "DEMO"),
    ], columns=[
        "from_object_id", "from_object_type", "to_object_id", "to_object_type",
        "relation_type", "source_table",
    ])

    return {"events": events, "objects": objects, "relations": relations}


def write_demo_ocel(out_dir: Path, overwrite: bool = False) -> None:
    targets = [out_dir / f"{name}.parquet" for name in ("events", "objects", "relations")]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path.relative_to(ROOT)) for path in existing)
        raise SystemExit(
            f"Refusing to overwrite existing OCEL files: {names}. "
            "Pass --overwrite if you intentionally want to replace them."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = build_demo_frames()
    for name, frame in frames.items():
        frame.to_parquet(out_dir / f"{name}.parquet", index=False)
        print(f"wrote {out_dir / f'{name}.parquet'} ({len(frame):,} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    write_demo_ocel(args.out, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
