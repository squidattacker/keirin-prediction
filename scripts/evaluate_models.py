#!/usr/bin/env python3
"""Evaluate venue-wide model metrics with optional condition breakdowns."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import predict_race as predictor  # type: ignore
from winticket.models import VenueModelStore


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate stored models on a feature parquet file")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--venue", required=True)
    parser.add_argument("--tipster-id", default="dsc-00")
    parser.add_argument("--model-type", choices=["lightgbm", "logistic", "lambdarank"], default="lightgbm")
    parser.add_argument("--schedule-date-from")
    parser.add_argument("--schedule-date-to")
    parser.add_argument("--output", type=Path, help="Optional path to save metrics CSV")
    return parser.parse_args(argv)


def load_frame(path: Path, tipster_id: str, start: str | None, end: str | None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[df["tipster_id"].fillna("") == tipster_id]
    if start:
        df = df[df["schedule_date"] >= start]
    if end:
        df = df[df["schedule_date"] <= end]
    df = df[~df["is_absent"].fillna(False)]
    return df


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    df = load_frame(args.features, args.tipster_id, args.schedule_date_from, args.schedule_date_to)
    if df.empty:
        raise SystemExit("No features matched the provided filters.")

    store = VenueModelStore()
    win_bundle = store.load(args.venue, target="win", model_type=args.model_type)
    top3_bundle = store.load(args.venue, target="top3", model_type=args.model_type)
    if win_bundle is None or top3_bundle is None:
        raise SystemExit("Missing stored models. Run training first.")
    win_bundle["tipster_id"] = args.tipster_id
    top3_bundle["tipster_id"] = args.tipster_id

    metrics = predictor.evaluate_hit_rates(df, win_bundle, top3_bundle)
    metrics_rows = pd.DataFrame([
        {"metric": key, "value": value} for key, value in metrics.items()
    ])
    print(metrics_rows.to_string(index=False))
    if args.output:
        metrics_rows.to_csv(args.output, index=False)
        print(f"saved metrics -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
