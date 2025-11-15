#!/usr/bin/env python3
"""Backtest 3-renpuku predictions on a venue dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.predict_race import infer_probabilities  # type: ignore
from winticket.models import VenueModelStore
from winticket.features import FEATURE_COLUMNS


METHODS = {
    "top3": "Pick top3 entries by top3_prob",
    "line_top3": "Line-aware selection that ensures strong lines keep their 3人目",
    "heat_top3": "Use heat score (1-top3_prob + momentum) to pick 3 entries",
    "hybrid": "Day1 uses top3, Day2+ uses heat_top3",
}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest 3-renpuku predictions")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--venue", required=True)
    parser.add_argument("--tipster-id", default="dsc-00")
    parser.add_argument("--model-type", choices=["lightgbm", "logistic", "lambdarank"], default="lightgbm")
    parser.add_argument("--method", choices=METHODS.keys(), default="top3")
    parser.add_argument(
        "--line-share-threshold",
        type=float,
        default=0.3,
        help="Minimum line_win_share to auto-include remaining members (line_top3 only)",
    )
    parser.add_argument(
        "--heat-weight",
        type=float,
        default=0.2,
        help="Weight for heat score when using heat_top3",
    )
    parser.add_argument(
        "--context-summary",
        action="store_true",
        help="Print breakdown by day_index and race_phase",
    )
    parser.add_argument("--schedule-date-from")
    parser.add_argument("--schedule-date-to")
    return parser.parse_args(argv)


def load_frame(path: Path, tipster_id: str, start: str | None, end: str | None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[(df["tipster_id"].fillna("") == tipster_id)]
    df = df[~df["is_absent"].fillna(False)]
    if start:
        df = df[df["schedule_date"] >= start]
    if end:
        df = df[df["schedule_date"] <= end]
    df = df[df["finish_order"].notna()].copy()
    if df.empty:
        raise SystemExit("No results available for the given filters.")
    return df


def predict_probs(df: pd.DataFrame, venue: str, model_type: str, tipster_id: str, heat_weight: float) -> pd.DataFrame:
    store = VenueModelStore()
    win_bundle = store.load(venue, target="win", model_type=model_type)
    top3_bundle = store.load(venue, target="top3", model_type=model_type)
    if win_bundle is None or top3_bundle is None:
        raise SystemExit("Missing stored models. Train them before backtesting.")
    win_bundle["tipster_id"] = tipster_id
    top3_bundle["tipster_id"] = tipster_id
    df = df.copy()
    df["win_prob"] = infer_probabilities(df, win_bundle)
    df["top3_prob"] = infer_probabilities(df, top3_bundle)
    df = add_line_features(df)
    df = add_heat_feature(df, heat_weight)
    return df


def add_line_features(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for race_id, race in df.groupby("race_id"):
        race = race.copy()
        line_scores = (
            race.groupby("line_id")[["top3_prob", "win_prob", "line_top3_expectation", "line_win_share"]]
            .sum()
            .rename(columns={
                "top3_prob": "line_top3_prob",
                "win_prob": "line_win_prob",
                "line_top3_expectation": "line_top3_expect",
                "line_win_share": "line_win_share_sum",
            })
        )
        if line_scores["line_top3_prob"].sum() == 0:
            line_scores["line_top3_prob_norm"] = 0.0
        else:
            line_scores["line_top3_prob_norm"] = line_scores["line_top3_prob"] / line_scores["line_top3_prob"].sum()
        race = race.join(line_scores, on="line_id")
        race["line_top3_prob_norm"] = race["line_top3_prob_norm"].fillna(0.0)
        race["line_top3_dom"] = race.groupby("line_id")["top3_prob"].transform(lambda s: s / (s.sum() or 1.0))
        frames.append(race)
    return pd.concat(frames, ignore_index=True)


def add_heat_feature(df: pd.DataFrame, heat_weight: float) -> pd.DataFrame:
    required = [
        "recent_win_rate",
        "recent_top3_rate",
        "line_win_share",
        "line_top3_expectation",
    ]
    for col in required:
        if col not in df.columns:
            df[col] = 0.0

    df["heat_score"] = (
        (1.0 - df["top3_prob"]) \
        + 0.5 * df["recent_win_rate"].fillna(0.0) \
        + 0.3 * df["recent_top3_rate"].fillna(0.0) \
        + 0.4 * df["line_win_share"].fillna(0.0) \
        + 0.1 * df["line_top3_expectation"].fillna(0.0)
    )
    df["line_heat"] = df.groupby(["race_id", "line_id"])['heat_score'].transform('sum')
    df["line_heat_scaled"] = df['line_heat'] / (df.groupby('race_id')['line_heat'].transform('sum') + 1e-6)
    df["heat_score"] = df["heat_score"] + 0.2 * df['line_heat_scaled']
    df["heat_adjusted"] = df["top3_prob"] + heat_weight * df["heat_score"]
    return df


def _select_line_aware(race: pd.DataFrame, threshold: float, context: dict | None = None) -> list[str]:
    race = race.copy()
    race["line_id"] = race["line_id"].fillna(-1.0)
    race["line_win_share"] = race["line_win_share"].fillna(0.0)
    race["line_top3_expectation"] = race["line_top3_expectation"].fillna(0.0)
    sorted_overall = race.sort_values("top3_prob", ascending=False)
    picks = list(sorted_overall.head(3)["player_id"].tolist())
    selected = set(picks)
    line_by_player = race.set_index("player_id")["line_id"].to_dict()
    prob_by_player = race.set_index("player_id")["top3_prob"].to_dict()

    adj_threshold = threshold
    if context:
        day = context.get("day_index")
        if day is not None and day >= 2:
            adj_threshold = min(adj_threshold, 0.25)
        phase = (context.get("race_phase") or "").lower()
        if phase in {"final", "semifinal"}:
            adj_threshold = min(adj_threshold, 0.2)
        elif phase in {"selection", "other"}:
            adj_threshold = min(adj_threshold, 0.28)

    line_stats = (
        race[["line_id", "line_win_share"]]
        .drop_duplicates()
        .sort_values("line_win_share", ascending=False)
    )

    counts = {}
    for pid in picks:
        lid = line_by_player.get(pid, -1.0)
        counts[lid] = counts.get(lid, 0) + 1

    for _, stats in line_stats.iterrows():
        line_id = stats["line_id"]
        share = stats["line_win_share"] or 0.0
        if share < adj_threshold:
            continue
        line_members = race[race["line_id"] == line_id].sort_values("top3_prob", ascending=False)
        if line_members.empty:
            continue
        if counts.get(line_id, 0) < 2:
            continue
        candidate = None
        for _, row in line_members.iterrows():
            pid = row["player_id"]
            if pid not in selected:
                candidate = pid
                break
        if not candidate:
            continue
        removal_candidates = [pid for pid in picks if line_by_player.get(pid) != line_id]
        if not removal_candidates:
            continue
        removal = min(removal_candidates, key=lambda pid: prob_by_player.get(pid, 0.0))
        picks.remove(removal)
        selected.remove(removal)
        removal_line = line_by_player.get(removal, -1.0)
        counts[removal_line] = counts.get(removal_line, 0) - 1
        if counts[removal_line] <= 0:
            counts.pop(removal_line, None)
        picks.append(candidate)
        selected.add(candidate)
        counts[line_id] = counts.get(line_id, 0) + 1

    picks = sorted(picks, key=lambda pid: prob_by_player.get(pid, 0.0), reverse=True)[:3]
    return picks


def select_horses(race: pd.DataFrame, method: str, threshold: float, context: dict | None = None) -> list[str]:
    if method == "top3":
        return race.sort_values("top3_prob", ascending=False).head(3)["player_id"].tolist()
    if method == "line_top3":
        return _select_line_aware(race, threshold, context)
    if method == "heat_top3":
        return race.sort_values("heat_adjusted", ascending=False).head(3)["player_id"].tolist()
    if method == "hybrid":
        if context and context.get("day_index", 1) <= 1:
            return race.sort_values("top3_prob", ascending=False).head(3)["player_id"].tolist()
        return race.sort_values("heat_adjusted", ascending=False).head(3)["player_id"].tolist()
    raise ValueError(f"unsupported method {method}")


def backtest(df: pd.DataFrame, method: str, threshold: float) -> tuple[float, pd.DataFrame]:
    rows = []
    correct = 0
    races = 0
    for race_id, race in df.groupby("race_id"):
        context = {
            "day_index": race.get("day_index").iloc[0] if "day_index" in race else None,
            "race_phase": race.get("race_phase_category").iloc[0] if "race_phase_category" in race else None,
        }
        predicted = list(select_horses(race, method, threshold, context))
        actual = race.loc[race["finish_order"] <= 3, "player_id"].tolist()
        races += 1
        success = set(predicted) == set(actual)
        correct += int(success)
        rows.append(
            {
                "race_id": race_id,
                "schedule_date": race["schedule_date"].iloc[0],
                "day_index": race.get("day_index").iloc[0] if "day_index" in race else None,
                "race_number": race["race_number"].iloc[0],
                "race_phase": race.get("race_phase_category").iloc[0] if "race_phase_category" in race else None,
                "predicted": ",".join(predicted),
                "actual": ",".join(actual),
                "hit": success,
            }
        )
    accuracy = correct / races if races else float("nan")
    return accuracy, pd.DataFrame(rows)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    df = load_frame(args.features, args.tipster_id, args.schedule_date_from, args.schedule_date_to)
    df = predict_probs(df, args.venue, args.model_type, args.tipster_id, args.heat_weight)
    accuracy, details = backtest(df, args.method, args.line_share_threshold)
    print(f"Method {args.method}: {accuracy:.1%} hit rate over {len(details)} races")
    misses = details[~details["hit"]]
    if not misses.empty:
        print("Sample misses:")
        print(misses.head().to_string(index=False))
    if args.context_summary:
        grouped = details.groupby("day_index")["hit"].agg(["mean", "count"]).reset_index()
        if not grouped.empty:
            print("\nBy day_index:")
            print(grouped.to_string(index=False))
        if "race_phase" in details.columns:
            phase_group = details.groupby("race_phase")["hit"].agg(["mean", "count"]).reset_index()
            if not phase_group.empty:
                print("\nBy race_phase:")
                print(phase_group.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
