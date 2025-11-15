#!/usr/bin/env python3
"""Predict win and top3 probabilities for a given race and report hit rates."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Iterable, Tuple

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from winticket.features import FEATURE_COLUMNS
from winticket.models import VenueModelStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict win/top3 probabilities and report current hit rates.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/winticket_lake/gold/features/komatsushima_2025111273.parquet"),
        help="Path to gold feature parquet file",
    )
    parser.add_argument("--venue", required=True, help="Venue slug (e.g. komatsushima)")
    parser.add_argument("--tipster-id", default="dsc-00")
    parser.add_argument("--race-number", type=int, help="Race number to show (defaults to first race found)")
    parser.add_argument("--day-index", type=int, help="Day index filter")
    parser.add_argument("--schedule-date", help="Schedule date (YYYYMMDD) filter")
    parser.add_argument("--model-type", choices=["lightgbm", "logistic", "lambdarank"], default="lightgbm")
    return parser.parse_args()


def load_bundle(venue: str, target: str, model_type: str):
    store = VenueModelStore()
    bundle = store.load(venue, target=target, model_type=model_type)
    if bundle is None:
        raise SystemExit(f"Missing model for {venue} target={target} model_type={model_type}")
    return bundle


def prepare_frame(df: pd.DataFrame, tipster_id: str, day_index: int | None, schedule_date: str | None,
                  race_number: int | None) -> Tuple[pd.DataFrame, int]:
    frame = df.copy()
    frame = frame[frame["tipster_id"].fillna("") == tipster_id]
    frame = frame[~frame["is_absent"].fillna(False)]
    if schedule_date:
        frame = frame[frame["schedule_date"] == schedule_date]
    if day_index is not None:
        frame = frame[frame["day_index"] == day_index]
    if frame.empty:
        raise SystemExit("No entries remaining after filters.")
    if race_number is None:
        ordered = frame.sort_values(["schedule_date", "race_number"], ascending=[False, True])
        race_number = int(ordered.iloc[0]["race_number"])
    race_frame = frame[frame["race_number"] == race_number]
    if race_frame.empty:
        raise SystemExit(f"Race number {race_number} not found with the given filters.")
    return race_frame, race_number


def ensure_features(frame: pd.DataFrame, feature_names: Iterable[str]) -> pd.DataFrame:
    for col in feature_names:
        if col not in frame.columns:
            frame[col] = 0.0
    return frame[list(feature_names)].fillna(0.0)


def _softmax_group(scores: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
    probs = np.zeros_like(scores, dtype=float)
    for rid in np.unique(race_ids):
        mask = race_ids == rid
        subset = scores[mask]
        shifted = subset - subset.max()
        exp = np.exp(shifted)
        denom = exp.sum()
        if denom <= 0:
            probs[mask] = 1.0 / len(subset)
        else:
            probs[mask] = exp / denom
    return probs


def infer_probabilities(frame: pd.DataFrame, bundle: Dict) -> np.ndarray:
    features = ensure_features(frame, bundle.get("feature_names", FEATURE_COLUMNS))
    estimator = bundle["estimator"]
    model_type = bundle.get("model_type")
    race_ids = frame["race_id"].to_numpy()
    if bundle.get("is_ranker") or model_type == "lambdarank":
        scores = estimator.predict(features)
        scores = np.asarray(scores, dtype=float)
        return _softmax_group(scores, race_ids)
    proba = estimator.predict_proba(features)
    if proba.ndim == 1:
        return proba
    return proba[:, 1]


def predict(frame: pd.DataFrame, win_bundle: Dict, top3_bundle: Dict) -> pd.DataFrame:
    out = frame.copy()
    out["win_prob"] = infer_probabilities(frame, win_bundle)
    out["top3_prob"] = infer_probabilities(frame, top3_bundle)
    return out


def evaluate_hit_rates(df: pd.DataFrame, win_bundle: Dict, top3_bundle: Dict) -> Dict[str, float]:
    eval_df = df[df["finish_order"].notna()].copy()
    if eval_df.empty:
        return {"win_hit_rate": float("nan"), "top3_coverage": float("nan")}
    eval_df = eval_df[eval_df["tipster_id"].fillna("") == win_bundle.get("tipster_id", "")]
    eval_df["win_prob"] = infer_probabilities(eval_df, win_bundle)
    eval_df["top3_prob"] = infer_probabilities(eval_df, top3_bundle)

    win_actual = (eval_df["finish_order"].astype(float) == 1).astype(int)
    try:
        win_logloss = log_loss(win_actual, eval_df["win_prob"], labels=[0, 1])
    except ValueError:
        win_logloss = float("nan")
    try:
        win_brier = brier_score_loss(win_actual, eval_df["win_prob"])
    except ValueError:
        win_brier = float("nan")

    win_correct = 0
    win_total = 0
    top3_hits = 0
    top3_total = 0

    for _, race in eval_df.groupby("race_id"):
        race = race.sort_values("win_prob", ascending=False)
        actual_winner = race.loc[race["finish_order"].astype(float) == 1, "player_id"].tolist()
        if not race.empty and actual_winner:
            win_total += 1
            if race.iloc[0]["player_id"] in actual_winner:
                win_correct += 1
        race_top3 = race.sort_values("top3_prob", ascending=False).head(3)["player_id"].tolist()
        actual_top3 = race.loc[race["finish_order"].astype(float) <= 3, "player_id"].tolist()
        if actual_top3:
            top3_total += len(actual_top3)
            top3_hits += len(set(race_top3) & set(actual_top3))

    win_hit_rate = win_correct / win_total if win_total else float("nan")
    top3_coverage = top3_hits / top3_total if top3_total else float("nan")

    wind_series = None
    if "wind_speed_post" in eval_df.columns:
        wind_series = eval_df["wind_speed_post"].fillna(eval_df.get("wind_speed"))
    elif "wind_speed" in eval_df.columns:
        wind_series = eval_df["wind_speed"].fillna(0.0)
    metrics = {
        "win_hit_rate": win_hit_rate,
        "top3_coverage": top3_coverage,
        "win_log_loss": win_logloss,
        "win_brier": win_brier,
    }
    if wind_series is not None and wind_series.notna().any():
        high = eval_df[wind_series >= 3.0]
        low = eval_df[wind_series < 3.0]
        for label, subset in ("high", high), ("low", low):
            if subset.empty:
                continue
            win_actual_subset = (subset["finish_order"].astype(float) == 1).astype(int)
            win_probs_subset = subset["win_prob"]
            wins = 0
            total = 0
            for _, race in subset.groupby("race_id"):
                total += 1
                actual_winner = race.loc[race["finish_order"].astype(float) == 1, "player_id"].tolist()
                if actual_winner and race.sort_values("win_prob", ascending=False).iloc[0]["player_id"] in actual_winner:
                    wins += 1
            metrics[f"win_hit_rate_wind_{label}"] = wins / total if total else float("nan")
    return metrics


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.features)
    race_frame, race_number = prepare_frame(
        df,
        tipster_id=args.tipster_id,
        day_index=args.day_index,
        schedule_date=args.schedule_date,
        race_number=args.race_number,
    )

    win_bundle = load_bundle(args.venue, target="win", model_type=args.model_type)
    top3_bundle = load_bundle(args.venue, target="top3", model_type=args.model_type)
    win_bundle["tipster_id"] = args.tipster_id
    top3_bundle["tipster_id"] = args.tipster_id

    race_preds = predict(race_frame, win_bundle, top3_bundle)
    metrics = evaluate_hit_rates(df, win_bundle, top3_bundle)

    def _recommendation(row) -> str:
        win_prob = row.get("win_prob", 0.0)
        top3_prob = row.get("top3_prob", 0.0)
        if win_prob >= 0.35 or (win_prob >= 0.28 and top3_prob >= 0.7):
            stars = "★★★"
        elif win_prob >= 0.2 or top3_prob >= 0.6:
            stars = "★★"
        elif win_prob >= 0.1 or top3_prob >= 0.45:
            stars = "★"
        else:
            stars = "☆"

        win_hit = metrics.get("win_hit_rate")
        top3_cov = metrics.get("top3_coverage")
        trust_values = [v for v in (win_hit, top3_cov) if v is not None and np.isfinite(v)]
        trust = float(np.mean(trust_values)) if trust_values else float("nan")
        if np.isnan(trust):
            trust_label = "信頼:不明"
        elif trust >= 0.7:
            trust_label = "信頼:高"
        elif trust >= 0.5:
            trust_label = "信頼:中"
        else:
            trust_label = "信頼:低"
        return f"{stars}/{trust_label}"

    race_preds["recommendation"] = race_preds.apply(_recommendation, axis=1)
    view_cols = [
        "number",
        "player_name",
        "win_prob",
        "top3_prob",
        "recommendation",
        "pick_rank",
        "recent_win_rate",
        "venue_win_rate",
        "line_id",
        "line_pos",
        "line_size",
        "line_win_share",
    ]
    print(f"Race #{race_number} predictions (tipster {args.tipster_id}):")
    print(race_preds[view_cols].sort_values("win_prob", ascending=False).to_string(index=False))

    print()
    print(
        "Current hit rates — win: {win:.1%}, top3 coverage: {top3:.1%}".format(
            win=metrics["win_hit_rate"], top3=metrics["top3_coverage"]
        )
    )
    win_log = metrics.get("win_log_loss")
    win_brier = metrics.get("win_brier")
    if win_log is not None and np.isfinite(win_log):
        print(f"Win logloss: {win_log:.4f}, Brier: {win_brier:.4f}")
    for label in ("high", "low"):
        key = f"win_hit_rate_wind_{label}"
        if key in metrics:
            print(f"Win hit rate (wind {label}): {metrics[key]:.1%}")


if __name__ == "__main__":
    main()
