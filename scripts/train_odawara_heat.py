#!/usr/bin/env python3
"""Train rolling HEAT model for Odawara cups."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from heat_utils import HEAT_FEATURES, augment_heat_features, ensure_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HEAT model for Odawara data")
    parser.add_argument("--data-root", type=Path, default=Path("data/winticket_lake/gold/features"))
    parser.add_argument("--pattern", default="odawara_*_pred.parquet")
    parser.add_argument("--min-train-rows", type=int, default=200)
    parser.add_argument("--max-train-rows", type=int, default=500)
    parser.add_argument("--history-out", type=Path, default=Path("odawara_heat_history_v4.csv"))
    parser.add_argument("--config-out", type=Path, default=Path("heat_configs/odawara_heat_v6.json"))
    return parser.parse_args()


def load_cup_df(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[(df["tipster_id"] == "dsc-00") & (~df["is_absent"].fillna(False))].copy()
    df = augment_heat_features(df)
    df = ensure_features(df, HEAT_FEATURES)
    df = df[df["finish_order"].notna()].copy()
    df["label"] = (df["finish_order"] <= 3).astype(float)
    return df


def evaluate(df: pd.DataFrame, score_col: str) -> tuple[float, float, int]:
    hits = total = exact = races = 0
    for _, race in df.groupby("race_id"):
        actual = set(race.loc[race["finish_order"] <= 3, "player_id"])
        if not actual:
            continue
        picks = set(race.sort_values(score_col, ascending=False).head(3)["player_id"])
        hits += len(picks & actual)
        total += len(actual)
        races += 1
        if picks == actual:
            exact += 1
    coverage = hits / total if total else float("nan")
    exact_rate = exact / races if races else float("nan")
    return coverage, exact_rate, races


def make_weight(series: pd.Series) -> pd.Series:
    min_date = series.min()
    max_date = series.max()
    span = (max_date - min_date).days or 1
    rel = (series - min_date).dt.days / span
    weights = 1.0 - 0.4 * rel
    return weights.clip(0.6, 1.0)


def main() -> None:
    args = parse_args()
    paths = sorted(args.data_root.glob(args.pattern), key=lambda p: (p.stem, p.name))
    records: list[dict] = []
    train_frames: list[pd.DataFrame] = []
    train_df = pd.DataFrame()
    scaler: StandardScaler | None = None
    model: LogisticRegression | None = None

    def predict_scores(df: pd.DataFrame) -> np.ndarray:
        assert scaler is not None and model is not None
        X = df[HEAT_FEATURES].astype(float).to_numpy()
        Xs = scaler.transform(X)
        return model.predict_proba(Xs)[:, 1]

    for path in paths:
        cup_df = load_cup_df(path)
        if cup_df.empty:
            continue
        cup_id = path.stem.replace("_pred", "").split("_")[1]
        cup_date = pd.to_datetime(cup_df["schedule_date"], format="%Y%m%d").min()
        base_cov, base_exact, races = evaluate(cup_df, "top3_prob")
        if model is not None and len(train_df) >= args.min_train_rows:
            cup_df["heat_score"] = predict_scores(cup_df)
            heat_cov, heat_exact, _ = evaluate(cup_df, "heat_score")
        else:
            heat_cov, heat_exact = base_cov, base_exact

        records.append(
            {
                "cup": cup_id,
                "schedule_date": cup_date.strftime("%Y%m%d"),
                "races": races,
                "base_coverage": base_cov,
                "base_exact": base_exact,
                "heat_coverage": heat_cov,
                "heat_exact": heat_exact,
            }
        )

        train_frames.append(cup_df)
        train_df = pd.concat(train_frames, ignore_index=True)
        if len(train_df) > args.max_train_rows:
            keep_ids = train_df.sort_values("schedule_date").tail(args.max_train_rows)["race_id"]
            keep_set = set(keep_ids)
            train_df = train_df[train_df["race_id"].isin(keep_set)].copy()
            trimmed: list[pd.DataFrame] = []
            for frame in train_frames:
                frame = frame[frame["race_id"].isin(keep_set)]
                if not frame.empty:
                    trimmed.append(frame)
            train_frames = trimmed

        if len(train_df) >= args.min_train_rows:
            X_train = train_df[HEAT_FEATURES].astype(float)
            y_train = train_df["label"]
            w = make_weight(pd.to_datetime(train_df["schedule_date"], format="%Y%m%d"))
            scaler = StandardScaler().fit(X_train)
            Xs_train = scaler.transform(X_train)
            model = LogisticRegression(max_iter=4000, solver="lbfgs", C=0.7)
            model.fit(Xs_train, y_train, sample_weight=w)

    pd.DataFrame(records).to_csv(args.history_out, index=False)
    if model is None or scaler is None:
        raise SystemExit("Insufficient data to train HEAT model")

    config = {
        "features": HEAT_FEATURES,
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "train_rows": int(len(train_df)),
        "train_races": int(train_df["race_id"].nunique()),
        "min_train_rows": args.min_train_rows,
        "max_train_rows": args.max_train_rows,
    }
    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.config_out, "w") as fh:
        import json

        json.dump(config, fh, indent=2)

    print(f"Saved history -> {args.history_out}")
    print(f"Saved config -> {args.config_out}")


if __name__ == "__main__":
    main()
