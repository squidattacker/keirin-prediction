"""Utilities for HEAT feature augmentation and scoring."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

HEAT_FEATURES: list[str] = [
    "top3_prob",
    "win_prob",
    "recent_win_rate",
    "recent_top3_rate",
    "recent_back_avg3",
    "line_win_share",
    "line_top3_expectation",
    "line_ev_gap_vs_next",
    "line_ev_rank",
    "line_size",
    "line_pos",
    "line_pos_ratio",
    "line_is_head",
    "line_tail_flag",
    "line_count",
    "line_type_code",
    "phase_importance_coeff",
    "comment_attack",
    "comment_support",
    # derived cross features / race context
    "line_relative_share",
    "line_peer_top3_avg",
    "line_peer_recent_win_avg",
    "line_peer_count",
    "line_pressure_score",
    "line_synergy_score",
    "line_lead_pressure",
    "line_support_combo",
    "line_attack_combo",
    "race_line_entropy",
    "race_top3_std",
]


def augment_heat_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns needed for HEAT scoring."""

    out = df.copy()
    if "line_size" not in out.columns:
        out["line_size"] = 0.0
    if "line_pos" not in out.columns:
        out["line_pos"] = 0.0

    out["line_pos_ratio"] = out["line_pos"] / out["line_size"].replace(0, np.nan)
    out["line_pos_ratio"] = out["line_pos_ratio"].fillna(0.0)
    out["line_tail_flag"] = (out["line_pos"] == out["line_size"]).astype(float)

    for col in ("recent_back_avg3", "comment_attack", "comment_support"):
        if col not in out.columns:
            out[col] = 0.0

    race_max_share = out.groupby("race_id")["line_win_share"].transform("max")
    race_max_share = race_max_share.replace(0.0, np.nan)
    out["line_relative_share"] = (out["line_win_share"] / race_max_share).fillna(0.0)

    group_cols = ["race_id", "line_id"]
    # line_id may contain NaN -> treat as -1 to keep groups isolated
    out["_line_group"] = out["line_id"].fillna(-1.0)
    group_cols = ["race_id", "_line_group"]

    peer_count = out.groupby(group_cols)["player_id"].transform("count") - 1
    out["line_peer_count"] = peer_count.clip(lower=0)

    top3_sum = out.groupby(group_cols)["top3_prob"].transform("sum") - out["top3_prob"]
    out["line_peer_top3_avg"] = np.where(
        peer_count > 0,
        top3_sum / peer_count.replace(0, np.nan),
        0.0,
    )
    out["line_peer_top3_avg"].fillna(0.0, inplace=True)

    win_sum = out.groupby(group_cols)["recent_win_rate"].transform("sum") - out["recent_win_rate"]
    out["line_peer_recent_win_avg"] = np.where(
        peer_count > 0,
        win_sum / peer_count.replace(0, np.nan),
        0.0,
    )
    out["line_peer_recent_win_avg"].fillna(0.0, inplace=True)

    out["line_pressure_score"] = out["line_relative_share"] * (1.0 - out["line_pos_ratio"])

    out["line_synergy_score"] = out["line_relative_share"] * out["line_peer_top3_avg"]
    out["line_lead_pressure"] = out["line_relative_share"] * (1.0 - out["line_pos_ratio"])
    out["line_support_combo"] = out["comment_support"] * (1.0 - out["line_pos_ratio"])
    out["line_attack_combo"] = out["comment_attack"] * out["line_pos_ratio"]

    # race-level entropy of line dominance
    line_share_unique = (
        out[["race_id", "_line_group", "line_win_share"]]
        .drop_duplicates()
        .rename(columns={"line_win_share": "_line_share"})
    )
    entropy = {}
    top3_spread = out.groupby("race_id")["top3_prob"].std().fillna(0.0)
    for race_id, group in line_share_unique.groupby("race_id"):
        shares = group["_line_share"].to_numpy()
        total = shares.sum()
        if total <= 0:
            entropy[race_id] = 0.0
            continue
        probs = shares / total
        entropy[race_id] = float(-np.sum(probs * np.log(probs + 1e-6)))

    out["race_line_entropy"] = out["race_id"].map(entropy).fillna(0.0)
    out["race_top3_std"] = out["race_id"].map(top3_spread).fillna(0.0)

    out.drop(columns="_line_group", inplace=True)
    return out


def ensure_features(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in features:
        if col not in out.columns:
            out[col] = 0.0
    return out


def predict_heat(df: pd.DataFrame, config: dict) -> np.ndarray:
    from sklearn.preprocessing import StandardScaler  # type: ignore

    features: list[str] = config["features"]
    mean = np.array(config["scaler"]["mean"], dtype=float)
    scale = np.array(config["scaler"]["scale"], dtype=float)
    coef = np.array(config["coefficients"], dtype=float)
    intercept = float(config["intercept"])

    df = augment_heat_features(df)
    df = ensure_features(df, features)
    X = df[features].fillna(0.0).astype(float).to_numpy()
    Xs = (X - mean) / scale
    logits = Xs.dot(coef) + intercept
    return 1 / (1 + np.exp(-logits))
