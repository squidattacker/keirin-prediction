"""Utilities for building datasets and training venue-level models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

from . import config
from .features import FEATURE_COLUMNS
from .models import VenueModelStore


@dataclass
class TrainingStats:
    rows: int
    positives: int
    negatives: int


class VenueModelTrainer:
    def __init__(
        self,
        venue_slug: str,
        *,
        tipster_id: str,
        train_race_numbers: Optional[List[int]] = None,
        model_type: str = "logistic",
    ) -> None:
        self.venue_slug = venue_slug
        self.tipster_id = tipster_id
        self.train_race_numbers = train_race_numbers
        self.model_type = model_type
        self.gold_dir = config.GOLD_FEATURES_DIR
        self.model_store = VenueModelStore()

    def _feature_paths(self) -> List[Path]:
        pattern = f"{self.venue_slug}_*.parquet"
        paths = sorted(self.gold_dir.glob(pattern))
        if not paths:
            raise RuntimeError(
                f"特徴量ファイルが見つかりません。dbt run で gold 層を更新してください: {self.gold_dir}/{pattern}"
            )
        return paths

    def _load_all_features(self) -> pd.DataFrame:
        frames = [pd.read_parquet(path) for path in self._feature_paths()]
        df = pd.concat(frames, ignore_index=True)
        df = df[df["venue_slug"] == self.venue_slug]
        if "tipster_id" in df.columns:
            df = df[df["tipster_id"] == self.tipster_id]
        if df.empty:
            raise RuntimeError("指定した tipster の特徴量がありません。dbt 側で predictions を確認してください。")
        return df

    def load_features(self) -> pd.DataFrame:
        df = self._load_all_features().copy()
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0
        df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0).astype(float)
        return df

    def build_dataset(self) -> pd.DataFrame:
        df = self._load_all_features()
        if self.train_race_numbers:
            df = df[df["race_number"].isin(self.train_race_numbers)]
        df = df[df["finish_order"].notna()].copy()
        if "is_winner" not in df.columns:
            raise RuntimeError("gold_entry_features に is_winner 列が必要です。")
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0
        df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0).astype(float)
        return df

    def _build_estimator(self):
        if self.model_type == "lightgbm":
            return LGBMClassifier(
                objective="binary",
                n_estimators=80,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                subsample=0.8,
                colsample_bytree=0.8,
                min_data_in_bin=1,
                min_data_in_leaf=1,
                feature_pre_filter=False,
                force_col_wise=True,
            )
        if self.model_type == "logistic":
            return LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced")
        raise ValueError(f"Unsupported model_type: {self.model_type}")

    def train(self) -> TrainingStats:
        df = self.build_dataset()
        if df.empty:
            raise RuntimeError("データセットが空です。harvest/backfill で結果付きレースを追加してください。")
        positives = int(df["is_winner"].sum())
        negatives = int(len(df) - positives)
        if positives == 0 or negatives == 0:
            raise RuntimeError("勝者/非勝者のサンプルが両方必要です。")

        X = df[FEATURE_COLUMNS]
        y = df["is_winner"]

        estimator = self._build_estimator()
        estimator.fit(X, y)

        bundle = {
            "estimator": estimator,
            "feature_names": FEATURE_COLUMNS,
            "model_type": self.model_type,
        }
        self.model_store.save(self.venue_slug, bundle)
        return TrainingStats(rows=len(df), positives=positives, negatives=negatives)
