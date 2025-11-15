"""Utilities for building datasets and training venue-level models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier, LGBMRanker

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
        target: str = "win",
    ) -> None:
        self.venue_slug = venue_slug
        self.tipster_id = tipster_id
        self.train_race_numbers = train_race_numbers
        self.model_type = model_type
        if target not in ("win", "top3"):
            raise ValueError("target must be 'win' or 'top3'")
        self.target = target
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
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        return df

    def build_dataset(self) -> pd.DataFrame:
        df = self._load_all_features()
        if self.train_race_numbers:
            df = df[df["race_number"].isin(self.train_race_numbers)]
        df = df[df["finish_order"].notna()].copy()
        label_col = "is_top3" if self.target == "top3" else "is_winner"
        if label_col not in df.columns:
            if label_col == "is_top3" and "finish_order" in df.columns:
                df[label_col] = df["finish_order"].apply(
                    lambda x: 1 if pd.notna(x) and int(x) in (1, 2, 3) else (None if pd.isna(x) else 0)
                )
            else:
                raise RuntimeError(f"gold_entry_features に {label_col} 列が必要です。")
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df[label_col] = df[label_col].fillna(0).astype(int)
        return df

    def _build_ranker_dataset(self) -> pd.DataFrame:
        df = self.load_features()
        if self.train_race_numbers:
            df = df[df["race_number"].isin(self.train_race_numbers)]
        df = df[df["finish_order"].notna()].copy()
        if df.empty:
            raise RuntimeError("順位学習用の finish_order 付きデータがありません。")
        df["finish_order"] = df["finish_order"].astype(float)
        df["entries_number"] = df.get("entries_number", 9).fillna(9).astype(float)

        def _rank_score(row: pd.Series) -> float:
            finish = row["finish_order"]
            entry_count = max(row["entries_number"], 1.0)
            base = entry_count - finish + 1.0
            if self.target == "top3" and finish > 3:
                return 0.0
            return max(base, 0.0)

        df["rank_label"] = df.apply(_rank_score, axis=1)
        df.sort_values(["race_id", "number"], inplace=True)
        return df

    def _build_sample_weights(self, df: pd.DataFrame) -> pd.Series | None:
        wind_cols = [col for col in ("wind_speed_post", "wind_speed") if col in df.columns]
        if not wind_cols:
            return None
        wind = df[wind_cols[0]].fillna(0.0)
        return 1.0 + (wind / 10.0)

    def load_model(self) -> Optional[Dict[str, Any]]:
        return self.model_store.load(
            self.venue_slug,
            target=self.target,
            model_type=self.model_type,
        )

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
        if self.model_type == "lambdarank":
            return LGBMRanker(
                objective="lambdarank",
                n_estimators=120,
                learning_rate=0.05,
                num_leaves=48,
                random_state=42,
                subsample=0.9,
                colsample_bytree=0.9,
                min_data_in_leaf=1,
                feature_pre_filter=False,
                force_col_wise=True,
            )
        raise ValueError(f"Unsupported model_type: {self.model_type}")

    def train(self) -> TrainingStats:
        if self.model_type == "lambdarank":
            df = self._build_ranker_dataset()
            y = df["rank_label"]
            groups = df.groupby("race_id").size().tolist()
            positives = int((y > 0).sum())
            negatives = int((y == 0).sum())
            X = df[FEATURE_COLUMNS]
            estimator = self._build_estimator()
            sample_weight = self._build_sample_weights(df)
            fit_kwargs = {"group": groups}
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight
            estimator.fit(X, y, **fit_kwargs)
        else:
            df = self.build_dataset()
            if df.empty:
                raise RuntimeError("データセットが空です。harvest/backfill で結果付きレースを追加してください。")
            label_col = "is_top3" if self.target == "top3" else "is_winner"
            positives = int(df[label_col].sum())
            negatives = int(len(df) - positives)
            if positives == 0 or negatives == 0:
                raise RuntimeError("勝者/非勝者のサンプルが両方必要です。")

            X = df[FEATURE_COLUMNS]
            y = df[label_col]

            estimator = self._build_estimator()
            sample_weight = self._build_sample_weights(df)
            fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
            estimator.fit(X, y, **fit_kwargs)

        bundle = {
            "estimator": estimator,
            "feature_names": FEATURE_COLUMNS,
            "model_type": self.model_type,
            "target": self.target,
            "is_ranker": self.model_type == "lambdarank",
        }
        self.model_store.save(
            self.venue_slug,
            target=self.target,
            model_type=self.model_type,
            bundle=bundle,
        )
        return TrainingStats(rows=len(df), positives=positives, negatives=negatives)
