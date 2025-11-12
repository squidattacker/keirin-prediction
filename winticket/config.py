"""Filesystem metadata for the winticket medallion pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def _ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

DATA_LAKE_ROOT = Path(
    os.environ.get(
        "WINTICKET_DATA_LAKE",
        PROJECT_ROOT / "data" / "winticket_lake",
    )
)
MODELS_DIR = Path(
    os.environ.get(
        "WINTICKET_MODELS_DIR",
        PROJECT_ROOT / "models" / "winticket",
    )
)

BRONZE_RACECARDS_DIR = DATA_LAKE_ROOT / "bronze" / "racecards"
BRONZE_RESULTS_DIR = DATA_LAKE_ROOT / "bronze" / "results"
SILVER_RACES_DIR = DATA_LAKE_ROOT / "silver" / "races"
GOLD_FEATURES_DIR = DATA_LAKE_ROOT / "gold" / "features"

_ensure_dirs(
    [
        BRONZE_RACECARDS_DIR,
        BRONZE_RESULTS_DIR,
        SILVER_RACES_DIR,
        GOLD_FEATURES_DIR,
        MODELS_DIR,
    ]
)

