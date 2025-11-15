"""Storage helpers for venue-specific models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from . import config


class VenueModelStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or config.MODELS_DIR)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, venue_slug: str, *, target: str, model_type: str) -> Path:
        return self.root / f"{venue_slug}_{target}_{model_type}.joblib"

    def save(self, venue_slug: str, *, target: str, model_type: str, bundle: Dict[str, Any]) -> None:
        joblib.dump(bundle, self.path_for(venue_slug, target=target, model_type=model_type))

    def load(self, venue_slug: str, *, target: str, model_type: str) -> Optional[Dict[str, Any]]:
        path = self.path_for(venue_slug, target=target, model_type=model_type)
        if not path.exists():
            return None
        return joblib.load(path)
