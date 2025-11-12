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

    def path_for(self, venue_slug: str) -> Path:
        return self.root / f"{venue_slug}.joblib"

    def save(self, venue_slug: str, bundle: Dict[str, Any]) -> None:
        joblib.dump(bundle, self.path_for(venue_slug))

    def load(self, venue_slug: str) -> Optional[Dict[str, Any]]:
        path = self.path_for(venue_slug)
        if not path.exists():
            return None
        return joblib.load(path)
