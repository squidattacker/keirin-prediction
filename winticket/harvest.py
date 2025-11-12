"""Harvest bronze-layer snapshots from winticket race/raceresult pages."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import requests

from . import config


PATTERN = re.compile(r"window.__PRELOADED_STATE__ = (\{.*?\});\nwindow.__CONFIG__", re.S)

QueryKey = Tuple[str, str]


TARGET_QUERIES: Dict[QueryKey, str] = {
    ("keirin/race/common", "FETCH_KEIRIN_RACE"): "race_common",
    ("keirin/race/odds", "FETCH_KEIRIN_RACE_ODDS"): "race_odds",
    ("keirin", "FETCH_KEIRIN_CUP_RACES"): "cup_overview",
}


@dataclass
class HarvestJob:
    cup_id: str
    venue_slug: str
    index: int
    race_number: int
    source_type: str

    @property
    def url(self) -> str:
        base = "racecard" if self.source_type == "racecard" else "raceresult"
        return f"https://www.winticket.jp/keirin/{self.venue_slug}/{base}/{self.cup_id}/{self.index}/{self.race_number}"

    @property
    def bronze_dir(self) -> Path:
        if self.source_type == "racecard":
            return config.BRONZE_RACECARDS_DIR
        if self.source_type == "raceresult":
            return config.BRONZE_RESULTS_DIR
        raise ValueError(f"unsupported source_type {self.source_type}")


def fetch_html(url: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def extract_preloaded_state(html: str) -> Dict[str, Any]:
    match = PATTERN.search(html)
    if not match:
        raise RuntimeError("window.__PRELOADED_STATE__ が見つかりませんでした。")
    return json.loads(match.group(1))


def collect_payload(state: Dict[str, Any]) -> Dict[str, Any]:
    queries = state.get("tanStackQuery", {}).get("queries", [])
    payload: Dict[str, Any] = {}
    for query in queries:
        key = tuple(query.get("queryKey", [])[:2])
        alias = TARGET_QUERIES.get(key)  # type: ignore[arg-type]
        if not alias:
            continue
        payload[alias] = query.get("state", {}).get("data")
    missing = [alias for alias in TARGET_QUERIES.values() if alias not in payload]
    if missing:
        raise RuntimeError(f"必要なクエリが不足しています: {missing}")
    return payload


def persist(job: HarvestJob, payload: Dict[str, Any]) -> Path:
    race = payload["race_common"]["race"]
    race_id = race["id"]
    metadata = {
        "cupId": job.cup_id,
        "index": job.index,
        "raceNumber": job.race_number,
        "raceId": race_id,
        "venueSlug": job.venue_slug,
        "capturedAt": datetime.now(timezone.utc).isoformat(),
        "sourceUrl": job.url,
        "sourceType": job.source_type,
    }
    data = {"meta": metadata, "state": payload}
    target = job.bronze_dir / f"{race_id}.json"
    target.write_text(json.dumps(data, ensure_ascii=True, indent=2))
    return target


def run(job: HarvestJob) -> Path:
    html = fetch_html(job.url)
    state = extract_preloaded_state(html)
    payload = collect_payload(state)
    return persist(job, payload)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download winticket race snapshots into the bronze layer")
    parser.add_argument("--cup-id", required=True)
    parser.add_argument("--venue", required=True, help="venue slug (e.g. ogaki)")
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--race-numbers", type=int, nargs="+", required=True)
    parser.add_argument("--sources", nargs="+", default=["racecard", "raceresult"], choices=["racecard", "raceresult"])
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    jobs = [
        HarvestJob(
            cup_id=args.cup_id,
            venue_slug=args.venue,
            index=args.index,
            race_number=number,
            source_type=source,
        )
        for number in args.race_numbers
        for source in args.sources
    ]
    for job in jobs:
        path = run(job)
        print(f"saved {job.source_type} race {job.race_number} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

