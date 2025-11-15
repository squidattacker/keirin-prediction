#!/usr/bin/env python3
"""Harvest all races for the given WinTicket cups by inspecting cup overview metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, List, Sequence, Tuple

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from winticket import harvest


def fetch_and_persist(job: harvest.HarvestJob):
    html = harvest.fetch_html(job.url)
    state = harvest.extract_preloaded_state(html)
    payload = harvest.collect_payload(state)
    harvest.persist(job, payload)
    return payload


def probe_initial_payload(cup_id: str, venue: str, max_index: int, max_race: int):
    last_error: Exception | None = None
    for idx in range(1, max_index + 1):
        for race in range(1, max_race + 1):
            job = harvest.HarvestJob(
                cup_id=cup_id,
                venue_slug=venue,
                index=idx,
                race_number=race,
                source_type="racecard",
            )
            try:
                payload = fetch_and_persist(job)
                print(f"seeded racecard idx={idx} race={race} -> {job.bronze_dir}")
                return job, payload
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status == 404:
                    continue
                last_error = exc
            except Exception as exc:  # pylint: disable=broad-except
                last_error = exc
    raise RuntimeError(f"Could not find any racecard for cup {cup_id}: {last_error}")


def enumerate_combos(payload) -> List[Tuple[int, int]]:
    overview = payload.get("cup_overview") or {}
    schedules = overview.get("schedules") or []
    races = overview.get("races") or []
    schedule_index = {sched.get("id"): int(sched.get("index") or 0) for sched in schedules}
    combos: set[Tuple[int, int]] = set()
    for race in races:
        schedule_id = race.get("scheduleId")
        idx = schedule_index.get(schedule_id)
        number = race.get("number")
        if not idx or not number:
            continue
        combos.add((int(idx), int(number)))
    return sorted(combos)


def download_job(job: harvest.HarvestJob) -> None:
    try:
        path = harvest.run(job)
        print(f"saved {job.source_type} idx={job.index} race={job.race_number} -> {path}")
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 404:
            print(f"missing {job.source_type} idx={job.index} race={job.race_number}: 404")
            return
        raise


def harvest_cup(cup_id: str, venue: str, sources: Sequence[str], max_index: int, max_race: int) -> None:
    seed_job, payload = probe_initial_payload(cup_id, venue, max_index, max_race)
    combos = enumerate_combos(payload)
    if not combos:
        combos = [(seed_job.index, seed_job.race_number)]
    seen = {(seed_job.index, seed_job.race_number, "racecard")}
    print(f"cup {cup_id}: {len(combos)} combos detected")
    for idx, race in combos:
        for source in sources:
            key = (idx, race, source)
            if key in seen:
                continue
            job = harvest.HarvestJob(
                cup_id=cup_id,
                venue_slug=venue,
                index=idx,
                race_number=race,
                source_type=source,
            )
            download_job(job)
            seen.add(key)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harvest all races for the provided cup ids")
    parser.add_argument("--venue", required=True, help="venue slug (e.g. komatsushima)")
    parser.add_argument("--cup-ids", nargs="+", help="cup ids to harvest")
    parser.add_argument("--sources", nargs="+", default=["racecard", "raceresult"], choices=["racecard", "raceresult"])
    parser.add_argument("--max-index", type=int, default=4)
    parser.add_argument("--max-race", type=int, default=12)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    for cup_id in args.cup_ids:
        print(f"=== Harvesting cup {cup_id} ===")
        harvest_cup(cup_id, args.venue, args.sources, args.max_index, args.max_race)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
