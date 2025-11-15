#!/usr/bin/env python3
"""Build venue-specific feature parquet files from Bronze JSON."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

FeatureRow = Dict[str, object]

ATTACK_KW = ["積極", "先行", "主導", "叩く", "攻", "前々", "仕掛", "逃げ"]
SUPPORT_KW = ["番手", "援護", "サポート", "任せ", "目標", "後ろ"]
CAUTIOUS_KW = ["様子", "無理せず", "慎重", "流れ", "辛い", "厳しい"]
NEGATIVE_KW = ["悪い", "不安", "故障", "疲れ", "重い"]
STYLE_CODE = {"逃": 3, "捲": 2, "差": 1, "追": 0, "両": 2}
PHASE_IMPORTANCE = {
    "final": 1.5,
    "semifinal": 1.25,
    "selection": 1.15,
    "qualifier": 1.05,
    "other": 1.0,
}

WEATHER_CODES = {
    "晴": 1.0,
    "曇": 0.6,
    "小雨": -0.4,
    "雨": -0.6,
    "雪": -0.8,
    "強風": -0.2,
}

LINE_TYPE_CODES = {
    "一列棒状": 3.0,
    "二分戦": 2.0,
    "三分戦": 1.0,
    "四分戦": 0.5,
}

CUP_GRADE_LABELS = {
    0: "ladies",
    1: "f2",
    2: "f1",
    3: "g3",
    4: "g2",
    5: "g1",
    6: "gp",
}
CUP_GRADE_RANK = {
    "unknown": 0.0,
    "ladies": 0.5,
    "f2": 1.0,
    "f1": 2.0,
    "g3": 3.0,
    "g2": 4.0,
    "g1": 5.0,
    "gp": 6.0,
}
F_CLASS_LABELS = {"ladies", "f2", "f1"}
G_CLASS_LABELS = {"g3", "g2", "g1", "gp"}


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().lower().replace("ｍ", "m")
        text = text.replace("m", "")
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _encode_weather(text: str | None) -> float:
    if not text:
        return 0.0
    text = text.strip()
    return WEATHER_CODES.get(text, 0.0)


def _encode_line_type(text: str | None) -> float:
    if not text:
        return 0.0
    return LINE_TYPE_CODES.get(text.strip(), 0.0)


def _encode_cup_grade(value: int | None) -> Dict[str, float | str]:
    label = CUP_GRADE_LABELS.get(value, "unknown")
    rank = CUP_GRADE_RANK[label]
    raw_value = float(value) if value is not None else -1.0
    return {
        "cup_grade_label": label,
        "cup_grade_value": raw_value,
        "cup_grade_rank": rank,
        "cup_grade_is_ladies": 1.0 if label == "ladies" else 0.0,
        "cup_grade_is_f2": 1.0 if label == "f2" else 0.0,
        "cup_grade_is_f1": 1.0 if label == "f1" else 0.0,
        "cup_grade_is_f_class": 1.0 if label in F_CLASS_LABELS else 0.0,
        "cup_grade_is_g3_or_higher": 1.0 if label in G_CLASS_LABELS else 0.0,
    }


def _build_exacta_lookup(rows: List[Dict] | None) -> Dict[int, Dict[str, List[float]]]:
    stats: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: {"odds": [], "pop": []})
    if not rows:
        return stats
    for odds_row in rows:
        if odds_row.get("absent"):
            continue
        key = odds_row.get("key") or []
        if len(key) < 2:
            continue
        first = int(key[0])
        odds_val = odds_row.get("odds")
        pop_val = odds_row.get("popularityOrder")
        if odds_val is not None:
            stats[first]["odds"].append(float(odds_val))
        if pop_val is not None:
            stats[first]["pop"].append(float(pop_val))
    return stats


def _track_length(distance: object, laps: object) -> float:
    dist = _safe_float(distance) or 0.0
    lap = _safe_float(laps) or 0.0
    return dist / lap if lap else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build feature parquet for a venue")
    parser.add_argument("--venue", required=True, help="venue slug (e.g. ogaki)")
    parser.add_argument("--cup-ids", nargs="*", help="optional list of cup ids to include")
    parser.add_argument("--data-root", type=Path, default=Path("data/winticket_lake"))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet (defaults to data_root/gold/features/{venue}_custom.parquet)",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open() as fh:
        return json.load(fh)


def flatten_results(blocks: List[Dict] | None) -> List[Dict]:
    rows: List[Dict] = []
    if not blocks:
        return rows
    for block in blocks:
        for rr in block.get("raceResults", []) or []:
            rows.append(rr)
    rows.sort(key=lambda r: int(str(r.get("raceId", "0"))), reverse=True)
    return rows


def aggregate_form(blocks: List[Dict] | None) -> Tuple[float, float, float, float]:
    recs = flatten_results(blocks)
    orders = [float(rr["order"]) for rr in recs if rr.get("order") is not None]
    if not orders:
        return (0.0, 0.0, 0.0, 0.0)
    n = float(len(orders))
    avg = sum(orders) / n
    win_rate = sum(1 for o in orders if o == 1.0) / n
    top3_rate = sum(1 for o in orders if o <= 3.0) / n
    return (n, avg, win_rate, top3_rate)


def recent_back_average(record: Dict | None, limit: int = 3) -> float:
    recs = flatten_results((record or {}).get("latestCupResults"))[:limit]
    vals = []
    for rr in recs:
        back = rr.get("back")
        if back is None:
            continue
        vals.append(1.0 if back else 0.0)
    return sum(vals) / len(vals) if vals else 0.0


def recent_final_half_average(record: Dict | None, limit: int = 3) -> float:
    recs = flatten_results((record or {}).get("latestCupResults"))[:limit]
    vals = []
    for rr in recs:
        try:
            vals.append(float(rr.get("finalHalfRecord")))
        except (TypeError, ValueError):
            continue
    return sum(vals) / len(vals) if vals else 0.0


def parse_line_map(race_common: Dict) -> Dict[int, Tuple[int, int, int]]:
    mapping: Dict[int, Tuple[int, int, int]] = {}
    for line_idx, line in enumerate(race_common.get("linePrediction", {}).get("lines") or [], start=1):
        numbers: List[int] = []
        for entry in line.get("entries") or []:
            numbers.extend(entry.get("numbers") or [])
        size = len(numbers)
        for pos, num in enumerate(numbers, start=1):
            mapping[num] = (line_idx, pos, size)
    return mapping


def comment_flags(text: str | None) -> Dict[str, float]:
    if not text:
        return {"comment_attack": 0.0, "comment_support": 0.0, "comment_cautious": 0.0, "comment_negative": 0.0}
    text = text.replace(" ", "")
    attack = float(any(kw in text for kw in ATTACK_KW))
    support = float(any(kw in text for kw in SUPPORT_KW))
    cautious = float(any(kw in text for kw in CAUTIOUS_KW))
    negative = float(any(kw in text for kw in NEGATIVE_KW))
    return {
        "comment_attack": attack,
        "comment_support": support,
        "comment_cautious": cautious,
        "comment_negative": negative,
    }


def categorize_phase(phase: str | None) -> Tuple[str, Dict[str, float]]:
    text = (phase or "").strip()
    if any(keyword in text for keyword in ("決勝", "優勝")):
        category = "final"
    elif "準決" in text:
        category = "semifinal"
    elif any(keyword in text for keyword in ("特選", "選抜")):
        category = "selection"
    elif "予選" in text or text.startswith("ガ予") or text.startswith("チ予"):
        category = "qualifier"
    else:
        category = "other"
    return category, {
        "is_final": 1.0 if category == "final" else 0.0,
        "is_semifinal": 1.0 if category == "semifinal" else 0.0,
        "is_selection": 1.0 if category == "selection" else 0.0,
        "is_qualifier": 1.0 if category == "qualifier" else 0.0,
        "importance": PHASE_IMPORTANCE.get(category, 1.0),
    }


def make_line_group_id(line_id: int | None, number: int) -> int:
    return int(line_id) if line_id else 100 + int(number)


def compute_line_stats(
    entries: List[Dict],
    records: Dict[str, Dict],
    players: Dict[str, Dict],
    line_map: Dict[int, Tuple[int, int, int]],
    finish_map: Dict[str, float],
) -> Dict[int, Dict[str, object]]:
    stats: Dict[int, Dict[str, object]] = {}
    for entry in entries:
        player_id = entry.get("playerId")
        if not player_id:
            continue
        number = int(entry.get("number") or 0)
        line_id, line_pos, line_size = line_map.get(number, (0, 0, 0))
        group_id = make_line_group_id(line_id, number)
        stat = stats.setdefault(
            group_id,
            {
                "line_id": float(line_id or 0),
                "line_size": float(line_size or 1),
                "members": [],
                "win_exp": 0.0,
                "top3_exp": 0.0,
            },
        )
        stat["line_size"] = float(max(stat["line_size"], float(line_size or 1)))
        player = players.get(player_id, {})
        record = records.get(player_id, {})
        stat.setdefault("members", []).append(
            {
                "player_id": player_id,
                "region_id": player.get("regionId") or "NA",
                "line_pos": line_pos or number,
                "finish_order": finish_map.get(player_id),
            }
        )
        stat["win_exp"] += float(record.get("firstRate") or 0.0) / 100.0
        stat["top3_exp"] += float(record.get("thirdRate") or 0.0) / 100.0

    if not stats:
        return stats

    race_line_total = sum(stat["win_exp"] for stat in stats.values())
    race_max_line_size = max(stat["line_size"] for stat in stats.values()) or 1.0
    sorted_groups = sorted(stats.items(), key=lambda kv: kv[1]["win_exp"], reverse=True)
    top_win = sorted_groups[0][1]["win_exp"] if sorted_groups else 0.0

    for idx, (group_id, stat) in enumerate(sorted_groups):
        next_win = sorted_groups[idx + 1][1]["win_exp"] if idx + 1 < len(sorted_groups) else 0.0
        stat["line_ev_rank"] = float(idx + 1)
        stat["line_ev_gap_vs_next"] = stat["win_exp"] - (next_win if idx == 0 else top_win)
        stat["line_win_share"] = stat["win_exp"] / race_line_total if race_line_total > 0 else 0.0
        stat["race_max_line_size"] = float(race_max_line_size)
        ordered_members = sorted(stat.get("members", []), key=lambda m: m["line_pos"])
        stat["line_signature_players"] = "-".join(m.get("player_id") or "" for m in ordered_members if m.get("player_id"))
        stat["line_signature_regions"] = "-".join(m.get("region_id") or "NA" for m in ordered_members)
        finishes = [m["finish_order"] for m in ordered_members if m.get("finish_order") is not None]
        stat["best_finish"] = min(finishes) if finishes else None

    return stats


def build_rows(venue: str, cup_ids: Iterable[str] | None, data_root: Path) -> List[FeatureRow]:
    rows: List[FeatureRow] = []
    player_line_history: Dict[str, Dict[str, float]] = defaultdict(lambda: {"wins": 0.0, "races": 0.0})
    region_line_history: Dict[str, Dict[str, float]] = defaultdict(lambda: {"wins": 0.0, "races": 0.0})
    racecards_dir = data_root / "bronze" / "racecards"
    results_dir = data_root / "bronze" / "results"

    for path in sorted(racecards_dir.glob("*.json")):
        data = load_json(path)
        meta = data.get("meta", {})
        if meta.get("venueSlug") != venue:
            continue
        cup_id = meta.get("cupId")
        if cup_ids and cup_id not in cup_ids:
            continue

        state_block = data.get("state", {})
        cup = (state_block.get("cup_overview") or {}).get("cup") or {}
        grade_features = _encode_cup_grade(cup.get("grade"))
        race_common = state_block.get("race_common", {})
        race_odds = state_block.get("race_odds", {})
        race_struct = race_common.get("race", {})
        schedule = race_common.get("schedule", {})
        entries = race_common.get("entries") or []
        players = {pl.get("id"): pl for pl in (race_common.get("players") or [])}
        records = {rec.get("playerId"): rec for rec in (race_common.get("records") or [])}
        line_map = parse_line_map(race_common)
        phase_category, phase_flags = categorize_phase(race_struct.get("raceType3"))
        race_importance = phase_flags["importance"]
        weather_pre = race_struct.get("weather")
        wind_speed_pre = _safe_float(race_struct.get("windSpeed"))
        line_prediction = race_common.get("linePrediction") or {}
        line_type_code = _encode_line_type(line_prediction.get("lineType"))
        line_count = float(len(line_prediction.get("lines") or []))
        track_length = _track_length(race_struct.get("distance"), race_struct.get("lap"))

        picks: Dict[int, int] = {}
        for pred in race_common.get("predictions") or []:
            if pred.get("tipsterId") != "dsc-00":
                continue
            for idx, num in enumerate(pred.get("order") or [], start=1):
                picks[int(num)] = idx

        field_ids = {entry.get("playerId") for entry in entries if entry}
        h2h = {pid: {"wins": 0, "losses": 0, "races": 0} for pid in field_ids if pid}
        for comp in race_common.get("competitionRecords") or []:
            pid = comp.get("playerId")
            opp = comp.get("opponentId")
            if pid in h2h and opp in field_ids:
                w = comp.get("wins") or 0
                l = comp.get("losses") or 0
                h2h[pid]["wins"] += w
                h2h[pid]["losses"] += l
                h2h[pid]["races"] += w + l

        exacta_stats = _build_exacta_lookup(race_odds.get("exacta"))

        race_id = meta.get("raceId")
        finish_map: Dict[str, float] = {}
        results_path = results_dir / f"{race_id}.json"
        result_race: Dict[str, object] = {}
        final_exacta_stats: Dict[int, Dict[str, List[float]]] | None = None
        if results_path.exists():
            res_data = load_json(results_path)
            state_block = res_data.get("state", {})
            race_common_result = state_block.get("race_common", {})
            result_race = race_common_result.get("race") or {}
            for res in race_common_result.get("results") or []:
                pid = res.get("playerId")
                order = res.get("order")
                if pid and order is not None:
                    finish_map[pid] = float(order)
            final_exacta_stats = _build_exacta_lookup(state_block.get("race_odds", {}).get("exacta"))

        line_stats = compute_line_stats(entries, records, players, line_map, finish_map)
        weather_post = (result_race or {}).get("weather")
        wind_speed_post = _safe_float((result_race or {}).get("windSpeed"))

        for entry in entries:
            player_id = entry.get("playerId")
            if not player_id:
                continue
            player = players.get(player_id, {})
            record = records.get(player_id, {})
            number = int(entry.get("number") or 0)
            odds_stats = exacta_stats[number]
            final_stats = final_exacta_stats[number] if final_exacta_stats is not None else odds_stats
            odds_vals = odds_stats["odds"]
            pop_vals = odds_stats["pop"]
            final_odds_vals = final_stats["odds"]
            final_pop_vals = final_stats["pop"]
            min_odds = min(odds_vals) if odds_vals else 0.0
            max_odds = max(odds_vals) if odds_vals else 0.0
            avg_odds = sum(odds_vals) / len(odds_vals) if odds_vals else 0.0
            spread_odds = max_odds - min_odds if odds_vals else 0.0
            min_pop = min(pop_vals) if pop_vals else 0.0
            avg_pop = sum(pop_vals) / len(pop_vals) if pop_vals else 0.0
            final_min_odds = min(final_odds_vals) if final_odds_vals else 0.0
            final_avg_odds = sum(final_odds_vals) / len(final_odds_vals) if final_odds_vals else 0.0
            final_min_pop = min(final_pop_vals) if final_pop_vals else 0.0
            final_avg_pop = sum(final_pop_vals) / len(final_pop_vals) if final_pop_vals else 0.0
            rec_races, rec_avg, rec_win, rec_top3 = aggregate_form((record or {}).get("latestCupResults"))
            ven_races, ven_avg, ven_win, ven_top3 = aggregate_form((record or {}).get("latestVenueResults"))
            recent_back = recent_back_average(record)
            recent_final = recent_final_half_average(record)
            line_id, line_pos, line_size = line_map.get(number, (0, 0, 0))
            line_is_head = 1.0 if line_pos == 1 and line_size > 0 else 0.0
            comment_info = comment_flags((record or {}).get("comment"))
            finish_order = finish_map.get(player_id)
            style_code = float(STYLE_CODE.get((record or {}).get("style"), 0))

            group_id = make_line_group_id(line_id, number)
            stats = line_stats.get(group_id, {})
            line_win_expectation = float(stats.get("win_exp", 0.0))
            line_top3_expectation = float(stats.get("top3_exp", 0.0))
            line_win_share = float(stats.get("line_win_share", 0.0))
            line_ev_rank = float(stats.get("line_ev_rank", 0.0))
            line_ev_gap = float(stats.get("line_ev_gap_vs_next", 0.0))
            race_max_line_size = float(stats.get("race_max_line_size", 0.0))
            signature_players = stats.get("line_signature_players") or ""
            signature_regions = stats.get("line_signature_regions") or ""
            player_hist = (
                player_line_history.get(signature_players, {"wins": 0.0, "races": 0.0})
                if signature_players
                else {"wins": 0.0, "races": 0.0}
            )
            region_hist = (
                region_line_history.get(signature_regions, {"wins": 0.0, "races": 0.0})
                if signature_regions
                else {"wins": 0.0, "races": 0.0}
            )
            line_signature_final_races = float(player_hist.get("races", 0.0))
            line_signature_final_win_rate = (player_hist["wins"] / player_hist["races"]) if player_hist.get("races") else 0.0
            line_region_final_races = float(region_hist.get("races", 0.0))
            line_region_final_win_rate = (region_hist["wins"] / region_hist["races"]) if region_hist.get("races") else 0.0

            line_priority_factor = 0.0
            if race_max_line_size > 0:
                size_ratio = (float(line_size) if line_size else 1.0) / race_max_line_size
                line_priority_factor = (size_ratio + line_win_share) / 2.0
            phase_line_priority_boost = race_importance * line_priority_factor
            phase_comment_support_boost = comment_info["comment_support"] * (1 + 0.5 * phase_flags["is_final"] + 0.3 * phase_flags["is_semifinal"])
            phase_comment_attack_boost = comment_info["comment_attack"] * (1 + 0.4 * phase_flags["is_final"] + 0.2 * phase_flags["is_semifinal"])
            sacrifice_role_score = comment_info["comment_support"] * race_importance * (1.0 if line_pos >= 2 else 0.0)
            tail_pressure = 0.6 if line_size >= 2 and line_pos == line_size else 0.0
            style_support = 0.1 if style_code <= 1 else 0.0
            phase_sacrifice_pressure = race_importance * (tail_pressure + 0.3 * comment_info["comment_support"] + style_support)

            row: FeatureRow = {
                "race_id": race_id,
                "cup_id": cup_id,
                "race_number": meta.get("raceNumber"),
                "day_index": meta.get("index"),
                "venue_slug": venue,
                "player_id": player_id,
                "player_name": player.get("name"),
                "number": number,
                "bracket_number": entry.get("bracketNumber"),
                "is_absent": entry.get("absent"),
                "schedule_date": schedule.get("date"),
                "schedule_day": schedule.get("day"),
                "cup_grade_label": grade_features["cup_grade_label"],
                "distance_m": race_struct.get("distance"),
                "weather": race_struct.get("weather"),
                "wind_speed": wind_speed_pre if wind_speed_pre is not None else 0.0,
                "wind_speed_post": wind_speed_post if wind_speed_post is not None else (wind_speed_pre if wind_speed_pre is not None else 0.0),
                "wind_speed_delta": (wind_speed_post - wind_speed_pre) if (wind_speed_post is not None and wind_speed_pre is not None) else 0.0,
                "weather_code": _encode_weather(weather_pre),
                "weather_post_code": _encode_weather(weather_post if weather_post is not None else weather_pre),
                "track_length_m": track_length,
                "line_type_code": line_type_code,
                "line_count": line_count,
                "entries_number": race_struct.get("entriesNumber"),
                "race_class": race_struct.get("class"),
                "race_phase": race_struct.get("raceType3"),
                "race_phase_category": phase_category,
                "phase_is_final": phase_flags["is_final"],
                "phase_is_semifinal": phase_flags["is_semifinal"],
                "phase_is_selection": phase_flags["is_selection"],
                "phase_is_qualifier": phase_flags["is_qualifier"],
                "phase_importance_coeff": race_importance,
                "age": player.get("age"),
                "player_class": player.get("class"),
                "player_group": player.get("group"),
                "region_id": player.get("regionId"),
                "recent_races": rec_races,
                "recent_avg_order": rec_avg,
                "recent_win_rate": rec_win,
                "recent_top3_rate": rec_top3,
                "venue_races": ven_races,
                "venue_avg_order": ven_avg,
                "venue_win_rate": ven_win,
                "venue_top3_rate": ven_top3,
                "racePoint": (record or {}).get("racePoint") or 0.0,
                "firstRate": (record or {}).get("firstRate") or 0.0,
                "secondRate": (record or {}).get("secondRate") or 0.0,
                "thirdRate": (record or {}).get("thirdRate") or 0.0,
                "exSpurt": ((record or {}).get("exSpurt") or {}).get("percentage") or 0.0,
                "exThrust": ((record or {}).get("exThrust") or {}).get("percentage") or 0.0,
                "exSplit": ((record or {}).get("exSplitLine") or {}).get("percentage") or 0.0,
                "exCompete": ((record or {}).get("exCompete") or {}).get("percentage") or 0.0,
                "exLeftBehind": ((record or {}).get("exLeftBehind") or {}).get("percentage") or 0.0,
                "exSnatch": ((record or {}).get("exSnatch") or {}).get("percentage") or 0.0,
                "lineFirstPct": ((record or {}).get("linePositionFirst") or {}).get("firstPercentage") or 0.0,
                "lineSinglePct": ((record or {}).get("lineSingleHorseman") or {}).get("firstPercentage") or 0.0,
                "min_exacta_odds_first": min_odds,
                "max_exacta_odds_first": max_odds,
                "avg_exacta_odds_first": avg_odds,
                "spread_exacta_odds_first": spread_odds,
                "min_exacta_popularity_first": min_pop,
                "avg_exacta_popularity_first": avg_pop,
                "final_min_exacta_odds_first": final_min_odds,
                "final_avg_exacta_odds_first": final_avg_odds,
                "final_min_exacta_popularity_first": final_min_pop,
                "final_avg_exacta_popularity_first": final_avg_pop,
                "field_head_to_head_wins": h2h.get(player_id, {}).get("wins", 0),
                "field_head_to_head_losses": h2h.get(player_id, {}).get("losses", 0),
                "field_head_to_head_races": h2h.get(player_id, {}).get("races", 0),
                "tipster_id": "dsc-00",
                "tipster_name": "AI予想",
                "pick_rank": picks.get(number),
                "recent_back_avg3": recent_back,
                "recent_final_half_avg3": recent_final,
                "line_id": float(line_id),
                "line_pos": float(line_pos),
                "line_size": float(line_size),
                "line_is_head": line_is_head,
                "style_code": style_code,
                "line_win_expectation": line_win_expectation,
                "line_top3_expectation": line_top3_expectation,
                "line_win_share": line_win_share,
                "line_ev_rank": line_ev_rank,
                "line_ev_gap_vs_next": line_ev_gap,
                "race_max_line_size": race_max_line_size,
                "line_signature_final_win_rate": line_signature_final_win_rate,
                "line_signature_final_races": line_signature_final_races,
                "line_region_final_win_rate": line_region_final_win_rate,
                "line_region_final_races": line_region_final_races,
                "phase_line_priority_boost": phase_line_priority_boost,
                "phase_comment_support_boost": phase_comment_support_boost,
                "phase_comment_attack_boost": phase_comment_attack_boost,
                "sacrifice_role_score": sacrifice_role_score,
                "phase_sacrifice_pressure": phase_sacrifice_pressure,
                **comment_info,
            }
            row.update(grade_features)
            row["comment_score"] = comment_info["comment_attack"] + comment_info["comment_support"] - comment_info["comment_cautious"] - comment_info["comment_negative"]
            row["finish_order"] = finish_order
            row["is_winner"] = 1 if finish_order == 1 else (None if finish_order is None else 0)
            row["is_top3"] = 1 if (finish_order is not None and finish_order <= 3) else (None if finish_order is None else 0)
            rows.append(row)

        if phase_category in ("final", "semifinal"):
            for stats in line_stats.values():
                best_finish = stats.get("best_finish")
                if best_finish is None:
                    continue
                signature_players = stats.get("line_signature_players")
                if signature_players:
                    hist = player_line_history[signature_players]
                    hist["races"] += 1.0
                    if best_finish == 1:
                        hist["wins"] += 1.0
                signature_regions = stats.get("line_signature_regions")
                if signature_regions:
                    hist = region_line_history[signature_regions]
                    hist["races"] += 1.0
                    if best_finish == 1:
                        hist["wins"] += 1.0

    return rows


def main() -> None:
    args = parse_args()
    cup_filter = set(args.cup_ids) if args.cup_ids else None
    rows = build_rows(args.venue, cup_filter, args.data_root)
    if not rows:
        raise SystemExit("No rows matched the given filters.")
    df = pd.DataFrame(rows)
    numeric_cols = [
        "number",
        "bracket_number",
        "entries_number",
        "distance_m",
        "track_length_m",
        "age",
        "player_class",
        "player_group",
        "recent_races",
        "recent_avg_order",
        "recent_win_rate",
        "recent_top3_rate",
        "venue_races",
        "venue_avg_order",
        "venue_win_rate",
        "venue_top3_rate",
        "racePoint",
        "firstRate",
        "secondRate",
        "thirdRate",
        "exSpurt",
        "exThrust",
        "exSplit",
        "exCompete",
        "exLeftBehind",
        "lineFirstPct",
        "lineSinglePct",
        "wind_speed",
        "wind_speed_post",
        "wind_speed_delta",
        "weather_code",
        "weather_post_code",
        "line_type_code",
        "line_count",
        "min_exacta_odds_first",
        "max_exacta_odds_first",
        "avg_exacta_odds_first",
        "spread_exacta_odds_first",
        "min_exacta_popularity_first",
        "avg_exacta_popularity_first",
        "final_min_exacta_odds_first",
        "final_avg_exacta_odds_first",
        "final_min_exacta_popularity_first",
        "final_avg_exacta_popularity_first",
        "field_head_to_head_wins",
        "field_head_to_head_losses",
        "field_head_to_head_races",
        "pick_rank",
        "recent_back_avg3",
        "recent_final_half_avg3",
        "line_id",
        "line_pos",
        "line_size",
        "line_is_head",
        "style_code",
        "comment_attack",
        "comment_support",
        "comment_cautious",
        "comment_negative",
        "comment_score",
        "phase_is_final",
        "phase_is_semifinal",
        "phase_is_selection",
        "phase_is_qualifier",
        "phase_importance_coeff",
        "line_win_expectation",
        "line_top3_expectation",
        "line_win_share",
        "line_ev_rank",
        "line_ev_gap_vs_next",
        "race_max_line_size",
        "line_signature_final_win_rate",
        "line_signature_final_races",
        "line_region_final_win_rate",
        "line_region_final_races",
        "phase_line_priority_boost",
        "phase_comment_support_boost",
        "phase_comment_attack_boost",
        "sacrifice_role_score",
        "phase_sacrifice_pressure",
        "cup_grade_value",
        "cup_grade_rank",
        "cup_grade_is_ladies",
        "cup_grade_is_f2",
        "cup_grade_is_f1",
        "cup_grade_is_f_class",
        "cup_grade_is_g3_or_higher",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    output = args.output or args.data_root / "gold" / "features" / f"{args.venue}_custom.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    print(f"saved {len(df)} rows -> {output}")


if __name__ == "__main__":
    main()
