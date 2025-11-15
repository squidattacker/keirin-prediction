"""Feature extraction helpers for training/inference."""

from __future__ import annotations

from typing import Any, Dict, List

FEATURE_COLUMNS = [
    "number",
    "bracket_number",
    "racePoint",
    "firstRate",
    "secondRate",
    "thirdRate",
    "exSpurt",
    "exThrust",
    "exSplit",
    "exCompete",
    "exLeftBehind",
    "exSnatch",
    "lineFirstPct",
    "lineSinglePct",
    "recent_races",
    "recent_avg_order",
    "recent_win_rate",
    "recent_top3_rate",
    "recent_back_avg3",
    "recent_final_half_avg3",
    "wind_speed",
    "wind_speed_post",
    "wind_speed_delta",
    "weather_code",
    "weather_post_code",
    "venue_races",
    "venue_avg_order",
    "venue_win_rate",
    "venue_top3_rate",
    "track_length_m",
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
    "age",
    "player_class",
    "player_group",
    "entries_number",
    "distance_m",
    "line_type_code",
    "line_count",
    "pick_rank",
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
FEATURE_COLUMNS = list(dict.fromkeys(FEATURE_COLUMNS))


def _extract_ex(entry: Dict[str, Any], key: str) -> float:
    ex = entry.get("ex", {}) or {}
    value = ex.get(key)
    if isinstance(value, dict):
        return float(value.get("percentage") or value.get("value") or 0.0)
    return float(value or 0.0)


def extract_entry_features(entry: Dict[str, Any]) -> Dict[str, float]:
    """Return numeric features for a single race entry."""

    return {
        "racePoint": float(entry.get("racePoint") or 0.0),
        "firstRate": float(entry.get("firstRate") or 0.0),
        "secondRate": float(entry.get("secondRate") or 0.0),
        "thirdRate": float(entry.get("thirdRate") or 0.0),
        "number": float(entry.get("number") or 0.0),
        "exSpurt": _extract_ex(entry, "exSpurt"),
        "exThrust": _extract_ex(entry, "exThrust"),
    }


def vectorize_features(feature_dict: Dict[str, float], columns: List[str]) -> List[float]:
    return [float(feature_dict.get(col, 0.0)) for col in columns]
