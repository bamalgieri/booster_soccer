from __future__ import annotations

from typing import Dict, Iterable, List


_COMP_PENALTY_KICK = {
    "robot_distance_ball": 0.25,
    "ball_vel_twd_goal": 1.5,
    "goal_scored": 2.5,
    "offside": -3.0,
    "ball_hits": -0.2,
    "robot_fallen": -1.5,
    "ball_blocked": -0.5,
    "steps": -1.0,
}

_COMP_KICK_TO_TARGET = {
    "offside": -1.0,
    "success": 2.0,
    "distance": 0.5,
    "steps": -0.3,
}

COMP_REWARD_CONFIG: Dict[str, Dict[str, float]] = {
    "goalie_penalty_kick": dict(_COMP_PENALTY_KICK),
    "obstacle_penalty_kick": dict(_COMP_PENALTY_KICK),
    "kick_to_target": dict(_COMP_KICK_TO_TARGET),
}


def _to_float(value) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def compute_competition_report(
    reward_terms: Dict[str, float],
    task_name: str,
    steps: int,
) -> Dict[str, float]:
    if task_name not in COMP_REWARD_CONFIG:
        raise ValueError(f"Unknown task_name '{task_name}' for competition scoring.")
    if not isinstance(reward_terms, dict):
        raise TypeError("reward_terms must be a dict of component rewards.")
    if steps is None:
        raise ValueError("steps must be provided for competition scoring.")

    config = COMP_REWARD_CONFIG[task_name]
    report: Dict[str, float] = {}
    missing = 0
    total = 0.0
    step_value = int(steps)

    for component, weight in config.items():
        if component == "steps":
            value = step_value
        else:
            if component in reward_terms:
                value = reward_terms[component]
            else:
                value = 0.0
                missing += 1
        contribution = float(weight) * _to_float(value)
        report[f"comp_{component}"] = contribution
        total += contribution

    report["comp_total"] = float(total)
    report["missing_component_count"] = int(missing)
    return report


def compute_competition_score(
    reward_terms: Dict[str, float],
    task_name: str,
    steps: int,
) -> float:
    report = compute_competition_report(reward_terms, task_name, steps)
    return float(report["comp_total"])


def aggregate_competition_reports(
    task_name: str,
    reports: Iterable[Dict[str, float]],
) -> Dict[str, float]:
    """Aggregate per-episode reports into per-episode means."""
    if task_name not in COMP_REWARD_CONFIG:
        raise ValueError(f"Unknown task_name '{task_name}' for competition scoring.")

    config = COMP_REWARD_CONFIG[task_name]
    component_keys: List[str] = [f"comp_{key}" for key in config.keys()]
    totals = {key: 0.0 for key in component_keys}
    total_sum = 0.0
    missing_sum = 0.0
    count = 0

    for report in reports:
        if not isinstance(report, dict):
            continue
        count += 1
        total_sum += float(report.get("comp_total", 0.0))
        missing_sum += float(report.get("missing_component_count", 0.0))
        for key in component_keys:
            totals[key] += float(report.get(key, 0.0))

    if count == 0:
        count = 1

    aggregated = {key: value / count for key, value in totals.items()}
    aggregated["comp_total"] = total_sum / count
    aggregated["missing_component_count"] = missing_sum / count
    return aggregated
