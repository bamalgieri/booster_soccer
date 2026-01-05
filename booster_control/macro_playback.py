from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class MacroSequence:
    name: str
    actions: np.ndarray


_MACRO_FILES = {
    "kick_ball1": "kick_ball1.npz",
    "kick_ball2": "kick_ball2.npz",
    "kick_ball3": "kick_ball3.npz",
    "powerful_kick": "powerful_kick.npz",
    "pass_ball1": "pass_ball1.npz",
    "goal_kick": "goal_kick.npz",
    "soccer_drill_run": "soccer_drill_run.npz",
}

_MACRO_CACHE: Dict[tuple, np.ndarray] = {}


def _load_npz(path_or_name: str, robot: str) -> np.lib.npyio.NpzFile:
    if os.path.exists(path_or_name):
        return np.load(path_or_name, allow_pickle=False)

    filename = path_or_name
    if not filename.endswith(".npz"):
        filename = f"{filename}.npz"

    from huggingface_hub import hf_hub_download

    file_name = hf_hub_download(
        repo_id="SaiResearch/booster_dataset",
        filename=f"soccer/{robot}/{filename}",
        repo_type="dataset",
    )
    return np.load(file_name, allow_pickle=False)


def _extract_dof_targets(qpos: np.ndarray, dof_count: int) -> np.ndarray:
    if qpos.ndim != 2:
        raise ValueError(f"qpos must be 2D (T, nq). Got shape {qpos.shape}.")
    if qpos.shape[1] < 7 + dof_count:
        raise ValueError(
            f"qpos width ({qpos.shape[1]}) is too small for {dof_count} DoF targets."
        )
    return qpos[:, 7 : 7 + dof_count]


def load_kick_macro(
    name: str,
    lower_control,
    robot: str = "booster_lower_t1",
    max_steps: int = 30,
) -> MacroSequence:
    macro_key = name.lower()
    npz_name = _MACRO_FILES.get(macro_key, name)
    cache_key = (npz_name, robot, max_steps)

    if cache_key in _MACRO_CACHE:
        return MacroSequence(name=macro_key, actions=_MACRO_CACHE[cache_key].copy())

    data = _load_npz(npz_name, robot=robot)
    if "qpos" not in data:
        raise KeyError(f"'qpos' not found in macro file '{npz_name}'.")

    qpos = np.array(data["qpos"], dtype=np.float32)
    dof_count = len(lower_control.default_dof_pos)
    dof_targets = _extract_dof_targets(qpos, dof_count)

    action_scale = float(lower_control.cfg["control"]["action_scale"])
    clip_actions = float(lower_control.cfg["normalization"]["clip_actions"])
    default_dof_pos = lower_control.default_dof_pos.astype(np.float32)

    actions = (dof_targets - default_dof_pos) / action_scale
    actions = np.clip(actions, -clip_actions, clip_actions)

    if max_steps is not None and max_steps > 0:
        actions = actions[:max_steps]

    _MACRO_CACHE[cache_key] = actions
    return MacroSequence(name=macro_key, actions=actions.copy())


def _as_vector(info: Dict, key: str) -> Optional[np.ndarray]:
    if key not in info:
        return None
    value = info.get(key)
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape[-1] != 3:
        return None
    return arr


def _pick_goal_vector(info: Dict) -> Optional[np.ndarray]:
    target = _as_vector(info, "target_xpos_rel_robot")
    if target is not None:
        return target

    g0 = _as_vector(info, "goal_team_0_rel_robot")
    g1 = _as_vector(info, "goal_team_1_rel_robot")
    if g0 is None:
        return g1
    if g1 is None:
        return g0
    return g0 if np.linalg.norm(g0) <= np.linalg.norm(g1) else g1


def _is_sentinel(command: np.ndarray, threshold: float, pattern: np.ndarray) -> bool:
    if command.shape[-1] != 3:
        return False
    return bool(np.all(command * pattern >= threshold))


def trigger_macro(
    command,
    info: Dict,
    trigger_threshold: float = 0.95,
    ball_radius: float = 0.6,
    alignment_threshold: float = 0.6,
    pattern: tuple[float, float, float] = (1.0, -1.0, 1.0),
) -> bool:
    cmd = np.asarray(command, dtype=np.float32)
    if not _is_sentinel(cmd, trigger_threshold, np.asarray(pattern, dtype=np.float32)):
        return False

    ball_vec = _as_vector(info, "ball_xpos_rel_robot")
    if ball_vec is None:
        return False

    if np.linalg.norm(ball_vec) > ball_radius:
        return False

    goal_vec = _pick_goal_vector(info)
    if goal_vec is None:
        return False

    ball_norm = np.linalg.norm(ball_vec)
    goal_norm = np.linalg.norm(goal_vec)
    if ball_norm < 1e-6 or goal_norm < 1e-6:
        return False

    alignment = float(np.dot(ball_vec, goal_vec) / (ball_norm * goal_norm))
    return alignment >= alignment_threshold
