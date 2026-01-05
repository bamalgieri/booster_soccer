import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from imitation_learning.scripts.preprocessor import Preprocessor


TASK_ONE_HOTS = {
    "goalie": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "obstacle": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "kick": np.array([0.0, 0.0, 1.0], dtype=np.float32),
}

DEFAULT_FILES = [
    "goal_kick.npz",
    "jogging.npz",
    "kick_ball1.npz",
    "kick_ball2.npz",
    "kick_ball3.npz",
    "pass_ball1.npz",
    "powerful_kick.npz",
    "running.npz",
    "soccer_drill_run.npz",
    "walking.npz",
]

DEFAULT_SENTINEL_FILES = [
    "kick_ball1.npz",
    "kick_ball2.npz",
    "kick_ball3.npz",
    "pass_ball1.npz",
    "powerful_kick.npz",
    "goal_kick.npz",
]


def load_mocap(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        qpos = np.array(data["qpos"], dtype=np.float32)
        qvel = np.array(data["qvel"], dtype=np.float32)
    return qpos, qvel


def _normalize_name(name: str) -> str:
    name = name.strip()
    if not name:
        return ""
    return name if name.endswith(".npz") else f"{name}.npz"


def _parse_name_list(value: str) -> List[str]:
    if not value:
        return []
    return [name for raw in value.split(",") if (name := _normalize_name(raw))]


def _parse_pattern(value: str) -> np.ndarray:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("Sentinel pattern must have 3 comma-separated values.")
    pattern = np.array([float(p) for p in parts], dtype=np.float32)
    if np.any(np.abs(pattern) > 1.0):
        raise ValueError("Sentinel pattern values must be within [-1, 1].")
    return pattern


def _parse_frame_spec(spec: str) -> List[int]:
    if not spec:
        return []
    frames: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid frame range '{token}'.")
            frames.update(range(start, end + 1))
        elif ":" in token:
            start_str, end_str = token.split(":", 1)
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else 0
            if end <= start:
                raise ValueError(f"Invalid frame slice '{token}'.")
            frames.update(range(start, end))
        else:
            frames.add(int(token))
    return sorted(frames)


def iter_paths(
    root: Path, names: Iterable[str], require_all: bool = False
) -> Tuple[List[Path], List[str]]:
    paths = []
    missing = []
    for name in names:
        candidate = root / name
        if candidate.exists():
            paths.append(candidate)
        else:
            missing.append(name)
    if missing and require_all:
        raise FileNotFoundError(f"Missing mocap files: {missing}")
    return paths, missing


def compute_scales(paths: List[Path]) -> Tuple[float, float]:
    max_lin = 0.0
    max_yaw = 0.0
    for path in paths:
        _, qvel = load_mocap(path)
        if qvel.shape[1] < 6:
            continue
        max_lin = max(max_lin, float(np.max(np.abs(qvel[:, 0:2]))))
        max_yaw = max(max_yaw, float(np.max(np.abs(qvel[:, 5]))))
    scale_lin = max(1.0, max_lin)
    scale_yaw = max(1.0, max_yaw)
    return scale_lin, scale_yaw


def build_dataset(
    paths: List[Path],
    task_one_hots: List[np.ndarray],
    stride: int,
    max_samples: int | None,
    scale_lin: float,
    scale_yaw: float,
    sentinel_files: set[str],
    sentinel_frames: set[int],
    sentinel_pattern: np.ndarray,
    context_mode: str,
    ball_radius: float,
    ball_far: float,
    goal_distance: float,
    sentinel_threshold: float,
    return_stats: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, Dict]:
    preprocessor = Preprocessor()
    obs_list = []
    act_list = []

    zeros_3 = np.zeros(3, dtype=np.float32)
    zeros_9 = np.zeros(9, dtype=np.float32)
    player_team = np.array([1.0, 0.0], dtype=np.float32)

    ball_near = np.array([ball_radius * 0.5, 0.0, 0.0], dtype=np.float32)
    goal_near = np.array([goal_distance, 0.0, 0.0], dtype=np.float32)
    ball_far_vec = np.array([ball_far, 0.0, 0.0], dtype=np.float32)
    goal_far = np.array([-goal_distance, 0.0, 0.0], dtype=np.float32)

    sentinel_total = 0
    file_counts: Dict[str, Dict[str, int]] = {}
    total = 0
    for path in paths:
        qpos, qvel = load_mocap(path)
        if qpos.shape[0] == 0:
            continue

        joint_pos = qpos[:, 7:19]
        joint_vel = qvel[:, 6:18]

        robot_quat = qpos[:, 3:7]
        robot_quat = robot_quat[:, [1, 2, 3, 0]]

        robot_gyro = qvel[:, 3:6]
        robot_velocimeter = qvel[:, 0:3]

        commands = np.stack(
            [
                qvel[:, 0] / scale_lin,
                qvel[:, 1] / scale_lin,
                qvel[:, 5] / scale_yaw,
            ],
            axis=1,
        )
        commands = np.clip(commands, -1.0, 1.0).astype(np.float32)

        use_sentinels = path.name in sentinel_files
        file_counts.setdefault(path.name, {"total": 0, "sentinel": 0})

        for idx in range(0, qpos.shape[0], stride):
            obs = np.concatenate([joint_pos[idx], joint_vel[idx]], dtype=np.float32)
            info = {
                "robot_quat": robot_quat[idx],
                "robot_gyro": robot_gyro[idx],
                "robot_accelerometer": zeros_3,
                "robot_velocimeter": robot_velocimeter[idx],
                "goal_team_0_rel_robot": zeros_3,
                "goal_team_1_rel_robot": zeros_3,
                "goal_team_0_rel_ball": zeros_3,
                "goal_team_1_rel_ball": zeros_3,
                "ball_xpos_rel_robot": zeros_3,
                "ball_velp_rel_robot": zeros_3,
                "ball_velr_rel_robot": zeros_3,
                "player_team": player_team,
                "goalkeeper_team_0_xpos_rel_robot": zeros_3,
                "goalkeeper_team_0_velp_rel_robot": zeros_3,
                "goalkeeper_team_1_xpos_rel_robot": zeros_3,
                "goalkeeper_team_1_velp_rel_robot": zeros_3,
                "target_xpos_rel_robot": zeros_3,
                "target_velp_rel_robot": zeros_3,
                "defender_xpos": zeros_9,
            }

            use_sentinel = use_sentinels and idx in sentinel_frames
            action = commands[idx]
            if use_sentinel:
                action = sentinel_pattern

            if context_mode != "keep":
                if use_sentinel:
                    info["ball_xpos_rel_robot"] = ball_near
                    info["goal_team_0_rel_robot"] = goal_near
                    info["goal_team_1_rel_robot"] = goal_near
                    info["target_xpos_rel_robot"] = goal_near
                    goal_rel_ball = goal_near - ball_near
                    info["goal_team_0_rel_ball"] = goal_rel_ball
                    info["goal_team_1_rel_ball"] = goal_rel_ball
                elif context_mode == "all":
                    info["ball_xpos_rel_robot"] = ball_far_vec
                    info["goal_team_0_rel_robot"] = goal_far
                    info["goal_team_1_rel_robot"] = goal_far
                    info["target_xpos_rel_robot"] = goal_far
                    goal_rel_ball = goal_far - ball_far_vec
                    info["goal_team_0_rel_ball"] = goal_rel_ball
                    info["goal_team_1_rel_ball"] = goal_rel_ball

            for task_one_hot in task_one_hots:
                obs_list.append(
                    preprocessor.modify_state(obs, info, task_one_hot).astype(
                        np.float32
                    )
                )
                act_list.append(action)
                file_counts[path.name]["total"] += 1
                if np.all(action * sentinel_pattern >= sentinel_threshold):
                    sentinel_total += 1
                    file_counts[path.name]["sentinel"] += 1
                total += 1
                if max_samples is not None and total >= max_samples:
                    break

            if max_samples is not None and total >= max_samples:
                break

    observations = np.stack(obs_list, axis=0) if obs_list else np.empty((0, 89))
    actions = np.stack(act_list, axis=0) if act_list else np.empty((0, 3))
    if return_stats:
        stats = {
            "total": total,
            "sentinel_total": sentinel_total,
            "by_file": file_counts,
        }
        return observations.astype(np.float32), actions.astype(np.float32), stats
    return observations.astype(np.float32), actions.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=str,
        default="booster_dataset/soccer/booster_lower_t1",
        help="Directory with Lower T1 mocap npz files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="booster_dataset/imitation_learning/bc_commands.npz",
        help="Output dataset path.",
    )
    parser.add_argument(
        "--task-mode",
        type=str,
        default="all",
        choices=["all", "goalie", "obstacle", "kick", "zeros"],
        help="Task one-hot assignment strategy.",
    )
    parser.add_argument(
        "--files",
        type=str,
        default="",
        help="Comma-separated list of mocap npz filenames to include.",
    )
    parser.add_argument(
        "--exclude-files",
        type=str,
        default="",
        help="Comma-separated list of mocap npz filenames to skip.",
    )
    parser.add_argument(
        "--strict-files",
        action="store_true",
        help="Fail if any requested files are missing.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Stride for subsampling motion frames.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on total samples.",
    )
    parser.add_argument(
        "--enable-sentinels",
        action="store_true",
        help="Inject sentinel actions into selected frames.",
    )
    parser.add_argument(
        "--sentinel-files",
        type=str,
        default="",
        help="Comma-separated list of mocap files to inject sentinels into.",
    )
    parser.add_argument(
        "--sentinel-frames",
        type=str,
        default="0-2",
        help="Frame indices per clip for sentinel injection (e.g. 0-2 or 0,1,2).",
    )
    parser.add_argument(
        "--sentinel-pattern",
        type=str,
        default="1,-1,1",
        help="Comma-separated sentinel action pattern.",
    )
    parser.add_argument(
        "--sentinel-threshold",
        type=float,
        default=0.95,
        help="Threshold used for reporting sentinel match counts.",
    )
    parser.add_argument(
        "--context-mode",
        type=str,
        default="keep",
        choices=["keep", "sentinel", "all"],
        help="Context shaping mode for ball/goal vectors.",
    )
    parser.add_argument(
        "--ball-radius",
        type=float,
        default=0.6,
        help="Ball radius used for sentinel context shaping.",
    )
    parser.add_argument(
        "--ball-far",
        type=float,
        default=3.0,
        help="Ball distance used for non-sentinel context shaping.",
    )
    parser.add_argument(
        "--goal-distance",
        type=float,
        default=2.0,
        help="Goal distance used for context shaping.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print per-file sentinel statistics.",
    )
    args = parser.parse_args()

    root = Path(args.input_root)
    include_names = _parse_name_list(args.files) or list(DEFAULT_FILES)
    exclude_names = set(_parse_name_list(args.exclude_files))
    include_names = [name for name in include_names if name not in exclude_names]

    paths, missing = iter_paths(root, include_names, require_all=args.strict_files)
    if not paths:
        raise FileNotFoundError(f"No mocap files found under {root}")
    if missing:
        print(f"Warning: missing mocap files: {missing}")

    if args.task_mode == "zeros":
        task_one_hots = [np.zeros(3, dtype=np.float32)]
    elif args.task_mode == "all":
        task_one_hots = list(TASK_ONE_HOTS.values())
    else:
        task_one_hots = [TASK_ONE_HOTS[args.task_mode]]

    sentinel_files = set()
    sentinel_frames: set[int] = set()
    sentinel_pattern = np.array([1.0, -1.0, 1.0], dtype=np.float32)
    if args.enable_sentinels:
        sentinel_list = _parse_name_list(args.sentinel_files)
        if not sentinel_list:
            sentinel_list = list(DEFAULT_SENTINEL_FILES)
        sentinel_files = set(sentinel_list)
        sentinel_frames = set(_parse_frame_spec(args.sentinel_frames))
        sentinel_pattern = _parse_pattern(args.sentinel_pattern)
        if args.context_mode == "keep":
            pass
        elif args.context_mode not in {"sentinel", "all"}:
            raise ValueError(f"Invalid context mode: {args.context_mode}")
    elif args.context_mode != "keep":
        raise ValueError("Context shaping requires --enable-sentinels.")

    if args.ball_far <= args.ball_radius:
        raise ValueError("--ball-far must be larger than --ball-radius.")

    scale_lin, scale_yaw = compute_scales(paths)
    observations, actions, stats = build_dataset(
        paths=paths,
        task_one_hots=task_one_hots,
        stride=args.stride,
        max_samples=args.max_samples,
        scale_lin=scale_lin,
        scale_yaw=scale_yaw,
        sentinel_files=sentinel_files,
        sentinel_frames=sentinel_frames,
        sentinel_pattern=sentinel_pattern,
        context_mode=args.context_mode,
        ball_radius=args.ball_radius,
        ball_far=args.ball_far,
        goal_distance=args.goal_distance,
        sentinel_threshold=args.sentinel_threshold,
        return_stats=True,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, observations=observations, actions=actions)
    sentinel_total = stats["sentinel_total"]
    total = stats["total"]
    sentinel_pct = (sentinel_total / total * 100.0) if total else 0.0
    print(
        f"Wrote {observations.shape[0]} samples to {output} "
        f"(scale_lin={scale_lin:.3f}, scale_yaw={scale_yaw:.3f})"
    )
    print(
        f"Sentinel samples: {sentinel_total}/{total} ({sentinel_pct:.2f}%) "
        f"| context_mode={args.context_mode}"
    )
    if args.report:
        for name, counts in stats["by_file"].items():
            if counts["total"] == 0:
                continue
            pct = counts["sentinel"] / counts["total"] * 100.0
            print(f"{name}: {counts['sentinel']}/{counts['total']} ({pct:.2f}%)")


if __name__ == "__main__":
    main()
