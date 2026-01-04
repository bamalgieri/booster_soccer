import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

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


def load_mocap(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        qpos = np.array(data["qpos"], dtype=np.float32)
        qvel = np.array(data["qvel"], dtype=np.float32)
    return qpos, qvel


def iter_paths(root: Path, names: Iterable[str]) -> List[Path]:
    paths = []
    for name in names:
        candidate = root / name
        if candidate.exists():
            paths.append(candidate)
    return paths


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
) -> Tuple[np.ndarray, np.ndarray]:
    preprocessor = Preprocessor()
    obs_list = []
    act_list = []

    zeros_3 = np.zeros(3, dtype=np.float32)
    zeros_9 = np.zeros(9, dtype=np.float32)
    player_team = np.array([1.0, 0.0], dtype=np.float32)

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

            for task_one_hot in task_one_hots:
                obs_list.append(
                    preprocessor.modify_state(obs, info, task_one_hot).astype(
                        np.float32
                    )
                )
                act_list.append(commands[idx])
                total += 1
                if max_samples is not None and total >= max_samples:
                    break

            if max_samples is not None and total >= max_samples:
                break

    observations = np.stack(obs_list, axis=0) if obs_list else np.empty((0, 89))
    actions = np.stack(act_list, axis=0) if act_list else np.empty((0, 3))
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
    args = parser.parse_args()

    root = Path(args.input_root)
    paths = iter_paths(root, DEFAULT_FILES)
    if not paths:
        raise FileNotFoundError(f"No mocap files found under {root}")

    if args.task_mode == "zeros":
        task_one_hots = [np.zeros(3, dtype=np.float32)]
    elif args.task_mode == "all":
        task_one_hots = list(TASK_ONE_HOTS.values())
    else:
        task_one_hots = [TASK_ONE_HOTS[args.task_mode]]

    scale_lin, scale_yaw = compute_scales(paths)
    observations, actions = build_dataset(
        paths=paths,
        task_one_hots=task_one_hots,
        stride=args.stride,
        max_samples=args.max_samples,
        scale_lin=scale_lin,
        scale_yaw=scale_yaw,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, observations=observations, actions=actions)
    print(
        f"Wrote {observations.shape[0]} samples to {output} "
        f"(scale_lin={scale_lin:.3f}, scale_yaw={scale_yaw:.3f})"
    )


if __name__ == "__main__":
    main()
