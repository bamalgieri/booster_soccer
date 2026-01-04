import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _append_arg(cmd: list[str], flag: str, value) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def _set_flag(cmd: list[str], flag: str, value) -> None:
    if value is None:
        return
    if flag in cmd:
        idx = cmd.index(flag)
        if idx + 1 < len(cmd):
            cmd[idx + 1] = str(value)
        else:
            cmd.append(str(value))
        return
    cmd.extend([flag, str(value)])


def _build_base_cmd(script_path: Path, args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, str(script_path)]
    _append_arg(cmd, "--n-envs", args.n_envs)
    _append_arg(cmd, "--device", args.device)
    _append_arg(cmd, "--net-arch", args.net_arch)
    _append_arg(cmd, "--buffer-size", args.buffer_size)
    _append_arg(cmd, "--batch-size", args.batch_size)
    _append_arg(cmd, "--eval-freq", args.eval_freq)
    _append_arg(cmd, "--steps-weight", args.steps_weight)
    return cmd


def _run_stage(stage_name: str, cmd: list[str], stage_dir: Path) -> Path:
    print(f"[overnight] Starting {stage_name}...")
    subprocess.run(cmd, check=True)
    best_model = stage_dir / "best_model.zip"
    if not best_model.exists():
        raise FileNotFoundError(f"{stage_name} did not produce {best_model}")
    print(f"[overnight] {stage_name} best model: {best_model}")
    return best_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-model", type=str, required=True, help="Stage 1 checkpoint zip.")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--net-arch", type=str, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--steps-weight", type=float, default=None)
    parser.add_argument("--stage1-timesteps", type=int, default=None)
    parser.add_argument("--stage2-timesteps", type=int, default=None)
    parser.add_argument("--stage3-timesteps", type=int, default=None)
    args = parser.parse_args()

    resume_path = Path(args.resume_model)
    if not resume_path.exists():
        raise FileNotFoundError(f"Stage 1 resume model not found: {resume_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = Path("training_runs") / f"overnight_{timestamp}"
    stage_dirs = {
        "stage1": root_dir / "stage1",
        "stage2": root_dir / "stage2",
        "stage3": root_dir / "stage3",
    }
    for stage_dir in stage_dirs.values():
        stage_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve().parent / "train_multitask.py"
    base_cmd = _build_base_cmd(script_path, args)

    stage1_cfg = {
        "lr": 1e-4,
        "tau": 0.003,
        "action_noise_sigma": 0.1,
        "learning_starts": 10_000,
        "eval_episodes": 20,
        "eval_seeds": "0,1,2",
        "reward_profile": "base",
        "seed": args.base_seed,
        "total_timesteps": args.stage1_timesteps,
        "steps_weight": -0.05,
        "stage_dir": stage_dirs["stage1"],
    }
    stage2_cfg = {
        "lr": 1e-4,
        "tau": 0.003,
        "action_noise_sigma": 0.1,
        "learning_starts": 10_000,
        "eval_episodes": 20,
        "eval_seeds": "0,1,2",
        "reward_profile": "pressure_shot",
        "seed": args.base_seed + 10,
        "total_timesteps": args.stage2_timesteps,
        "stage_dir": stage_dirs["stage2"],
    }
    stage3_cfg = {
        "lr": 5e-5,
        "tau": 0.002,
        "action_noise_sigma": 0.05,
        "learning_starts": 10_000,
        "eval_episodes": 50,
        "eval_seeds": "0,1,2,3,4",
        "reward_profile": "tight",
        "seed": args.base_seed + 20,
        "total_timesteps": args.stage3_timesteps,
        "steps_weight": -0.5,
        "stage_dir": stage_dirs["stage3"],
    }

    def stage_cmd(
        base_cmd: list[str],
        stage_cfg: dict,
        resume_model: str | Path | None = None,
        stage_steps: int | None = None,
    ) -> list[str]:
        cmd = list(base_cmd)
        stage_dir = stage_cfg["stage_dir"]
        cmd.extend(
            [
                "--save-dir",
                str(stage_dir),
                "--seed",
                str(stage_cfg["seed"]),
                "--lr",
                str(stage_cfg["lr"]),
                "--tau",
                str(stage_cfg["tau"]),
                "--action-noise-sigma",
                str(stage_cfg["action_noise_sigma"]),
                "--learning-starts",
                str(stage_cfg["learning_starts"]),
                "--eval-episodes",
                str(stage_cfg["eval_episodes"]),
                "--eval-seeds",
                stage_cfg["eval_seeds"],
                "--reward-profile",
                stage_cfg["reward_profile"],
            ]
        )
        if resume_model is not None:
            cmd.extend(["--resume-model", str(resume_model)])
        sw = stage_cfg.get("steps_weight", -0.01)
        _set_flag(cmd, "--steps-weight", sw)
        total_steps = stage_steps if stage_steps is not None else stage_cfg.get("total_timesteps")
        if total_steps is not None:
            cmd.extend(["--total-timesteps", str(total_steps)])
        return cmd

    stage1_best = _run_stage(
        "stage1",
        stage_cmd(base_cmd, stage1_cfg, resume_path),
        stage_dirs["stage1"],
    )
    stage2_best = _run_stage(
        "stage2",
        stage_cmd(base_cmd, stage2_cfg, stage1_best),
        stage_dirs["stage2"],
    )
    stage3_best = _run_stage(
        "stage3",
        stage_cmd(base_cmd, stage3_cfg, stage2_best),
        stage_dirs["stage3"],
    )

    print("[overnight] Complete.")
    print(f"[overnight] Run dir: {root_dir}")
    print(f"[overnight] Stage 1 best: {stage1_best}")
    print(f"[overnight] Stage 2 best: {stage2_best}")
    print(f"[overnight] Stage 3 best: {stage3_best}")


if __name__ == "__main__":
    main()
