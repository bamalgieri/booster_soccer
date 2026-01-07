import argparse
from pathlib import Path

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from multitask_utils import (
    TASK_SPECS,
    evaluate_selected_tasks,
    evaluate_selected_tasks_competition,
    make_env,
)


def generate_report_card(per_task_scores):
    """
    Converts Env-Style terminal scores to Estimated Competition Scores.
    Target Benchmark: ~8.0 Total (Sum of 3 tasks)
    """
    header = f"{'task':<24} {'RAW (ENV)':>10} {'PROJ (COMP)':>12}"
    print("Competition Score Projection Report Card")
    print(header)
    print("-" * len(header))
    total_projected = 0.0
    for task in TASK_SPECS:
        if task.name not in per_task_scores:
            continue
        raw_score = float(per_task_scores[task.name])
        if "penalty_kick" in task.name:
            if raw_score > 15.0:
                projected = 2.0
            else:
                projected = -1.0 + (raw_score * 0.05)
                projected = max(-4.0, min(-0.5, projected))
        elif "kick_to_target" in task.name:
            projected = raw_score * 0.1
        else:
            projected = raw_score
        total_projected += projected
        print(f"{task.name:<24} {raw_score:>10.4f} {projected:>12.4f}")
    print("-" * len(header))
    print(f"{'TOTAL PROJ':<24} {'':>10} {total_projected:>12.4f}")
    print(f"{'BENCHMARK':<24} {'':>10} {8.0:>12.4f}")


def _parse_tasks(raw: str | None):
    if not raw:
        return TASK_SPECS
    requested = [token.strip() for token in raw.split(",") if token.strip()]
    if not requested:
        return TASK_SPECS
    task_map = {task.name: task for task in TASK_SPECS}
    missing = [name for name in requested if name not in task_map]
    if missing:
        available = ", ".join(sorted(task_map.keys()))
        raise ValueError(f"Unknown tasks: {missing}. Available: {available}")
    return [task_map[name] for name in requested]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to SB3 zip.")
    parser.add_argument(
        "--algo",
        type=str,
        default="td3",
        choices=["td3", "sac", "ppo"],
        help="SB3 algorithm used to train the model.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task names (default: all tasks).",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps-weight", type=float, default=-0.001)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--vecnormalize-path",
        dest="vecnormalize_path",
        type=str,
        default=None,
        help="Path to VecNormalize stats (vecnormalize.pkl) to normalize observations.",
    )
    parser.add_argument(
        "--vecnormalize",
        dest="vecnormalize_path",
        type=str,
        default=None,
        help="Deprecated alias for --vecnormalize-path.",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    algo_map = {"td3": TD3, "sac": SAC, "ppo": PPO}
    model = algo_map[args.algo.lower()].load(model_path, device=args.device)
    task_specs = _parse_tasks(args.tasks)
    if not task_specs:
        raise ValueError("No tasks selected for evaluation.")
    obs_rms = None
    obs_rms_epsilon = 1e-8
    if args.vecnormalize_path:
        vecnormalize_path = Path(args.vecnormalize_path)
        dummy_env = DummyVecEnv([make_env(task_specs[0])])
        vec_norm = VecNormalize.load(vecnormalize_path, dummy_env)
        vec_norm.training = False
        vec_norm.norm_reward = False
        obs_rms = vec_norm.obs_rms
        obs_rms_epsilon = vec_norm.epsilon
        print(f"[eval] Loaded VecNormalize stats from {vecnormalize_path}")

    scores = evaluate_selected_tasks(
        model,
        task_specs,
        episodes=args.episodes,
        steps_weight=args.steps_weight,
        deterministic=args.deterministic,
        seed=0,
        obs_rms=obs_rms,
        obs_rms_epsilon=obs_rms_epsilon,
    )
    comp_scores = evaluate_selected_tasks_competition(
        model,
        task_specs,
        episodes=args.episodes,
        deterministic=args.deterministic,
        seed=0,
        obs_rms=obs_rms,
        obs_rms_epsilon=obs_rms_epsilon,
    )

    print("Per-task terminal scores:")
    for task in task_specs:
        print(f"  {task.name}: {scores[task.name]:.4f}")
    print(f"S_overall: {scores['S_overall']:.4f}")
    generate_report_card(scores)
    print("")
    print("Competition scoring (episode-end, website-aligned):")
    for task in task_specs:
        key = f"{task.name}/comp_total"
        if key in comp_scores:
            print(f"  {task.name}: {comp_scores[key]:.4f}")
    print(f"comp_total_sum: {comp_scores['comp_total_sum']:.4f}")


if __name__ == "__main__":
    main()
