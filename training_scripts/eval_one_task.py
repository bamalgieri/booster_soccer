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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--algo",
        type=str,
        default="td3",
        choices=["td3", "sac", "ppo"],
        help="SB3 algorithm used to train the model.",
    )
    parser.add_argument("--episodes", type=int, default=1)
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

    algo_map = {"td3": TD3, "sac": SAC, "ppo": PPO}
    model_path = Path(args.model)
    model = algo_map[args.algo.lower()].load(model_path, device=args.device)

    task_specs = [ts for ts in TASK_SPECS if ts.name == args.task]
    if not task_specs:
        available = ", ".join([ts.name for ts in TASK_SPECS])
        raise ValueError(f"Unknown task: {args.task}. Available: {available}")
    task_spec = task_specs[0]

    obs_rms = None
    obs_rms_epsilon = 1e-8
    if args.vecnormalize_path:
        vecnormalize_path = Path(args.vecnormalize_path)
        dummy_env = DummyVecEnv([make_env(task_spec)])
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
    print(f"{task_spec.name}: {scores[task_spec.name]:.4f}")
    print(f"S_overall: {scores['S_overall']:.4f}")
    comp_key = f"{task_spec.name}/comp_total"
    if comp_key in comp_scores:
        print(f"{task_spec.name} comp_total: {comp_scores[comp_key]:.4f}")


if __name__ == "__main__":
    main()
