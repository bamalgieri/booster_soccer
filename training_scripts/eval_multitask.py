import argparse
from pathlib import Path

from stable_baselines3 import TD3

from multitask_utils import TASK_SPECS, evaluate_all_tasks, evaluate_all_tasks_competition


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to TD3 zip.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps-weight", type=float, default=-0.001)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model)
    model = TD3.load(model_path, device=args.device)

    scores = evaluate_all_tasks(
        model,
        episodes=args.episodes,
        steps_weight=args.steps_weight,
        deterministic=args.deterministic,
        seed=0,
    )
    comp_scores = evaluate_all_tasks_competition(
        model,
        episodes=args.episodes,
        deterministic=args.deterministic,
        seed=0,
    )

    print("Per-task terminal scores:")
    for task in TASK_SPECS:
        print(f"  {task.name}: {scores[task.name]:.4f}")
    print(f"S_overall: {scores['S_overall']:.4f}")
    generate_report_card(scores)
    print("")
    print("Competition scoring (episode-end, website-aligned):")
    for task in TASK_SPECS:
        key = f"{task.name}/comp_total"
        if key in comp_scores:
            print(f"  {task.name}: {comp_scores[key]:.4f}")
    print(f"comp_total_sum: {comp_scores['comp_total_sum']:.4f}")


if __name__ == "__main__":
    main()
