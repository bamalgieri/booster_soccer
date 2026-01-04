import argparse
import numpy as np
from stable_baselines3 import TD3
from multitask_utils import TASK_SPECS, evaluate_all_tasks  # reuse existing scoring

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--steps-weight", type=float, default=-0.001)
    p.add_argument("--deterministic", action="store_true")
    args = p.parse_args()

    # Load model (env will be created inside evaluate_all_tasks if that's how yours works;
    # otherwise you'd need make_env here)
    model = TD3.load(args.model)

    # Filter tasks
    task_specs = [ts for ts in TASK_SPECS if ts.get("name") == args.task or ts.get("task_name") == args.task]
    if not task_specs:
        raise ValueError(f"Unknown task: {args.task}. Available: {[ts.get('name', ts.get('task_name')) for ts in TASK_SPECS]}")

    # Temporarily eval only that task by calling your evaluator in a task-scoped way.
    # If evaluate_all_tasks always loops TASK_SPECS internally, you’ll need to add a parameter there.
    scores = evaluate_all_tasks(
        model,
        episodes=args.episodes,
        steps_weight=args.steps_weight,
        deterministic=args.deterministic,
        seed=0,
        task_specs=task_specs,  # add support if needed
    )
    print(scores)

if __name__ == "__main__":
    main()
