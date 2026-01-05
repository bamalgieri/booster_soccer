import argparse

import numpy as np

from multitask_utils import evaluate_all_tasks, evaluate_all_tasks_competition


class ZeroPolicy:
    def __init__(self, action_dim: int = 3) -> None:
        self._action = np.zeros(action_dim, dtype=np.float32)

    def predict(self, obs, deterministic: bool = True):
        return self._action.copy(), None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps-weight", type=float, default=-0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = ZeroPolicy()

    scores = evaluate_all_tasks(
        model,
        episodes=args.episodes,
        steps_weight=args.steps_weight,
        deterministic=True,
        seed=args.seed,
    )
    comp_scores = evaluate_all_tasks_competition(
        model,
        episodes=args.episodes,
        deterministic=True,
        seed=args.seed,
    )

    print("[smoke] eval keys:", sorted(scores.keys()))
    print("[smoke] eval_comp keys:", sorted(comp_scores.keys()))


if __name__ == "__main__":
    main()
