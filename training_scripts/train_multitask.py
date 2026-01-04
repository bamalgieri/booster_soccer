import argparse
from pathlib import Path

import numpy as np
import torch

from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from multitask_utils import TASK_SPECS, evaluate_all_tasks, make_env


class MultiTaskEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_episodes: int,
        eval_freq: int,
        steps_weight: float,
        save_dir: Path,
        eval_seeds: list[int],
        reward_profile: str,
        deterministic: bool = True,
    ):
        super().__init__()
        self.eval_episodes = eval_episodes
        self.eval_freq = eval_freq
        self.steps_weight = steps_weight
        self.save_dir = save_dir
        self.eval_seeds = eval_seeds
        self.reward_profile = reward_profile
        self.deterministic = deterministic
        self.best_score = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        scores = evaluate_all_tasks(
            self.model,
            episodes=self.eval_episodes,
            steps_weight=self.steps_weight,
            deterministic=self.deterministic,
            seed=0,
            eval_seeds=self.eval_seeds,
            reward_profile=self.reward_profile,
        )
        s_overall = scores["S_overall"]

        for task_name, score in scores.items():
            self.logger.record(f"eval/{task_name}", score)
        if "ep_len_mean" not in scores:
            rollout_len = getattr(self.logger, "name_to_value", {}).get("rollout/ep_len_mean")
            if rollout_len is not None:
                self.logger.record("eval/ep_len_mean", rollout_len)
        self.logger.record("eval/best_S_overall", self.best_score)

        if s_overall > self.best_score:
            self.best_score = s_overall
            best_path = self.save_dir / "best_model.zip"
            self.model.save(best_path)
            print(f"[eval] New best S_overall={s_overall:.4f} -> {best_path}")

        return True


def load_bc_weights(model: TD3, path: Path) -> None:
    payload = torch.load(path, map_location=model.device)
    state_dict = payload["state_dict"] if isinstance(payload, dict) else payload
    model.policy.actor.mu.load_state_dict(state_dict)
    model.policy.actor_target.mu.load_state_dict(state_dict)


def parse_eval_seeds(raw: str | None) -> list[int]:
    if raw is None:
        return [0]
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        return [0]
    return [int(token) for token in tokens]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--net-arch", type=str, default="256,256,128")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--action-noise-sigma", type=float, default=0.0)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--eval-seeds", type=str, default="0")
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--steps-weight", type=float, default=-0.05)
    parser.add_argument("--bc-weights", type=str, default=None)
    parser.add_argument("--resume-model", type=str, default=None)
    parser.add_argument(
        "--reward-profile",
        type=str,
        default="base",
        choices=["base", "pressure_shot", "tight"],
    )
    parser.add_argument("--save-dir", type=str, default="training_runs/multitask_td3")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    net_arch = [int(v.strip()) for v in args.net_arch.split(",") if v.strip()]
    eval_seeds = parse_eval_seeds(args.eval_seeds)

    env_fns = []
    for i in range(args.n_envs):
        task_spec = TASK_SPECS[i % len(TASK_SPECS)]
        env_fns.append(make_env(task_spec, seed=args.seed + i, reward_profile=args.reward_profile))

    vec_env = VecMonitor(DummyVecEnv(env_fns))

    n_actions = int(vec_env.action_space.shape[0])
    action_noise = None
    if args.action_noise_sigma > 0:
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=args.action_noise_sigma * np.ones(n_actions),
        )

    if args.resume_model:
        model = TD3.load(args.resume_model, env=vec_env, device=args.device)
        model.learning_rate = args.lr
        model.lr_schedule = get_schedule_fn(args.lr)
        model.tau = args.tau
        model.learning_starts = args.learning_starts
        model.action_noise = action_noise
        print(f"Resumed TD3 from {args.resume_model}")
        if args.bc_weights:
            print("Ignoring --bc-weights when resuming from a checkpoint.")
    else:
        policy_kwargs = dict(net_arch=net_arch)
        model = TD3(
            "MlpPolicy",
            vec_env,
            learning_rate=args.lr,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            policy_kwargs=policy_kwargs,
            device=args.device,
            seed=args.seed,
            tau=args.tau,
            action_noise=action_noise,
            verbose=1,
        )

        if args.bc_weights:
            load_bc_weights(model, Path(args.bc_weights))
            print(f"Loaded BC weights from {args.bc_weights}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    eval_cb = MultiTaskEvalCallback(
        eval_episodes=args.eval_episodes,
        eval_freq=args.eval_freq,
        steps_weight=args.steps_weight,
        save_dir=save_dir,
        eval_seeds=eval_seeds,
        reward_profile=args.reward_profile,
        deterministic=True,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=eval_cb)

    final_path = save_dir / "final_model.zip"
    model.save(final_path)
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()
