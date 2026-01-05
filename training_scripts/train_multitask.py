import argparse
import os
from pathlib import Path

import numpy as np
import torch

from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from multitask_utils import (
    TASK_SPECS,
    TaskSpec,
    build_task_list,
    evaluate_selected_tasks,
    evaluate_selected_tasks_competition,
    make_env,
)


_TASK_SUFFIXES = {
    "goalie_penalty_kick": "gpk",
    "obstacle_penalty_kick": "opk",
    "kick_to_target": "ktt",
}


class MultiTaskEvalCallback(BaseCallback):
    def __init__(
        self,
        task_specs: list[TaskSpec],
        eval_episodes: int,
        eval_freq: int,
        steps_weight: float,
        save_dir: Path,
        best_model_name: str,
        write_compat_copy: bool,
        eval_seeds: list[int],
        reward_profile: str,
        macro_reward_bonus: float,
        macro_reward_radius: float,
        macro_reward_alignment_threshold: float,
        macro_reward_cap: float,
        competition_only_eval: bool,
        legacy_only_eval: bool,
        deterministic: bool = True,
    ):
        super().__init__()
        self.eval_episodes = eval_episodes
        self.eval_freq = eval_freq
        self.steps_weight = steps_weight
        self.save_dir = save_dir
        self.best_model_name = best_model_name
        self.write_compat_copy = write_compat_copy
        self.task_specs = task_specs
        self.eval_seeds = eval_seeds
        self.reward_profile = reward_profile
        self.macro_reward_bonus = macro_reward_bonus
        self.macro_reward_radius = macro_reward_radius
        self.macro_reward_alignment_threshold = macro_reward_alignment_threshold
        self.macro_reward_cap = macro_reward_cap
        self.competition_only_eval = competition_only_eval
        self.legacy_only_eval = legacy_only_eval
        self.deterministic = deterministic
        self.best_score = -np.inf
        self.best_comp_score = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        scores = None
        if not self.competition_only_eval:
            scores = evaluate_selected_tasks(
                self.model,
                self.task_specs,
                episodes=self.eval_episodes,
                steps_weight=self.steps_weight,
                deterministic=self.deterministic,
                seed=0,
                eval_seeds=self.eval_seeds,
                reward_profile=self.reward_profile,
                macro_reward_bonus=self.macro_reward_bonus,
                macro_reward_radius=self.macro_reward_radius,
                macro_reward_alignment_threshold=self.macro_reward_alignment_threshold,
                macro_reward_cap=self.macro_reward_cap,
            )
        comp_scores = None
        if not self.legacy_only_eval:
            comp_scores = evaluate_selected_tasks_competition(
                self.model,
                self.task_specs,
                episodes=self.eval_episodes,
                deterministic=self.deterministic,
                seed=0,
                eval_seeds=self.eval_seeds,
                reward_profile=self.reward_profile,
                macro_reward_bonus=self.macro_reward_bonus,
                macro_reward_radius=self.macro_reward_radius,
                macro_reward_alignment_threshold=self.macro_reward_alignment_threshold,
                macro_reward_cap=self.macro_reward_cap,
            )
        s_overall = scores["S_overall"] if scores is not None else None
        comp_total_sum = comp_scores["comp_total_sum"] if comp_scores is not None else None
        comp_overall = None

        if scores is not None:
            for task_name, score in scores.items():
                self.logger.record(f"eval/{task_name}", score)
        if comp_scores is not None:
            for key, value in comp_scores.items():
                self.logger.record(f"eval_comp/{key}", value, exclude=("stdout",))
            for task_spec in self.task_specs:
                task_key = f"{task_spec.name}/comp_total"
                task_score = float(comp_scores.get(task_key, 0.0))
                self.logger.record(f"comp_eval/{task_spec.name}", task_score)
            comp_overall = float(comp_total_sum) if comp_total_sum is not None else 0.0
            self.logger.record("comp_eval/C_overall", comp_overall)
        if scores is not None and "ep_len_mean" not in scores:
            rollout_len = getattr(self.logger, "name_to_value", {}).get("rollout/ep_len_mean")
            if rollout_len is not None:
                self.logger.record("eval/ep_len_mean", rollout_len)
        macro_bonus_mean = getattr(self.logger, "name_to_value", {}).get(
            "rollout/macro_reward_bonus_mean"
        )
        if macro_bonus_mean is not None:
            self.logger.record("macro/reward_bonus_mean", macro_bonus_mean)
        if scores is not None:
            self.logger.record("eval/best_S_overall", self.best_score)
        if comp_scores is not None:
            self.logger.record("eval_comp/best_comp_total_sum", self.best_comp_score)

        if s_overall is not None and s_overall > self.best_score:
            self.best_score = s_overall
            best_path = self.save_dir / self.best_model_name
            self.model.save(best_path)
            if self.write_compat_copy and best_path.name != "best_model.zip":
                compat_path = self.save_dir / "best_model.zip"
                self.model.save(compat_path)
            print(f"[eval] New best S_overall={s_overall:.4f} -> {best_path}")

        if comp_overall is not None and comp_overall > self.best_comp_score:
            self.best_comp_score = comp_overall
            best_path = self.save_dir / "best_model_competition.zip"
            self.model.save(best_path)
            s_overall_str = f"{s_overall:.4f}" if s_overall is not None else "n/a"
            print(
                "[eval] New best C_overall="
                f"{comp_overall:.4f} (S_overall={s_overall_str}) -> {best_path}"
            )

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


def parse_task_weights(raw: str | None) -> dict[str, float] | None:
    if raw is None:
        return None
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        return None
    weights: dict[str, float] = {}
    valid_names = {spec.name for spec in TASK_SPECS}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"Invalid task weight entry '{token}'. Use name=value.")
        name, value = token.split("=", 1)
        name = name.strip()
        if name not in valid_names:
            raise ValueError(
                f"Unknown task name '{name}' in --task-weights. Valid: {sorted(valid_names)}"
            )
        weights[name] = float(value)
    return weights


def parse_tasks(raw: str | None) -> list[TaskSpec]:
    if raw is None:
        return list(TASK_SPECS)
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        raise ValueError("--tasks must include at least one task name.")
    spec_map = {spec.name: spec for spec in TASK_SPECS}
    invalid = [token for token in tokens if token not in spec_map]
    if invalid:
        raise ValueError(f"Unknown task name(s) in --tasks: {invalid}. Valid: {sorted(spec_map)}")
    active_specs: list[TaskSpec] = []
    seen = set()
    for token in tokens:
        if token in seen:
            continue
        active_specs.append(spec_map[token])
        seen.add(token)
    return active_specs


def resolve_best_model_name(task_specs: list[TaskSpec]) -> tuple[str, bool]:
    if len(task_specs) != 1:
        return "best_model.zip", False
    task_name = task_specs[0].name
    suffix = _TASK_SUFFIXES.get(task_name)
    if suffix is None:
        raise ValueError(f"No best model suffix configured for task '{task_name}'.")
    return f"best_model_{suffix}.zip", True


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
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--hf-home", type=str, default=None)
    parser.add_argument("--macro-reward-bonus", type=float, default=0.0)
    parser.add_argument("--macro-reward-radius", type=float, default=0.6)
    parser.add_argument("--macro-reward-alignment-threshold", type=float, default=0.6)
    parser.add_argument("--macro-reward-cap", type=float, default=0.5)
    parser.add_argument(
        "--competition-only-eval",
        action="store_true",
        help="Skip legacy eval scoring; only run competition evaluation.",
    )
    parser.add_argument(
        "--legacy-only-eval",
        action="store_true",
        help="Skip competition eval scoring; only run legacy evaluation.",
    )
    parser.add_argument(
        "--task-weights",
        type=str,
        default=None,
        help="Comma-separated task weights, e.g. goalie_penalty_kick=1,obstacle_penalty_kick=2.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task names to train/eval (default: all tasks).",
    )
    parser.add_argument(
        "--reward-profile",
        type=str,
        default="base",
        choices=["base", "pressure_shot", "tight"],
    )
    parser.add_argument("--save-dir", type=str, default="training_runs/multitask_td3")
    args = parser.parse_args()

    if args.competition_only_eval and args.legacy_only_eval:
        raise ValueError("Only one of --competition-only-eval or --legacy-only-eval may be set.")

    if args.hf_home:
        hub_cache = os.path.join(args.hf_home, "hub")
        datasets_cache = os.path.join(args.hf_home, "datasets")
        os.environ["HF_HOME"] = args.hf_home
        os.environ["HUGGINGFACE_HUB_CACHE"] = hub_cache
        os.environ["HF_HUB_CACHE"] = hub_cache
        os.environ["HF_DATASETS_CACHE"] = datasets_cache
        print(f"[train] HF_HOME set to {args.hf_home}")
        print(f"[train] HF caches set under {args.hf_home}")
    if args.hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
        os.environ["HF_TOKEN"] = args.hf_token

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    net_arch = [int(v.strip()) for v in args.net_arch.split(",") if v.strip()]
    eval_seeds = parse_eval_seeds(args.eval_seeds)

    active_task_specs = parse_tasks(args.tasks)
    task_weights = parse_task_weights(args.task_weights)
    task_list = build_task_list(args.n_envs, task_weights, task_specs=active_task_specs)
    env_fns = []
    for i, task_spec in enumerate(task_list):
        env_fns.append(
            make_env(
                task_spec,
                seed=args.seed + i,
                reward_profile=args.reward_profile,
                macro_reward_bonus=args.macro_reward_bonus,
                macro_reward_radius=args.macro_reward_radius,
                macro_reward_alignment_threshold=args.macro_reward_alignment_threshold,
                macro_reward_cap=args.macro_reward_cap,
            )
        )

    has_reward_wrapper = args.reward_profile != "base" or args.macro_reward_bonus > 0.0
    info_keywords = ("macro_reward_bonus_mean",) if has_reward_wrapper else ()
    vec_env = VecMonitor(
        DummyVecEnv(env_fns),
        info_keywords=info_keywords,
    )

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
    best_model_name, write_compat_copy = resolve_best_model_name(active_task_specs)

    eval_cb = MultiTaskEvalCallback(
        task_specs=active_task_specs,
        eval_episodes=args.eval_episodes,
        eval_freq=args.eval_freq,
        steps_weight=args.steps_weight,
        save_dir=save_dir,
        best_model_name=best_model_name,
        write_compat_copy=write_compat_copy,
        eval_seeds=eval_seeds,
        reward_profile=args.reward_profile,
        macro_reward_bonus=args.macro_reward_bonus,
        macro_reward_radius=args.macro_reward_radius,
        macro_reward_alignment_threshold=args.macro_reward_alignment_threshold,
        macro_reward_cap=args.macro_reward_cap,
        competition_only_eval=args.competition_only_eval,
        legacy_only_eval=args.legacy_only_eval,
        deterministic=True,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=eval_cb)

    final_path = save_dir / "final_model.zip"
    model.save(final_path)
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()
