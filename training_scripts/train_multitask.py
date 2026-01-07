import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

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

_ALGO_CLASSES = {
    "td3": TD3,
    "ppo": PPO,
    "sac": SAC,
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
        obs_rms=None,
        obs_rms_epsilon: float = 1e-8,
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
        self.obs_rms = obs_rms
        self.obs_rms_epsilon = obs_rms_epsilon
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
                obs_rms=self.obs_rms,
                obs_rms_epsilon=self.obs_rms_epsilon,
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
                obs_rms=self.obs_rms,
                obs_rms_epsilon=self.obs_rms_epsilon,
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


def freeze_linear_layers(
    module: nn.Module, max_layers: int
) -> tuple[list[torch.nn.Parameter], int]:
    frozen_params: list[torch.nn.Parameter] = []
    frozen_layers = 0
    if max_layers <= 0:
        return frozen_params, frozen_layers
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            frozen_layers += 1
            for param in layer.parameters():
                param.requires_grad = False
                frozen_params.append(param)
            if frozen_layers >= max_layers:
                break
    return frozen_params, frozen_layers


class FreezeBCLayersCallback(BaseCallback):
    def __init__(self, frozen_params: list[torch.nn.Parameter], unfreeze_step: int):
        super().__init__()
        self.frozen_params = frozen_params
        self.unfreeze_step = unfreeze_step
        self._unfrozen = False

    def _on_step(self) -> bool:
        if self._unfrozen or not self.frozen_params:
            return True
        if self.num_timesteps > self.unfreeze_step:
            for param in self.frozen_params:
                param.requires_grad = True
            self._unfrozen = True
            print(
                f"[train] Unfroze {len(self.frozen_params)} parameters at step "
                f"{self.num_timesteps}."
            )
        return True


class EntropyFloorCallback(BaseCallback):
    def __init__(self, min_ent_coef: float):
        super().__init__()
        self.min_ent_coef = float(min_ent_coef)
        self._log_min = math.log(self.min_ent_coef)

    def _on_step(self) -> bool:
        log_ent_coef = getattr(self.model, "log_ent_coef", None)
        if log_ent_coef is None:
            return True
        with torch.no_grad():
            current = torch.exp(log_ent_coef)
            if torch.any(current < self.min_ent_coef):
                log_ent_coef.copy_(torch.full_like(log_ent_coef, self._log_min))
        return True


def split_bc_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    layer_indices = sorted({int(key.split(".", 1)[0]) for key in state_dict})
    if not layer_indices:
        raise ValueError("BC weights missing layer indices.")
    last_index = layer_indices[-1]
    hidden_state = {
        key: value
        for key, value in state_dict.items()
        if int(key.split(".", 1)[0]) < last_index
    }
    action_state = {
        key.split(".", 1)[1]: value
        for key, value in state_dict.items()
        if int(key.split(".", 1)[0]) == last_index
    }
    return hidden_state, action_state


def load_bc_weights(model, path: Path, algo: str) -> None:
    payload = torch.load(path, map_location=model.device)
    state_dict = payload["state_dict"] if isinstance(payload, dict) else payload
    algo = algo.lower()
    if algo == "td3":
        model.policy.actor.mu.load_state_dict(state_dict)
        model.policy.actor_target.mu.load_state_dict(state_dict)
        return
    hidden_state, action_state = split_bc_state_dict(state_dict)
    if algo == "ppo":
        model.policy.mlp_extractor.policy_net.load_state_dict(hidden_state)
        model.policy.action_net.load_state_dict(action_state)
        return
    if algo == "sac":
        model.policy.actor.latent_pi.load_state_dict(hidden_state)
        model.policy.actor.mu.load_state_dict(action_state)
        return
    raise ValueError(f"Unsupported algo for BC weight loading: {algo}")


def parse_eval_seeds(raw: str | None) -> list[int]:
    if raw is None:
        return [0]
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        return [0]
    return [int(token) for token in tokens]


def resolve_eval_seeds(raw: str | None, random_count: int, seed: int) -> list[int]:
    if random_count and random_count > 0:
        rng = np.random.default_rng(seed)
        return list(map(int, rng.integers(0, 10_000, size=random_count)))
    return parse_eval_seeds(raw)


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
    parser.add_argument(
        "--algo",
        type=str,
        default="td3",
        choices=sorted(_ALGO_CLASSES.keys()),
        help="RL algorithm to use for multitask training (default: td3).",
    )
    parser.add_argument("--net-arch", type=str, default="256,256,128")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=None,
        help="Fixed entropy coefficient for PPO/SAC (default: PPO=0.01, SAC=auto).",
    )
    parser.add_argument(
        "--ent-coef-auto",
        action="store_true",
        help="Enable SAC automatic entropy tuning (SAC only).",
    )
    parser.add_argument(
        "--ent-coef-min",
        type=float,
        default=None,
        help="Lower bound for SAC auto entropy coefficient (requires --ent-coef-auto).",
    )
    parser.add_argument("--action-noise-sigma", type=float, default=0.0)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument(
        "--eval-seeds",
        type=str,
        default="0,1,2,3",
        help="Comma-separated eval seeds (default: 0,1,2,3; more seeds increase eval time).",
    )
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--steps-weight", type=float, default=-0.05)
    parser.add_argument("--bc-weights", type=str, default=None)
    parser.add_argument(
        "--freeze-bc-layers",
        type=int,
        default=0,
        help="Freeze first N actor Linear layers after loading BC weights (requires --bc-weights).",
    )
    parser.add_argument(
        "--freeze-until-step",
        type=int,
        default=0,
        help="Timesteps to keep BC-frozen layers frozen (requires --bc-weights).",
    )
    parser.add_argument(
        "--bc-replay-dataset",
        type=str,
        default=None,
        help="Path to a .npz dataset with observations/actions for replay prefill.",
    )
    parser.add_argument(
        "--bc-replay-fraction",
        type=float,
        default=0.0,
        help="Fraction of the replay buffer to prefill with BC transitions (0-1).",
    )
    parser.add_argument("--resume-model", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--hf-home", type=str, default=None)
    parser.add_argument("--macro-reward-bonus", type=float, default=0.0)
    parser.add_argument("--macro-reward-radius", type=float, default=0.6)
    parser.add_argument("--macro-reward-alignment-threshold", type=float, default=0.6)
    parser.add_argument("--macro-reward-cap", type=float, default=0.5)
    parser.add_argument("--eval-seeds-random", type=int, default=0)
    parser.add_argument(
        "--normalize-obs",
        action="store_true",
        help="Normalize observations with VecNormalize (obs only).",
    )
    parser.add_argument(
        "--vecnormalize-path",
        type=str,
        default=None,
        help="Path to VecNormalize stats to load or save (default: <save_dir>/vecnormalize.pkl).",
    )
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
        choices=["base", "env", "competition", "pressure_shot", "tight"],
    )
    parser.add_argument("--save-dir", type=str, default="training_runs/multitask_td3")
    args = parser.parse_args()

    if args.competition_only_eval and args.legacy_only_eval:
        raise ValueError("Only one of --competition-only-eval or --legacy-only-eval may be set.")
    if not 0.0 <= args.bc_replay_fraction <= 1.0:
        raise ValueError("--bc-replay-fraction must be within [0, 1].")
    if args.ent_coef is not None and args.ent_coef < 0.0:
        raise ValueError("--ent-coef must be non-negative.")
    if args.ent_coef_auto and args.ent_coef is not None:
        raise ValueError("--ent-coef-auto cannot be used with --ent-coef.")
    if args.ent_coef_min is not None:
        if args.ent_coef_min < 0.0:
            raise ValueError("--ent-coef-min must be non-negative.")
        if not args.ent_coef_auto:
            raise ValueError("--ent-coef-min requires --ent-coef-auto.")
    if args.ent_coef_auto and args.algo.lower() != "sac":
        raise ValueError("--ent-coef-auto is only supported for SAC.")

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
    eval_seeds = resolve_eval_seeds(args.eval_seeds, args.eval_seeds_random, args.seed)

    active_task_specs = parse_tasks(args.tasks)
    task_weights = parse_task_weights(args.task_weights)
    reward_mode = "env" if args.reward_profile == "base" else args.reward_profile
    if reward_mode in ("env", "competition") and args.macro_reward_bonus > 0.0:
        raise ValueError("macro_reward_bonus requires reward_profile=pressure_shot or tight.")
    task_rng = np.random.default_rng(args.seed)
    task_list = build_task_list(
        args.n_envs,
        task_weights,
        task_specs=active_task_specs,
        rng=task_rng,
    )
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

    has_reward_wrapper = reward_mode in ("pressure_shot", "tight")
    info_keywords = ("macro_reward_bonus_mean",) if has_reward_wrapper else ()
    vec_env = VecMonitor(
        DummyVecEnv(env_fns),
        info_keywords=info_keywords,
    )
    vec_normalize = None
    vecnormalize_load_path = Path(args.vecnormalize_path) if args.vecnormalize_path else None
    if args.normalize_obs:
        if vecnormalize_load_path:
            vec_normalize = VecNormalize.load(vecnormalize_load_path, vec_env)
            vec_normalize.training = True
            vec_normalize.norm_reward = False
            vec_env = vec_normalize
            print(f"[train] Loaded VecNormalize stats from {vecnormalize_load_path}")
        else:
            vec_normalize = VecNormalize(
                vec_env,
                norm_obs=True,
                norm_reward=False,
                clip_obs=10.0,
            )
            vec_env = vec_normalize
            print("[train] VecNormalize enabled (obs only).")

    algo = args.algo.lower()
    algo_cls = _ALGO_CLASSES[algo]
    if algo != "td3" and args.action_noise_sigma > 0:
        print(f"[train] Ignoring action noise for algo={algo}.")

    n_actions = int(vec_env.action_space.shape[0])
    action_noise = None
    if algo == "td3" and args.action_noise_sigma > 0:
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=args.action_noise_sigma * np.ones(n_actions),
        )

    bc_loaded = False
    if args.resume_model:
        model = algo_cls.load(args.resume_model, env=vec_env, device=args.device)
        model.learning_rate = args.lr
        model.lr_schedule = get_schedule_fn(args.lr)
        if hasattr(model, "tau"):
            model.tau = args.tau
        if hasattr(model, "learning_starts"):
            model.learning_starts = args.learning_starts
        if algo == "td3":
            model.action_noise = action_noise
        print(f"Resumed {algo.upper()} from {args.resume_model}")
        if args.bc_weights:
            print("Ignoring --bc-weights when resuming from a checkpoint.")
    else:
        policy_kwargs = dict(net_arch=net_arch)
        if algo == "td3":
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
        elif algo == "ppo":
            ent_coef = args.ent_coef if args.ent_coef is not None else 0.01
            model = PPO(
                "MlpPolicy",
                vec_env,
                n_steps=2048,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=ent_coef,
                learning_rate=args.lr,
                clip_range=0.2,
                batch_size=args.batch_size,
                policy_kwargs=policy_kwargs,
                device=args.device,
                seed=args.seed,
                verbose=1,
            )
        elif algo == "sac":
            sac_kwargs = {}
            if args.ent_coef is not None:
                sac_kwargs["ent_coef"] = args.ent_coef
            elif args.ent_coef_auto:
                sac_kwargs["ent_coef"] = "auto"
            model = SAC(
                "MlpPolicy",
                vec_env,
                buffer_size=args.buffer_size,
                batch_size=args.batch_size,
                learning_starts=args.learning_starts,
                learning_rate=args.lr,
                tau=args.tau,
                train_freq=1,
                gradient_steps=1,
                policy_kwargs=policy_kwargs,
                device=args.device,
                seed=args.seed,
                verbose=1,
                **sac_kwargs,
            )
        else:
            raise ValueError(f"Unsupported algo selection: {algo}")

        if args.bc_weights:
            load_bc_weights(model, Path(args.bc_weights), algo)
            print(f"Loaded BC weights from {args.bc_weights}")
            bc_loaded = True

    frozen_params: list[torch.nn.Parameter] = []
    frozen_layers = 0
    if args.freeze_bc_layers > 0 or args.freeze_until_step > 0:
        if not bc_loaded:
            print("[train] Freeze options require --bc-weights; skipping layer freezing.")
        elif args.freeze_bc_layers <= 0:
            print("[train] --freeze-bc-layers is 0; skipping layer freezing.")
        else:
            if algo in ("td3", "sac"):
                target_module = model.policy.actor.mu
            elif algo == "ppo":
                target_module = model.policy.mlp_extractor.policy_net
            else:
                raise ValueError(f"Unsupported algo for BC layer freezing: {algo}")
            frozen_params, frozen_layers = freeze_linear_layers(
                target_module, args.freeze_bc_layers
            )
            if frozen_params:
                layer_word = "layer" if frozen_layers == 1 else "layers"
                print(
                    f"[train] Froze {frozen_layers} actor {layer_word} "
                    f"({len(frozen_params)} params) until step {args.freeze_until_step}."
                )
                if frozen_layers < args.freeze_bc_layers:
                    print(
                        f"[train] Requested {args.freeze_bc_layers} layers but only "
                        f"found {frozen_layers} Linear layers to freeze."
                    )
            else:
                print("[train] No Linear layers found to freeze in actor.")

    prefill_requested = args.bc_replay_dataset and args.bc_replay_fraction > 0.0
    if algo == "ppo":
        if args.bc_replay_dataset or args.bc_replay_fraction > 0.0:
            print("[train] PPO has no replay buffer; ignoring BC replay prefill options.")
    elif args.bc_replay_fraction > 0.0 and not args.bc_replay_dataset:
        raise ValueError("--bc-replay-dataset is required when --bc-replay-fraction > 0.")
    elif prefill_requested:
        dataset_path = Path(args.bc_replay_dataset)
        print(
            f"[train] Loading BC replay dataset from {dataset_path} "
            "(large files may slow startup)."
        )
        with np.load(dataset_path, allow_pickle=False) as data:
            if "observations" not in data or "actions" not in data:
                raise ValueError("BC replay dataset must include 'observations' and 'actions'.")
            observations = np.asarray(data["observations"])
            actions = np.asarray(data["actions"])
        if observations.shape[0] != actions.shape[0]:
            raise ValueError("BC replay observations/actions row count mismatch.")
        total_samples = observations.shape[0]
        if total_samples == 0:
            raise ValueError("BC replay dataset is empty.")
        num_demo = int(args.buffer_size * args.bc_replay_fraction)
        n_envs = model.n_envs
        num_demo = (num_demo // n_envs) * n_envs
        if num_demo <= 0:
            print("[train] BC replay fraction too small; skipping replay prefill.")
        else:
            rng = np.random.default_rng(args.seed)
            replace = num_demo > total_samples
            indices = rng.choice(total_samples, size=num_demo, replace=replace)
            indices = indices.reshape(-1, n_envs)
            reward = np.zeros(n_envs, dtype=np.float32)
            done = np.zeros(n_envs, dtype=np.float32)
            infos = [{"TimeLimit.truncated": False} for _ in range(n_envs)]
            for batch_idx in indices:
                obs_batch = observations[batch_idx]
                action_batch = actions[batch_idx]
                model.replay_buffer.add(
                    obs_batch,
                    obs_batch,
                    action_batch,
                    reward,
                    done,
                    infos,
                )
            print(f"[train] Pre-filled replay buffer with {num_demo} BC transitions.")
    elif args.bc_replay_dataset and args.bc_replay_fraction <= 0.0:
        print("[train] BC replay dataset provided with zero fraction; skipping prefill.")

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
        obs_rms=vec_normalize.obs_rms if vec_normalize is not None else None,
        obs_rms_epsilon=vec_normalize.epsilon if vec_normalize is not None else 1e-8,
        macro_reward_radius=args.macro_reward_radius,
        macro_reward_alignment_threshold=args.macro_reward_alignment_threshold,
        macro_reward_cap=args.macro_reward_cap,
        competition_only_eval=args.competition_only_eval,
        legacy_only_eval=args.legacy_only_eval,
        deterministic=True,
    )

    callbacks = [eval_cb]
    if frozen_params:
        callbacks.append(FreezeBCLayersCallback(frozen_params, args.freeze_until_step))
    if args.ent_coef_auto and args.ent_coef_min is not None and algo == "sac":
        callbacks.append(EntropyFloorCallback(args.ent_coef_min))
    callback = callbacks[0] if len(callbacks) == 1 else callbacks
    model.learn(total_timesteps=args.total_timesteps, callback=callback)

    final_path = save_dir / "final_model.zip"
    model.save(final_path)
    if vec_normalize is not None:
        vecnormalize_save_path = (
            vecnormalize_load_path if vecnormalize_load_path else save_dir / "vecnormalize.pkl"
        )
        vecnormalize_save_path.parent.mkdir(parents=True, exist_ok=True)
        vec_normalize.save(vecnormalize_save_path)
        print(f"[train] Saved VecNormalize stats to {vecnormalize_save_path}")
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()
