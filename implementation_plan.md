# Booster Soccer RL - Implementation Checklist + Diffs

## Checklist (apply in order)

- [ ] 1) Enable observation normalization with SB3 VecNormalize (train + eval)
  - Files: `training_scripts/train_multitask.py`, `training_scripts/multitask_utils.py`, `training_scripts/eval_multitask.py`
  - Diffs: Diff A, Diff B, Diff C
- [ ] 2) Clip actions and relax wrapper-order checks to allow optional wrappers
  - Files: `training_scripts/multitask_utils.py`
  - Diffs: Diff B
- [ ] 3) Reward mode selection (env vs competition vs shaped) with backward-compatible `base`
  - Files: `training_scripts/multitask_utils.py`, `training_scripts/train_multitask.py`, `README.md`
  - Diffs: Diff A, Diff B, Diff D
- [ ] 4) Randomize per-env task assignment while keeping tasks fixed per env slot
  - Files: `training_scripts/multitask_utils.py`, `training_scripts/train_multitask.py`
  - Diffs: Diff A, Diff B
- [ ] 5) Add per-episode success metrics logging (goal_scored, success, ball_hits, offside, distance)
  - Files: `training_scripts/multitask_utils.py`, `training_scripts/train_multitask.py`, `training_scripts/training.py`
  - Diffs: Diff A, Diff B, Diff E
- [ ] 6) Optional: random eval seeds via CLI flag (reproducible)
  - Files: `training_scripts/train_multitask.py`
  - Diffs: Diff A

## Diffs

### Diff A: `training_scripts/train_multitask.py`

```diff
diff --git a/training_scripts/train_multitask.py b/training_scripts/train_multitask.py
index 43b0c2a..b2e0b48 100644
--- a/training_scripts/train_multitask.py
+++ b/training_scripts/train_multitask.py
@@ -9,7 +9,7 @@ import torch
 from stable_baselines3 import TD3
 from stable_baselines3.common.callbacks import BaseCallback
 from stable_baselines3.common.noise import NormalActionNoise
 from stable_baselines3.common.utils import get_schedule_fn
-from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
+from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
 
 from multitask_utils import (
     TASK_SPECS,
@@ -44,6 +44,8 @@ class MultiTaskEvalCallback(BaseCallback):
         macro_reward_cap: float,
         competition_only_eval: bool,
         legacy_only_eval: bool,
+        obs_rms=None,
+        obs_rms_epsilon: float = 1e-8,
         deterministic: bool = True,
     ):
         super().__init__()
@@ -62,6 +64,8 @@ class MultiTaskEvalCallback(BaseCallback):
         self.competition_only_eval = competition_only_eval
         self.legacy_only_eval = legacy_only_eval
         self.deterministic = deterministic
+        self.obs_rms = obs_rms
+        self.obs_rms_epsilon = obs_rms_epsilon
         self.best_score = -np.inf
         self.best_comp_score = -np.inf
 
@@ -82,6 +86,8 @@ class MultiTaskEvalCallback(BaseCallback):
                 reward_profile=self.reward_profile,
                 macro_reward_bonus=self.macro_reward_bonus,
                 macro_reward_radius=self.macro_reward_radius,
+                obs_rms=self.obs_rms,
+                obs_rms_epsilon=self.obs_rms_epsilon,
                 macro_reward_alignment_threshold=self.macro_reward_alignment_threshold,
                 macro_reward_cap=self.macro_reward_cap,
             )
@@ -98,6 +104,8 @@ class MultiTaskEvalCallback(BaseCallback):
                 reward_profile=self.reward_profile,
                 macro_reward_bonus=self.macro_reward_bonus,
                 macro_reward_radius=self.macro_reward_radius,
+                obs_rms=self.obs_rms,
+                obs_rms_epsilon=self.obs_rms_epsilon,
                 macro_reward_alignment_threshold=self.macro_reward_alignment_threshold,
                 macro_reward_cap=self.macro_reward_cap,
             )
@@ -141,6 +149,15 @@ def parse_eval_seeds(raw: str | None) -> list[int]:
     if not tokens:
         return [0]
     return [int(token) for token in tokens]
+
+
+def resolve_eval_seeds(raw: str | None, random_count: int, seed: int) -> list[int]:
+    if random_count and random_count > 0:
+        rng = np.random.default_rng(seed)
+        return list(map(int, rng.integers(0, 10_000, size=random_count)))
+    return parse_eval_seeds(raw)
@@ -199,7 +216,7 @@ def main() -> None:
     parser.add_argument("--steps-weight", type=float, default=-0.05)
     parser.add_argument("--bc-weights", type=str, default=None)
     parser.add_argument("--resume-model", type=str, default=None)
-    parser.add_argument("--hf-token", type=str, default=None)
+    parser.add_argument("--hf-token", type=str, default=None)
     parser.add_argument("--hf-home", type=str, default=None)
     parser.add_argument("--macro-reward-bonus", type=float, default=0.0)
     parser.add_argument("--macro-reward-radius", type=float, default=0.6)
     parser.add_argument("--macro-reward-alignment-threshold", type=float, default=0.6)
     parser.add_argument("--macro-reward-cap", type=float, default=0.5)
+    parser.add_argument("--eval-seeds-random", type=int, default=0)
+    parser.add_argument("--normalize-obs", action="store_true")
+    parser.add_argument("--vecnormalize-path", type=str, default=None)
     parser.add_argument(
         "--competition-only-eval",
         action="store_true",
@@ -229,9 +246,9 @@ def main() -> None:
         "--tasks",
         type=str,
         default=None,
         help="Comma-separated task names to train/eval (default: all tasks).",
     )
     parser.add_argument(
         "--reward-profile",
         type=str,
         default="base",
-        choices=["base", "pressure_shot", "tight"],
+        choices=["base", "env", "competition", "pressure_shot", "tight"],
     )
     parser.add_argument("--save-dir", type=str, default="training_runs/multitask_td3")
     args = parser.parse_args()
@@ -259,12 +276,17 @@ def main() -> None:
     torch.backends.cudnn.benchmark = True
 
     net_arch = [int(v.strip()) for v in args.net_arch.split(",") if v.strip()]
-    eval_seeds = parse_eval_seeds(args.eval_seeds)
+    eval_seeds = resolve_eval_seeds(args.eval_seeds, args.eval_seeds_random, args.seed)
 
     active_task_specs = parse_tasks(args.tasks)
     task_weights = parse_task_weights(args.task_weights)
-    task_list = build_task_list(args.n_envs, task_weights, task_specs=active_task_specs)
+    reward_mode = "env" if args.reward_profile == "base" else args.reward_profile
+    if reward_mode in ("env", "competition") and args.macro_reward_bonus > 0.0:
+        raise ValueError("macro_reward_bonus requires reward_profile=pressure_shot or tight.")
+    task_rng = np.random.default_rng(args.seed)
+    task_list = build_task_list(
+        args.n_envs, task_weights, task_specs=active_task_specs, rng=task_rng
+    )
     env_fns = []
     for i, task_spec in enumerate(task_list):
         env_fns.append(
@@ -281,12 +303,26 @@ def main() -> None:
             )
         )
 
-    has_reward_wrapper = args.reward_profile != "base" or args.macro_reward_bonus > 0.0
+    has_reward_wrapper = reward_mode in ("pressure_shot", "tight")
     info_keywords = ("macro_reward_bonus_mean",) if has_reward_wrapper else ()
     vec_env = VecMonitor(
         DummyVecEnv(env_fns),
         info_keywords=info_keywords,
     )
+    vec_normalize = None
+    if args.normalize_obs:
+        if args.vecnormalize_path:
+            vec_normalize = VecNormalize.load(args.vecnormalize_path, vec_env)
+            vec_normalize.training = True
+            vec_normalize.norm_reward = False
+            vec_env = vec_normalize
+        else:
+            vec_normalize = VecNormalize(
+                vec_env,
+                norm_obs=True,
+                norm_reward=False,
+                clip_obs=10.0,
+            )
+            vec_env = vec_normalize
@@ -330,6 +366,8 @@ def main() -> None:
         eval_seeds=eval_seeds,
         reward_profile=args.reward_profile,
         macro_reward_bonus=args.macro_reward_bonus,
+        obs_rms=vec_normalize.obs_rms if vec_normalize is not None else None,
+        obs_rms_epsilon=vec_normalize.epsilon if vec_normalize is not None else 1e-8,
         macro_reward_radius=args.macro_reward_radius,
         macro_reward_alignment_threshold=args.macro_reward_alignment_threshold,
         macro_reward_cap=args.macro_reward_cap,
@@ -340,6 +378,10 @@ def main() -> None:
     model.learn(total_timesteps=args.total_timesteps, callback=eval_cb)
 
     final_path = save_dir / "final_model.zip"
     model.save(final_path)
+    if vec_normalize is not None:
+        vec_normalize.save(save_dir / "vecnormalize.pkl")
     print(f"Saved final model to {final_path}")
```

### Diff B: `training_scripts/multitask_utils.py`

```diff
diff --git a/training_scripts/multitask_utils.py b/training_scripts/multitask_utils.py
index 3b45b2b..f1c9b5a 100644
--- a/training_scripts/multitask_utils.py
+++ b/training_scripts/multitask_utils.py
@@ -6,6 +6,7 @@ from typing import Callable, Dict, List, Tuple
 
 import gymnasium as gym
 from gymnasium import spaces
+from gymnasium.wrappers import ClipAction
 import numpy as np
@@ -53,6 +54,15 @@ COMP_REWARD_CONFIGS: Dict[str, Dict[str, float]] = {
     name: dict(config) for name, config in COMP_REWARD_CONFIG.items()
 }
 
+_EVAL_TERM_KEYS = (
+    "goal_scored",
+    "success",
+    "ball_hits",
+    "offside",
+    "robot_distance_ball",
+    "distance",
+)
+
 
 def build_task_list(
     n_envs: int,
     task_weights: Dict[str, float] | None = None,
     task_specs: List[TaskSpec] | None = None,
+    rng: np.random.Generator | None = None,
 ) -> List[TaskSpec]:
     if n_envs <= 0:
         return []
     active_specs = task_specs if task_specs is not None else TASK_SPECS
     if not active_specs:
         raise ValueError("task_specs must include at least one task.")
+    rng = rng or np.random.default_rng()
     if not task_weights:
-        return [active_specs[i % len(active_specs)] for i in range(n_envs)]
+        return list(rng.choice(active_specs, size=n_envs, replace=True))
 
     weights = {spec.name: float(task_weights.get(spec.name, 0.0)) for spec in active_specs}
     total = sum(weights.values())
     if total <= 0.0:
         raise ValueError("task_weights must include at least one positive value.")
 
-    normalized = {name: value / total for name, value in weights.items()}
-    raw_counts = {name: normalized[name] * n_envs for name in normalized}
-    counts = {name: int(raw_counts[name]) for name in raw_counts}
-    remainder = n_envs - sum(counts.values())
-
-    if remainder > 0:
-        fractional = [
-            (name, raw_counts[name] - counts[name]) for name in normalized
-        ]
-        order = {spec.name: index for index, spec in enumerate(active_specs)}
-        fractional.sort(key=lambda item: (-item[1], order[item[0]]))
-        for name, _ in fractional:
-            if remainder <= 0:
-                break
-            counts[name] += 1
-            remainder -= 1
-
-    task_list: List[TaskSpec] = []
-    remaining = counts.copy()
-    while len(task_list) < n_envs:
-        for spec in active_specs:
-            if remaining.get(spec.name, 0) > 0:
-                task_list.append(spec)
-                remaining[spec.name] -= 1
-            if len(task_list) >= n_envs:
-                break
-    return task_list
+    probs = np.array([weights[spec.name] for spec in active_specs], dtype=np.float64)
+    probs = probs / probs.sum()
+    return list(rng.choice(active_specs, size=n_envs, replace=True, p=probs))
@@ -305,6 +315,10 @@ def _matches_keywords(key: str, keywords: tuple[str, ...]) -> bool:
     key_lower = key.lower()
     return any(word in key_lower for word in keywords)
 
+
+def _normalize_reward_profile(reward_profile: str) -> str:
+    return "env" if reward_profile == "base" else reward_profile
+
 
 def _adjust_reward_config(reward_config: Dict[str, float], reward_profile: str) -> Dict[str, float]:
-    if reward_profile == "base":
+    if reward_profile not in ("pressure_shot", "tight"):
         return reward_config
     if reward_profile == "pressure_shot":
         scale_goal = 1.1
@@ -349,34 +363,63 @@ def _resolve_max_steps(env: gym.Env) -> int | None:
     return None
 
 
-def _assert_wrapper_order(env: gym.Env, has_reward_wrapper: bool) -> None:
+def _safe_float(value) -> float:
+    try:
+        return float(value)
+    except (TypeError, ValueError):
+        return 0.0
+
+
+def _maybe_normalize_obs(obs: np.ndarray, obs_rms, epsilon: float) -> np.ndarray:
+    if obs_rms is None:
+        return obs
+    return (obs - obs_rms.mean) / (np.sqrt(obs_rms.var) + epsilon)
+
+
+def _aggregate_reward_terms(
+    term_list: List[Dict[str, float]],
+    keys: tuple[str, ...],
+) -> Dict[str, float]:
+    if not term_list:
+        return {key: 0.0 for key in keys}
+    totals = {key: 0.0 for key in keys}
+    for terms in term_list:
+        if not isinstance(terms, dict):
+            continue
+        for key in keys:
+            totals[key] += _safe_float(terms.get(key, 0.0))
+    count = max(1, len(term_list))
+    return {key: value / count for key, value in totals.items()}
+
+
+def _assert_wrapper_order(
+    env: gym.Env,
+    has_reward_wrapper: bool,
+    has_competition_wrapper: bool = True,
+) -> None:
     order = []
     current = env
     while isinstance(current, gym.Wrapper):
         order.append(type(current))
         current = current.env
 
-    expected = [CompetitionAlignmentWrapper]
-    if has_reward_wrapper:
-        expected.append(RewardProfileWrapper)
-    expected.extend([PreprocessObsWrapper, CommandActionWrapper])
-
-    for index, wrapper_type in enumerate(expected):
-        if index >= len(order) or order[index] is not wrapper_type:
-            names = [cls.__name__ for cls in order]
-            expected_names = [cls.__name__ for cls in expected]
-            raise AssertionError(
-                f"Wrapper order mismatch. Expected {expected_names}, got {names}."
-            )
+    expected: List[type] = []
+    if has_competition_wrapper:
+        expected.append(CompetitionAlignmentWrapper)
+    if has_reward_wrapper:
+        expected.append(RewardProfileWrapper)
+    expected.extend([PreprocessObsWrapper, CommandActionWrapper])
+    indices = []
+    for wrapper_type in expected:
+        if wrapper_type not in order:
+            names = [cls.__name__ for cls in order]
+            raise AssertionError(f"Missing wrapper {wrapper_type.__name__}. Got {names}.")
+        indices.append(order.index(wrapper_type))
+    if indices != sorted(indices):
+        names = [cls.__name__ for cls in order]
+        expected_names = [cls.__name__ for cls in expected]
+        raise AssertionError(
+            f"Wrapper order mismatch. Expected {expected_names} in order, got {names}."
+        )
@@ -425,15 +468,22 @@ def make_env(
 
     def _init():
         env = gym.make(task_spec.env_id, render_mode=render_mode)
         base_env = env.unwrapped
+        reward_mode = _normalize_reward_profile(reward_profile)
+        if reward_mode not in ("pressure_shot", "tight") and macro_reward_bonus > 0.0:
+            raise ValueError("macro_reward_bonus requires reward_profile=pressure_shot or tight.")
         if hasattr(base_env, "goal_site"):
             base_env.goal_site = _resolve_site_name(base_env, base_env.goal_site)
         if hasattr(base_env, "target_name"):
             base_env.target_name = _resolve_site_name(base_env, base_env.target_name)
         reward_config = getattr(base_env, "reward_config", None)
-        if isinstance(reward_config, dict):
-            base_env.reward_config = _adjust_reward_config(reward_config, reward_profile)
+        if isinstance(reward_config, dict) and reward_mode in ("pressure_shot", "tight"):
+            base_env.reward_config = _adjust_reward_config(reward_config, reward_mode)
         env = CommandActionWrapper(env)
+        env = ClipAction(env)
         env = PreprocessObsWrapper(env, task_spec.one_hot)
-        if reward_profile != "base" or macro_reward_bonus > 0.0:
+        if reward_mode in ("pressure_shot", "tight"):
             reward_config = getattr(base_env, "reward_config", None)
             has_timeout_term = isinstance(reward_config, dict) and any(
                 _matches_keywords(key, _TIMEOUT_KEYWORDS) for key in reward_config
             )
             has_shot_term = isinstance(reward_config, dict) and any(
                 _matches_keywords(key, _SHOT_KEYWORDS) for key in reward_config
             )
             env = RewardProfileWrapper(
                 env,
                 reward_profile=reward_profile,
                 apply_timeout=not has_timeout_term,
                 apply_shot=not has_shot_term,
                 macro_reward_bonus=macro_reward_bonus,
                 macro_reward_radius=macro_reward_radius,
                 macro_reward_alignment_threshold=macro_reward_alignment_threshold,
                 macro_reward_cap=macro_reward_cap,
             )
-        env = CompetitionAlignmentWrapper(env, task_spec.name)
-        _assert_wrapper_order(env, reward_profile != "base" or macro_reward_bonus > 0.0)
+        if reward_mode == "competition":
+            env = CompetitionAlignmentWrapper(env, task_spec.name)
+        _assert_wrapper_order(
+            env,
+            has_reward_wrapper=reward_mode in ("pressure_shot", "tight"),
+            has_competition_wrapper=reward_mode == "competition",
+        )
         if seed is not None:
             env.reset(seed=seed)
         return env
@@ -553,6 +603,8 @@ def evaluate_policy(
     reward_profile: str = "base",
     macro_reward_bonus: float = 0.0,
     macro_reward_radius: float = 0.6,
+    obs_rms=None,
+    obs_rms_epsilon: float = 1e-8,
     macro_reward_alignment_threshold: float = 0.6,
     macro_reward_cap: float = 0.5,
-) -> Tuple[float, List[float], List[int], int | None]:
+) -> Tuple[float, List[float], List[int], int | None, List[Dict[str, float]]]:
     env = make_env(
         task_spec,
         seed=seed,
@@ -568,6 +620,7 @@ def evaluate_policy(
     scores: List[float] = []
     lengths: List[int] = []
     max_steps = _resolve_max_steps(env)
+    reward_terms_list: List[Dict[str, float]] = []
 
     for ep in range(episodes):
         obs, info = env.reset(seed=seed + ep)
+        model_obs = _maybe_normalize_obs(obs, obs_rms, obs_rms_epsilon)
         terminated = truncated = False
         steps = 0
         reward_terms = {}
 
         while not (terminated or truncated):
             if hasattr(model, "predict"):
-                action, _ = model.predict(obs, deterministic=deterministic)
+                action, _ = model.predict(model_obs, deterministic=deterministic)
             else:
-                action = model(obs)
+                action = model(model_obs)
             obs, _, terminated, truncated, info = env.step(action)
             reward_terms = info.get("reward_terms", reward_terms)
+            model_obs = _maybe_normalize_obs(obs, obs_rms, obs_rms_epsilon)
             steps += 1
 
         reward_config = getattr(env.unwrapped, "reward_config", {})
         if not isinstance(reward_config, dict):
             reward_config = {}
         score = compute_terminal_score(reward_terms, reward_config, steps, steps_weight)
         scores.append(score)
         lengths.append(steps)
+        reward_terms_list.append(reward_terms if isinstance(reward_terms, dict) else {})
 
     env.close()
-    return float(np.mean(scores)), scores, lengths, max_steps
+    return float(np.mean(scores)), scores, lengths, max_steps, reward_terms_list
@@ -586,6 +639,8 @@ def evaluate_policy_competition(
     reward_profile: str = "base",
     macro_reward_bonus: float = 0.0,
     macro_reward_radius: float = 0.6,
+    obs_rms=None,
+    obs_rms_epsilon: float = 1e-8,
     macro_reward_alignment_threshold: float = 0.6,
     macro_reward_cap: float = 0.5,
-) -> Tuple[float, List[Dict[str, float]]]:
+) -> Tuple[float, List[Dict[str, float]], List[Dict[str, float]]]:
     env = make_env(
         task_spec,
         seed=seed,
@@ -601,10 +656,12 @@ def evaluate_policy_competition(
         macro_reward_cap=macro_reward_cap,
     )()
     reports: List[Dict[str, float]] = []
+    reward_terms_list: List[Dict[str, float]] = []
 
     for ep in range(episodes):
         obs, info = env.reset(seed=seed + ep)
+        model_obs = _maybe_normalize_obs(obs, obs_rms, obs_rms_epsilon)
         terminated = truncated = False
         steps = 0
         reward_terms: Dict[str, float] = {}
 
         while not (terminated or truncated):
             if hasattr(model, "predict"):
-                action, _ = model.predict(obs, deterministic=deterministic)
+                action, _ = model.predict(model_obs, deterministic=deterministic)
             else:
-                action = model(obs)
+                action = model(model_obs)
             obs, _, terminated, truncated, info = env.step(action)
             current_terms = info.get("reward_terms", reward_terms)
             reward_terms = current_terms if isinstance(current_terms, dict) else reward_terms
+            model_obs = _maybe_normalize_obs(obs, obs_rms, obs_rms_epsilon)
             steps += 1
 
         report = compute_competition_report(reward_terms, task_spec.name, steps)
         reports.append(report)
+        reward_terms_list.append(reward_terms if isinstance(reward_terms, dict) else {})
 
     env.close()
     mean_comp_total = float(np.mean([report["comp_total"] for report in reports])) if reports else 0.0
-    return mean_comp_total, reports
+    return mean_comp_total, reports, reward_terms_list
@@ -620,6 +677,8 @@ def evaluate_all_tasks(
     reward_profile: str = "base",
     macro_reward_bonus: float = 0.0,
     macro_reward_radius: float = 0.6,
+    obs_rms=None,
+    obs_rms_epsilon: float = 1e-8,
     macro_reward_alignment_threshold: float = 0.6,
     macro_reward_cap: float = 0.5,
 ) -> Dict[str, float]:
@@ -634,6 +693,8 @@ def evaluate_all_tasks(
         reward_profile=reward_profile,
         macro_reward_bonus=macro_reward_bonus,
         macro_reward_radius=macro_reward_radius,
+        obs_rms=obs_rms,
+        obs_rms_epsilon=obs_rms_epsilon,
         macro_reward_alignment_threshold=macro_reward_alignment_threshold,
         macro_reward_cap=macro_reward_cap,
     )
@@ -649,6 +710,8 @@ def evaluate_selected_tasks(
     reward_profile: str = "base",
     macro_reward_bonus: float = 0.0,
     macro_reward_radius: float = 0.6,
+    obs_rms=None,
+    obs_rms_epsilon: float = 1e-8,
     macro_reward_alignment_threshold: float = 0.6,
     macro_reward_cap: float = 0.5,
 ) -> Dict[str, float]:
@@ -659,6 +722,7 @@ def evaluate_selected_tasks(
     per_task_scores: Dict[str, List[float]] = {task.name: [] for task in task_specs}
     all_lengths: List[int] = []
     max_steps = None
+    per_task_terms: Dict[str, List[Dict[str, float]]] = {task.name: [] for task in task_specs}
 
     for seed_value in seeds:
         for task_index, task_spec in enumerate(task_specs):
-            mean_score, _, lengths, task_max_steps = evaluate_policy(
+            mean_score, _, lengths, task_max_steps, reward_terms_list = evaluate_policy(
                 model,
                 task_spec,
                 episodes=episodes,
                 steps_weight=steps_weight,
                 deterministic=deterministic,
                 seed=seed_value + task_index * 1000,
                 reward_profile=reward_profile,
                 macro_reward_bonus=macro_reward_bonus,
                 macro_reward_radius=macro_reward_radius,
+                obs_rms=obs_rms,
+                obs_rms_epsilon=obs_rms_epsilon,
                 macro_reward_alignment_threshold=macro_reward_alignment_threshold,
                 macro_reward_cap=macro_reward_cap,
             )
             per_task_scores[task_spec.name].append(mean_score)
             all_lengths.extend(lengths)
+            per_task_terms[task_spec.name].extend(reward_terms_list)
             if task_max_steps is not None:
                 max_steps = task_max_steps if max_steps is None else max(max_steps, task_max_steps)
@@ -672,6 +739,12 @@ def evaluate_selected_tasks(
     per_task = {name: float(np.mean(scores)) for name, scores in per_task_scores.items()}
     overall = float(np.mean(list(per_task.values()))) if per_task else 0.0
     per_task["S_overall"] = overall
+    for task_name, term_list in per_task_terms.items():
+        term_means = _aggregate_reward_terms(term_list, _EVAL_TERM_KEYS)
+        for term_key, term_value in term_means.items():
+            per_task[f"{task_name}/{term_key}_mean"] = float(term_value)
     if all_lengths:
         ep_len_mean = float(np.mean(all_lengths))
         per_task["ep_len_mean"] = ep_len_mean
@@ -700,6 +773,8 @@ def evaluate_all_tasks_competition(
     reward_profile: str = "base",
     macro_reward_bonus: float = 0.0,
     macro_reward_radius: float = 0.6,
+    obs_rms=None,
+    obs_rms_epsilon: float = 1e-8,
     macro_reward_alignment_threshold: float = 0.6,
     macro_reward_cap: float = 0.5,
 ) -> Dict[str, float]:
@@ -714,6 +789,8 @@ def evaluate_all_tasks_competition(
         reward_profile=reward_profile,
         macro_reward_bonus=macro_reward_bonus,
         macro_reward_radius=macro_reward_radius,
+        obs_rms=obs_rms,
+        obs_rms_epsilon=obs_rms_epsilon,
         macro_reward_alignment_threshold=macro_reward_alignment_threshold,
         macro_reward_cap=macro_reward_cap,
     )
@@ -729,6 +806,8 @@ def evaluate_selected_tasks_competition(
     reward_profile: str = "base",
     macro_reward_bonus: float = 0.0,
     macro_reward_radius: float = 0.6,
+    obs_rms=None,
+    obs_rms_epsilon: float = 1e-8,
     macro_reward_alignment_threshold: float = 0.6,
     macro_reward_cap: float = 0.5,
 ) -> Dict[str, float]:
@@ -739,6 +818,7 @@ def evaluate_selected_tasks_competition(
         seeds = [seed]
 
     per_task_reports: Dict[str, List[Dict[str, float]]] = {task.name: [] for task in task_specs}
+    per_task_terms: Dict[str, List[Dict[str, float]]] = {task.name: [] for task in task_specs}
 
     for seed_value in seeds:
         for task_index, task_spec in enumerate(task_specs):
-            _, reports = evaluate_policy_competition(
+            _, reports, reward_terms_list = evaluate_policy_competition(
                 model,
                 task_spec,
                 episodes=episodes,
                 deterministic=deterministic,
                 seed=seed_value + task_index * 1000,
                 reward_profile=reward_profile,
                 macro_reward_bonus=macro_reward_bonus,
                 macro_reward_radius=macro_reward_radius,
+                obs_rms=obs_rms,
+                obs_rms_epsilon=obs_rms_epsilon,
                 macro_reward_alignment_threshold=macro_reward_alignment_threshold,
                 macro_reward_cap=macro_reward_cap,
             )
             per_task_reports[task_spec.name].extend(reports)
+            per_task_terms[task_spec.name].extend(reward_terms_list)
@@ -764,6 +844,11 @@ def evaluate_selected_tasks_competition(
         for component in COMP_REWARD_CONFIG[task_name].keys():
             key = f"comp_{component}"
             results[f"{task_name}/{key}"] = float(aggregated.get(key, 0.0))
+        term_means = _aggregate_reward_terms(per_task_terms[task_name], _EVAL_TERM_KEYS)
+        for term_key, term_value in term_means.items():
+            results[f"{task_name}/{term_key}_mean"] = float(term_value)
         comp_total_sum += float(aggregated["comp_total"])
 
     results["comp_total_sum"] = float(comp_total_sum)
     return results
```

### Diff C: `training_scripts/eval_multitask.py`

```diff
diff --git a/training_scripts/eval_multitask.py b/training_scripts/eval_multitask.py
index 18e8df0..f9e4f39 100644
--- a/training_scripts/eval_multitask.py
+++ b/training_scripts/eval_multitask.py
@@ -2,8 +2,9 @@ import argparse
 from pathlib import Path
 
 from stable_baselines3 import TD3
+from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
 
-from multitask_utils import TASK_SPECS, evaluate_all_tasks, evaluate_all_tasks_competition
+from multitask_utils import TASK_SPECS, evaluate_all_tasks, evaluate_all_tasks_competition, make_env
@@ -34,12 +35,26 @@ def main() -> None:
     parser.add_argument("--steps-weight", type=float, default=-0.001)
     parser.add_argument("--device", type=str, default="cpu")
     parser.add_argument("--deterministic", action="store_true")
+    parser.add_argument("--vecnormalize", type=str, default=None)
     args = parser.parse_args()
 
     model_path = Path(args.model)
     model = TD3.load(model_path, device=args.device)
+    obs_rms = None
+    obs_rms_epsilon = 1e-8
+    if args.vecnormalize:
+        dummy_env = DummyVecEnv([make_env(TASK_SPECS[0])])
+        vec_norm = VecNormalize.load(args.vecnormalize, dummy_env)
+        vec_norm.training = False
+        vec_norm.norm_reward = False
+        obs_rms = vec_norm.obs_rms
+        obs_rms_epsilon = vec_norm.epsilon
 
     scores = evaluate_all_tasks(
         model,
         episodes=args.episodes,
         steps_weight=args.steps_weight,
         deterministic=args.deterministic,
         seed=0,
+        obs_rms=obs_rms,
+        obs_rms_epsilon=obs_rms_epsilon,
     )
     comp_scores = evaluate_all_tasks_competition(
         model,
         episodes=args.episodes,
         deterministic=args.deterministic,
         seed=0,
+        obs_rms=obs_rms,
+        obs_rms_epsilon=obs_rms_epsilon,
     )
```

### Diff D: `README.md`

```diff
diff --git a/README.md b/README.md
index 6a8c4e6..4f6a5c1 100644
--- a/README.md
+++ b/README.md
@@ -142,6 +142,14 @@ python training_scripts/train_multitask.py \
   --n-envs 6 \
   --save-dir training_runs/multitask_td3
 ```
+
+Notes:
+- `--reward-profile` now supports `env` and `competition`; `base` is treated as `env`.
+- `--normalize-obs` enables VecNormalize and writes `vecnormalize.pkl` to the save dir.
+  Pass `--vecnormalize-path` when resuming or `--vecnormalize` in eval scripts.
+- `--eval-seeds-random N` draws N random eval seeds (seeded by `--seed`).
 
 ### 4) Terminal-Only Evaluation (S1/S2/S3, S_overall)
```

### Diff E: `training_scripts/training.py`

```diff
diff --git a/training_scripts/training.py b/training_scripts/training.py
index 0a9a0c6..1b6a5d1 100644
--- a/training_scripts/training.py
+++ b/training_scripts/training.py
@@ -83,6 +83,17 @@ def training_loop(
 
         episode_count += 1
+
+        reward_terms = info.get("reward_terms", {}) if isinstance(info, dict) else {}
+        if isinstance(reward_terms, dict) and reward_terms:
+            goal_scored = float(reward_terms.get("goal_scored", 0.0))
+            success = float(reward_terms.get("success", 0.0))
+            ball_hits = float(reward_terms.get("ball_hits", 0.0))
+            pbar.write(
+                f"Episode {episode_count} metrics: goal_scored={goal_scored:.1f} "
+                f"success={success:.1f} ball_hits={ball_hits:.1f}"
+            )
 
     pbar.close()
     env.close()
```

## Migration Note (short)

- **Reward profile**: `--reward-profile base` now aliases `env` (raw rewards only). To train on competition-aligned rewards, pass `--reward-profile competition`. Shaped profiles remain `pressure_shot`/`tight`.
- **Normalization**: when `--normalize-obs` is enabled, a `vecnormalize.pkl` file is saved alongside checkpoints. Resume with `--vecnormalize-path` and evaluate with `--vecnormalize` to keep normalization consistent.
- **Task assignment**: per-env task selection is now randomized (but reproducible with `--seed`). Each env slot keeps its task for the full run.
- **Macro bonus**: `--macro-reward-*` is only valid with shaped profiles (`pressure_shot`/`tight`).
