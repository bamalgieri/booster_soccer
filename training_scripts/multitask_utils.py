import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

import sai_mujoco  # noqa: F401  # registers envs
from booster_control.t1_utils import LowerT1JoyStick
from booster_control.macro_playback import load_kick_macro, trigger_macro
from imitation_learning.scripts.preprocessor import Preprocessor
from competition_scoring import (
    COMP_REWARD_CONFIG,
    aggregate_competition_reports,
    compute_competition_report,
    compute_competition_score,
)


@dataclass(frozen=True)
class TaskSpec:
    env_id: str
    name: str
    one_hot: np.ndarray


TASK_SPECS: List[TaskSpec] = [
    TaskSpec(
        env_id="LowerT1GoaliePenaltyKick-v0",
        name="goalie_penalty_kick",
        one_hot=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    ),
    TaskSpec(
        env_id="LowerT1ObstaclePenaltyKick-v0",
        name="obstacle_penalty_kick",
        one_hot=np.array([0.0, 1.0, 0.0], dtype=np.float32),
    ),
    TaskSpec(
        env_id="LowerT1KickToTarget-v0",
        name="kick_to_target",
        one_hot=np.array([0.0, 0.0, 1.0], dtype=np.float32),
    ),
]

COMP_REWARD_CONFIGS: Dict[str, Dict[str, float]] = {
    name: dict(config) for name, config in COMP_REWARD_CONFIG.items()
}


def build_task_list(
    n_envs: int,
    task_weights: Dict[str, float] | None = None,
    task_specs: List[TaskSpec] | None = None,
) -> List[TaskSpec]:
    if n_envs <= 0:
        return []
    active_specs = task_specs if task_specs is not None else TASK_SPECS
    if not active_specs:
        raise ValueError("task_specs must include at least one task.")
    if not task_weights:
        return [active_specs[i % len(active_specs)] for i in range(n_envs)]

    weights = {spec.name: float(task_weights.get(spec.name, 0.0)) for spec in active_specs}
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("task_weights must include at least one positive value.")

    normalized = {name: value / total for name, value in weights.items()}
    raw_counts = {name: normalized[name] * n_envs for name in normalized}
    counts = {name: int(raw_counts[name]) for name in raw_counts}
    remainder = n_envs - sum(counts.values())

    if remainder > 0:
        fractional = [
            (name, raw_counts[name] - counts[name]) for name in normalized
        ]
        order = {spec.name: index for index, spec in enumerate(active_specs)}
        fractional.sort(key=lambda item: (-item[1], order[item[0]]))
        for name, _ in fractional:
            if remainder <= 0:
                break
            counts[name] += 1
            remainder -= 1

    task_list: List[TaskSpec] = []
    remaining = counts.copy()
    while len(task_list) < n_envs:
        for spec in active_specs:
            if remaining.get(spec.name, 0) > 0:
                task_list.append(spec)
                remaining[spec.name] -= 1
            if len(task_list) >= n_envs:
                break
    return task_list



class CommandActionWrapper(gym.ActionWrapper):
    def __init__(
        self,
        env: gym.Env,
        macro_enabled: bool = True,
        macro_name: str = "goal_kick",
        macro_robot: str = "booster_lower_t1",
        macro_max_steps: int = 30,
        macro_trigger_threshold: float = 0.95,
        macro_ball_radius: float = 0.6,
        macro_alignment_threshold: float = 0.6,
    ):
        super().__init__(env)
        self.base_env = env.unwrapped
        self.lower_control = LowerT1JoyStick(self.base_env)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.macro_enabled = macro_enabled
        self.macro_name = macro_name
        self.macro_robot = macro_robot
        self.macro_max_steps = macro_max_steps
        self.macro_trigger_threshold = macro_trigger_threshold
        self.macro_ball_radius = macro_ball_radius
        self.macro_alignment_threshold = macro_alignment_threshold
        self._reset_macro_state(reset_counters=True)

    def _reset_macro_state(self, reset_counters: bool = False) -> None:
        self._macro_active = False
        self._macro_actions = None
        self._macro_index = 0
        self._macro_triggered = False
        self._macro_step_index = -1
        self._macro_end_after_step = False
        if reset_counters:
            self._macro_trigger_count = 0
            self._macro_active_steps = 0
            self._macro_aborted_count = 0

    def reset(self, **kwargs):
        self._reset_macro_state(reset_counters=True)
        return self.env.reset(**kwargs)

    def _should_abort_macro(self, info: Dict) -> bool:
        reward_terms = info.get("reward_terms", {})
        if not isinstance(reward_terms, dict):
            return False
        for key in ("ball_hits", "goal_scored", "success"):
            if key in reward_terms and float(reward_terms[key]) > 0.0:
                return True
        return False

    def action(self, command):
        observation = self.base_env._get_obs()
        info = self.base_env._get_info()
        self._macro_triggered = False
        self._macro_step_index = -1
        self._macro_end_after_step = False

        if self.macro_enabled and self._macro_active:
            macro_action = np.asarray(self._macro_actions[self._macro_index], dtype=np.float32)
            self._macro_step_index = self._macro_index
            self._macro_index += 1
            self._macro_active_steps += 1
            if self._macro_index >= len(self._macro_actions):
                self._macro_end_after_step = True
            return self.lower_control.get_torque(observation, macro_action)

        if self.macro_enabled and trigger_macro(
            command,
            info,
            trigger_threshold=self.macro_trigger_threshold,
            ball_radius=self.macro_ball_radius,
            alignment_threshold=self.macro_alignment_threshold,
        ):
            try:
                sequence = load_kick_macro(
                    self.macro_name,
                    self.lower_control,
                    robot=self.macro_robot,
                    max_steps=self.macro_max_steps,
                )
                if sequence.actions.size > 0:
                    self._macro_actions = sequence.actions
                    self._macro_active = True
                    self._macro_index = 0
                    self._macro_triggered = True
                    self._macro_trigger_count += 1
                    macro_action = np.asarray(self._macro_actions[self._macro_index], dtype=np.float32)
                    self._macro_step_index = self._macro_index
                    self._macro_index += 1
                    self._macro_active_steps += 1
                    if self._macro_index >= len(self._macro_actions):
                        self._macro_end_after_step = True
                    return self.lower_control.get_torque(observation, macro_action)
            except Exception:
                self._macro_aborted_count += 1
                self._reset_macro_state(reset_counters=False)

        ctrl, _ = self.lower_control.get_actions(command, observation, info)
        return ctrl

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info = dict(info)
        info["macro_active"] = self._macro_active
        info["macro_triggered"] = self._macro_triggered
        info["macro_name"] = self.macro_name if (self._macro_active or self._macro_triggered) else ""
        info["macro_step_index"] = self._macro_step_index
        info["macro_trigger_count"] = self._macro_trigger_count
        info["macro_active_steps"] = self._macro_active_steps
        info["macro_aborted_count"] = self._macro_aborted_count

        if self._macro_active and self._should_abort_macro(info):
            self._macro_aborted_count += 1
            self._reset_macro_state(reset_counters=False)
        elif self._macro_end_after_step:
            self._reset_macro_state(reset_counters=False)

        self._macro_triggered = False
        self._macro_step_index = -1
        self._macro_end_after_step = False
        return obs, reward, terminated, truncated, info


class PreprocessObsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, task_one_hot: np.ndarray):
        super().__init__(env)
        self.preprocessor = Preprocessor()
        self.task_one_hot = task_one_hot.astype(np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(89,), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self.preprocessor.modify_state(obs, info, self.task_one_hot)
        info = dict(info)
        info["task_index"] = self.task_one_hot.copy()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self.preprocessor.modify_state(obs, info, self.task_one_hot)
        info = dict(info)
        info["task_index"] = self.task_one_hot.copy()
        return obs, reward, terminated, truncated, info


_GOAL_KEYWORDS = ("goal", "score", "success")
_SHOT_KEYWORDS = ("shot", "kick", "strike")
_TIMEOUT_KEYWORDS = ("timeout", "time_out", "time_penalty", "stall", "stalling")


class RewardProfileWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        reward_profile: str,
        apply_timeout: bool,
        apply_shot: bool,
        macro_reward_bonus: float = 0.0,
        macro_reward_radius: float = 0.6,
        macro_reward_alignment_threshold: float = 0.6,
        macro_reward_cap: float = 0.5,
    ):
        super().__init__(env)
        self.reward_profile = reward_profile
        self.apply_timeout = apply_timeout
        self.apply_shot = apply_shot
        self.macro_reward_bonus = float(macro_reward_bonus)
        self.macro_reward_radius = float(macro_reward_radius)
        self.macro_reward_alignment_threshold = float(macro_reward_alignment_threshold)
        self.macro_reward_cap = float(macro_reward_cap)
        self._macro_bonus_sum = 0.0
        self._macro_steps = 0
        if reward_profile == "pressure_shot":
            self.timeout_penalty = -0.05
            self.shot_bonus = 0.02
        elif reward_profile == "tight":
            self.timeout_penalty = -0.1
            self.shot_bonus = 0.05
        else:
            self.timeout_penalty = 0.0
            self.shot_bonus = 0.0

    def _matches_keywords(self, key: str, keywords: tuple[str, ...]) -> bool:
        key_lower = key.lower()
        return any(word in key_lower for word in keywords)

    def _has_shot_attempt(self, info: Dict) -> bool:
        reward_terms = info.get("reward_terms", {})
        if isinstance(reward_terms, dict):
            for key, value in reward_terms.items():
                if self._matches_keywords(key, _SHOT_KEYWORDS) and float(value) > 0.0:
                    return True
        for key, value in info.items():
            if not isinstance(key, str):
                continue
            if self._matches_keywords(key, _SHOT_KEYWORDS):
                if isinstance(value, (int, float)) and float(value) > 0.0:
                    return True
                if isinstance(value, bool) and value:
                    return True
        return False

    def _macro_bonus(self, info: Dict) -> float:
        if self.macro_reward_bonus <= 0.0:
            return 0.0
        if not info.get("macro_triggered", False):
            return 0.0
        ball_vec = info.get("ball_xpos_rel_robot")
        if ball_vec is None:
            return 0.0
        ball_vec = np.asarray(ball_vec, dtype=np.float32)
        if ball_vec.shape[-1] != 3:
            return 0.0
        if np.linalg.norm(ball_vec) > self.macro_reward_radius:
            return 0.0

        goal_vec = info.get("target_xpos_rel_robot")
        if goal_vec is None:
            goal_0 = info.get("goal_team_0_rel_robot")
            goal_1 = info.get("goal_team_1_rel_robot")
            if goal_0 is None:
                goal_vec = goal_1
            elif goal_1 is None:
                goal_vec = goal_0
            else:
                goal_vec = goal_0 if np.linalg.norm(goal_0) <= np.linalg.norm(goal_1) else goal_1
        if goal_vec is None:
            return 0.0

        goal_vec = np.asarray(goal_vec, dtype=np.float32)
        if goal_vec.shape[-1] != 3:
            return 0.0

        ball_norm = np.linalg.norm(ball_vec)
        goal_norm = np.linalg.norm(goal_vec)
        if ball_norm < 1e-6 or goal_norm < 1e-6:
            return 0.0
        alignment = float(np.dot(ball_vec, goal_vec) / (ball_norm * goal_norm))
        if alignment < self.macro_reward_alignment_threshold:
            return 0.0

        remaining = self.macro_reward_cap - self._macro_bonus_sum
        if remaining <= 0.0:
            return 0.0
        return min(self.macro_reward_bonus, remaining)

    def reset(self, **kwargs):
        self._macro_bonus_sum = 0.0
        self._macro_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = 0.0
        if self.apply_timeout and truncated and not terminated:
            shaped += self.timeout_penalty
        if self.apply_shot and self._has_shot_attempt(info):
            shaped += self.shot_bonus
        self._macro_steps += 1
        macro_bonus = self._macro_bonus(info)
        if macro_bonus > 0.0:
            self._macro_bonus_sum += macro_bonus
            shaped += macro_bonus
            info = dict(info)
            info["macro_reward_bonus"] = macro_bonus
        if terminated or truncated:
            info = dict(info)
            info["macro_reward_bonus_mean"] = (
                self._macro_bonus_sum / max(1, self._macro_steps)
            )
        return obs, reward + shaped, terminated, truncated, info


class CompetitionAlignmentWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, task_name: str):
        super().__init__(env)
        self.task_name = task_name or ""
        self.task_name_lower = self.task_name.lower()

    def _safe_float(self, value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _get_term(self, reward_terms: Dict[str, float], key: str) -> float:
        return self._safe_float(reward_terms.get(key, 0.0))

    def _compute_reward(self, reward_terms: Dict[str, float], reward: float) -> float:
        if "penalty_kick" in self.task_name_lower:
            return (
                1.0 * self._get_term(reward_terms, "robot_distance_ball")
                + 30.0 * self._get_term(reward_terms, "goal_scored")
                + 2.0 * self._get_term(reward_terms, "ball_vel_twd_goal")
                - 0.05
            )
        if "kick_to_target" in self.task_name_lower:
            return (
                0.5 * self._get_term(reward_terms, "distance")
                + 20.0 * self._get_term(reward_terms, "success")
                - 0.05
            )
        return reward

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward_terms = info.get("reward_terms", {})
        if isinstance(reward_terms, dict) and reward_terms:
            return (
                obs,
                self._compute_reward(reward_terms, reward),
                terminated,
                truncated,
                info,
            )
        return obs, reward, terminated, truncated, info


def _matches_keywords(key: str, keywords: tuple[str, ...]) -> bool:
    key_lower = key.lower()
    return any(word in key_lower for word in keywords)


def _adjust_reward_config(reward_config: Dict[str, float], reward_profile: str) -> Dict[str, float]:
    if reward_profile == "base":
        return reward_config
    if reward_profile == "pressure_shot":
        scale_goal = 1.1
        scale_timeout = 1.5
        scale_shot = 1.1
    elif reward_profile == "tight":
        scale_goal = 1.2
        scale_timeout = 2.0
        scale_shot = 1.2
    else:
        return reward_config

    adjusted = dict(reward_config)
    for key, weight in reward_config.items():
        if _matches_keywords(key, _GOAL_KEYWORDS) and weight > 0:
            adjusted[key] = weight * scale_goal
        if _matches_keywords(key, _SHOT_KEYWORDS) and weight > 0:
            adjusted[key] = weight * scale_shot
        if _matches_keywords(key, _TIMEOUT_KEYWORDS) and weight < 0:
            adjusted[key] = weight * scale_timeout
    return adjusted


def _resolve_max_steps(env: gym.Env) -> int | None:
    spec = getattr(env, "spec", None)
    max_steps = getattr(spec, "max_episode_steps", None) if spec is not None else None
    if max_steps is not None:
        return int(max_steps)
    for attr in ("_max_episode_steps", "max_episode_steps"):
        if hasattr(env, attr):
            value = getattr(env, attr)
            if value is not None:
                return int(value)
    base_env = getattr(env, "unwrapped", None)
    if base_env is not None:
        for attr in ("_max_episode_steps", "max_episode_steps"):
            if hasattr(base_env, attr):
                value = getattr(base_env, attr)
                if value is not None:
                    return int(value)
    return None


def _assert_wrapper_order(env: gym.Env, has_reward_wrapper: bool) -> None:
    order = []
    current = env
    while isinstance(current, gym.Wrapper):
        order.append(type(current))
        current = current.env

    expected = [CompetitionAlignmentWrapper]
    if has_reward_wrapper:
        expected.append(RewardProfileWrapper)
    expected.extend([PreprocessObsWrapper, CommandActionWrapper])

    for index, wrapper_type in enumerate(expected):
        if index >= len(order) or order[index] is not wrapper_type:
            names = [cls.__name__ for cls in order]
            expected_names = [cls.__name__ for cls in expected]
            raise AssertionError(
                f"Wrapper order mismatch. Expected {expected_names}, got {names}."
            )


def make_env(
    task_spec: TaskSpec,
    seed: int | None = None,
    render_mode: str | None = None,
    reward_profile: str = "base",
    macro_reward_bonus: float = 0.0,
    macro_reward_radius: float = 0.6,
    macro_reward_alignment_threshold: float = 0.6,
    macro_reward_cap: float = 0.5,
) -> Callable[[], gym.Env]:
    def _resolve_site_name(base_env: gym.Env, site_name: str) -> str:
        if not isinstance(site_name, str) or site_name == "":
            return site_name
        sim = getattr(base_env, "sim", None)
        model = getattr(sim, "model", None)
        if model is None:
            return site_name

        def _site_exists(name: str) -> bool:
            try:
                model.site_name2id(name)
            except Exception:
                return False
            return True

        if _site_exists(site_name):
            return site_name
        if not site_name.startswith("/"):
            slash_name = f"/{site_name}"
            if _site_exists(slash_name):
                return slash_name
        return site_name

    def _init():
        env = gym.make(task_spec.env_id, render_mode=render_mode)
        base_env = env.unwrapped
        if hasattr(base_env, "goal_site"):
            base_env.goal_site = _resolve_site_name(base_env, base_env.goal_site)
        if hasattr(base_env, "target_name"):
            base_env.target_name = _resolve_site_name(base_env, base_env.target_name)
        reward_config = getattr(base_env, "reward_config", None)
        if isinstance(reward_config, dict):
            base_env.reward_config = _adjust_reward_config(reward_config, reward_profile)
        env = CommandActionWrapper(env)
        env = PreprocessObsWrapper(env, task_spec.one_hot)
        if reward_profile != "base" or macro_reward_bonus > 0.0:
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
        env = CompetitionAlignmentWrapper(env, task_spec.name)
        _assert_wrapper_order(env, reward_profile != "base" or macro_reward_bonus > 0.0)
        if seed is not None:
            env.reset(seed=seed)
        return env

    return _init


def compute_terminal_score(
    reward_terms: Dict[str, float],
    reward_config: Dict[str, float],
    steps: int,
    steps_weight: float,
) -> float:
    score = 0.0
    for key, weight in reward_config.items():
        if key in reward_terms:
            score += float(weight) * float(reward_terms[key])
    score += float(steps_weight) * float(steps)
    return float(score)


def compute_competition_terminal_score(
    task_name: str,
    reward_terms: Dict[str, float],
    steps: int,
) -> float:
    return float(compute_competition_score(reward_terms, task_name, steps))


def evaluate_policy(
    model,
    task_spec: TaskSpec,
    episodes: int,
    steps_weight: float,
    deterministic: bool = True,
    seed: int = 0,
    reward_profile: str = "base",
    macro_reward_bonus: float = 0.0,
    macro_reward_radius: float = 0.6,
    macro_reward_alignment_threshold: float = 0.6,
    macro_reward_cap: float = 0.5,
) -> Tuple[float, List[float], List[int], int | None]:
    env = make_env(
        task_spec,
        seed=seed,
        reward_profile=reward_profile,
        macro_reward_bonus=macro_reward_bonus,
        macro_reward_radius=macro_reward_radius,
        macro_reward_alignment_threshold=macro_reward_alignment_threshold,
        macro_reward_cap=macro_reward_cap,
    )()
    scores: List[float] = []
    lengths: List[int] = []
    max_steps = _resolve_max_steps(env)

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        terminated = truncated = False
        steps = 0
        reward_terms = {}

        while not (terminated or truncated):
            if hasattr(model, "predict"):
                action, _ = model.predict(obs, deterministic=deterministic)
            else:
                action = model(obs)
            obs, _, terminated, truncated, info = env.step(action)
            reward_terms = info.get("reward_terms", reward_terms)
            steps += 1

        reward_config = getattr(env.unwrapped, "reward_config", {})
        if not isinstance(reward_config, dict):
            reward_config = {}
        score = compute_terminal_score(reward_terms, reward_config, steps, steps_weight)
        scores.append(score)
        lengths.append(steps)

    env.close()
    return float(np.mean(scores)), scores, lengths, max_steps


def evaluate_policy_competition(
    model,
    task_spec: TaskSpec,
    episodes: int,
    deterministic: bool = True,
    seed: int = 0,
    reward_profile: str = "base",
    macro_reward_bonus: float = 0.0,
    macro_reward_radius: float = 0.6,
    macro_reward_alignment_threshold: float = 0.6,
    macro_reward_cap: float = 0.5,
) -> Tuple[float, List[Dict[str, float]]]:
    env = make_env(
        task_spec,
        seed=seed,
        reward_profile=reward_profile,
        macro_reward_bonus=macro_reward_bonus,
        macro_reward_radius=macro_reward_radius,
        macro_reward_alignment_threshold=macro_reward_alignment_threshold,
        macro_reward_cap=macro_reward_cap,
    )()
    reports: List[Dict[str, float]] = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        terminated = truncated = False
        steps = 0
        reward_terms: Dict[str, float] = {}

        while not (terminated or truncated):
            if hasattr(model, "predict"):
                action, _ = model.predict(obs, deterministic=deterministic)
            else:
                action = model(obs)
            obs, _, terminated, truncated, info = env.step(action)
            current_terms = info.get("reward_terms", reward_terms)
            reward_terms = current_terms if isinstance(current_terms, dict) else reward_terms
            steps += 1

        report = compute_competition_report(reward_terms, task_spec.name, steps)
        reports.append(report)

    env.close()
    mean_comp_total = float(np.mean([report["comp_total"] for report in reports])) if reports else 0.0
    return mean_comp_total, reports


def evaluate_all_tasks(
    model,
    episodes: int,
    steps_weight: float,
    deterministic: bool = True,
    seed: int = 0,
    eval_seeds: List[int] | None = None,
    reward_profile: str = "base",
    macro_reward_bonus: float = 0.0,
    macro_reward_radius: float = 0.6,
    macro_reward_alignment_threshold: float = 0.6,
    macro_reward_cap: float = 0.5,
) -> Dict[str, float]:
    return evaluate_selected_tasks(
        model,
        TASK_SPECS,
        episodes=episodes,
        steps_weight=steps_weight,
        deterministic=deterministic,
        seed=seed,
        eval_seeds=eval_seeds,
        reward_profile=reward_profile,
        macro_reward_bonus=macro_reward_bonus,
        macro_reward_radius=macro_reward_radius,
        macro_reward_alignment_threshold=macro_reward_alignment_threshold,
        macro_reward_cap=macro_reward_cap,
    )


def evaluate_selected_tasks(
    model,
    task_specs: List[TaskSpec],
    episodes: int,
    steps_weight: float,
    deterministic: bool = True,
    seed: int = 0,
    eval_seeds: List[int] | None = None,
    reward_profile: str = "base",
    macro_reward_bonus: float = 0.0,
    macro_reward_radius: float = 0.6,
    macro_reward_alignment_threshold: float = 0.6,
    macro_reward_cap: float = 0.5,
) -> Dict[str, float]:
    seeds = eval_seeds if eval_seeds is not None else [seed]
    if not seeds:
        seeds = [seed]

    per_task_scores: Dict[str, List[float]] = {task.name: [] for task in task_specs}
    all_lengths: List[int] = []
    max_steps = None

    for seed_value in seeds:
        for task_index, task_spec in enumerate(task_specs):
            mean_score, _, lengths, task_max_steps = evaluate_policy(
                model,
                task_spec,
                episodes=episodes,
                steps_weight=steps_weight,
                deterministic=deterministic,
                seed=seed_value + task_index * 1000,
                reward_profile=reward_profile,
                macro_reward_bonus=macro_reward_bonus,
                macro_reward_radius=macro_reward_radius,
                macro_reward_alignment_threshold=macro_reward_alignment_threshold,
                macro_reward_cap=macro_reward_cap,
            )
            per_task_scores[task_spec.name].append(mean_score)
            all_lengths.extend(lengths)
            if task_max_steps is not None:
                max_steps = task_max_steps if max_steps is None else max(max_steps, task_max_steps)

    per_task = {name: float(np.mean(scores)) for name, scores in per_task_scores.items()}
    overall = float(np.mean(list(per_task.values()))) if per_task else 0.0
    per_task["S_overall"] = overall
    if all_lengths:
        ep_len_mean = float(np.mean(all_lengths))
        per_task["ep_len_mean"] = ep_len_mean
        if max_steps is not None and ep_len_mean >= 0.98 * max_steps:
            print(
                f"[eval] Warning: mean episode length {ep_len_mean:.1f} near max_steps={max_steps} "
                "(timeouts likely)."
            )
    return per_task


def evaluate_all_tasks_competition(
    model,
    episodes: int,
    deterministic: bool = True,
    seed: int = 0,
    eval_seeds: List[int] | None = None,
    reward_profile: str = "base",
    macro_reward_bonus: float = 0.0,
    macro_reward_radius: float = 0.6,
    macro_reward_alignment_threshold: float = 0.6,
    macro_reward_cap: float = 0.5,
) -> Dict[str, float]:
    return evaluate_selected_tasks_competition(
        model,
        TASK_SPECS,
        episodes=episodes,
        deterministic=deterministic,
        seed=seed,
        eval_seeds=eval_seeds,
        reward_profile=reward_profile,
        macro_reward_bonus=macro_reward_bonus,
        macro_reward_radius=macro_reward_radius,
        macro_reward_alignment_threshold=macro_reward_alignment_threshold,
        macro_reward_cap=macro_reward_cap,
    )


def evaluate_selected_tasks_competition(
    model,
    task_specs: List[TaskSpec],
    episodes: int,
    deterministic: bool = True,
    seed: int = 0,
    eval_seeds: List[int] | None = None,
    reward_profile: str = "base",
    macro_reward_bonus: float = 0.0,
    macro_reward_radius: float = 0.6,
    macro_reward_alignment_threshold: float = 0.6,
    macro_reward_cap: float = 0.5,
) -> Dict[str, float]:
    seeds = eval_seeds if eval_seeds is not None else [seed]
    if not seeds:
        seeds = [seed]

    per_task_reports: Dict[str, List[Dict[str, float]]] = {task.name: [] for task in task_specs}

    for seed_value in seeds:
        for task_index, task_spec in enumerate(task_specs):
            _, reports = evaluate_policy_competition(
                model,
                task_spec,
                episodes=episodes,
                deterministic=deterministic,
                seed=seed_value + task_index * 1000,
                reward_profile=reward_profile,
                macro_reward_bonus=macro_reward_bonus,
                macro_reward_radius=macro_reward_radius,
                macro_reward_alignment_threshold=macro_reward_alignment_threshold,
                macro_reward_cap=macro_reward_cap,
            )
            per_task_reports[task_spec.name].extend(reports)

    results: Dict[str, float] = {}
    comp_total_sum = 0.0

    for task_spec in task_specs:
        task_name = task_spec.name
        reports = per_task_reports[task_name]
        aggregated = aggregate_competition_reports(task_name, reports)
        results[f"{task_name}/comp_total"] = float(aggregated["comp_total"])
        results[f"{task_name}/missing_component_count"] = float(
            aggregated["missing_component_count"]
        )
        for component in COMP_REWARD_CONFIG[task_name].keys():
            key = f"comp_{component}"
            results[f"{task_name}/{key}"] = float(aggregated.get(key, 0.0))
        comp_total_sum += float(aggregated["comp_total"])

    results["comp_total_sum"] = float(comp_total_sum)
    return results
