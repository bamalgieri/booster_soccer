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
from imitation_learning.scripts.preprocessor import Preprocessor


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



class CommandActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.base_env = env.unwrapped
        self.lower_control = LowerT1JoyStick(self.base_env)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

    def action(self, command):
        observation = self.base_env._get_obs()
        info = self.base_env._get_info()
        ctrl, _ = self.lower_control.get_actions(command, observation, info)
        return ctrl


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
    def __init__(self, env: gym.Env, reward_profile: str, apply_timeout: bool, apply_shot: bool):
        super().__init__(env)
        self.reward_profile = reward_profile
        self.apply_timeout = apply_timeout
        self.apply_shot = apply_shot
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

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = 0.0
        if self.apply_timeout and truncated and not terminated:
            shaped += self.timeout_penalty
        if self.apply_shot and self._has_shot_attempt(info):
            shaped += self.shot_bonus
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


def make_env(
    task_spec: TaskSpec,
    seed: int | None = None,
    render_mode: str | None = None,
    reward_profile: str = "base",
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
        if reward_profile != "base":
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
            )
        env = CompetitionAlignmentWrapper(env, task_spec.name)
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


def evaluate_policy(
    model,
    task_spec: TaskSpec,
    episodes: int,
    steps_weight: float,
    deterministic: bool = True,
    seed: int = 0,
    reward_profile: str = "base",
) -> Tuple[float, List[float], List[int], int | None]:
    env = make_env(task_spec, seed=seed, reward_profile=reward_profile)()
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


def evaluate_all_tasks(
    model,
    episodes: int,
    steps_weight: float,
    deterministic: bool = True,
    seed: int = 0,
    eval_seeds: List[int] | None = None,
    reward_profile: str = "base",
) -> Dict[str, float]:
    seeds = eval_seeds if eval_seeds is not None else [seed]
    if not seeds:
        seeds = [seed]

    per_task_scores: Dict[str, List[float]] = {task.name: [] for task in TASK_SPECS}
    all_lengths: List[int] = []
    max_steps = None

    for seed_value in seeds:
        for task_index, task_spec in enumerate(TASK_SPECS):
            mean_score, _, lengths, task_max_steps = evaluate_policy(
                model,
                task_spec,
                episodes=episodes,
                steps_weight=steps_weight,
                deterministic=deterministic,
                seed=seed_value + task_index * 1000,
                reward_profile=reward_profile,
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
