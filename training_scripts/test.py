import sys, os
import argparse
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import glfw
from stable_baselines3 import TD3, SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from huggingface_hub import hf_hub_download

# Make repo root importable without absolute paths
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_root)

import sai_mujoco  # noqa: F401  # registers envs
from booster_control.t1_utils import LowerT1JoyStick

# ---------- Command→Action wrapper ----------
class CommandActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.base_env = env.unwrapped
        self.lower_control = LowerT1JoyStick(self.base_env)
        # RL policy outputs 3-dim command (vx, vy, yaw_rate) in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def action(self, command):
        # NOTE: relies on base_env private getters; works but is brittle.
        observation = self.base_env._get_obs()
        info = self.base_env._get_info()
        ctrl, _ = self.lower_control.get_actions(command, observation, info)
        return ctrl


def resolve_algo(env_id: str, algo: str | None) -> str:
    if algo:
        return algo.lower()
    return "sac" if "kicktotarget" in env_id.lower() else "td3"


def maybe_normalize_obs(obs, obs_rms, epsilon: float):
    if obs_rms is None:
        return obs
    return (obs - obs_rms.mean) / (np.sqrt(obs_rms.var) + epsilon)


def load_vecnormalize_stats(vecnormalize_path: str, env_fn):
    vec_env = DummyVecEnv([env_fn])
    vec_norm = VecNormalize.load(vecnormalize_path, vec_env)
    vec_norm.training = False
    vec_norm.norm_reward = False
    obs_rms = vec_norm.obs_rms
    epsilon = vec_norm.epsilon
    vec_env.close()
    return obs_rms, epsilon


def get_viewer_window(env):
    base_env = getattr(env, "base_env", None)
    if base_env is None:
        base_env = env.unwrapped
    renderer = getattr(base_env, "mujoco_renderer", None)
    viewer = getattr(renderer, "viewer", None) if renderer is not None else None
    return getattr(viewer, "window", None) if viewer is not None else None


def resolve_model_filename(env_id: str) -> str:
    """
    Pick the model filename based on env name.
    - If 'goalie' in env_id (case-insensitive) -> goalie model
    - Else if 'kick' in env_id -> kicker model
    - Else raise a helpful error
    """
    env_lc = env_id.lower()
    if "goalie" in env_lc:
        return "models/td3_goalie_penalty_kick.zip"
    if "kick" in env_lc:
        # Adjust if your repo uses a different filename for the kicker model
        return "models/sac_kick_to_target.zip"
    raise ValueError(
        f"Could not resolve a model for env '{env_id}'. "
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="LowerT1GoaliePenaltyKick-v0",
        help="Gym env ID (e.g., LowerT1GoaliePenaltyKick-v0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device for SB3 (e.g., 'cpu', 'mps', 'cuda')",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of rollout episodes",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a local SB3 zip (default: download pretrained model).",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        choices=["td3", "sac", "ppo"],
        help="SB3 algorithm used to train the model (default: infer from env).",
    )
    parser.add_argument(
        "--vecnormalize-path",
        dest="vecnormalize_path",
        type=str,
        default=None,
        help="Path to VecNormalize stats (vecnormalize.pkl) to normalize observations.",
    )
    parser.add_argument(
        "--multitask",
        action="store_true",
        help="Use multitask wrappers from training_scripts/multitask_utils.",
    )
    args = parser.parse_args()

    if args.multitask:
        from multitask_utils import TASK_SPECS, make_env

        task_spec = next((spec for spec in TASK_SPECS if spec.env_id == args.env), None)
        if task_spec is None:
            print(f"[ERROR] Unknown env for multitask wrapper: {args.env}")
            sys.exit(1)
        render_env_fn = make_env(task_spec, render_mode="human")
        eval_env_fn = make_env(task_spec, render_mode=None)
    else:
        def render_env_fn():
            base_env = gym.make(args.env, render_mode="human")
            return CommandActionWrapper(base_env)

        def eval_env_fn():
            base_env = gym.make(args.env, render_mode=None)
            return CommandActionWrapper(base_env)

    obs_rms = None
    obs_rms_epsilon = 1e-8
    if args.vecnormalize_path:
        vec_path = os.path.expanduser(args.vecnormalize_path)
        if not os.path.exists(vec_path):
            print(f"[ERROR] VecNormalize stats not found: {vec_path}")
            sys.exit(1)
        obs_rms, obs_rms_epsilon = load_vecnormalize_stats(vec_path, eval_env_fn)
        print(f"[INFO] Loaded VecNormalize stats from {vec_path}")

    if args.model:
        model_file = os.path.expanduser(args.model)
        if not os.path.exists(model_file):
            print(f"[ERROR] Model file not found: {model_file}")
            sys.exit(1)
    else:
        # Resolve model path from env name
        try:
            model_relpath = resolve_model_filename(args.env)
        except ValueError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)

        # Try to fetch the model from Hugging Face
        try:
            model_file = hf_hub_download(
                repo_id="SaiResearch/booster_soccer_models",
                filename=model_relpath,
                repo_type="model",
            )
        except Exception as e:
            print(
                f"[ERROR] Model file '{model_relpath}' not found in SaiResearch/booster_soccer_models.\n"
                f"Details: {e}"
            )
            sys.exit(1)

    algo = resolve_algo(args.env, args.algo)
    algo_map = {"td3": TD3, "sac": SAC, "ppo": PPO}
    if algo not in algo_map:
        print(f"[ERROR] Unsupported algo: {algo}")
        sys.exit(1)
    model = algo_map[algo].load(model_file, device=args.device)

    # Build env
    env = render_env_fn()
    use_lowlevel_obs = (not args.multitask) and ("KickToTarget" in args.env)

    def get_model_obs(obs, info):
        if use_lowlevel_obs:
            obs = env.lower_control.get_obs(np.zeros(3), obs, info)
        return maybe_normalize_obs(obs, obs_rms, obs_rms_epsilon)

    # ---- Rollout ----
    for ep in range(args.episodes):
        obs, info = env.reset(seed=42 + ep)
        terminated = truncated = False
        ep_return = 0.0
        print(f"[Episode {ep+1}] Running. Press ESC to stop.")

        while not (terminated or truncated):
            window = get_viewer_window(env)
            if window is not None and glfw.get_current_context() is not None:
                # Stop if user hit ESC inside the MuJoCo window
                if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                    print("\n[INFO] ESC pressed — stopping and closing.")
                    env.close()
                    sys.exit(0)

                # Stop if user clicked the window close button (red X)
                if glfw.window_should_close(window):
                    print("\n[INFO] Window closed — stopping and exiting.")
                    env.close()
                    sys.exit(0)
            model_obs = get_model_obs(obs, info)
            action, _ = model.predict(model_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)

        print(f"[Episode {ep+1}] return = {ep_return:.3f}")

    env.close()
    print("[INFO] Environment closed. Exiting.")


if __name__ == "__main__":
    main()
