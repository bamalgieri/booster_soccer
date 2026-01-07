# Quick Start Examples

This guide shows common ways to use the new competition scoring, evaluation, and macro features.
It is written for people who are new to both this project and programming.

If you have never run Python before, you can still follow the examples as "recipes":

- Look for the file paths.
- Copy the exact code into a new file if needed.
- Run the commands from the project root.

---

## 1) Single-episode competition scoring (manual)

Use this when you already have the terminal `reward_terms` from the environment and the number of steps.
This is the core scoring function used by evaluation and checkpoint selection (official scoring uses
`steps=1.0` for the steps component regardless of episode length).

Create a file named `examples_single_episode_score.py`:

```python
from training_scripts.competition_scoring import compute_competition_report

# Example terminal reward terms from an episode
reward_terms = {
    "robot_distance_ball": 1.2,
    "ball_vel_twd_goal": 0.4,
    "goal_scored": 1.0,
    "offside": 0.0,
    "ball_hits": 0.0,
    "robot_fallen": 0.0,
    "ball_blocked": 0.0,
}

# The number of steps in that episode (stored as episode_steps metadata)
steps = 350

# Task name must match the TaskSpec name
task_name = "goalie_penalty_kick"

report = compute_competition_report(reward_terms, task_name, steps)
print("comp_total:", report["comp_total"])
print("comp_steps:", report["comp_steps"])
print("missing_component_count:", report["missing_component_count"])
```

What to expect:

- `comp_total` is the final competition score for that episode.
- `comp_steps` is the step penalty (weight \* steps).
- `missing_component_count` tells you how many expected reward components were missing.

Common mistakes:

- Using the wrong `task_name`. It must be one of:
  - `goalie_penalty_kick`
  - `obstacle_penalty_kick`
  - `kick_to_target`
- Expecting the steps component to scale with episode length; the official evaluator injects
  `steps=1.0`, so only `episode_steps` uses the provided step count.

---

## 2) Multi-task evaluation (automatic)

### Option A: Use the evaluation script (no coding)

If you already have a trained model:

```powershell
python training_scripts/eval_multitask.py --model path\to\best_model.zip --episodes 5 --deterministic
```

If the model was trained with `--normalize-obs`, load the matching stats:

```powershell
python training_scripts/eval_multitask.py --model path\to\best_model.zip --episodes 5 --deterministic --vecnormalize-path training_runs\multitask_td3\vecnormalize.pkl
```

Use the `vecnormalize.pkl` from the same training run to avoid shape mismatches.

This prints:

- Per-task terminal scores (old env-style metrics)
- The legacy "Competition Score Projection Report Card"
- The new competition scoring block (per-task comp_total and comp_total_sum)

### Option B: Use the evaluation utilities (small script)

Create a file named `examples_multitask_eval.py`:

```python
import numpy as np
from training_scripts.multitask_utils import evaluate_all_tasks_competition

# A tiny dummy policy that always outputs zeros (used for smoke tests)
class ZeroPolicy:
    def predict(self, obs, deterministic=True):
        return np.zeros(3, dtype=np.float32), None

model = ZeroPolicy()

scores = evaluate_all_tasks_competition(
    model,
    episodes=1,
    deterministic=True,
    seed=0,
)

# Print the overall competition sum
print("comp_total_sum:", scores["comp_total_sum"])

# Print a per-task total
print("goalie comp_total:", scores["goalie_penalty_kick/comp_total"])
```

What to expect:

- The `scores` dict includes keys for each task and each component.
- The keys are stable, even if some reward components are missing.

---

## 3) Macro triggering (sentinel-based, proximity-gated)

Macros are short, pre-recorded motion sequences that can be triggered by a special command.
The trigger only fires if:

- The command matches the sentinel pattern.
- The ball is close to the robot.
- The ball is aligned with the goal or target.

### A) Trigger logic (pure function)

You can test the trigger decision without running the simulator:

```python
import numpy as np
from booster_control.macro_playback import trigger_macro

# Sentinel command pattern (1, -1, 1)
command = np.array([1.0, -1.0, 1.0], dtype=np.float32)

# Example info dict with required keys
info = {
    "ball_xpos_rel_robot": np.array([0.2, 0.0, 0.0], dtype=np.float32),
    "goal_team_0_rel_robot": np.array([1.0, 0.0, 0.0], dtype=np.float32),
}

should_trigger = trigger_macro(command, info)
print("trigger:", should_trigger)
```

### B) In the environment (automatic)

You do not need to call `trigger_macro()` yourself during training.
It is wired into the `CommandActionWrapper` inside `make_env()`.

When a macro triggers, the `info` dict gains these fields:

- `macro_active`: True while the macro is playing
- `macro_triggered`: True on the step it starts
- `macro_name`: the macro name, e.g. `goal_kick`
- `macro_step_index`: current step within the macro
- `macro_trigger_count`: count of triggers so far in the episode
- `macro_active_steps`: total steps played so far
- `macro_aborted_count`: count of aborted macros so far

### C) Loading a macro sequence manually

If you want to inspect the motion:

```python
from booster_control.macro_playback import load_kick_macro
from booster_control.t1_utils import LowerT1JoyStick
import gymnasium as gym

# Create an env to access the lower-level controller config
env = gym.make("LowerT1GoaliePenaltyKick-v0")
lower_control = LowerT1JoyStick(env.unwrapped)

macro = load_kick_macro("goal_kick", lower_control, max_steps=30)
print("macro length:", macro.actions.shape[0])
```

---

## 4) Optional: macro reward bonus (small shaping)

If you want to encourage the policy to trigger macros at good times,
set a small bonus in training (requires `reward_profile=pressure_shot` or `tight` by default):

```powershell
python training_scripts/train_multitask.py --reward-profile pressure_shot --macro-reward-bonus 0.5
```

For competition-shaped training, you must opt in to the macro bonus:

```powershell
python training_scripts/train_multitask.py --reward-profile competition --allow-macro-bonus-with-competition --macro-reward-bonus 0.1
```

The wrapper only applies the bonus when:

- A macro is triggered
- The ball is within the radius
- The alignment threshold is met

The per-episode mean is logged as `macro/reward_bonus_mean`.

---

## 5) CLI flag reference (train_multitask.py and overnight_3stage.py)

This section lists every flag you can pass to these scripts, their defaults,
and simple "light load" vs "heavier load" suggestions.

"Light load" means faster runs with less evaluation time.
"Heavier load" means slower runs with more evaluation or bigger models.

### A) train_multitask.py flags

| Flag                                 | Default                         | What it does                                                  | Light load option | Heavier load option            |
| ------------------------------------ | ------------------------------- | ------------------------------------------------------------- | ----------------- | ------------------------------ |
| `--total-timesteps`                  | `1000000`                       | Total training steps.                                         | `200000`          | `3000000`                      |
| `--n-envs`                           | `6`                             | Number of parallel environments.                              | `2` or `4`        | `8` or `12`                    |
| `--seed`                             | `0`                             | Random seed for reproducibility.                              | keep default      | keep default                   |
| `--device`                           | `"cuda"`                        | Compute device (`cuda` or `cpu`).                             | `cpu` if no GPU   | `cuda`                         |
| `--algo`                             | `"td3"`                         | RL algorithm (`td3`, `ppo`, `sac`).                           | `"td3"`           | `"ppo"` or `"sac"`             |
| `--net-arch`                         | `"256,256,128"`                 | Policy MLP sizes.                                             | `"128,128"`       | `"256,256,256"`                |
| `--lr`                               | `3e-4`                          | Learning rate.                                                | keep default      | keep default                   |
| `--tau`                              | `0.005`                         | Target network update rate.                                   | keep default      | keep default                   |
| `--action-noise-sigma`               | `0.0`                           | Exploration noise.                                            | `0.0`             | `0.05` to `0.1`                |
| `--buffer-size`                      | `500000`                        | Replay buffer size.                                           | `100000`          | `1000000`                      |
| `--batch-size`                       | `256`                           | Training batch size.                                          | `128`             | `512`                          |
| `--learning-starts`                  | `20000`                         | Steps before learning starts.                                 | `5000`            | `50000`                        |
| `--eval-episodes`                    | `8`                             | Episodes per task per seed.                                   | `2`               | `20`                           |
| `--eval-seeds`                       | `"0,1,2,3"`                     | Eval seed list (comma-separated).                             | `"0"`             | `"0,1,2,3,4"`                  |
| `--eval-freq`                        | `50000`                         | How often to run eval (timesteps).                            | `200000`          | `10000`                        |
| `--steps-weight`                     | `-0.05`                         | Legacy eval step penalty only.                                | keep default      | keep default                   |
| `--normalize-obs`                    | `False`                         | Normalize observations with VecNormalize.                     | `False`           | `True`                         |
| `--vecnormalize-path`                | `None`                          | Load/save VecNormalize stats.                                 | n/a               | n/a                            |
| `--bc-weights`                       | `None`                          | Path to BC weights for init.                                  | n/a               | n/a                            |
| `--freeze-bc-layers`                 | `0`                             | Freeze first N actor Linear layers (requires `--bc-weights`). | `0`               | `2`                            |
| `--freeze-until-step`                | `0`                             | Timesteps to keep BC layers frozen.                           | `0`               | `200000`                       |
| `--bc-replay-dataset`                | `None`                          | Demo replay dataset (`.npz`) for prefill.                     | n/a               | n/a                            |
| `--bc-replay-fraction`               | `0.0`                           | Fraction of replay buffer to prefill.                         | `0.0`             | `0.1`                          |
| `--resume-model`                     | `None`                          | Path to resume checkpoint.                                    | n/a               | n/a                            |
| `--macro-reward-bonus`               | `0.0`                           | Extra reward for good macro use.                              | `0.0`             | `0.25` to `0.5`                |
| `--macro-reward-radius`              | `0.6`                           | Ball distance gate for bonus.                                 | keep default      | keep default                   |
| `--macro-reward-alignment-threshold` | `0.6`                           | Alignment gate for bonus.                                     | keep default      | keep default                   |
| `--macro-reward-cap`                 | `0.5`                           | Max bonus per episode.                                        | keep default      | keep default                   |
| `--allow-macro-bonus-with-competition` | `False`                       | Allow macro bonus with `reward_profile=competition`.          | `False`           | `True`                         |
| `--competition-only-eval`            | `False`                         | Skip legacy eval pass.                                        | `True`            | `False`                        |
| `--task-weights`                     | `None`                          | Bias env slots by task.                                       | n/a               | n/a                            |
| `--reward-profile`                   | `"base"`                        | Reward shaping profile.                                       | keep default      | `"pressure_shot"` or `"tight"` |
| `--save-dir`                         | `"training_runs/multitask_td3"` | Output directory.                                             | n/a               | n/a                            |

python training_scripts/train_multitask.py --total-timesteps 200000 --n-envs 2 --buffer-size 100000 --batch-size 128 --learning-starts 10000 --eval-episodes 2 --eval-freq 40000 --macro-reward-bonus 0.5 --save-dir "training_runs/multitask_td3/multitask_td3_2" --resume-model training_runs\overnight_20260102_231947\stage2\best_model.zip

python train_multitask.py --total-timesteps --n-envs --action-noise-sigma --buffer-size --batch-size --learning-starts --eval-episodes --eval-freq --macro-reward-bonus --competition-only-eval --save-dir

Tip: Defaults use 4 eval seeds (`0,1,2,3`), so evaluation is slower. If you want faster runs,
lower `eval-episodes`, set `--eval-seeds "0"`, and increase `eval-freq` so evaluation happens
less often.

### B) 2-day workflow examples (condensed)

Day 1: build the BC dataset and train a BC policy.

```powershell
python training_scripts/build_bc_dataset.py --input-root booster_dataset/soccer/booster_lower_t1 --output booster_dataset/imitation_learning/bc_commands.npz --enable-sentinels --context-mode sentinel
python training_scripts/bc_train.py --dataset booster_dataset/imitation_learning/bc_commands.npz --out training_runs/bc_actor.pt --net-arch 256,256 --epochs 40 --batch-size 1024 --lr 3e-4 --seed 0
```

Day 2: multi-task RL fine-tuning and evaluation.

```powershell
python training_scripts/train_multitask.py --algo ppo --total-timesteps 5000000 --n-envs 32 --task-weights goalie_penalty_kick=1,obstacle_penalty_kick=1,kick_to_target=1 --bc-weights training_runs/bc_actor.pt --freeze-bc-layers 2 --freeze-until-step 200000 --bc-replay-dataset booster_dataset/imitation_learning/bc_commands.npz --bc-replay-fraction 0.1 --normalize-obs --save-dir training_runs/multitask_ppo --competition-only-eval --eval-freq 100000 --eval-episodes 4 --eval-seeds 0,1,2,3
python training_scripts/eval_multitask.py --model training_runs/multitask_ppo/best_model_competition.zip --episodes 20 --deterministic --vecnormalize-path training_runs/multitask_ppo/vecnormalize.pkl
```

### C) overnight_3stage.py flags

Note: This script sets many training hyperparameters internally per stage.
The flags below only override the exposed knobs.

| Flag                 | Default  | What it does                       | Light load option | Heavier load option |
| -------------------- | -------- | ---------------------------------- | ----------------- | ------------------- |
| `--resume-model`     | required | Starting checkpoint for stage 1.   | n/a               | n/a                 |
| `--base-seed`        | `0`      | Base seed for stages.              | keep default      | keep default        |
| `--n-envs`           | `None`   | Pass-through to train_multitask.   | `4`               | `8` or `12`         |
| `--device`           | `None`   | Pass-through device.               | `cpu` if no GPU   | `cuda`              |
| `--net-arch`         | `None`   | Pass-through network sizes.        | `"128,128"`       | `"256,256,256"`     |
| `--buffer-size`      | `None`   | Pass-through replay buffer size.   | `100000`          | `1000000`           |
| `--batch-size`       | `None`   | Pass-through batch size.           | `128`             | `512`               |
| `--eval-freq`        | `None`   | Pass-through eval frequency.       | `200000`          | `10000`             |
| `--steps-weight`     | `None`   | Pass-through legacy step penalty.  | keep default      | keep default        |
| `--task-weights`     | `None`   | Task slot bias (env counts).       | n/a               | n/a                 |
| `--eval-seeds`       | `None`   | Override all stage eval seeds.     | `"0"`             | `"0,1,2,3,4"`       |
| `--hf-token`         | `None`   | HF token for downloads (redacted). | n/a               | n/a                 |
| `--hf-home`          | `None`   | HF cache directory.                | n/a               | n/a                 |
| `--stage1-timesteps` | `None`   | Override stage 1 timesteps.        | `200000`          | `2000000`           |
| `--stage2-timesteps` | `None`   | Override stage 2 timesteps.        | `200000`          | `2000000`           |
| `--stage3-timesteps` | `None`   | Override stage 3 timesteps.        | `200000`          | `3000000`           |

If you omit a flag here, the script uses the defaults built into the stage configs
or the defaults from `train_multitask.py`.

---

## 6) Where these live in the codebase

- Competition scoring: `training_scripts/competition_scoring.py`
- Competition evaluation: `training_scripts/multitask_utils.py`
- Macro playback helper: `booster_control/macro_playback.py`
- Training/eval scripts: `training_scripts/train_multitask.py`, `training_scripts/eval_multitask.py`
- Smoke test: `training_scripts/smoke_eval_competition.py`

### 7) Single-Task calls with train_multitask.py

python training_scripts/train_multitask.py --resume-model training_runs\overnight_20260102_231947\stage2\best_model.zip --total-timesteps 50000 --hf-token "" --legacy-only-eval --n-envs 6 --tasks kick_to_target --save-dir training_runs/ktt_single

python training_scripts/train_multitask.py --total-timesteps 200000 --n-envs 6 --buffer-size 100000 --batch-size 128 --learning-starts 10000 --eval-episodes 2 --eval-freq 40000 --macro-reward-bonus 0.5 --tasks kick_to_target --save-dir training_runs/ktt_single --resume-model training_runs\overnight_20260102_231947\stage2\best_model.zip

### 8) Training

python training_scripts/eval_multitask.py --model training_runs/ktt_sac_gpu_500k/best_model_competition.zip --episodes 10 --steps-weight -0.001 --deterministic --task kick_to_target
