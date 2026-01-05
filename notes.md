# Notes: Competition Scoring Alignment + Macros

## Sources

- SKILL.md
- reference.md
- examples.md

## Findings

### Evaluation and scoring

- Evaluation loop in `training_scripts/multitask_utils.py` uses a step counter in `evaluate_policy` and applies `compute_terminal_score(reward_terms, reward_config, steps, steps_weight)`.
- `reward_terms` are pulled from `info.get("reward_terms", reward_terms)` each step; terminal scoring uses the latest dict.

### Competition weights

- `combined_website_evaluation.md` defines per-task weight configs and adds `steps: 1.0` per timestep in the website evaluator.
- Task 1/2 weights include: `robot_distance_ball`, `ball_vel_twd_goal`, `goal_scored`, `offside`, `ball_hits`, `robot_fallen`, `ball_blocked`, `steps`.
- Task 3 weights include: `offside`, `success`, `distance`, `steps`.

### Tests

- Existing test files: `training_scripts/test.py`, `imitation_learning/test.py` (no standard test framework observed yet).
- Added `tests/test_competition_scoring.py` using `unittest` for pure scoring logic.
- Added `training_scripts/smoke_eval_competition.py` to exercise eval + comp eval paths with a zero-action policy.
- Added `tests/test_macro_playback.py` for sentinel gating and macro length cap.

### Macros + reward gate

- Macro playback implemented in `booster_control/macro_playback.py` using mimic qpos -> low-level action targets.
- CommandActionWrapper now supports sentinel-triggered macro playback with proximity + alignment gating, plus info counters.
- RewardProfileWrapper includes optional macro reward bonus gated on macro_triggered, ball proximity, and alignment; per-episode cap tracked.

### BC pretrain

- Existing `training_scripts/build_bc_dataset.py` and `training_scripts/bc_train.py` already provide a velocity-based BC pipeline; no new changes needed.

### Training/eval updates

- Competition evaluation utilities added in `training_scripts/multitask_utils.py` and wired into `train_multitask.py`/`eval_multitask.py`.
- Wrapper order assertion added in `training_scripts/multitask_utils.py` to guard `CommandActionWrapper -> PreprocessObsWrapper -> RewardProfileWrapper -> CompetitionAlignmentWrapper`.
- Task-weighted env slot allocation implemented via `build_task_list` and `--task-weights` in `train_multitask.py`.
- `training_scripts/overnight_3stage.py` now selects resume checkpoints using competition scoring and supports `--eval-seeds`, `--task-weights`, `--hf-token`, `--hf-home`.
- Added `--competition-only-eval` to `training_scripts/train_multitask.py` to skip legacy eval and only run competition scoring.

### Docs

- Added `quick_start_examples.md` with beginner-friendly usage examples and a full CLI flag reference.

## Validation Results

- `python -m unittest tests.test_competition_scoring tests.test_macro_playback`

---

# Notes: Single-Task Training Support

## Findings

- `training_scripts/train_multitask.py` builds envs via `build_task_list` and evaluates via `evaluate_all_tasks` and `evaluate_all_tasks_competition`.
- `training_scripts/multitask_utils.py` owns `TASK_SPECS`, `build_task_list`, and evaluation helpers that currently iterate all tasks.
- `training_scripts/overnight_3stage.py` builds the train command in `_build_base_cmd` and uses `evaluate_all_tasks_competition` for resume selection.
- Added single-task selection plumbing: `--tasks` parsing, `evaluate_selected_tasks`, and task-specific best model naming with a `best_model.zip` compatibility copy.

---

# Notes: HF Flags + Eval Mode Chaining

## Findings

- `training_scripts/train_multitask.py` now wires `--hf-home`/`--hf-token` into `HF_HOME`, hub/datasets caches, and HF token env vars without printing tokens.
- `MultiTaskEvalCallback` tracks legacy `S_overall` best separately from competition best, saving `best_model.zip` (legacy) and `best_model_competition.zip` (competition).
- Competition logs add `comp_eval/<task>` and `comp_eval/C_overall` alongside existing `eval_comp/*` metrics.
- `training_scripts/overnight_3stage.py` adds `--competition-only-eval`/`--legacy-only-eval`, enforces mutual exclusivity, and chains stages via `pick_best_checkpoint` with safe fallbacks.

---

# Notes: Implementation Checklist + Diffs

## Sources

- `training_scripts/multitask_utils.py`
- `training_scripts/train_multitask.py`
- `training_scripts/eval_multitask.py`
- `training_scripts/training.py`
- `README.md`

## Findings

- Task assignment is centralized in `build_task_list` and currently deterministic based on weights.
- Wrapper order is hard-coded; adding `ClipAction` or normalization wrappers will fail without relaxing checks.
- Evaluation utilities return only aggregate scores; adding per-episode term metrics requires extending the return path.
- SB3 `VecNormalize` stats are not saved/loaded; a separate save/load path is needed for eval parity.

## Decisions

- Use `VecNormalize` for training and pass `obs_rms` into evaluation to normalize observations for policy inference.
- Randomize per-env task assignment at startup using a seeded RNG; tasks remain fixed per env slot.
- Extend `--reward-profile` to include `env` and `competition`, while preserving `base` as an alias for `env`.
