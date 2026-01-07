# Notes: Competition Scoring Alignment + Diagnostics + Macro Bonus

## Sources

### Source 1: combined_website_evaluation.md
- URL: local file
- Key points:
  - Official evaluator injects `steps = 1.0` into raw_reward every timestep.
  - Score is computed only at episode end from raw_reward * reward_config.
  - Task weights: penalty kicks use steps=-1.0; kick_to_target uses steps=-0.3.

### Source 2: training_scripts/*
- URL: local files
- Key points:
  - `compute_competition_report` in `competition_scoring.py` already uses `step_value = 1.0` and records `episode_steps`.
  - `compute_competition_terminal_score` and `evaluate_policy_competition` pass episode steps into competition scorer.
  - Competition eval aggregates reward_terms but only logs comp_total metrics under `comp_eval`.
  - `make_env` rejects `macro_reward_bonus` unless reward_profile is `pressure_shot` or `tight`.

## Synthesized Findings

### Scoring + evaluation flow
- Competition eval collects `reward_terms` per episode and uses `compute_competition_report` for scoring.
- `evaluate_selected_tasks_competition` aggregates `comp_*` and a small set of term means (`_EVAL_TERM_KEYS`).
- Added competition diagnostics: per-task term means under `term_mean/<term>` plus success/offside rates.
- Competition eval now passes `steps=1` into competition scoring/reporting for website alignment.

### Macro bonus handling
- Macro trigger info is emitted by `CommandActionWrapper` (`macro_triggered`, `macro_active`, counts).
- Macro bonus shaping currently only exists in `RewardProfileWrapper` for non-competition profiles.
- Added opt-in competition macro bonus in `CompetitionAlignmentWrapper` with per-episode cap.
