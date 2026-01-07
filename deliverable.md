# Deliverable: Competition Scoring Alignment + Diagnostics + Macro Bonus

## Direct Result
- Competition scoring now enforces steps=1 in terminal scoring and reporting paths to match the official evaluator.
- Competition eval adds per-task term means and success/offside rates under `comp_eval/<task>/...` without removing existing keys.
- Added per-task macro usage rate (`macro_trigger_rate`) for competition eval diagnostics.
- Introduced an opt-in competition macro bonus via `--allow-macro-bonus-with-competition`, applied once per macro trigger with a per-episode cap.

## Diff Summary
- `training_scripts/competition_scoring.py`: Hard-code steps component to 1.0 with official-evaluator comment.
- `training_scripts/multitask_utils.py`: Align competition steps semantics, add competition diagnostics/rates, track macro triggers, and allow opt-in competition macro bonus.
- `training_scripts/train_multitask.py`: Add CLI flag, wire allow flag through env/eval, and log new comp_eval diagnostics.

## Commands (Exact Order)
1. `python -c "from training_scripts.competition_scoring import compute_competition_score; print(compute_competition_score({}, 'kick_to_target', 5000))"`
2. `python training_scripts/smoke_eval_competition.py --episodes 1`
3. `python training_scripts/train_multitask.py --reward-profile competition --allow-macro-bonus-with-competition --macro-reward-bonus 0.1 --total-timesteps 1000 --n-envs 1 --eval-episodes 1 --eval-freq 1000 --competition-only-eval --save-dir training_runs/comp_macro_smoke`

## Notes / Risks
- `compute_competition_report` now receives steps=1 from competition eval, so its `episode_steps` field no longer reflects actual episode length.
- Competition eval adds new `comp_eval/<task>/...` keys but leaves existing keys unchanged.
