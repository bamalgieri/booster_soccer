# Implementation Report: Competition Scoring Alignment + Macros

## Summary
- Added competition scoring module and evaluation utilities aligned to website weights, with new eval_comp logging and best checkpoint selection by comp_total_sum.
- Implemented sentinel-triggered kick macros with proximity/alignment gating, per-episode counters, and optional macro reward shaping.
- Updated overnight_3stage to select resumes via competition scoring, add task-weighted env allocation, and pass HF token/home safely.
- Added beginner-friendly quick-start examples for scoring, evaluation, and macro triggering.
- Added a competition-only eval option to skip legacy scoring during training.

## Files Touched
- training_scripts/competition_scoring.py
- training_scripts/multitask_utils.py
- training_scripts/train_multitask.py
- training_scripts/eval_multitask.py
- training_scripts/smoke_eval_competition.py
- training_scripts/overnight_3stage.py
- booster_control/macro_playback.py
- tests/test_competition_scoring.py
- tests/test_macro_playback.py
- quick_start_examples.md
- task_plan.md
- notes.md

## Tests Run
- python -m unittest tests.test_competition_scoring tests.test_macro_playback
