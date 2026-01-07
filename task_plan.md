# Task Plan: Competition Scoring Alignment + Diagnostics + Macro Bonus

## Goal
Align competition steps semantics with the official evaluator, add per-task competition diagnostics, and add opt-in macro bonus for competition-shaped training without changing official weights.

## Phases
- [x] Phase 1: Plan and setup
- [x] Phase 2: Research/gather information
- [x] Phase 3: Execute/build
- [x] Phase 4: Review and deliver

## Key Questions
1. Where are steps and reward_terms handled in the competition scoring and eval paths?
2. Where is macro reward bonus currently rejected and how should it be allowed with opt-in flag?

## Decisions Made
- Added comp_eval diagnostics using per-task term means/rates from competition eval results.
- Added macro_trigger_rate diagnostic to reflect macro usage per task.
- Documented the new competition macro bonus flag and steps semantics in quick start docs.

## Errors Encountered
- None yet.

## Status
**Completed** - Changes implemented, summarized, and verification commands prepared.
