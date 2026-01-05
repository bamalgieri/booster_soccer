# Task Plan: Competition Scoring Alignment + Macros

## Goal

Implement competition-style scoring/evaluation and macro support end-to-end while preserving existing outputs and adding required tests.

## Phases

- [x] Phase 1: Plan and setup
- [x] Phase 2: Inspect current evaluation/scoring code paths
- [x] Phase 3: Implement competition scoring module + tests
- [x] Phase 4: Integrate competition eval into train/eval + tests
- [x] Phase 5: Macro trigger/playback + reward gate + tests
- [x] Phase 6: Optional BC pretrain pipeline (if straightforward)
- [x] Phase 7: overnight_3stage updates + HF passthrough
- [x] Phase 8: Wrap-up, notes, and verification

## Key Questions

1. Where is evaluation step counting implemented and how do reward_terms flow into info?
2. Which files control wrapper order and where to assert invariants?
3. Where should macro playback live to avoid controller changes?

## Decisions Made

- Plan to keep competition scoring in a pure module with explicit report schema.
- Preserve existing eval stdout/log keys; add eval_comp/\* alongside.

## Errors Encountered

- [2025-02-14] apply_patch rejected while updating task_plan.md; will update via subsequent patch.
- [2025-02-14] unittest failed due to missing huggingface_hub import in macro_playback; switched to lazy import in \_load_npz.
- [2025-02-14] smoke_eval_competition failed due to escaped quotes in wrapper order assertion; fixed string literal.
- [2025-02-14] smoke_eval_competition failed due to CommandActionWrapper.step bypassing action() transform; now calling super().step.
- [2025-02-14] overnight_3stage run hit VecMonitor KeyError for macro_reward_bonus_mean; now only requests info_keywords when reward wrapper is active.
- [2025-02-14] overnight_3stage run hit SB3 logger truncation collision for eval_comp keys; exclude eval_comp from stdout output.

## Status

**All phases complete** - Ready for review

---

# Task Plan: Single-Task Training Support

## Goal

Add single-task selection support to training/evaluation while preserving multitask defaults and CLI compatibility.

## Phases

- [x] Phase 1: Inspect train/eval entrypoints and task utilities for integration points
- [x] Phase 2: Implement task selection, evaluation filtering, and best model naming
- [x] Phase 3: Thread --tasks through overnight_3stage and document/runbook updates

## Notes

- Keep multi-task default behavior unchanged when --tasks is omitted.

---

# Task Plan: HF Flags + Eval Mode Chaining

## Goal

Add HF auth/cache passthrough to training and evaluation-mode checkpoint chaining with legacy/competition support while preserving backward compatibility.

## Phases

- [x] Phase 1: Inspect current CLI/eval flow and checkpoint usage
- [x] Phase 2: Add HF env wiring + legacy-only eval support in training
- [x] Phase 3: Add eval-mode flags + checkpoint selection in overnight runner
- [x] Phase 4: Update notes and verification guidance
