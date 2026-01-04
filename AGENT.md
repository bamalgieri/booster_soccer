# Agent Instructions (Codex / Repo Contributor)

## Mission
Build ONE high-level policy that solves:
- LowerT1GoaliePenaltyKick-v0
- LowerT1ObstaclePenaltyKick-v0
- LowerT1KickToTarget-v0
WITHOUT per-task tuning (single architecture + single config + single checkpoint format).

## Scoring & Checkpoint Selection (Critical)
- Compute per-task mean terminal-only weighted reward: S1, S2, S3
- Overall score: S_overall = (S1 + S2 + S3) / 3  (equal task weights = 1)
- Always pick “best” checkpoint by S_overall, and always report S1/S2/S3/S_overall.

## Non-Negotiables
- Do not modify the low-level controller unless absolutely required.
- No env-specific heads, branches, or per-env hyperparameters.
- Ground all claims in code: cite file path + symbol + approximate line numbers.
- Prefer minimal diffs that reuse existing scripts.

## Work Order (Do in this exact order)
1) Repo recon map (training entrypoints, env registration, policy/controller contracts).
2) Extract env contracts (obs/action/reward dict keys/termination).
3) Implement evaluation-parity script matching the provided terminal-only weighting.
4) Implement BC pretrain from provided npz datasets.
5) Implement balanced multi-env RL fine-tuning (≈1/3 sampling each).
6) Add performance settings for a single consumer GPU.

## Output Format for Every Change
- What changed + why
- Diff(s) by file
- Command(s) to run
- How to verify (expected logs/metrics)
