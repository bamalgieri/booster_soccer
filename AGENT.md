# AGENT.md — Booster Soccer RL (Competition-Score Alignment + Diagnostics)

You are **Codex**, operating inside this repository. You will make code changes directly in the repo and return **clean diffs + exact commands to run**.

This repo already uses a “planning-with-files” workflow. You must **work with**:

- `SKILL.md` (workflow rules)
- `task_plan.md` (phases, decisions, error log, completion)
- `reference.md` (context-engineering principles + guardrails)
- `combined_website_evaluation.md` (**source-of-truth** for official competition scoring)

---

## 0) Non-Negotiables (Hard Constraints)

1. **Source of truth**

   - `combined_website_evaluation.md` is the _authoritative_ competition scoring logic.
   - If anything conflicts, match `combined_website_evaluation.md`.

2. **Do NOT change competition weights**

   - Do **not** modify the numerical weight dictionaries in `training_scripts/competition_scoring.py`.

3. **No breaking changes**

   - Keep function signatures stable unless explicitly instructed otherwise.
   - Do **not** remove or rename existing log keys. You may add new keys.

4. **Competition scoring semantics**

   - Competition score is computed at **episode end** only.
   - **Steps term semantics must match the official evaluator** (see `combined_website_evaluation.md`).

5. **Work style**
   - Read before deciding (per `SKILL.md`).
   - Update `task_plan.md` after each phase; log errors in “Errors Encountered”.

---

## 1) Primary Objective (What “Done” Means)

Implement the repo changes specified by the user’s “**4) Single Codex prompt to implement all recommended changes**” section.

If the user message includes that block, treat it as an **implementation spec** and follow it precisely.

At minimum, deliver:

### A) Steps semantics alignment (must match official evaluator)

- Ensure the `"steps"` component uses a constant `1.0` value (ignore actual episode length) in:
  - competition scoring report generation
  - terminal-score computation path

### B) Competition-eval diagnostics (additive only)

- Add per-task:
  - `success_rate`, `offside_rate`
  - per-term means under `comp_eval/<task>/term_mean/<term>`
- Keep existing `comp_eval/C_overall` and current keys intact.

### C) Optional macro bonus during competition-shaped training (opt-in flag)

- Add a CLI flag:
  - `--allow-macro-bonus-with-competition` (default false)
- When enabled, allow `macro_reward_bonus` under `reward_profile=competition` as training-only shaping:
  - do not alter official competition weights/scoring.

---

## 2) Required Workflow (Follow This Every Time)

### Phase 1 — Re-ground in goals

1. Open and re-read:
   - `task_plan.md`
   - `combined_website_evaluation.md`
   - any files named in the user spec (usually: `competition_scoring.py`, `multitask_utils.py`, `train_multitask.py`)
2. Confirm where the “steps” penalty is computed and how reward_terms flow.

### Phase 2 — Implement changes in small commits

Make changes in tight loops:

- Edit → run quick check → adjust → run check again

### Phase 3 — Verification + acceptance checks

You must run (or at least provide exact commands for) the acceptance checks described in the user spec.

### Phase 4 — Deliverables

Return:

- A concise summary of what changed
- A bullet list of impacted files
- The exact commands to verify locally
- Any risks / gotchas

Also update `task_plan.md`:

- Mark relevant phases complete
- Record decisions made
- Log any errors encountered + resolution

---

## 3) Guardrails (Common Failure Modes)

### Don’t accidentally “improve” the official scorer

- You may improve **training-time shaping** and **diagnostics**,
  but **competition scoring must remain identical** to `combined_website_evaluation.md`.

### Steps term gotcha

- If local code currently uses `steps = episode_length` (or similar) for the `"steps"` component, that will **misalign** with the official evaluator.
- Your fix must enforce `"steps" = 1.0` consistently in both:
  - score computation path
  - report generation path

### Logging gotcha

- Add keys, don’t rename.
- Keep prefixes stable: `comp_eval/...`

### Macro bonus gotcha

- Macro bonus must trigger **once per macro trigger event**, not per playback step.
- Make it visible in `info` so it can be logged.

---

## 4) Implementation Checklist (Use as Your Working TODO)

### A) Steps semantics

- [ ] Locate local competition scoring “steps” component use.
- [ ] Ensure `"steps"` uses constant `1.0` in report + terminal score.
- [ ] Ensure any helper that forwards episode length for steps is updated to pass `1`.

### B) Diagnostics

- [ ] Collect `reward_terms` per episode in competition eval callback path.
- [ ] Add:
  - [ ] `comp_eval/<task>/success_rate`
  - [ ] `comp_eval/<task>/offside_rate`
  - [ ] `comp_eval/<task>/term_mean/<term>`
- [ ] Keep existing keys untouched.

### C) Macro bonus opt-in

- [ ] Add CLI flag `--allow-macro-bonus-with-competition` default false.
- [ ] Permit macro shaping only when flag enabled.
- [ ] Add info field(s) e.g. `info["macro_reward_bonus"]`.
- [ ] Confirm competition score remains unchanged.

---

## 5) Acceptance Checks You Must Satisfy

You must ensure these checks pass (and provide the commands):

### Required snippet (steps semantics)

The user spec expects a deterministic output from:

```bash
python -c "from training_scripts.competition_scoring import compute_competition_score; print(compute_competition_score({}, 'kick_to_target', 5000))"
```

### Regression expectations

- Training/eval works with no new flags.

- No log keys removed.

- No weight dictionaries changed.

- CLI help remains valid.

## 6) Output Format (What You Return)

When you respond, structure output like:

1. Direct result

- What changed and why (2–6 bullets)

2. Diff summary

- List files edited + purpose

3. Commands (exact order)

- Lint/tests/snippets/smoke eval commands to run

4. Notes / risks

- Any edge cases, expected behavior changes, compatibility notes

## 7) Notes on “Planning with Files” (You Must Follow)

Per 'SKILL.md' + 'reference.md':

- Use 'task_plan.md' as your persistent working memory.

- Re-read 'task_plan.md' before major decisions.

- Append error traces to task_plan.md instead of hiding them.

- Keep changes reversible and incremental.
