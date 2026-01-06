# Optimized 2-Day Workflow Implementation Plan

## Repo Reality Check

## Existing components found

- **Environment & wrappers** — The repo registers all three soccer tasks via 'sai_mujoco' and builds environments in 'training_scripts/multitask_utils.py'. The 'make_env' factory applies 'CommandActionWrapper' for macro-triggered kicks, 'PreprocessObsWrapper' to append the task one-hot, and optionally 'RewardProfileWrapper' or 'CompetitionAlignmentWrapper' to handle shaped rewards or competition-aligned scoring. Wrapper order is asserted so the competition reward always overwrites the base reward.

- **Reward extraction & competition scoring** — Reward terms are collected from 'info["reward_terms"]' and combined with a step penalty. 'CompetitionAlignmentWrapper' computes the weighted sum of key terms (distance to ball, ball velocity, goal-scored, success, etc.) and subtracts a fixed step penalty. 'training_scripts/competition_scoring.py' defines the official weights and aggregation for competition scoring.

- **Imitation learning** — 'training_scripts/build_bc_dataset.py' loads motion capture '.npz' files, preprocesses joint and task-context features, injects sentinel commands for macro triggering, and outputs a command dataset. 'training_scripts/bc_train.py' trains a simple MLP policy on this dataset and saves a PyTorch state dict. There is no IQL implementation in this repo.

- **Multi-task training** — 'training_scripts/train_multitask.py' instantiates a vectorised environment using 'build_task_list' and runs the TD3 algorithm. It accepts '--task-weights', loads BC weights into the TD3 actor, evaluates via 'MultiTaskEvalCallback', logs per-task and competition metrics, and saves 'best_model_competition.zip' when the competition score improves. Single-task mode is supported via '--tasks'.

- **Macro playback** — Macros are triggered via sentinel commands inside 'CommandActionWrapper'. The wrapper plays pre-recorded kick sequences from '.npz' files and logs macro statistics, while 'RewardProfileWrapper' can grant a small reward bonus for good macro use.

- **Evaluation & scoring** — 'evaluate_policy' and 'evaluate_policy_competition' compute env-style scores and official competition scores for a given policy. 'evaluate_all_tasks_competition' aggregates per-task competition reports and produces a 'comp_total_sum' metric. 'eval_multitask.py' loads a TD3 model and prints env and competition metrics.

- **Export/conversion** — JAX conversion is hinted at in documentation, but no 'jax2torch.py' exists in this repo. The BC and RL training scripts produce PyTorch/Stable-Baselines checkpoints directly.

## Missing or inconsistent components

- **Algorithm choice** — 'train_multitask.py' hardcodes the TD3 algorithm. There is no support for PPO or SAC, which the optimized workflow prescribes for efficient on-policy learning and better stability. There is also no way to switch algorithms without editing the code.

- **Observation normalisation** — The training script does not wrap environments with 'VecNormalize' or any running normaliser. Without normalisation, high-dimensional observations produced by 'PreprocessObsWrapper' can destabilise learning.

- **Demonstration replay mixing** — While BC weights can initialise the actor , there is no option to pre-fill or mix demonstration transitions into the TD3 replay buffer. The optimized workflow suggests mixing off-policy BC data during RL fine-tuning to accelerate convergence.

- **Flexible evaluation seeds** — Evaluation seeds are specified manually via '--eval-seeds'; there is no default of multiple seeds. The optimized workflow calls for a diverse seed list to reduce overfitting.

- **Layer freezing** — There is no mechanism to freeze lower layers of the network after loading BC weights. Freezing initial layers during early RL training can preserve imitation knowledge.

- **PPO/SAC hyperparameter defaults** — Since only TD3 is implemented, there are no algorithm-specific hyperparameters or network architectures for PPO/SAC.

- **Observation scaling & reward normalisation** — Reward shaping and competition reward are computed, but there is no reward or return normaliser; the baseline has to handle widely-varying reward scales.

## Delta List

| Gap                                     | Where                                               | Fix strategy                                                                                                                                                     |
| --------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| No algorithm selection (TD3 only)       | 'training_scripts/train_multitask.py'               | Add a `--algo {td3,ppo,sac}` flag; instantiate the chosen Stable-Baselines3 algorithm with appropriate defaults. Keep TD3 as default for backward compatibility. |
| No observation normalisation            | 'training_scripts/train_multitask.py'; env creation | Add `--normalize-obs`; when set, wrap the vectorised env in `VecNormalize` (`norm_obs=True`, `norm_reward=False`). Save/load normalisation stats.                |
| No demonstration replay mixing          | 'training_scripts/train_multitask.py'               | Add `--bc-replay-dataset` and `--bc-replay-fraction`. If provided, load demos from `.npz` and pre-fill a fraction of the replay buffer before RL starts.         |
| No layer freezing after BC init         | 'training_scripts/train_multitask.py'               | Add `--freeze-bc-layers` and `--freeze-until-step`. Freeze first N policy layers after BC init, then unfreeze via a callback at the specified timestep.          |
| Lack of PPO/SAC hyperparameter defaults | 'training_scripts/train_multitask.py'               | Provide sensible defaults for PPO (e.g., `n_steps`, `gae_lambda`, `clip_range`) and SAC (e.g., `target_entropy`, `tau`) when those algorithms are selected.      |
| Limited evaluation seeds                | 'training_scripts/train_multitask.py'               | Change default `--eval-seeds` to multiple seeds (e.g., `0,1,2,3`) when not explicitly provided; document the eval-time tradeoff.                                 |
| No runbook automation                   | N/A                                                 | Provide a runbook: dataset prep, BC training, optional IQL, multi-task RL fine-tune with new options, evaluation, export.                                        |
| Observation/routine documentation       | N/A                                                 | Document new flags and the 2-day workflow in `quick_start_examples.md`; update CLI help strings.                                                                 |

## Codex Prompt Pack

### Prompt 1: Add algorithm selection to train_multitask.py

**Objective:**

Allow the user to choose between
TD3, PPO, or SAC for multi-task training so the workflow can use PPO/SAC for faster on-policy learning.

**Files to inspect/edit:**

'training_scripts/train_multitask.py'; verify that TD3 is currently hardcoded at lines 346-359.

**Plan:**

1. Introduce a new CLI flag '--algo' with choices 'td3', 'ppo', and 'sac' (default 'td3').

2. Based on the flag, instantiate the corresponding Stable-Baselines3 class ('TD3', 'PPO', or 'SAC') with sensible defaults:

- **TD3**: keep current parameters.

- **PPO**: set 'n_steps=2048', 'gamma=0.99', 'gae_lambda=0.95', 'ent_coef=0.01', 'learning_rate=args.lr', 'clip_range=0.2', 'batch_size=args.batch_size'.

- **SAC**: set 'buffer_size=args.buffer_size', 'batch_size=args.batch_size', 'learning_starts=args.learning_starts', 'learning_rate=args.lr', 'tau=args.tau', 'train_freq=1', 'gradient_steps=1'.

3. When using PPO or SAC, ignore action noise (since those algorithms handle exploration internally).

4. Continue to load BC weights into the actor if '--bc-weights' is provided and the selected algorithm has an actor (works for all three algorithms). For PPO, load weights into 'model.policy.mlp_extractor.policy_net' and 'model.policy.action_net'.

5. Ensure 'MultiTaskEvalCallback' still functions for all algorithms.

**Acceptance criteria:**

Running 'python training_scripts/train_multitask.py --algo ppo' should instantiate PPO without errors and begin training. The CLI help should display the new flag. Existing TD3 behaviour must remain unchanged when '--algo' is omitted.

**Commands to run:**

'python training_scripts/train_multitask.py --help' should list '--algo'.

A smoke test with a small number of timesteps (e.g., '--total-timesteps 1000 --n-envs 2 --algo ppo --eval-episodes 1 --eval-freq 500') should run without crashing.

**Risk notes:**

Changing the algorithm may break metrics logging if the policy has a different interface; test with each algorithm. Ensure that 'eval_multitask.py' still loads TD3 checkpoints; PPO/SAC checkpoints should also be loadable for evaluation. Maintain backward compatibility by defaulting to TD3.

**Output requirements:**

Show a unified diff of 'train_multitask.py' with the new flag, conditional model construction, and updated help string. Summarise the changes and report smoke-test results.

### Prompt 2: Implement observation normalisation option

**Objective:**

Provide an optional '--normalize-obs' flag to wrap the vectorised environment in 'VecNormalize', improving training stability by scaling observations.

**Files to inspect/edit:**

'training_scripts/train_multitask.py' (around the environment creation lines 318-327) and possibly 'eval_multitask.py' for loading normalisation stats.

**Plan:**

1. Add a boolean CLI flag '--normalize-obs' (default 'False') and an optional '--vecnormalize-path' to load/save normalisation statistics.

2. After constructing the 'VecMonitor', if '--normalize-obs' is true, wrap the environment with 'VecNormalize' ('from stable_baselines3.common.vec_env import VecNormalize') with 'norm_obs=True', 'norm_reward=False', 'clip_obs=10.0'.

3. When saving the final model, also save the normaliser stats to '<save_dir>/vecnormalize.pkl' via 'vec_env.save(vecnormalize_path)'. When resuming or evaluating, load stats from the provided path.

4. Update 'eval_multitask.py' to accept '--vecnormalize-path' and wrap the environment when evaluating.

5. Update CLI help strings and 'quick_start_examples.md' to explain how to use the option.

**Acceptance criteria:**

When running 'train_multitask.py --normalize-obs', the log should show that observations are normalised; training should proceed.

Running 'eval_multitask.py' with the saved 'vecnormalize.pkl' should reproduce normalised evaluation.The default behaviour (no normalisation) must not change.

** Commands to run:**

Train a small model with '--normalize-obs', verify that 'vecnormalize.pkl' is created, then evaluate using 'eval_multitask.py --vecnormalize-path'.

**Risk notes:**

Saving/loading normalisation stats must be conditioned on the flag to avoid breaking previous runs. Normalisation can change reward scales; keep 'norm_reward=False' to avoid altering competition rewards.

**Output requirements:**

Show diffs adding the flags and environment wrapping; summarise how saving/loading is handled.

### Prompt 3: Add demonstration replay pre-filling

**Objective:**

Accelerate RL fine-tuning by pre-filling a fraction of the replay buffer with BC demonstration transitions.

**Files to inspect/edit:**

'training_scripts/train_multitask.py' (after model creation and before training starts).

**Plan:**

1. Add CLI flags '--bc-replay-dataset' (path to the '.npz' file) and '--bc-replay-fraction' (float in '[0,1]', default '0.0').

2. If both flags are provided and the selected algorithm uses a replay buffer (TD3 or SAC), load the dataset using 'np.load' and compute 'num_demo = int(buffer_size\* fraction)'. Randomly select 'num_demo' rows from the dataset and add them to 'model.replay_buffer' via 'add(obs, next_obs, action, reward, done, truncated)'; use zeros for reward and done since we only want the behavioural mapping. For simplicity, set 'next_obs=obs' and 'done=False' so the algorithm treats them as single-step transitions.

3. Print how many transitions were inserted. Skip pre-fill if the algorithm is PPO (which has no replay buffer).

**Acceptance criteria:**

When running 'train_multitask.py --algo td3 --bc-replay-dataset path/to/bc_commands.npz --bc-replay-fraction 0.1', the script should load ~10 % of the replay buffer with demonstration pairs and continue training. The CLI help should show the new options. Running without these flags must behave as before.

**Commands to run:**

A small training run with '--bc-replay-fraction 0.1' should print the number of pre-filled transitions. For PPO, the script should warn and ignore the option.

**Risk notes:**

Pre-filling with demonstration transitions may bias the early gradient updates; we intentionally set zero rewards so they do not affect critic learning. However, if the dataset is large, loading it may slow start-up; document the expected cost. Ensure that 'np.load' uses 'allow_pickle=False' for safety.

**Output requirements:**

Show the diff adding flags and the pre-fill logic; summarise the number of transitions inserted during smoke test.

### Prompt 4: Implement BC layer freezing option

**Objective:**

Preserve imitation-learned representations during the first phase of RL training by freezing initial network layers and unfreezing them after a specified number of steps.

**Files to inspect/edit:**

'training_scripts/train_multitask.py' (after loading BC weights and before calling 'model.learn').

**Plan:**

1. Add CLI flags '--freeze-bc-layers' (integer, number of initial layers to freeze; default '0') and '--freeze-until-step' (integer, number of training timesteps to keep layers frozen; default '0').

2. After loading BC weights, if 'freeze_bc_layers > 0', traverse 'model.policy.actor.mu' (for TD3/SAC) or 'model.policy.mlp_extractor.policy_net' (for PPO) and set 'requires_grad = False' for the first 'freeze_bc_layers' 'nn.Linear' layers. Store a list of the frozen parameters.

3. Wrap the learning loop or callback to unfreeze these parameters when 'self.num_timesteps' exceeds 'freeze_until_step'. This can be done by creating a subclass of 'BaseCallback' that checks 'self.num_timesteps' in '\_on_step' and re-enables gradients by setting 'p.requires_grad=True' once. Register this callback alongside 'MultiTaskEvalCallback'.

4. Document in the help that freezing only works when BC weights are loaded.

**Acceptance criteria:**

Running 'train_multitask.py --bc-weights path/to/bc.pt --freeze-bc-layers 1 --freeze-until-step 50000' should freeze the first layer for the first 50k steps.

A debug print should show when layers are unfrozen.

Default behaviour (no freezing) must remain unchanged.

**Commands to run:**

Train a small model with the freeze options and inspect logs for unfreeze message.

**Risk notes:**

Freezing layers for too long may impede RL learning; emphasise that the user must set a reasonable 'freeze-until-step'.

Ensure compatibility with all algorithms; if the selected algorithm is PPO or SAC, adjust layer access accordingly.

Do not freeze critic networks.

**Output requirements:**

Provide diffs adding the flags, the freezing callback class, and the integration into the training loop. Summarise freeze/unfreeze behaviour.

### Prompt 5: Improve evaluation defaults and documentation

**Objective:**

Make evaluation more robust by using multiple default seeds and documenting new training options.

**Files to inspect/edit:**

'training_scripts/train_multitask.py' (default 'eval_seeds'), 'training_scripts/eval_multitask.py', and 'quick_start_examples.md'.

**Plan:**

1. Change the default value of '--eval-seeds' from '"0"' to '"0,1,2,3"' in 'train_multitask.py'. Update help text to reflect multiple seeds.

2. Add '--vecnormalize-path' option to 'eval_multitask.py' so that evaluation can load observation normalisation stats when present. Wrap evaluation environments with 'VecNormalize' if a path is
   provided.

3. Update 'quick_start_examples.md' to document the new '--algo', '--normalize-obs', '--vecnormalize-path', '--bc-replay-dataset', '--bc-replay-fraction', '--freeze-bc-layers', and '--freeze-until-step' flags and show example commands for the 2-day workflow.

**Acceptance criteria:**

Running 'train_multitask.py' without specifying '--eval-seeds' should evaluate on seeds '[0,1,2,3]'. 'eval_multitask.py --vecnormalize-path path/to/vecnormalize.pkl' should load normalisation stats.

Documentation in 'quick_start_examples.md' should clearly list and explain the new flags.

**Commands to run:**

Check '--help' outputs.

Generate 'quick_start_examples.md' diff.

Run evaluation with '--vecnormalize-path' to ensure no errors.

**Risk notes:**

Changing default eval seeds increases evaluation time; emphasise this in documentation and allow users to override.

Loading 'VecNormalize' must match the training normaliser to avoid shape errors.

**Output requirements:**

Provide diffs for the default seed change, 'eval_multitask.py' update, and documentation. Summarise the new instructions.

## Runbook (2-day workflow, repo-specific)

### Day 0.5 - setup & data preparation

1. _Clone repository and apply patches._ Run the Codex prompts in order to modify the codebase. Verify
   that 'train_multitask.py' now supports '--algo', '--normalize-obs', demonstration replay,
   and freezing. Check updated help messages.

2. _Create imitation datasets._ Place raw mocap '.npz' files in 'booster_dataset/soccer/booster_lower_t1'. Build the dataset with:

```bash
python training_scripts/build_bc_dataset.py --input-root booster_dataset/soccer/booster_lower_t1 \
--output booster_dataset/imitation_learning/bc_commands.npz --enable-sentinels \
--context-mode sentinel
```

Inspect the output; ensure that sentinel counts look reasonable.

3. _Split dataset._ Optionally split the dataset into training and validation sets using 'numpy'. For example:

```bash
import numpy as np
data = np.load('booster_dataset/imitation_learning/bc_commands.npz')
obs, acts = data['observations'], data['actions']
idx = np.random.permutation(len(obs))
train_idx, val_idx = idx[:int(0.8*len(obs))], idx[int(0.8*len(obs)):]
np.savez('bc_train.npz', observations=obs[train_idx], actions=acts[train_idx])
np.savez('bc_val.npz', observations=obs[val_idx], actions=acts[val_idx])
```

### Day 1 - BC training (and optional IQL)

1. _Behavioural cloning training._ Train a BC policy for each task:

```bash
python training_scripts/bc_train.py --dataset bc_train.npz --out training_runs/bc_actor.pt \
--net-arch 256,256 --epochs 40 --batch-size 1024 --lr 3e-4 --seed 0
```

Monitor the loss per epoch. Evaluate the BC actor on each task using a short script:

```bash
from stable_baselines3 import TD3
from training_scripts.multitask_utils import evaluate_policy, TASK_SPECS
import torch, numpy as np
payload = torch.load('training_runs/bc_actor.pt')
# reconstruct a simple policy using the saved state_dict and dimensions
class BCPolicy:
	def__init__(self, sd, obs_dim, action_dim, net_arch):
		import torch.nn as nn
		from stable_baselines3.common.torch_layers import create_mlp
		self.net= nn.Sequential(*create_mlp(obs_dim, action_dim, net_arch, nn.ReLU, True))
		self.net.load_state_dict(sd)
	def__call__(self, obs):
		with torch.no_grad():
			return self.net(torch.tensor(obs)).cpu().numpy()
acto = BCPolicy(payload['state_dict'], payload['obs_dim'], payload['action_dim'], payload['net_arch'])
score, _, _, _ = evaluate_policy(actor, TASK_SPECS[0], episodes=5, steps_weight=-0.05)
print('BC score:', score)
```

2. _Optional IQL._ If the repository adds an IQL implementation in the future, train it using 'imitation_learning/train.py'. Otherwise, proceed to RL fine-tuning.

3. _Aggregate BC weights._ Use the BC weight file as the '--bc-weights' input when running multi-task RL. The actor will be initialised with these weights.

### Day 1.5 - Multi-task RL fine-tuning

1. _Train with PPO or SAC._ Launch multi-task training using the new flags:

```bash
python training_scripts/train_multitask.py \
	--algo ppo \
	--total-timesteps 5000000 \
	--n-envs 32 \
	--task-weights goalie_penalty_kick=1,obstacle_penalty_kick=1,kick_to_target=1 \
	--bc-weights training_runs/bc_actor.pt\
	--freeze-bc-layers1 --freeze-until-step200000\
	--bc-replay-dataset booster_dataset/imitation_learning/bc_commands.npz \
	--bc-replay-fraction 0.1\
	--normalize-obs \
	--save-dir training_runs/multitask_ppo \
	--competition-only-eval --eval-freq 100000 --eval-episodes 4 --eval-seeds 0,1,2,3
```

For SAC, replace '--algo ppo' with '--algo sac' and adjust 'buffer-size' if needed. Training
should run for ~12 hours on a GPU, recording 'comp_eval/<task>' metrics and saving the best
model when 'C_overall' improves .

2. _Monitor progress._ Periodically check the logs or TensorBoard. Expect the competition score to climb into the 4-8 range after a few million steps. If certain tasks lag, you can fine-tune them individually:

```bash
python training_scripts/train_multitask.py --tasks kick_to_target --total-timesteps 1000000 \
	--resume-model training_runs/multitask_ppo/best_model_competition.zip --algoppo \
	--n-envs32 --save-dir training_runs/ktt_finetune
```

### Day 2 - Final tuning & evaluation

1. _Hyperparameter sweep._ If resources permit, run a few shorter experiments with different 'learning_rate', 'ent_coef' (for PPO), or 'tau' (for SAC). Use the competition score to select the best run.

2. _Final evaluation._ Evaluate the best checkpoint on a large number of episodes:

```bash
python training_scripts/eval_multitask.py --model training_runs/multitask_ppo/best_model_competition.zip \
	--episodes 50 --device cuda --vecnormalize-path training_runs/multitask_ppo/vecnormalize.pkl --deterministic
```

Confirm that 'comp_total_sum' is close to or exceeds 10.0.

3. _Export._ The competition submissions accept Stable-Baselines3 models. If a different format is
   required, convert the actor network using a custom script (not included here). Otherwise, submit
   'best_model_competition.zip' and the accompanying 'vecnormalize.pkl'.

### **Expected metrics:**

BC policies typically achieve terminal scores of 4-6 per task. After fine-tuning with PPO/SAC and macro-assisted exploration, the competition sum should rise to ≈10. Watch the 'comp_eval/C_overall' metric; improvements should be monotonic if the reward alignment is correct.

## Guardrails

- **Do not remove TD3 support.** Keep TD3 as the default algorithm to maintain backward compatibility. New flags must be optional.
- **Do not alter existing log keys or evaluation outputs.** The format of 'comp_eval/<task>' and 'C_overall' must remain unchanged so existing tooling can parse them.
- **Do not change reward weights.** Competition scoring weights are defined in 'competition_scoring.py'; modifications would misalign with the official evaluator .
- **Do not hard-code dataset paths.** Always expose file paths as CLI arguments and document them; avoid referencing hidden user directories.
- **Rollback strategy:** Before applying patches, create a Git commit or backup. To rollback, revert the commit or restore the backup. Each Codex prompt should generate diffs that can be individually reverted with 'git checkout <file>' if needed.
