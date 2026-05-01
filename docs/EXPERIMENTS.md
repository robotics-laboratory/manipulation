# Experiment log (manipulation / SO-ARM101)

Chronological notes on RL / transfer runs: what we tried, how it was configured, and what we learned.  
**Path:** `manipulation/docs/EXPERIMENTS.md` (versioned in git).

**References:** curated papers — [`KEY_PAPERS.md`](KEY_PAPERS.md); sparse reward deep dive — [`SPARSE_REWARD.md`](SPARSE_REWARD.md).

---

## How to add an entry

1. Copy the **template** below into a new `###` section under **Log** (newest on top).
2. Keep one block per **distinct run or config sweep**; link paths relative to repo root or `manipulation/`.
3. If a run failed fast, still log it with **Outcome** and **Insight** (even one line).

---

## Template (copy me)

```markdown
### YYYY-MM-DD — short title

- **Goal / hypothesis:** …
- **Setup:**
  - **Env / task:** e.g. `Isaac-SO-ARM101-…-v0`
  - **Image / container:** e.g. `manipulation/docker/...` + compose tag
  - **Code / config:** e.g. `isaac_so_arm101/.../guided_env_cfg.py`, PPO cfg, commit SHA
  - **Env vars:** e.g. `ISAAC_SO_ARM101_TRAJECTORY_DISCRIMINATOR_FILE=…`
  - **Seeds / CLI:** full train command if possible
- **What changed vs previous:** bullets
- **Metrics / artifacts:** TensorBoard dir, `output/…`, checkpoint path
- **Outcome:** success / partial / fail; qualitative behavior
- **Insights:** 1–3 bullets (what we’d do next or avoid)
```

---

## Behavioral cloning (BC) → PPO

Scripts (Isaac + Hydra, same pattern as `train.py`):

1. **Collect** `(actor_obs, action)` from the **same teacher checkpoint** you use for trajectories / discriminator data, with the **same** `--task`, `--agent`, and camera flags you want at RL time:
   - `python -m isaac_so_arm101.scripts.rsl_rl.collect_bc_dataset --task ... --checkpoint .../model.pt --num_episodes 500 [--disable_task_cameras]`
   - Saves `.pt` with `actor_obs`, `actions`, `transition_in_success_episode`, and `meta` (`obs_groups`, shapes, teacher path).
2. **Train BC** on the **downstream** task/agent (matching obs/action dims):  
   - `python -m isaac_so_arm101.scripts.rsl_rl.train_bc --task ... --dataset .../bc.pt [--disable_task_cameras] --bc_epochs 50`
   - Writes an **RSL-RL checkpoint** (`runner.save`) suitable for `train.py --resume True --checkpoint ...` (or your usual run directory resolution).
3. **PPO** from the BC checkpoint, then continue as usual.

Notes: uses teacher **`act_inference`** (mean actions) as BC targets; only **`ActorCritic`** / `OnPolicyRunner` is supported; enable `--include_failed_episode_transitions` on `train_bc` if you collected with `--keep_failed` and want all data.

---

## Backlog / ideas not tried

- **Shaping signal:** Replace hard floor-only saturation with bounded mapping or **center** `log D` (e.g. minus batch/EMA mean) so relative preference remains when D is confident.
- **Sketch-to-Skill-style stack:** Optional **IL+RL** action mix; consider **rollout / fake** negatives for D (not only shuffled teacher pairs). *(BC warm-start: see **Behavioral cloning (BC) → PPO** above.)*
- **Ablations:** Trajectory guidance on vs off; discriminator weight / `log_d_min` sweep; verify EE frame and Δp timestep match teacher collection.

---

## Log

*(Newest first.)*

### 2026-05-01 — Lift-cube anti-flip stability rewards

- **Goal / hypothesis:** Stop the learned lift policy from flipping the wrist/cube upward while preserving successful stable lifts.
- **Setup:**
  - **Env / task:** `LeIsaac-SO101-LiftCube-RewardDense-Train-v0` and matching reward-dense play/collect variants.
  - **Code / config:** `manipulation/scripts/leisaac/source/leisaac/leisaac/tasks/lift_cube/mdp/rewards.py`; `lift_cube_env_cfg.py`.
  - **Seeds / CLI:** Retrain fresh PPO; compare videos against the over-the-top wrist-lift behavior.
- **What changed vs previous:** Added `lifted_angular_stillness_dense` to reward low cube angular velocity after lift starts, and `wrist_flip_penalty` to penalize `abs(wrist_flex) > 1.05` rad once the cube is lifted. Enabled them as `lifted_angular_stillness` weight `4.0` and `wrist_flip` weight `-4.0`.
- **Metrics / artifacts:** Pending new run under `logs/rsl_rl/leisaac_lift_cube_reward_dense/...`; inspect `Episode_Reward/lifted_angular_stillness`, `Episode_Reward/wrist_flip`, lift success, and rollout videos.
- **Outcome:** Code ready; training/evaluation pending.
- **Insights:** The posture penalty is lift-gated so it discourages the exploit during lifting without constraining reach/grasp exploration too early.

### 2026-04-30 — PickOrange Direct Eureka entry point

- **Goal / hypothesis:** Make the existing direct PickOrange task usable with IsaacLabEureka for automated reward proposal and short PPO feedback loops.
- **Setup:**
  - **Env / task:** `LeIsaac-SO101-PickOrange-Eureka-Direct-v0`.
  - **Code / config:** `manipulation/scripts/leisaac/source/leisaac/leisaac/tasks/pick_orange/direct/pick_orange_env.py`; `pick_orange/__init__.py`; `pick_orange/agents/rsl_rl_ppo_cfg.py`; `pick_orange/eureka_task_cfg.py`; launcher `manipulation/scripts/leisaac/scripts/eureka/train_pick_orange.py`.
  - **Env vars:** requires `OPENAI_API_KEY` or Azure OpenAI env vars for IsaacLabEureka.
  - **Seeds / CLI:** `python manipulation/scripts/leisaac/scripts/eureka/train_pick_orange.py --max_eureka_iterations 1 --max_training_iterations 10`.
- **What changed vs previous:** Added a state-only direct config, RSL-RL runner config, oracle dense reward for Eureka reward-correlation logging, and `self._eureka_success_metric(env_ids)` task score.
- **Metrics / artifacts:** Expected under `logs/eureka/LeIsaac-SO101-PickOrange-Eureka-Direct-v0/...` and `logs/rl_runs/rsl_rl_eureka/leisaac_pick_orange_eureka_direct/...`.
- **Outcome:** Code ready; full Eureka run pending IsaacLabEureka installation/API key and simulator smoke test.
- **Insights:** The wrapper patches IsaacLabEureka's worker environment creation to import `leisaac`, avoiding edits to the external `isaaclab_eureka` package.

### 2026-04-30 — Lift-cube XY spawn stability reward

- **Goal / hypothesis:** Encourage human-like lift behavior where the cube is lifted near its spawn XY instead of dragged laterally before lifting.
- **Setup:**
  - **Env / task:** `LeIsaac-SO101-LiftCube-RewardDense-Train-v0` and matching reward-dense play/collect variants.
  - **Code / config:** `manipulation/scripts/leisaac/source/leisaac/leisaac/tasks/lift_cube/mdp/rewards.py`; `lift_cube_env_cfg.py`.
  - **Seeds / CLI:** Retrain fresh PPO with rate-limited action, lift-only objective, and post-lift stillness.
- **What changed vs previous:** Added `xy_position_stability_dense`, which snapshots cube reset-time XY per env and rewards `1 - tanh(distance_xy / 0.08)`; enabled as `xy_position_stability` with weight `3.0`.
- **Metrics / artifacts:** Pending new run under `logs/rsl_rl/leisaac_lift_cube_reward_dense/...`.
- **Outcome:** Code ready; training/evaluation pending.
- **Insights:** The reset-time snapshot follows object randomization, unlike static USD/default coordinates.

### 2026-04-29 — Lift-only observation/objective cleanup

- **Goal / hypothesis:** For stable lift-only dataset collection, remove target-conditioned signals that encourage lateral motion after lifting.
- **Setup:**
  - **Env / task:** `LeIsaac-SO101-LiftCube-RewardDense-Train-v0` and matching reward-dense play/collect variants.
  - **Code / config:** `manipulation/scripts/leisaac/source/leisaac/leisaac/tasks/lift_cube/lift_cube_env_cfg.py`.
  - **Seeds / CLI:** Retrain fresh PPO with rate-limited action and post-lift stillness.
- **What changed vs previous:** Removed `target_cube_position` from `LiftCubeMlpObservationsCfg.PolicyCfg`; set `goal_tracking`, `goal_tracking_fine`, and `goal_success_bonus` default weights to `0.0` so goal conditioning is opt-in via CLI.
- **Metrics / artifacts:** Pending new run under `logs/rsl_rl/leisaac_lift_cube_reward_dense/...`.
- **Outcome:** Code ready; training/evaluation pending.
- **Insights:** If the objective is lift + settle, random `object_pose` commands are nuisance variables unless goal rewards are intentionally enabled.

### 2026-04-29 — Lift-cube post-lift stabilization reward

- **Goal / hypothesis:** Encourage the policy to settle and hold the cube after lifting instead of ending episodes with continuing motion.
- **Setup:**
  - **Env / task:** `LeIsaac-SO101-LiftCube-RewardDense-Train-v0` and matching reward-dense play/collect variants.
  - **Code / config:** `manipulation/scripts/leisaac/source/leisaac/leisaac/tasks/lift_cube/lift_cube_env_cfg.py`; `mdp.lifted_stillness_dense`.
  - **Seeds / CLI:** Retrain fresh PPO after rate-limited action + goal-tracking changes.
- **What changed vs previous:** Added `lifted_stillness` reward (`lifted_height_delta=0.15`, `velocity_std=0.08`, weight `4.0`) and reduced height-only `success_bonus` from `20.0` to `10.0`.
- **Metrics / artifacts:** Pending new run under `logs/rsl_rl/leisaac_lift_cube_reward_dense/...`.
- **Outcome:** Code ready; training/evaluation pending.
- **Insights:** Stability is rewarded only after a meaningful lift, avoiding extra pressure during reach/grasp exploration.

### 2026-04-29 — Restore lift-cube goal tracking reward

- **Goal / hypothesis:** Make the smoothed lift-cube policy optimize commanded cube placement again instead of only lifting quickly.
- **Setup:**
  - **Env / task:** `LeIsaac-SO101-LiftCube-RewardDense-Train-v0` and matching reward-dense play/collect variants.
  - **Code / config:** `manipulation/scripts/leisaac/source/leisaac/leisaac/tasks/lift_cube/lift_cube_env_cfg.py`; existing reward funcs in `mdp/rewards.py`.
  - **Seeds / CLI:** Retrain fresh PPO after the rate-limited action experiment.
- **What changed vs previous:** Added `goal_tracking` (`std=0.30`, weight `8.0`), `goal_tracking_fine` (`std=0.05`, weight `3.0`), and `goal_success_bonus` (`position_tolerance=0.05`, weight `10.0`) using the `object_pose` command; terms are gated once cube lift exceeds `0.025` m from reset height.
- **Metrics / artifacts:** Pending new run under `logs/rsl_rl/leisaac_lift_cube_reward_dense/...`.
- **Outcome:** Code ready; training/evaluation pending.
- **Insights:** The policy observation already contained `target_cube_position`; without these terms the target was not part of the objective.

### 2026-04-29 — Lift-cube action target rate limiter

- **Goal / hypothesis:** Make the SO-101 lift-cube policy less reactive without changing simulated robot actuator physics.
- **Setup:**
  - **Env / task:** `LeIsaac-SO101-LiftCube-RewardDense-Train-v0` for training; matching reward-dense play/collect variants for evaluation/export.
  - **Code / config:** `manipulation/scripts/leisaac/source/leisaac/leisaac/tasks/lift_cube/mdp/actions.py`; `lift_cube_env_cfg.py`.
  - **Seeds / CLI:** Retrain fresh PPO; optional overrides: `env.actions.arm_action.max_delta=...`, `env.actions.arm_action.scale=...`.
- **What changed vs previous:** Arm action now uses `RateLimitedJointPositionActionCfg` with `scale=0.25` and `max_delta=0.025` rad/control-step; command target changes are clamped before applying joint-position targets.
- **Metrics / artifacts:** Pending new run under `logs/rsl_rl/leisaac_lift_cube_reward_dense/...`.
- **Outcome:** Code ready; training/evaluation pending.
- **Insights:** Prefer command-interface smoothing over actuator gain/velocity edits when the target is real-robot transfer.

### 2026-03-19 — Discriminator metrics “constant” while raw varies

- **Observed:** `log_d_raw` / `logit` vary (e.g. std ~10) but `log_d_shaped_std: 0` and `log_d_shaped_mean: -3`.
- **Cause:** Inner `log(eps)` floor maps ultra-negative `log_d` to one constant; then clamp `[-3,0]` wipes what little spread remained.
- **Fix:** Reward uses **`log_d_raw`**; default **`center_per_env_batch=True`**; symmetric clamp **`[-3, 3]`**; log **`Metrics/discriminator/log_d_centered_std`**.

### 2026-03-19 — PPO `std >= 0` crash (discriminator runs)

- **Symptom:** `RuntimeError: normal expects all elements of std >= 0.0` in `ActorCritic.act`.
- **Mitigations applied:** `noise_std_type="log"`; **ultra-conservative** PPO for discrim env (`lr=1e-5`, `clip_param=0.1`, `max_grad_norm=0.15`, `normalize_advantage_per_mini_batch=True`, `value_loss_coef=0.35`, 3 epochs); λ=`0.002`, shaped clamp **±1.5**; `train.py` **resume warning** + `_sanitize_rsl_rl_policy_std` after `runner.load`.
- **Note:** Prefer **fresh** runs when switching std parametrization; mid-training crashes = reduce LR / λ further or disable discrim term to confirm.

### 2026-03-19 — Log unclamped discriminator stats (TensorBoard `Metrics/discriminator/*`)

- **Goal:** Verify `log D` / logits vary across training (not artifact of clamp/centering).
- **Code:** `TrajectoryDiscriminator.log_d_scores`; `discriminator_guidance_reward` sets `_discriminator_metrics_pending`; `flush_discriminator_metrics_to_log` interval event merges into `extras['log']` after reset (`DiscriminatorDiagnosticsEventCfg` on sparse discriminator env).
- **Metrics:** `log_d_raw_*` (no eps floor), `logit_*` (pre-temperature), `log_d_shaped_*` (after center+outer clamp, used for reward).

### 2026-03-19 — Discriminator shaping: fix “always zero” (temperature + centering)

- **Goal / hypothesis:** After temperature+centering pass, **`Episode_Reward/discriminator_guidance` printed 0.0000** — unusable for monitoring and (**if** mean logged) misleading.
- **Cause:** **`logit_temperature=4`** drives all logits → **0**, hence **log D → log(0.5)** for every env (identical). **`log_d -= mean(log_d)`** then → **exactly 0** each step.
- **What changed vs previous:** Default **`logit_temperature=1.5`** (mild); **`center_per_env_batch=False`** by default; optional centering only if **`std(log_d) > center_min_std`**; clamp **`[-5, 0]`**, **`weight=0.02`** again.
- **Outcome:** Non-zero, **absolute** `log D` shaping (may still be **flat negative** if student OOD — distinct issue).
- **Insights:** Do not use **large T** with **batch centering**; see `discriminator_guidance_reward` docstring.

### 2026-03-19 — Discriminator shaping: temperature + batch-relative signal (superseded)

- **Note:** First attempt used **T=4** + centering → **zero** logged reward; see entry above for fix.

### 2026-03-19 — Discriminator `g`: episode-initial layout (train/serve alignment)

- **Goal / hypothesis:** Offline D was trained with `g` = **initial** object (+ goal) per teacher episode; RL previously passed **live** `root_pos_w`, causing OOD conditioning and flat `log D`.
- **Setup:** `manipulation/isaac_so_arm101/src/isaac_so_arm101/tasks/lift/mdp/rewards.py` — `_get_or_create_discriminator_state` + `discriminator_guidance_reward`.
- **What changed vs previous:** Per-env buffers `initial_object_pos_w`, `initial_goal_pos_w`; on the same **new-episode mask** used for `prev_ee_pos`, snapshot current `object_pos_w` and `goal_pos_w`; build `g` from these buffers (`object_only` / `object_goal`) instead of moving object pose. Migration path for old in-process state dicts missing keys.
- **Metrics / artifacts:** N/A — **re-run** sparse discriminator training and compare `Episode_Reward/discriminator_guidance` variance / task metrics vs pre-fix.
- **Outcome:** **Code landed** — behavioral validation pending.
- **Insights:** If metrics still flat, next suspects: clamp saturation, Δp/sim-step vs teacher sample rate, D negatives only.

### 2026-03-19 — Sparse discriminator run (~iter 1207): constant shaping, no sparse success

- **Goal / hypothesis:** Student RL on sparse lift with `log D` shaping toward teacher EE transitions; discriminator trained offline on teacher `.pt`.
- **Setup:**
  - **Env / task:** `Isaac-SO-ARM101-Guided-Lift-Cube-Sparse-Discriminator-v0` (inferred from guided + sparse discriminator stack).
  - **Code / config:** `manipulation/isaac_so_arm101/.../guided_env_cfg.py` (`GuidedDiscriminatorRewardsCfg`: trajectory terms weight 0; `discriminator_guidance` weight `0.02`, `log_d_min=-3`, `log_d_max=0`); `mdp/rewards.py` `discriminator_guidance_reward`; PPO via `GuidedDiscriminatorLiftCubePPORunnerCfg` (`rsl_rl_guided_ppo_cfg.py`).
  - **Env vars:** `ISAAC_SO_ARM101_TRAJECTORY_DISCRIMINATOR_FILE` → e.g. `.../trajectory_discriminator_lift_cube.pt` (Docker: `/workspace/isaac-bridge/output/...`).
  - **Discriminator train:** `scripts/train_trajectory_discriminator.py`; positives = aligned `(p, Δp)`; negatives = shuffled teacher transitions with **same `g`** batch as positives; `g` from **`initial_object_pos`** (and goal if `object_goal`).
- **What changed vs earlier attempts:** Stability: clamp on `log D`, lower λ, dedicated PPO hyperparameters for discriminator envs (lower LR, grad norm, init noise).
- **Metrics / artifacts:** Console ~iter **1207/1500**: `Mean reward ~ -0.30`, `Mean episode length 250`, `Episode_Termination/time_out 1.0`; episodic task rewards (`reaching_object`, `lifting_object`, etc.) **0**; `Episode_Reward/discriminator_guidance` **-0.0600** flat.
- **Outcome:** **Partial / fail** — training stable but **no task progress**; discriminator term **useless as signal** (saturated).
- **Insights:**
  - **-0.06 == 0.02 × (−3):** logged term is `weight × clamped log D`; stuck on **`log_d_min`** ⇒ raw `log D` almost always &lt; −3.
  - **No gradient of preference** when shaping is constant; sparse task alone insufficient in this regime.
  - **Bug (since fixed):** RL fed **`g = current object pose`**; D trained on **`g = initial reset layout`**. See newer log entry **“Discriminator `g`: episode-initial layout”**.
  - **vs Sketch-to-Skill:** Same broad idea (D-guided exploration) but they add **BC warm start** and **IL+RL mixing**; we also use **teacher-only** negatives — D may be **sharp** on policy OOD.

### 2026-03-19 — Guided discriminator PPO: stability and config import fixes

- **Goal / hypothesis:** Stop PPO crash (`normal expects all elements of std >= 0`) when using large negative `log D`; make discriminator runner cfg load in-container.
- **Setup:** Same stack as above; Isaac Lab `RewardManager` logs per-term `raw * weight` (not × `dt`).
- **What changed vs previous:**
  - `discriminator_guidance_reward`: **clamp** `log D` to `[log_d_min, log_d_max]`; config exposes limits; λ default **0.02**.
  - **`GuidedDiscriminatorLiftCubePPORunnerCfg`:** explicit `RslRlPpoActorCriticCfg` / `RslRlPpoAlgorithmCfg` blocks (`lr=5e-5`, `max_grad_norm=0.5`, `init_noise_std=0.8`) — **`LiftCubePPORunnerCfg.policy.replace(...)` removed** (`@configclass` has no class-level `.policy`).
  - `tasks/lift/__init__.py`: sparse discriminator gym ids point to **`GuidedDiscriminatorLiftCubePPORunnerCfg`**.
- **Metrics / artifacts:** N/A (engineering pass).
- **Outcome:** **Success** for crash/config; long-run **learning** still blocked by shaping/task issues (see log entry above).
- **Insights:** Clamp fixed scale but contributes to **flat** shaping when D is always below floor; need **aligned `g`** + softer or **relative** reward if D is to teach.
