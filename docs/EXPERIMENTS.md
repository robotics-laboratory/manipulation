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
