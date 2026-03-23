# Sparse reward: papers & practical tricks

Deep dive on **sparse / binary rewards**: what the literature actually did, why it worked, and knobs people use. The broader bibliography lives in `[KEY_PAPERS.md](KEY_PAPERS.md)`.

**Path:** `manipulation/docs/SPARSE_REWARD.md`

---

## Theory & safe dense shaping


| Reference                                                                                                                                               | Why it matters for us                                                                                                                                                                     |
| ------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Policy invariance under reward transformations](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) (Ng, Harada & Russell, ICML 1999) | **Potential-based shaping (PBRS):** add a dense term $\Phi(s)$ so the *optimal policy stays the same* while gradients become easier—justifies “extra” potentials if they’re path-correct. |
| [Curriculum learning](https://dl.acm.org/doi/10.1145/1553374.1553380) (Bengio et al., ICML 2009)                                                        | Train on **easier distributions first**; classic motivation for staged tasks (reach → grasp → lift).                                                                                      |


*Action:* When our “Sparse vs Dense” story is documented in `[LIFT_CUBE_ENV_SO101.md](LIFT_CUBE_ENV_SO101.md)`, align naming of terms with the shaping literature you trust.

---

## Sparse reward: examples that *did* work (and *why*)

**Problem:** with **binary / rare** success, random exploration seldom sees reward → no gradient → PPO/DQN stalls (“sparse reward + long horizon” is especially bad in manipulation).


| Paper                                                                                                                                         | What was sparse                             | Why it worked (mechanism)                                                                                                                                                             |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) (Andrychowicz et al., 2017)                                                   | Goal-conditioned success                    | **Relabel failed trajectories** as successes for *alternative* goals in the replay buffer → every rollout gives supervised signal for *some* goal; off-policy replay scales it.       |
| [Reverse Curriculum Generation for RL](https://arxiv.org/abs/1707.05300) (Florensa et al., 2017)                                              | Goal reaching / manipulation assembly       | **Start near the goal**, widen initial-state distribution only when the policy succeeds—automatic **curriculum** without hand-shaped potentials; needs **reset to arbitrary states**. |
| [Go-Explore](https://arxiv.org/abs/1901.10995) (Ecoffet et al.; see also [Nature summary](https://www.nature.com/articles/s41586-020-2441-0)) | Hard exploration (e.g. Montezuma’s Revenge) | **Archive** promising states, **return** to them, then explore locally → solves “never visit success” exploration failure.                                                            |
| [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741) (Christiano et al., 2017)                                                  | No hand-coded reward                        | **Learn a reward model** from human comparisons on short clips → dense proxy for a sparse or ill-defined objective.                                                                   |


**Contrast (honest):** many **sim-to-real manipulation** papers that report high success use **dense** distance / contact / velocity terms (or large offline demos), not pure sparse binary reward from scratch—cite them as *engineering reality*, not as “sparse RL alone solved it.”

---

## Practical tricks (often combined)

These are standard **knobs** people use when sparse training is unstable; match them to what we actually enable in code (see `[LIFT_CUBE_ENV_SO101.md](LIFT_CUBE_ENV_SO101.md)` for our Dense/Sparse split).


| Trick                                         | Idea                                                                                                                                                                                                        |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **PBRS / distance shaping**                   | Potential based on distance to goal or object—keeps optimum if potentials are chosen correctly (Ng et al.).                                                                                                 |
| **Curriculum**                                | Easier goals, shorter distance, or lighter physics first (manual or automatic, e.g. reverse curriculum).                                                                                                    |
| **HER-style relabeling**                      | If the task is multi-goal or can be reinterpreted, mine failed episodes as positives for other goals.                                                                                                       |
| **Imitation warm-start**                      | BC or behavior cloning from a teacher / human, *then* RL with sparse success—reduces exploration burden (see `[EXPERIMENTS.md](EXPERIMENTS.md)` BC → PPO).                                                  |
| **Trajectory / expert alignment**             | Extra dense term that pulls policy toward teacher waypoints (our **trajectory_guidance** / related ideas).                                                                                                  |
| **Exploration bonuses**                       | Intrinsic curiosity or count-based bonuses when extrinsic reward is almost always zero ([Pathak et al.](KEY_PAPERS.md#exploration-bonuses-when-reward-is-almost-zero) in `[KEY_PAPERS.md](KEY_PAPERS.md)`). |
| **Reward / value normalization**              | Stabilize scales when rare +1 spikes dominate (e.g. [PopArt](https://arxiv.org/abs/1809.04474) for adaptive value normalization; verify what RSL-RL / our PPO config does).                                 |
| **Longer training + curriculum on penalties** | Slowly increase magnitude of action / velocity penalties so the agent first learns *success*, then *style* (similar spirit to our curriculum on `action_rate` / `joint_vel` in Dense).                      |
| **Reset distribution tricks**                 | If the sim allows, start closer to success or randomize only *after* first success (related to reverse curriculum).                                                                                         |


**Reality check:** “Sparse-only PPO from tabula rasa” on a **high-DOF arm + contact** often fails; papers that *look* sparse usually hide **one** of: dense shaping, demos, curriculum, or off-policy replay tricks.

---

## See also


| Doc                                                | Role                                                      |
| -------------------------------------------------- | --------------------------------------------------------- |
| `[KEY_PAPERS.md](KEY_PAPERS.md)`                   | Full paper index (HER row, exploration, imitation, VLAs). |
| `[LIFT_CUBE_ENV_SO101.md](LIFT_CUBE_ENV_SO101.md)` | Our Dense vs Sparse lift-cube definitions in code.        |
| `[EXPERIMENTS.md](EXPERIMENTS.md)`                 | What we actually ran (BC → PPO, guided tasks, …).         |


---

## Changelog


| Date       | Change                                         |
| ---------- | ---------------------------------------------- |
| 2026-03-19 | Split out from `KEY_PAPERS.md` into this file. |


