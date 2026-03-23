# Key papers & reports (manipulation / RL / VLAs)

Curated references to **ground design choices** beyond pure intuition. This is not a survey—add papers as you actually read them and note what you took away in [`EXPERIMENTS.md`](EXPERIMENTS.md).

**Path:** `manipulation/docs/KEY_PAPERS.md`

---

## How to use

1. When you try an approach (e.g. sparse-only PPO, trajectory shaping, BC warm-start), **link the closest paper** here and log the run in `EXPERIMENTS.md`.
2. Prefer **primary sources** (arXiv / conference PDF / official blog) over second-hand summaries.
3. Keep one line **“Why it matters for us”** when you add an entry—otherwise the list becomes noise.

---

## On-policy RL & stability (baselines)

| Reference | Why it matters for us |
| --------- | --------------------- |
| [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017) | Default family for `rsl_rl` / PPO in Isaac Lab; baseline when comparing sparse vs shaped rewards. |
| [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) (Haarnoja et al., 2018) | Alternative when exploration / entropy matters under sparse signal (not always wired in our stack). |

---

## Sparse rewards, shaping & curriculum

**Deep dive:** [`SPARSE_REWARD.md`](SPARSE_REWARD.md) — PBRS & curriculum (theory), **examples that worked** (HER, reverse curriculum, Go-Explore, preference RL), **practical tricks** table, reality check on manipulation papers.

---

## Hindsight & relabeling (sparse goals)

| Reference | Why it matters for us |
| --------- | --------------------- |
| [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) (Andrychowicz et al., NIPS 2017) | Classic way to learn from failures when the reward is sparse in **goal** space; compare to our trajectory matching / teacher alignment. |

---

## Imitation, offline data & warm-start

| Reference | Why it matters for us |
| --------- | --------------------- |
| [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476) (Ho & Ermon, NIPS 2016) | Discriminator-style signal (“is this behavior like the expert?”)—conceptual neighbor to **trajectory / discriminator guidance** in our guided configs. |
| [A Reduction of Imitation Learning to No-Regret Online Learning](https://arxiv.org/abs/1011.0686) (Ross, Gordon & Bagnell, 2011) — **DAgger** | Iterative correction when BC drifts; relevant if we mix BC and on-policy rollouts. |
| Conservative Q-learning / offline RL line | If we ever train value-based policies from logged data only—cite **CQL** (Kumar et al.) or a follow-up you use. |

*Action:* Our BC pipeline is described in [`EXPERIMENTS.md`](EXPERIMENTS.md) (`collect_bc_dataset`, `train_bc`).

---

## Human preferences, corrections & “RLHF-adjacent” robotics

| Reference | Why it matters for us |
| --------- | --------------------- |
| [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) (Christiano et al., NeurIPS 2017) | Canonical **preference-based** reward modeling (often what people mean by “RLHF” in RL). |
| [π\*₀.₆: a VLA That Learns From Experience](https://arxiv.org/abs/2511.14759) (Physical Intelligence, 2025) — **RECAP** | Post-training with **experience + human corrections**; *not* identical to classic RLHF, but addresses “how do strong policies improve after BC?” |

Use precise wording: **RLHF** usually implies a **learned reward model from comparisons**; **coaching / interventions / RECAP** are related but not the same recipe.

---

## Vision–language–action (VLA) models & manipulation

| Reference | Why it matters for us |
| --------- | --------------------- |
| [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246) (Kim et al., 2024) | Open baseline for **VLA + tokenized actions**; good reference for multi-embodiment imitation. |
| [SmolVLA](https://arxiv.org/abs/2506.01844) (Cadene et al., 2025) | **Small** VLA aligned with our SmolVLA inference path in `manipulation/` (efficiency, flow-matching action head). |
| RT-1 / RT-2 line (Brohan et al.) | Large-scale robot transformers from Google DeepMind—cite the exact paper version if we compare scaling or language conditioning. |

---

## Exploration bonuses (when reward is almost zero)

| Reference | Why it matters for us |
| --------- | --------------------- |
| [Curiosity-driven exploration by self-supervised prediction](https://arxiv.org/abs/1705.05363) (Pathak et al., ICML 2017) | Intrinsic motivation when **extrinsic** reward is rare. |
| [Unifying Count-Based Exploration](https://arxiv.org/abs/1611.04717) (Bellemare et al., 2016) | Density / pseudo-count exploration—optional add-on for hard exploration. |

---

## Mapping to this repo (quick)

| Topic | Where in repo |
| ----- | ------------- |
| Dense vs sparse lift-cube | [`LIFT_CUBE_ENV_SO101.md`](LIFT_CUBE_ENV_SO101.md), [`SPARSE_REWARD.md`](SPARSE_REWARD.md), `lift_env_cfg.py` |
| Trajectory guidance / discriminator rewards | `guided_env_cfg.py`, `mdp/rewards.py` |
| BC → PPO | [`EXPERIMENTS.md`](EXPERIMENTS.md) |
| VLA inference | `scripts/run_smolvla_isaac.py`, `manipulation/README.md` |

---

## Changelog

| Date | Change |
| ---- | ------ |
| 2026-03-19 | Initial list; sparse deep dive moved to [`SPARSE_REWARD.md`](SPARSE_REWARD.md). |
