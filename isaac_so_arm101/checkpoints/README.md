# Final RSL-RL checkpoints (curated)

Use this directory for **stable copies** of `model_*.pt` files so you do not depend on timestamped training runs under `logs/rsl_rl/`.

## Naming

Use one file per task, descriptive names, e.g.:

| File | Typical task |
|------|----------------|
| `so101_lift_cube.pt` | `Isaac-SO-ARM101-Lift-Cube-v0` (see below) |
| `so101_lift_cube_play.pt` | Trained on dense task, used with `*-Lift-Cube-Play-v0` for rollout |
| `so100_reach.pt` | `Isaac-SO-ARM100-Reach-v0` |

### `so101_lift_cube.pt` (first curated)

Copied from training run:

`logs/rsl_rl/lift/2026-04-11_09-42-39/model_1499.pt` → `checkpoints/so101_lift_cube.pt`.

After training, copy (or symlink) the best checkpoint:

```bash
cp isaac_so_arm101/logs/rsl_rl/<experiment>/<run>/model_1499.pt \
   isaac_so_arm101/checkpoints/so101_lift_cube.pt
```

## Play (host or container)

From project dir (`manipulation/` on host, `/workspace/isaac-bridge` in Docker):

```bash
isaaclab -p isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py \
  --task Isaac-SO-ARM101-Lift-Cube-Play-v0 \
  --checkpoint isaac_so_arm101/checkpoints/so101_lift_cube.pt
```

`isaaclab.sh` often sets the process cwd to the Isaac Lab repo, so relative paths would fail without help. **`resolve_checkpoint_cli_path`** (in `log_paths.py`) prepends **`PROJECT_DIR`** (`/workspace/isaac-bridge` in Docker; set in `docker-compose.yaml`) so the same relative path works.

If you still hit `FileNotFoundError`, use an absolute path inside the container:

`/workspace/isaac-bridge/isaac_so_arm101/checkpoints/so101_lift_cube.pt`

By default, `play.py` does **not** write JIT/ONNX under `exported/`. Pass **`--export_policy`** if you want `policy.pt` and `policy.onnx` next to the checkpoint.

Override directory with env **`ISAAC_SO_ARM101_FINAL_CHECKPOINTS_DIR`** (see `log_paths.py`).

Weight files (`*.pt`) are gitignored; only this README is tracked.
