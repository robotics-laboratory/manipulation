# Fall-restore for LeRobot dataset collection

When collecting with `collect_lerobot_dataset.py`, you can enable **fall-restore** so the cube is respawned on the table after a drop instead of ending the episode via `object_dropping`.

## Enabling

- Pass `--enable-fall-restore` to the collector.
- The script removes the `object_dropping` termination and registers interval + reset events (`LiftEnvCfg.events.fall_restore_*`).
- MDP implementation: `isaac_so_arm101/tasks/lift/mdp/fall_restore_episode.py` (`FallRestoreRatioBand` / `FallRestoreRatioCap` are documentation aliases for the ratio logic in the collector).

## Ratio band + stall breaker

Optional filters on **fall_restores / episode_length** (in environment steps):

- `--min-fall-restore-fraction` (default `-1` = off): discard episodes below this ratio.
- `--max-fall-restore-fraction` (default `0.10`): discard episodes above this ratio (set `-1` to disable upper bound).
- `--fall-restore-stall-limit`: if many episodes in a row are discarded by the band, the next episode is **saved anyway** (avoids collecting zero episodes when the band is too tight).
- `--max-fall-restores-per-episode` (default `16`): after this many respawns, the MDP forces a time-out.
- `--keep-force-commit-episodes` (default off): keep episodes that hit `max-fall-restores-per-episode`. By default, strict mode discards them (still subject to stall breaker).

## Clean-success rule (non-restore mode)

When `--enable-fall-restore` is **not** used, collector now applies a strict clean-success check:

- lift is detected at `--clean-lift-threshold` (default `0.025`)
- if cube later falls to `--clean-drop-threshold` or below (default `0.020`) in the same episode,
  the successful episode is marked **dirty** and discarded

This rule is intentionally **disabled** in fall-restore mode, where drop/recovery is expected behavior.

## Target restore/clean mix

To keep the final saved dataset near **20% restore episodes / 80% clean episodes**:

- `--target-restore-episode-fraction` (default `0.20`)
- `--restore-fraction-tolerance` (default `0.05`)
- `--restore-quota-warmup-episodes` (default `20`)

Collector tags an episode as **restore** if it had at least one restore (`fall_n > 0`).
When projected restore share exceeds `target + tolerance`, additional restore episodes are discarded.

## Manual review mode

You can manually decide every completed episode:

- `--manual-review`: enable interactive accept/skip/quit prompt
- `--manual-review-camera-key` (default `observation.images.top`): which camera stream to render
- `--manual-review-dir`: where to write preview MP4s (default: temporary file per episode)
- `--manual-review-keep-mp4`: keep preview MP4s after decision (default: delete)
- `--manual-review-gui`: play preview directly in an OpenCV GUI window and use `r/c/s/q` keys (falls back to terminal prompt if GUI is unavailable)

Flow per episode:
1) collector renders preview MP4
2) prints auto suggestion (from current filters)
3) prints counters: `restore`, `clean`, `skipped`
4) prompts: `r` (save as restore) / `c` (save as clean) / `s` (skip) / `q` (quit)

## Simulation rate

Lift tasks use **30 Hz** control (`sim.dt = 1/60`, `decimation = 2`). Use `--fps 30` (default) for LeRobot metadata unless you intentionally decimate.
