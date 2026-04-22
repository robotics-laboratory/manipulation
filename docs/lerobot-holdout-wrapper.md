# LeRobot Holdout Wrapper

`manipulation/scripts/lerobot_train_wrapper.py` extends `lerobot_train` with:

- deterministic train/holdout split by episode
- periodic holdout loss evaluation
- optional action comparison against teacher labels from dataset `action`

The wrapper is intended for your SO-101 dataset where student and teacher can use different observations.

## Run (inside container)

```bash
python3 scripts/lerobot_train_wrapper.py \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id=igor-saprygin/so101-lift-cube-smolvla \
  --policy.push_to_hub=true \
  --dataset.repo_id=igor-saprygin/so101-lift-cube \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2", "observation.images.side": "observation.images.camera3"}' \
  --steps=20000 \
  --eval_freq=2000 \
  --holdout_ratio=0.1 \
  --holdout_min_episodes=1 \
  --holdout_seed=1000 \
  --teacher_compare=true \
  --batch_size=32 \
  --output_dir=output/train/so101-lift-cube-smolvla-holdout \
  --job_name=so101-lift-cube-smolvla-holdout \
  --policy.device=cuda \
  --wandb.enable=false
```

## Wrapper arguments

- `--holdout_ratio`: fraction of selected episodes used for holdout (`0.1` default)
- `--holdout_min_episodes`: minimum holdout episodes (`1` default)
- `--holdout_seed`: deterministic split seed (`1000` default)
- `--holdout_max_batches`: cap holdout eval batches per eval pass (`0` = full holdout)
- `--teacher_compare`: compute action MSE/MAE/cosine from holdout batches (`true` default)

## Logged metrics

At each `eval_freq` step the wrapper logs:

- `holdout/loss`
- `holdout/action_mse`
- `holdout/action_mae`
- `holdout/action_cosine`

Teacher metrics are skipped if policy action prediction cannot be extracted for the active policy type.

## Reproducibility artifact

Each run writes split metadata to:

- `output_dir/holdout_episode_ids.json`

This file stores selected/train/holdout episode IDs and split/eval settings.
