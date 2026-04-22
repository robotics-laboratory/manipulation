#!/usr/bin/env python3
"""LeRobot train wrapper with holdout evaluation and teacher-action comparison.

This wrapper keeps the standard LeRobot offline training flow and adds:
1) deterministic episode-level holdout split
2) periodic holdout loss evaluation
3) optional action-agreement metrics against teacher actions in holdout data
"""

import copy
import dataclasses
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from termcolor import colored

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.scripts.lerobot_train import update_policy
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import format_big_number, init_logging


@dataclass
class HoldoutTrainPipelineConfig(TrainPipelineConfig):
    holdout_ratio: float = 0.1
    holdout_min_episodes: int = 1
    holdout_seed: int = 1000
    holdout_max_batches: int = 0
    teacher_compare: bool = True

    def validate(self) -> None:
        super().validate()
        if self.holdout_ratio < 0.0 or self.holdout_ratio >= 1.0:
            raise ValueError("holdout_ratio must be in [0.0, 1.0).")
        if self.holdout_min_episodes < 0:
            raise ValueError("holdout_min_episodes must be >= 0.")
        if self.holdout_max_batches < 0:
            raise ValueError("holdout_max_batches must be >= 0.")


def _compute_episode_split(cfg: HoldoutTrainPipelineConfig) -> tuple[list[int], list[int], int]:
    meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        revision=cfg.dataset.revision,
    )
    all_episodes = list(range(meta.total_episodes))
    selected_episodes = list(cfg.dataset.episodes) if cfg.dataset.episodes is not None else all_episodes
    if len(selected_episodes) < 2:
        raise ValueError(
            f"Need at least 2 episodes for train/holdout split, got {len(selected_episodes)} "
            "(after applying dataset.episodes filter)."
        )

    desired = int(round(len(selected_episodes) * cfg.holdout_ratio))
    holdout_count = max(cfg.holdout_min_episodes, desired)
    holdout_count = min(holdout_count, len(selected_episodes) - 1)

    rng = random.Random(cfg.holdout_seed)
    shuffled = selected_episodes.copy()
    rng.shuffle(shuffled)
    holdout_ids = sorted(shuffled[:holdout_count])
    train_ids = sorted(shuffled[holdout_count:])
    return train_ids, holdout_ids, len(selected_episodes)


def _write_holdout_metadata(
    cfg: HoldoutTrainPipelineConfig,
    selected_episode_count: int,
    train_ids: list[int],
    holdout_ids: list[int],
) -> None:
    payload = {
        "dataset_repo_id": cfg.dataset.repo_id,
        "dataset_root": cfg.dataset.root,
        "dataset_revision": cfg.dataset.revision,
        "selected_episode_count": selected_episode_count,
        "train_episode_count": len(train_ids),
        "holdout_episode_count": len(holdout_ids),
        "holdout_ratio": cfg.holdout_ratio,
        "holdout_min_episodes": cfg.holdout_min_episodes,
        "holdout_seed": cfg.holdout_seed,
        "holdout_max_batches": cfg.holdout_max_batches,
        "teacher_compare": cfg.teacher_compare,
        "train_episode_ids": train_ids,
        "holdout_episode_ids": holdout_ids,
    }
    out_path = Path(cfg.output_dir) / "holdout_episode_ids.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _split_train_holdout_cfg(
    cfg: HoldoutTrainPipelineConfig,
    train_ids: list[int],
    holdout_ids: list[int],
) -> tuple[HoldoutTrainPipelineConfig, HoldoutTrainPipelineConfig]:
    train_cfg = copy.deepcopy(cfg)
    holdout_cfg = copy.deepcopy(cfg)
    train_cfg.dataset.episodes = train_ids
    holdout_cfg.dataset.episodes = holdout_ids
    return train_cfg, holdout_cfg


def _batch_to_inference_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in batch.items() if k != ACTION}


def _extract_target_action(batch: dict[str, Any]) -> torch.Tensor | None:
    target = batch.get(ACTION)
    if not isinstance(target, torch.Tensor):
        return None
    if target.ndim == 3:
        return target[:, 0, :]
    if target.ndim == 2:
        return target
    return None


def _extract_pred_action(
    unwrapped_policy: Any,
    inference_batch: dict[str, Any],
) -> torch.Tensor | None:
    if hasattr(unwrapped_policy, "predict_action_chunk"):
        try:
            pred = unwrapped_policy.predict_action_chunk(inference_batch)
            if isinstance(pred, torch.Tensor):
                if pred.ndim == 3:
                    return pred[:, 0, :]
                if pred.ndim == 2:
                    return pred
        finally:
            if hasattr(unwrapped_policy, "reset"):
                unwrapped_policy.reset()

    if hasattr(unwrapped_policy, "select_action"):
        try:
            pred = unwrapped_policy.select_action(inference_batch)
            if isinstance(pred, torch.Tensor) and pred.ndim == 2:
                return pred
        finally:
            if hasattr(unwrapped_policy, "reset"):
                unwrapped_policy.reset()
    return None


def evaluate_holdout(
    cfg: HoldoutTrainPipelineConfig,
    policy: torch.nn.Module,
    accelerator: Accelerator,
    holdout_loader: torch.utils.data.DataLoader,
    preprocessor: Any,
) -> dict[str, float]:
    policy.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_cosine = 0.0
    count = 0
    compared = 0

    unwrapped_policy = accelerator.unwrap_model(policy)
    with torch.no_grad():
        for batch_idx, batch in enumerate(holdout_loader):
            if cfg.holdout_max_batches > 0 and batch_idx >= cfg.holdout_max_batches:
                break
            batch = preprocessor(dict(batch))
            with accelerator.autocast():
                loss, _ = policy.forward(batch)
                loss_val = loss.mean() if isinstance(loss, torch.Tensor) else torch.tensor(loss, device=accelerator.device)
            total_loss += float(loss_val.item())
            count += 1

            if cfg.teacher_compare:
                target = _extract_target_action(batch)
                if target is None:
                    continue
                pred = _extract_pred_action(unwrapped_policy, _batch_to_inference_inputs(batch))
                if pred is None:
                    continue
                dim = min(pred.shape[-1], target.shape[-1])
                pred = pred[..., :dim]
                target = target[..., :dim]
                total_mse += float(F.mse_loss(pred, target).item())
                total_mae += float(F.l1_loss(pred, target).item())
                total_cosine += float(F.cosine_similarity(pred, target, dim=-1).mean().item())
                compared += 1

    totals = torch.tensor(
        [total_loss, float(count), total_mse, total_mae, total_cosine, float(compared)],
        device=accelerator.device,
        dtype=torch.float64,
    )
    totals = accelerator.reduce(totals, reduction="sum")

    out: dict[str, float] = {}
    global_count = int(totals[1].item())
    global_compared = int(totals[5].item())
    if global_count > 0:
        out["holdout/loss"] = float((totals[0] / totals[1]).item())
    if global_compared > 0:
        out["holdout/action_mse"] = float((totals[2] / totals[5]).item())
        out["holdout/action_mae"] = float((totals[3] / totals[5]).item())
        out["holdout/action_cosine"] = float((totals[4] / totals[5]).item())
    return out


@parser.wrap()
def train(cfg: HoldoutTrainPipelineConfig, accelerator: Accelerator | None = None):
    cfg.validate()

    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        force_cpu = cfg.policy.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    init_logging(accelerator=accelerator)
    is_main_process = accelerator.is_main_process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        from lerobot.rl.wandb_utils import WandBLogger

        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    train_ids, holdout_ids, selected_episode_count = _compute_episode_split(cfg)
    train_cfg, holdout_cfg = _split_train_holdout_cfg(cfg, train_ids, holdout_ids)

    if is_main_process:
        _write_holdout_metadata(cfg, selected_episode_count, train_ids, holdout_ids)
        logging.info(
            "Episode split: selected=%d train=%d holdout=%d (ratio=%.3f, seed=%d)",
            selected_episode_count,
            len(train_ids),
            len(holdout_ids),
            cfg.holdout_ratio,
            cfg.holdout_seed,
        )

    if is_main_process:
        logging.info("Creating training dataset")
        dataset = make_dataset(train_cfg)
        logging.info("Creating holdout dataset")
        holdout_dataset = make_dataset(holdout_cfg)
    accelerator.wait_for_everyone()
    if not is_main_process:
        dataset = make_dataset(train_cfg)
        holdout_dataset = make_dataset(holdout_cfg)

    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("Creating policy")
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)

    if cfg.peft is not None:
        logging.info("Using PEFT! Wrapping model.")
        peft_cli_overrides = dataclasses.asdict(cfg.peft)
        policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    accelerator.wait_for_everyone()

    processor_kwargs: dict[str, Any] = {}
    postprocessor_kwargs: dict[str, Any] = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.policy.type == "sarm":
        processor_kwargs["dataset_meta"] = dataset.meta

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            }
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    rabc_weights = None
    if cfg.use_rabc:
        from lerobot.utils.rabc import RABCWeights

        chunk_size = getattr(policy.config, "chunk_size", None)
        if chunk_size is None:
            raise ValueError("Chunk size is not found in policy config")
        head_mode = getattr(cfg, "rabc_head_mode", "sparse")
        logging.info(f"Loading SARM progress for RA-BC from {cfg.rabc_progress_path}")
        logging.info(f"Using chunk_size={chunk_size} from policy config, head_mode={head_mode}")
        rabc_weights = RABCWeights(
            progress_path=cfg.rabc_progress_path,
            chunk_size=chunk_size,
            head_mode=head_mode,
            kappa=getattr(cfg, "rabc_kappa", 0.01),
            epsilon=getattr(cfg, "rabc_epsilon", 1e-6),
            device=device,
        )

    step = 0
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)
        else:
            env_preprocessor, env_postprocessor = None, None
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        logging.info(f"{holdout_dataset.num_frames=} ({format_big_number(holdout_dataset.num_frames)})")
        logging.info(f"{holdout_dataset.num_episodes=}")
        effective_bs = cfg.batch_size * accelerator.num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {accelerator.num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")
    else:
        env_preprocessor, env_postprocessor = None, None

    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    holdout_loader = torch.utils.data.DataLoader(
        holdout_dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(policy, optimizer, dataloader, lr_scheduler)
    holdout_loader = accelerator.prepare(holdout_loader)
    dl_iter = cycle(dataloader)

    policy.train()
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            rabc_weights_provider=rabc_weights,
        )

        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                if rabc_weights is not None:
                    rabc_stats = rabc_weights.get_stats()
                    wandb_log_dict.update(
                        {
                            "rabc_delta_mean": rabc_stats["delta_mean"],
                            "rabc_delta_std": rabc_stats["delta_std"],
                            "rabc_num_frames": rabc_stats["num_frames"],
                        }
                    )
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)
            accelerator.wait_for_everyone()

        if is_eval_step:
            holdout_metrics = evaluate_holdout(cfg, policy, accelerator, holdout_loader, preprocessor)
            if is_main_process and holdout_metrics:
                logging.info("Holdout metrics at step %d: %s", step, holdout_metrics)
                if wandb_logger:
                    wandb_logger.log_dict(holdout_metrics, step, mode="eval")

        if cfg.env and is_eval_step and is_main_process:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with torch.no_grad(), accelerator.autocast():
                eval_info = eval_policy_all(
                    envs=eval_env,
                    policy=accelerator.unwrap_model(policy),
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    n_episodes=cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                    max_parallel_tasks=cfg.env.max_parallel_tasks,
                )
            aggregated = eval_info["overall"]
            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size,
                dataset.num_frames,
                dataset.num_episodes,
                eval_metrics,
                initial_step=step,
                accelerator=accelerator,
            )
            eval_tracker.eval_s = aggregated.pop("eval_s")
            eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
            eval_tracker.pc_success = aggregated.pop("pc_success")
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")
        accelerator.wait_for_everyone()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")
        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            if cfg.policy.use_peft:
                unwrapped_policy.push_model_to_hub(cfg, peft_model=unwrapped_policy)
            else:
                unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
