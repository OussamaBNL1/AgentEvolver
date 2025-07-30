#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
advantage.py – offline script to recompute (and dump) advantage tensors for a
trained BeyondAgent *actor* checkpoint.

The previous crash came from passing an obsolete `enable_log` keyword into
`BeyondAgentRayPPOTrainer`.  This version removes that argument entirely.

Run example
-----------
```bash
cd /mnt/data/taoshuchang.tsc/beyondagent/BeyondAgent
python beyondagent/advantage.py \
  --config-path config \
  --config-name beyond_agent_dataflow \
  --ckpt /…/global_step_20/actor \
  --train-files /…/train.parquet \
  --val-files   /…/dev.parquet \
  --micro-batch-gpu 1 \
  --outdir advantage_dump \
  --num-batches 10
```
The script writes `adv_before.pt` & `adv_after.pt` under *outdir*.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping

import hydra
import torch
from hydra.core.global_hydra import GlobalHydra

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _init_hydra(cfg_path: str | os.PathLike[str], version_base: str = "1.3") -> None:
    """Allow both relative *directories* and absolute paths when initialising
    Hydra (avoids the earlier config_path error)."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    p = Path(cfg_path)
    if p.is_absolute():
        hydra.initialize_config_dir(version_base=version_base, config_dir=str(p))
    else:
        hydra.initialize(version_base=version_base, config_path=str(p))


def _expanduser(cfg: Any) -> None:
    if isinstance(cfg, Mapping):
        for k, v in cfg.items():
            if isinstance(v, str) and "~" in v:
                cfg[k] = os.path.expanduser(v)
            else:
                _expanduser(v)
    elif isinstance(cfg, list):
        for i in range(len(cfg)):
            _expanduser(cfg[i])

# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AdvantageAnalyzer:
    def __init__(
        self,
        cfg_dir: str | os.PathLike[str],
        cfg_name: str,
        ckpt_dir: str | os.PathLike[str],
        train_files: str | None = None,
        val_files: str | None = None,
        micro_batch_gpu: int | None = None,
        outdir: str | os.PathLike[str] = "advantage_dump",
    ) -> None:
        _init_hydra(cfg_dir)
        self.cfg = hydra.compose(config_name=cfg_name)

        # Optional CLI overrides
        if train_files:
            self.cfg.data.train_files = os.path.expanduser(train_files)
        if val_files:
            self.cfg.data.val_files = os.path.expanduser(val_files)
        self.cfg.actor_rollout_ref.model.path = str(ckpt_dir)

        actor_cfg = self.cfg.actor_rollout_ref.actor
        if not actor_cfg.get("ppo_micro_batch_size") and not actor_cfg.get(
            "ppo_micro_batch_size_per_gpu"):
            actor_cfg.ppo_micro_batch_size_per_gpu = micro_batch_gpu or 1
        if not actor_cfg.get("ppo_mini_batch_size"):
            actor_cfg.ppo_mini_batch_size = 8

        _expanduser(self.cfg)

        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.trainer = None  # lazy init

    # ------------------------------------------------------------------
    def _build_trainer(self):
        """Construct a minimal BeyondAgentRayPPOTrainer (no RL loops)."""
        from verl.single_controller.ray import RayWorkerGroup  # type: ignore
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn  # type: ignore
        from verl.utils import hf_tokenizer
        from verl.utils.fs import copy_to_local
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role  # type: ignore
        from verl.workers.fsdp_workers import (  # type: ignore
            ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker)
        import ray  # delayed import keeps log noise minimal
        from beyondagent.module.trainer.ba_ray_trainer import BeyondAgentRayPPOTrainer

        ar_cls = (
            AsyncActorRolloutRefWorker
            if self.cfg.actor_rollout_ref.rollout.mode == "async"
            else ActorRolloutRefWorker
        )
        role_mapping = {Role.ActorRollout: ray.remote(ar_cls), Role.Critic: ray.remote(CriticWorker)}

        pool = "pool"
        spec = {pool: [self.cfg.trainer.n_gpus_per_node] * self.cfg.trainer.nnodes}
        rpm = ResourcePoolManager(spec, {Role.ActorRollout: pool, Role.Critic: pool})

        local_model = copy_to_local(self.cfg.actor_rollout_ref.model.path)
        tok = hf_tokenizer(local_model, trust_remote_code=self.cfg.data.get("trust_remote_code", False))

        train_ds = RLHFDataset(self.cfg.data.train_files, tok, None, self.cfg.data)
        val_ds = RLHFDataset(self.cfg.data.val_files, tok, None, self.cfg.data)

        self.trainer = BeyondAgentRayPPOTrainer(
            config=self.cfg,
            tokenizer=tok,
            processor=None,
            role_worker_mapping=role_mapping,
            resource_pool_manager=rpm,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=None,
            val_reward_fn=None,
            train_dataset=train_ds,
            val_dataset=val_ds,
            collate_fn=collate_fn,
            train_sampler=None,
            device_name=self.cfg.trainer.device,
        )

    # ------------------------------------------------------------------
    def run(self, num_batches: int = 10):
        if self.trainer is None:
            print("[INFO] building trainer …")
            self._build_trainer()

        from verl.trainer.ppo.core_algos import compute_advantage  # type: ignore

        dl = self.trainer.get_train_dataloader()
        raw, norm = [], []
        for idx, batch in enumerate(dl):
            if num_batches > 0 and idx >= num_batches:
                break
            r, v = batch["rewards"].float(), batch["values"].float()
            adv = compute_advantage(r, v, gamma=1.0, gae_lambda=0.95)
            raw.append(adv.cpu())
            norm.append(((adv - adv.mean()) / (adv.std() + 1e-8)).cpu())

        torch.save(torch.cat(raw), self.outdir / "adv_before.pt")
        torch.save(torch.cat(norm), self.outdir / "adv_after.pt")
        print(f"[✓] saved to {self.outdir}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser("Advantage recompute util")
    p.add_argument("--config-path", required=True)
    p.add_argument("--config-name", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--train-files")
    p.add_argument("--val-files")
    p.add_argument("--micro-batch-gpu", type=int)
    p.add_argument("--outdir", default="advantage_dump")
    p.add_argument("--num-batches", type=int, default=10)
    args = p.parse_args()

    AdvantageAnalyzer(
        cfg_dir=args.config_path,
        cfg_name=args.config_name,
        ckpt_dir=args.ckpt,
        train_files=args.train_files,
        val_files=args.val_files,
        micro_batch_gpu=args.micro_batch_gpu,
        outdir=args.outdir,
    ).run(num_batches=args.num_batches)

if __name__ == "__main__":
    main()

    """
python beyondagent/advantage.py \
  --config-path ../config \
  --config-name beyond_agent_dataflow \
  --ckpt /mnt/data/taoshuchang.tsc/beyondagent/BeyondAgent/checkpoints/beyondagent/qwen3_14b_sparse_baseline_trbs8_ppobs8/global_step_20/actor \
  --train-files /mnt/data_aisys_cpfs/zouanni.zan/data/appworld_parquet/train.parquet \
  --val-files   /mnt/data_aisys_cpfs/zouanni.zan/data/appworld_parquet/dev.parquet \
  --micro-batch-gpu 1 \
  --outdir advantage_dump \
  --num-batches 10
  
  
  python beyondagent/advantage.py \
  --config-path ../config \
  --config-name beyond_agent_dataflow \
  --ckpt /mnt/data_aisys_cpfs/xielipeng.xlp/models/Qwen3-14B \
  --train-files /mnt/data_aisys_cpfs/zouanni.zan/data/appworld_parquet/train.parquet \
  --val-files   /mnt/data_aisys_cpfs/zouanni.zan/data/appworld_parquet/dev.parquet \
  --micro-batch-gpu 1 \
  --outdir advantage_dump \
  --num-batches 10
    """
