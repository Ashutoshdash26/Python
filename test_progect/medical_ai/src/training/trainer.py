"""
src/training/trainer.py
────────────────────────
Production training loop for MedicalAI.

Features:
  - Mixed-precision training (torch.amp) for A100/V100
  - Cosine LR schedule with warmup
  - Early stopping on macro AUROC
  - Weights & Biases experiment tracking
  - Gradient clipping
  - Per-epoch validation with full metrics
  - Checkpoint saving (best + last)
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from src.models.full_model import MedicalAIModel
from src.training.losses import CombinedLoss
from src.utils.metrics import compute_epoch_metrics
from src.utils.logger import get_logger

log = get_logger(__name__)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class EarlyStopping:
    """Stop training when validation metric stops improving."""

    def __init__(self, patience: int = 7, min_delta: float = 1e-4, mode: str = "max"):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.best      = -float("inf") if mode == "max" else float("inf")
        self.counter   = 0
        self.triggered = False

    def step(self, metric: float) -> bool:
        improved = (self.mode == "max" and metric > self.best + self.min_delta) or \
                   (self.mode == "min" and metric < self.best - self.min_delta)
        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return improved


class MedicalAITrainer:
    """
    Trainer for the full MedicalAI model.

    Args:
        model         : MedicalAIModel instance
        train_loader  : training DataLoader
        val_loader    : validation DataLoader
        cfg           : config dict (from config.yaml)
        output_dir    : where to save checkpoints
        icd_vocab     : dict {icd10_code: idx} for metric labelling
        device        : 'cuda' | 'cpu'
        use_wandb     : log to Weights & Biases
    """

    def __init__(
        self,
        model: MedicalAIModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        output_dir: str | Path,
        icd_vocab: dict[str, int],
        device: str = "cuda",
        use_wandb: bool = True,
    ):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.output_dir   = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.icd_vocab    = icd_vocab
        self.device       = device
        self.use_wandb    = use_wandb and HAS_WANDB

        train_cfg = cfg["training"]
        self.epochs       = train_cfg["epochs"]
        self.grad_clip    = train_cfg["gradient_clip"]
        self.mixed_prec   = train_cfg["mixed_precision"] and device == "cuda"

        self.criterion = CombinedLoss(
            gamma_neg=4.0, gamma_pos=1.0, clip=0.05, entropy_weight=1e-3,
        )

        self.optimizer = self._build_optimizer(train_cfg)
        self.scheduler = self._build_scheduler(train_cfg)
        self.scaler    = GradScaler(enabled=self.mixed_prec)
        self.early_stopping = EarlyStopping(patience=train_cfg["early_stopping_patience"])

        self.best_metric   = -float("inf")
        self.global_step   = 0

        if self.use_wandb:
            wandb.init(project="medical-ai", config=cfg, name="run_1")
            wandb.watch(model, log="gradients", log_freq=100)

    def _build_optimizer(self, train_cfg: dict) -> AdamW:
        """Layer-wise learning rate decay for BERT layers."""
        no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}

        # NLP encoder: lower LR multiplier
        nlp_params_decay    = [p for n, p in self.model.nlp_encoder.named_parameters()
                                if p.requires_grad and not any(nd in n for nd in no_decay)]
        nlp_params_no_decay = [p for n, p in self.model.nlp_encoder.named_parameters()
                                if p.requires_grad and any(nd in n for nd in no_decay)]

        # Other modules: standard LR
        other_decay    = [p for n, p in self.model.named_parameters()
                          if p.requires_grad and "nlp_encoder" not in n
                          and not any(nd in n for nd in no_decay)]
        other_no_decay = [p for n, p in self.model.named_parameters()
                          if p.requires_grad and "nlp_encoder" not in n
                          and any(nd in n for nd in no_decay)]

        lr = train_cfg["learning_rate"]
        return AdamW([
            {"params": nlp_params_decay,    "lr": lr * 0.1, "weight_decay": train_cfg["weight_decay"]},
            {"params": nlp_params_no_decay, "lr": lr * 0.1, "weight_decay": 0.0},
            {"params": other_decay,         "lr": lr,       "weight_decay": train_cfg["weight_decay"]},
            {"params": other_no_decay,      "lr": lr,       "weight_decay": 0.0},
        ])

    def _build_scheduler(self, train_cfg: dict) -> OneCycleLR:
        steps_per_epoch = len(self.train_loader)
        return OneCycleLR(
            self.optimizer,
            max_lr=train_cfg["learning_rate"],
            epochs=train_cfg["epochs"],
            steps_per_epoch=steps_per_epoch,
            pct_start=train_cfg["warmup_ratio"],
            anneal_strategy="cos",
        )

    # ── Training epoch ─────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = asl_total = entr_total = 0.0
        n_batches  = len(self.train_loader)
        t0         = time.time()

        for step, batch in enumerate(self.train_loader):
            numerical      = batch["numerical"].to(self.device)
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.mixed_prec):
                output = self.model(numerical, input_ids, attention_mask)
                losses = self.criterion(
                    output["logits"], labels, output["entropy_loss"]
                )
                loss = losses["total"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            asl_total  += losses["asl"].item()
            entr_total += losses["entropy"].item()
            self.global_step += 1

            if step % 50 == 0:
                lr = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                log.info(
                    "Train step",
                    epoch=epoch, step=f"{step}/{n_batches}",
                    loss=f"{loss.item():.4f}", lr=f"{lr:.2e}",
                    elapsed=f"{elapsed:.1f}s",
                )
                if self.use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/asl_loss": losses["asl"].item(),
                        "train/lr": lr,
                        "step": self.global_step,
                    })

        return {
            "loss":     total_loss / n_batches,
            "asl":      asl_total  / n_batches,
            "entropy":  entr_total / n_batches,
        }

    # ── Validation epoch ───────────────────────────────────────────────

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        all_logits = []
        all_labels = []
        total_loss = 0.0

        for batch in self.val_loader:
            numerical      = batch["numerical"].to(self.device)
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"].to(self.device)

            with autocast(enabled=self.mixed_prec):
                output = self.model(numerical, input_ids, attention_mask)
                losses = self.criterion(output["logits"], labels, output["entropy_loss"])

            total_loss += losses["total"].item()
            all_logits.append(output["logits"].cpu())
            all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_probs  = torch.sigmoid(all_logits)

        metrics = compute_epoch_metrics(
            all_probs.numpy(), all_labels.numpy(),
            threshold=self.cfg["classifier"]["threshold"],
        )
        metrics["loss"] = total_loss / len(self.val_loader)

        log.info("Validation", epoch=epoch, **{k: f"{v:.4f}" for k, v in metrics.items()})
        if self.use_wandb:
            wandb.log({"val/" + k: v for k, v in metrics.items()} | {"epoch": epoch})

        return metrics

    # ── Main training loop ─────────────────────────────────────────────

    def train(self) -> MedicalAIModel:
        log.info("Starting training", epochs=self.epochs, device=self.device)
        params = self.model.parameter_count()
        log.info("Model parameters", **params)

        for epoch in range(1, self.epochs + 1):
            log.info(f"── Epoch {epoch}/{self.epochs} ──")

            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._val_epoch(epoch)

            monitor_metric = val_metrics.get("auroc_macro", val_metrics.get("loss", 0.0))
            improved = self.early_stopping.step(monitor_metric)

            if improved:
                ckpt_path = self.output_dir / "best_model.pt"
                self._save_checkpoint(ckpt_path, epoch, val_metrics)
                log.info("New best model saved", metric=f"{monitor_metric:.4f}", path=str(ckpt_path))

            # Always save latest
            self._save_checkpoint(self.output_dir / "last_model.pt", epoch, val_metrics)

            if self.early_stopping.triggered:
                log.info("Early stopping triggered", epoch=epoch)
                break

        if self.use_wandb:
            wandb.finish()

        # Return best model
        best_ckpt = self.output_dir / "best_model.pt"
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=self.device)
            self.model.load_state_dict(ckpt["state_dict"])
        return self.model

    def _save_checkpoint(
        self, path: Path, epoch: int, metrics: dict
    ) -> None:
        torch.save({
            "epoch":      epoch,
            "state_dict": self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "scheduler":  self.scheduler.state_dict(),
            "metrics":    metrics,
            "global_step": self.global_step,
        }, path)
