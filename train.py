"""Train PPG/GSR transformer encoders with a CLIP-style contrastive loss."""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from config import CFG, Config
from dataset import build_dataloaders
from model import PPGGSRCLIP, clip_contrastive_loss, retrieval_accuracy


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CLIP-style PPG/GSR transformer encoders")
    parser.add_argument("--data_dir", type=Path, default=CFG.data.data_dir)
    parser.add_argument("--window_size", type=int, default=CFG.data.window_size)
    parser.add_argument("--stride", type=int, default=CFG.data.stride)
    parser.add_argument("--batch_size", type=int, default=CFG.data.batch_size)
    parser.add_argument("--epochs", type=int, default=CFG.train.epochs)
    parser.add_argument("--lr", type=float, default=CFG.train.learning_rate)
    parser.add_argument("--device", type=str, default=CFG.train.device)
    parser.add_argument("--checkpoint_dir", type=Path, default=CFG.train.checkpoint_dir)
    return parser.parse_args()


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    cfg.data.data_dir = args.data_dir
    cfg.data.window_size = args.window_size
    cfg.data.stride = args.stride
    cfg.data.batch_size = args.batch_size
    cfg.train.epochs = args.epochs
    cfg.train.learning_rate = args.lr
    cfg.train.device = args.device
    cfg.train.checkpoint_dir = args.checkpoint_dir
    return cfg


def run_epoch(
    model: PPGGSRCLIP,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_ppg_acc = 0.0
    total_gsr_acc = 0.0
    total_batches = 0

    for batch in loader:
        ppg = batch["ppg"].to(device)
        gsr = batch["gsr"].to(device)

        with torch.set_grad_enabled(is_train):
            logits, _, _ = model(ppg, gsr)
            loss = clip_contrastive_loss(logits)
            ppg_acc, gsr_acc = retrieval_accuracy(logits)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_ppg_acc += ppg_acc.item()
        total_gsr_acc += gsr_acc.item()
        total_batches += 1

    return {
        "loss": total_loss / max(total_batches, 1),
        "ppg_to_gsr_acc": total_ppg_acc / max(total_batches, 1),
        "gsr_to_ppg_acc": total_gsr_acc / max(total_batches, 1),
    }


def save_checkpoint(
    path: Path,
    model: PPGGSRCLIP,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: Config,
    metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(cfg),
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(CFG, args)
    set_seed(cfg.train.seed)

    device = resolve_device(cfg.train.device)
    train_loader, val_loader = build_dataloaders(cfg.data, seed=cfg.train.seed)

    model = PPGGSRCLIP(cfg.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    best_val_loss = float("inf")
    for epoch in range(1, cfg.train.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        message = (
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_ppg2gsr={train_metrics['ppg_to_gsr_acc']:.4f} "
            f"train_gsr2ppg={train_metrics['gsr_to_ppg_acc']:.4f}"
        )

        val_metrics = None
        if val_loader is not None:
            val_metrics = run_epoch(model, val_loader, device)
            message += (
                f" val_loss={val_metrics['loss']:.4f} "
                f"val_ppg2gsr={val_metrics['ppg_to_gsr_acc']:.4f} "
                f"val_gsr2ppg={val_metrics['gsr_to_ppg_acc']:.4f}"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(
                    cfg.train.checkpoint_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    cfg,
                    val_metrics,
                )

        print(message)

        if epoch % cfg.train.save_every == 0:
            save_checkpoint(
                cfg.train.checkpoint_dir / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                epoch,
                cfg,
                val_metrics or train_metrics,
            )

    save_checkpoint(
        cfg.train.checkpoint_dir / "last.pt",
        model,
        optimizer,
        cfg.train.epochs,
        cfg,
        val_metrics or train_metrics,
    )


if __name__ == "__main__":
    main()
