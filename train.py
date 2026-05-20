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
    parser.add_argument("--num_workers", type=int, default=CFG.data.num_workers)
    parser.add_argument("--val_ratio", type=float, default=CFG.data.val_ratio)
    parser.add_argument("--epochs", type=int, default=CFG.train.epochs)
    parser.add_argument("--lr", type=float, default=CFG.train.learning_rate)
    parser.add_argument("--device", type=str, default=CFG.train.device)
    parser.add_argument("--checkpoint_dir", type=Path, default=CFG.train.checkpoint_dir)
    parser.add_argument("--save_every", type=int, default=CFG.train.save_every)
    parser.add_argument(
        "--save_steps",
        type=int,
        default=CFG.train.save_steps,
        help="Save an extra checkpoint every N optimizer steps. Use 0 to disable step checkpoints.",
    )
    parser.add_argument(
        "--inspect_batches",
        type=int,
        default=CFG.train.inspect_batches,
        help="Print shape/stat checks for the first N training batches before training. Use 0 to disable.",
    )
    return parser.parse_args()


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    cfg.data.data_dir = args.data_dir
    cfg.data.window_size = args.window_size
    cfg.data.stride = args.stride
    cfg.data.batch_size = args.batch_size
    cfg.data.num_workers = args.num_workers
    cfg.data.val_ratio = args.val_ratio
    cfg.train.epochs = args.epochs
    cfg.train.learning_rate = args.lr
    cfg.train.device = args.device
    cfg.train.checkpoint_dir = args.checkpoint_dir
    cfg.train.save_every = args.save_every
    cfg.train.save_steps = args.save_steps
    cfg.train.inspect_batches = args.inspect_batches
    return cfg


def _tensor_stats(name: str, x: torch.Tensor) -> str:
    finite = torch.isfinite(x)
    finite_ratio = finite.float().mean().item()
    return (
        f"{name}: shape={tuple(x.shape)} "
        f"mean={x.float().mean().item():.4f} std={x.float().std().item():.4f} "
        f"min={x.float().min().item():.4f} max={x.float().max().item():.4f} "
        f"finite={finite_ratio:.3f}"
    )


def inspect_dataloader(loader: torch.utils.data.DataLoader, num_batches: int) -> None:
    """Print quick checks to verify DataLoader output before training."""
    if num_batches <= 0:
        return

    dataset = loader.dataset
    print(f"[inspect] dataset_windows={len(dataset)} batch_size={loader.batch_size}")
    base_dataset = getattr(dataset, "dataset", dataset)
    if hasattr(base_dataset, "describe"):
        print(f"[inspect] dataset_summary={base_dataset.describe()}")

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break

        ppg = batch["ppg"]
        gsr = batch["gsr"]
        timestamp = batch["timestamp"]
        monotonic_ratio = (timestamp[:, 1:] >= timestamp[:, :-1]).float().mean().item()
        print(f"[inspect] batch={batch_idx}")
        print(f"[inspect]   {_tensor_stats('ppg', ppg)}")
        print(f"[inspect]   {_tensor_stats('gsr', gsr)}")
        print(f"[inspect]   timestamp_shape={tuple(timestamp.shape)} monotonic_ratio={monotonic_ratio:.3f}")
        print(f"[inspect]   first_file={batch['file'][0]}")


def save_checkpoint(
    path: Path,
    model: PPGGSRCLIP,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: Config,
    metrics: Dict[str, float],
    global_step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(cfg),
            "metrics": metrics,
        },
        path,
    )
    print(f"[checkpoint] saved {path}")


def run_epoch(
    model: PPGGSRCLIP,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: Config,
    epoch: int,
    global_step: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[Dict[str, float], int]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_ppg_acc = 0.0
    total_gsr_acc = 0.0
    total_batches = 0

    for batch_idx, batch in enumerate(loader, start=1):
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
                global_step += 1

        total_loss += loss.item()
        total_ppg_acc += ppg_acc.item()
        total_gsr_acc += gsr_acc.item()
        total_batches += 1

        if is_train and cfg.train.log_every > 0 and batch_idx % cfg.train.log_every == 0:
            print(
                f"[train] epoch={epoch:03d} batch={batch_idx:05d}/{len(loader):05d} "
                f"step={global_step} loss={loss.item():.4f} "
                f"ppg2gsr={ppg_acc.item():.4f} gsr2ppg={gsr_acc.item():.4f}"
            )

        if is_train and cfg.train.save_steps > 0 and global_step % cfg.train.save_steps == 0:
            metrics = {
                "loss": total_loss / max(total_batches, 1),
                "ppg_to_gsr_acc": total_ppg_acc / max(total_batches, 1),
                "gsr_to_ppg_acc": total_gsr_acc / max(total_batches, 1),
            }
            save_checkpoint(
                cfg.train.checkpoint_dir / f"step_{global_step:08d}.pt",
                model,
                optimizer,
                epoch,
                cfg,
                metrics,
                global_step,
            )

    return (
        {
            "loss": total_loss / max(total_batches, 1),
            "ppg_to_gsr_acc": total_ppg_acc / max(total_batches, 1),
            "gsr_to_ppg_acc": total_gsr_acc / max(total_batches, 1),
        },
        global_step,
    )


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(CFG, args)
    set_seed(cfg.train.seed)

    device = resolve_device(cfg.train.device)
    train_loader, val_loader = build_dataloaders(cfg.data, seed=cfg.train.seed)
    inspect_dataloader(train_loader, cfg.train.inspect_batches)

    model = PPGGSRCLIP(cfg.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    best_val_loss = float("inf")
    global_step = 0
    val_metrics = None
    train_metrics = {}
    for epoch in range(1, cfg.train.epochs + 1):
        train_metrics, global_step = run_epoch(model, train_loader, device, cfg, epoch, global_step, optimizer)
        message = (
            f"epoch={epoch:03d} "
            f"step={global_step} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_ppg2gsr={train_metrics['ppg_to_gsr_acc']:.4f} "
            f"train_gsr2ppg={train_metrics['gsr_to_ppg_acc']:.4f}"
        )

        val_metrics = None
        if val_loader is not None:
            val_metrics, global_step = run_epoch(model, val_loader, device, cfg, epoch, global_step)
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
                    global_step,
                )

        print(message)

        if cfg.train.save_every > 0 and epoch % cfg.train.save_every == 0:
            save_checkpoint(
                cfg.train.checkpoint_dir / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                epoch,
                cfg,
                val_metrics or train_metrics,
                global_step,
            )

    save_checkpoint(
        cfg.train.checkpoint_dir / "last.pt",
        model,
        optimizer,
        cfg.train.epochs,
        cfg,
        val_metrics or train_metrics,
        global_step,
    )


if __name__ == "__main__":
    main()
