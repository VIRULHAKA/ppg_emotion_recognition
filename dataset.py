"""Dataset utilities for npy files containing timestamp/GSR/PPG columns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from config import DataConfig


@dataclass(frozen=True)
class WindowRecord:
    file_idx: int
    start: int
    end: int


class PPGGSRNpyDataset(Dataset):
    """Sliding-window Dataset for aligned daily npy files.

    Expected npy content: one array with three columns in this order:
    [timestamp, GSR, PPG]. A header row such as ["timestamp", "GSR", "PPG"]
    is allowed and will be ignored automatically.

    Each item is a positive PPG/GSR pair from the same time window:
        ppg: Tensor[window_size, 1]
        gsr: Tensor[window_size, 1]
    """

    def __init__(
        self,
        data_dir: str | Path,
        window_size: int = 400,
        stride: int = 200,
        normalize: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize

        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")

        self.files = self._discover_files(self.data_dir)
        self.arrays: List[np.ndarray] = []
        self.records: List[WindowRecord] = []

        self._build_index()

    @staticmethod
    def _discover_files(data_dir: Path) -> List[Path]:
        files = sorted(data_dir.rglob("*.npy"))
        if not files:
            raise FileNotFoundError(f"No .npy files found under {data_dir}")
        return files

    @staticmethod
    def _load_npy(path: Path) -> np.ndarray:
        raw = np.load(path, allow_pickle=True)
        if raw.ndim != 2 or raw.shape[1] < 3:
            raise ValueError(f"{path} must be a 2D array with at least 3 columns")

        # Use pandas numeric conversion so both pure numeric arrays and object/string
        # arrays with a header row are handled consistently.
        df = pd.DataFrame(raw[:, :3], columns=["timestamp", "gsr", "ppg"])
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["gsr"] = pd.to_numeric(df["gsr"], errors="coerce")
        df["ppg"] = pd.to_numeric(df["ppg"], errors="coerce")
        df = df.dropna(subset=["gsr", "ppg"]).reset_index(drop=True)

        if df.empty:
            raise ValueError(f"{path} has no valid GSR/PPG rows")

        return df[["timestamp", "gsr", "ppg"]].to_numpy(dtype=np.float32)

    def _build_index(self) -> None:
        for path in self.files:
            arr = self._load_npy(path)
            if len(arr) < self.window_size:
                continue

            file_idx = len(self.arrays)
            self.arrays.append(arr)

            for start in range(0, len(arr) - self.window_size + 1, self.stride):
                self.records.append(WindowRecord(file_idx=file_idx, start=start, end=start + self.window_size))

        if not self.records:
            raise ValueError(
                "No training windows were created. Try a smaller window_size/stride "
                "or check that the npy files contain enough rows."
            )

    @staticmethod
    def _zscore(x: np.ndarray) -> np.ndarray:
        mean = x.mean()
        std = x.std()
        if std < 1e-8:
            return x - mean
        return (x - mean) / std


    def describe(self) -> Dict[str, int | float]:
        """Return a compact summary for checking dataset construction."""
        rows_per_file = [len(arr) for arr in self.arrays]
        return {
            "num_npy_files_found": len(self.files),
            "num_npy_files_used": len(self.arrays),
            "num_windows": len(self.records),
            "window_size": self.window_size,
            "stride": self.stride,
            "min_rows_per_used_file": min(rows_per_file) if rows_per_file else 0,
            "max_rows_per_used_file": max(rows_per_file) if rows_per_file else 0,
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        rec = self.records[idx]
        arr = self.arrays[rec.file_idx][rec.start:rec.end]

        timestamp = arr[:, 0].astype(np.float32)
        gsr = arr[:, 1].astype(np.float32)
        ppg = arr[:, 2].astype(np.float32)

        if self.normalize:
            gsr = self._zscore(gsr)
            ppg = self._zscore(ppg)

        source_file = self.files[rec.file_idx]
        return {
            "ppg": torch.from_numpy(ppg).unsqueeze(-1),
            "gsr": torch.from_numpy(gsr).unsqueeze(-1),
            "timestamp": torch.from_numpy(timestamp),
            "file": str(source_file),
        }


def split_dataset(
    dataset: Dataset,
    val_ratio: float,
    seed: int,
) -> Tuple[Subset, Subset]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")

    val_len = int(len(dataset) * val_ratio)
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_len, val_len], generator=generator)


def build_dataloaders(cfg: DataConfig, seed: int) -> Tuple[DataLoader, DataLoader | None]:
    dataset = PPGGSRNpyDataset(
        data_dir=cfg.data_dir,
        window_size=cfg.window_size,
        stride=cfg.stride,
        normalize=cfg.normalize,
    )
    train_set, val_set = split_dataset(dataset, cfg.val_ratio, seed)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=cfg.drop_last,
    )

    val_loader = None
    if len(val_set) > 0:
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            drop_last=False,
        )

    return train_loader, val_loader
