from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from build_dataset import _align_on_ppg_timestamps, _basename_without_modality, _crop_to_overlap, _read_signal_csv


@dataclass
class WindowIndex:
    subject_id: str
    session_id: str
    start: int
    end: int
    cache_id: int


class PPGGSRDataset(Dataset):
    """PyTorch dataset for CLIP-style PPG/GSR dual-encoder training.

    Input structure (raw mode):
      root_dir/<subject>/<time_range>_GSR.csv
      root_dir/<subject>/<time_range>_PPG.csv

    The dataset aligns signals on the PPG timeline and then slices fixed-length windows.
    Each __getitem__ returns:
      {
        "ppg": Tensor[1, window_size],
        "gsr": Tensor[1, window_size],
        "subject_id": str,
        "session_id": str,
        "timestamps": Tensor[window_size],
      }
    """

    def __init__(
        self,
        root_dir: str | Path,
        window_size: int = 200,
        stride: int = 100,
        tolerance_s: float = 0.03,
        min_valid_ratio: float = 1.0,
        normalize: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.window_size = window_size
        self.stride = stride
        self.tolerance_s = tolerance_s
        self.min_valid_ratio = min_valid_ratio
        self.normalize = normalize

        self._aligned_cache: List[pd.DataFrame] = []
        self._window_index: List[WindowIndex] = []

        self._build_index()

    def _build_index(self) -> None:
        for subject_dir in sorted(self.root_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            subject_id = subject_dir.name

            gsr_files = sorted(subject_dir.glob("*_GSR.csv"))
            ppg_files = sorted(subject_dir.glob("*_PPG.csv"))
            ppg_map = {_basename_without_modality(p): p for p in ppg_files}

            for gsr_path in gsr_files:
                session_id = _basename_without_modality(gsr_path)
                ppg_path = ppg_map.get(session_id)
                if ppg_path is None:
                    continue

                gsr = _read_signal_csv(gsr_path, "gsr")
                ppg = _read_signal_csv(ppg_path, "ppg")
                gsr_c, ppg_c, overlap = _crop_to_overlap(gsr, ppg)
                if overlap is None:
                    continue

                aligned = _align_on_ppg_timestamps(gsr_c, ppg_c, tolerance_s=self.tolerance_s)
                if aligned.empty:
                    continue

                valid_needed = int(self.window_size * self.min_valid_ratio)
                if len(aligned) < valid_needed:
                    continue

                cache_id = len(self._aligned_cache)
                self._aligned_cache.append(aligned)

                for start in range(0, len(aligned) - self.window_size + 1, self.stride):
                    end = start + self.window_size
                    self._window_index.append(
                        WindowIndex(
                            subject_id=subject_id,
                            session_id=session_id,
                            start=start,
                            end=end,
                            cache_id=cache_id,
                        )
                    )

    def __len__(self) -> int:
        return len(self._window_index)

    def _normalize_signal(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean()
        std = x.std()
        if std < 1e-8:
            return x - mean
        return (x - mean) / std

    def __getitem__(self, idx: int):
        item = self._window_index[idx]
        aligned = self._aligned_cache[item.cache_id]

        w = aligned.iloc[item.start:item.end]
        ppg = w["ppg"].to_numpy(dtype=np.float32)
        gsr = w["gsr"].to_numpy(dtype=np.float32)
        ts = w["timestamp"].to_numpy(dtype=np.float32)

        if self.normalize:
            ppg = self._normalize_signal(ppg)
            gsr = self._normalize_signal(gsr)

        return {
            "ppg": torch.from_numpy(ppg).unsqueeze(0),
            "gsr": torch.from_numpy(gsr).unsqueeze(0),
            "timestamps": torch.from_numpy(ts),
            "subject_id": item.subject_id,
            "session_id": item.session_id,
        }


if __name__ == "__main__":
    # quick smoke-check
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=100)
    args = parser.parse_args()

    ds = PPGGSRDataset(root_dir=args.root_dir, window_size=args.window_size, stride=args.stride)
    print(f"num_windows={len(ds)}")
    if len(ds):
        sample = ds[0]
        print(sample["ppg"].shape, sample["gsr"].shape, sample["timestamps"].shape)
