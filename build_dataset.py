#!/usr/bin/env python3
"""Build paired PPG/GSR samples from raw subject-day CSV files.

Expected folder layout (root_dir):
  root_dir/
    1001/
      day_xxx/
        time_GSR.csv
        time_PPG.csv
        time_ACC.csv   (optional, ignored here)
    ...

CSV format assumptions:
  - First column: signal value
  - Second column: timestamp
  - No strict header requirement (script auto-detects)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


VALID_SUBJECT_PREFIXES = ("10", "20")


@dataclass
class PairRecord:
    subject_id: str
    session_id: str
    start_time: float
    end_time: float
    n_gsr: int
    n_ppg: int
    n_aligned: int


def _read_signal_csv(path: Path, value_name: str) -> pd.DataFrame:
    """Read CSV and normalize to columns: [timestamp, value_name]."""
    df = pd.read_csv(path)

    # Handle headerless files: fallback to raw read with numeric column ids.
    if df.shape[1] < 2:
        df = pd.read_csv(path, header=None)
    if df.shape[1] < 2:
        raise ValueError(f"{path} has fewer than 2 columns")

    col0, col1 = df.columns[:2]
    out = pd.DataFrame(
        {
            value_name: pd.to_numeric(df[col0], errors="coerce"),
            "timestamp": pd.to_numeric(df[col1], errors="coerce"),
        }
    ).dropna(subset=[value_name, "timestamp"])

    out = out.sort_values("timestamp", kind="mergesort").drop_duplicates(
        subset=["timestamp"], keep="first"
    )
    out = out[["timestamp", value_name]].reset_index(drop=True)
    return out


def _overlap_range(gsr: pd.DataFrame, ppg: pd.DataFrame) -> Optional[Tuple[float, float]]:
    start = max(float(gsr["timestamp"].iloc[0]), float(ppg["timestamp"].iloc[0]))
    end = min(float(gsr["timestamp"].iloc[-1]), float(ppg["timestamp"].iloc[-1]))
    if end <= start:
        return None
    return start, end


def _crop_to_overlap(gsr: pd.DataFrame, ppg: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Tuple[float, float]]]:
    overlap = _overlap_range(gsr, ppg)
    if overlap is None:
        return gsr.iloc[0:0], ppg.iloc[0:0], None

    start, end = overlap
    gsr_c = gsr[(gsr["timestamp"] >= start) & (gsr["timestamp"] <= end)].copy()
    ppg_c = ppg[(ppg["timestamp"] >= start) & (ppg["timestamp"] <= end)].copy()
    return gsr_c.reset_index(drop=True), ppg_c.reset_index(drop=True), overlap


def _align_on_ppg_timestamps(
    gsr_cropped: pd.DataFrame,
    ppg_cropped: pd.DataFrame,
    tolerance_s: float,
) -> pd.DataFrame:
    """Align GSR to PPG timeline using nearest timestamp within tolerance."""
    if gsr_cropped.empty or ppg_cropped.empty:
        return pd.DataFrame(columns=["timestamp", "ppg", "gsr"])

    aligned = pd.merge_asof(
        ppg_cropped.sort_values("timestamp"),
        gsr_cropped.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=tolerance_s,
    )
    aligned = aligned.dropna(subset=["ppg", "gsr"]).reset_index(drop=True)
    return aligned[["timestamp", "ppg", "gsr"]]


def _iter_subject_dirs(root_dir: Path) -> Iterable[Path]:
    for p in sorted(root_dir.iterdir()):
        if p.is_dir() and p.name.isdigit() and p.name.startswith(VALID_SUBJECT_PREFIXES):
            yield p


def _iter_session_dirs(subject_dir: Path) -> Iterable[Path]:
    for p in sorted(subject_dir.iterdir()):
        if p.is_dir():
            yield p


def build_dataset(root_dir: Path, out_dir: Path, tolerance_s: float) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir = out_dir / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    records: List[PairRecord] = []

    for subj_dir in _iter_subject_dirs(root_dir):
        subject_id = subj_dir.name
        for ses_dir in _iter_session_dirs(subj_dir):
            session_id = ses_dir.name
            gsr_path = ses_dir / "time_GSR.csv"
            ppg_path = ses_dir / "time_PPG.csv"

            if not gsr_path.exists() or not ppg_path.exists():
                continue

            gsr = _read_signal_csv(gsr_path, "gsr")
            ppg = _read_signal_csv(ppg_path, "ppg")
            gsr_c, ppg_c, overlap = _crop_to_overlap(gsr, ppg)

            if overlap is None:
                records.append(
                    PairRecord(subject_id, session_id, np.nan, np.nan, 0, 0, 0)
                )
                continue

            start, end = overlap
            aligned = _align_on_ppg_timestamps(gsr_c, ppg_c, tolerance_s=tolerance_s)

            out_path = aligned_dir / f"{subject_id}__{session_id}.csv"
            aligned.to_csv(out_path, index=False)

            records.append(
                PairRecord(
                    subject_id=subject_id,
                    session_id=session_id,
                    start_time=start,
                    end_time=end,
                    n_gsr=len(gsr_c),
                    n_ppg=len(ppg_c),
                    n_aligned=len(aligned),
                )
            )

    meta = pd.DataFrame([r.__dict__ for r in records])
    meta.to_csv(out_dir / "metadata.csv", index=False)

    summary: Dict[str, float] = {
        "num_pairs": int(len(meta)),
        "num_pairs_with_overlap": int(meta["n_aligned"].gt(0).sum()) if len(meta) else 0,
        "mean_aligned_len": float(meta["n_aligned"].mean()) if len(meta) else 0.0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build aligned PPG/GSR dataset from raw folders")
    parser.add_argument("--root_dir", type=Path, required=True, help="Root folder containing subject folders")
    parser.add_argument("--out_dir", type=Path, default=Path("dataset_output"), help="Output folder")
    parser.add_argument(
        "--tolerance_s",
        type=float,
        default=0.03,
        help="Max timestamp difference for nearest-neighbor alignment (seconds)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = build_dataset(args.root_dir, args.out_dir, args.tolerance_s)
    print(f"Done. Built {len(meta)} subject-session pairs.")
    if len(meta):
        print(meta[["subject_id", "session_id", "n_gsr", "n_ppg", "n_aligned"]].head())


if __name__ == "__main__":
    main()
