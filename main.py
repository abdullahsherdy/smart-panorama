"""
Smart Panorama — Stages 1–4 pipeline (project root entry point)
=================================================================
Run from the repository root::

    python main.py

Stages run **in order**, not in parallel: each stage needs the previous
stage’s outputs (arrays for the next computation). You *could* later
parallelize **across unrelated image pairs** (different workers per pair),
but within one pair the chain is always preprocess → detect → match → stitch.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

# Import modules from `src/` when launching `python main.py` from repo root
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from match import find_preprocessed_pair_ids, run_pair
from preprocess import run_batch
from stitch import stitch_pair


def _output_path_for_csv(abs_path: str) -> str:
    """Repo-relative path using forward slashes (no machine-specific absolute paths)."""
    return os.path.relpath(abs_path, _REPO_ROOT).replace("\\", "/")


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline — Stages 1 → 4
# ──────────────────────────────────────────────────────────────────────────────


def run_pipeline_stages(
    input1_dir: str,
    input2_dir: str,
    preprocessed_dir: str,
    panorama_dir: str,
    max_pairs: int,
    nfeatures: int = 5000,
    match_ratio: float = 0.75,
    ransac_thresh: float = 5.0,
    skip_stage1_if_present: bool = False,
) -> tuple[pd.DataFrame, List[str]]:
    """
    Execute Stage 1 (batch), then for each pair Stages 2–3 via ``run_pair``
    and Stage 4 ``stitch_pair``.

    Returns
    -------
    metrics : DataFrame with per-pair statistics
    saved_paths : list of panorama JPG paths that were written
    """
    preprocessed_dir = os.path.abspath(preprocessed_dir)
    panorama_dir = os.path.abspath(panorama_dir)
    os.makedirs(panorama_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  STAGE 1 — Preprocessing (batch)")
    print("=" * 60)

    have_npy = bool(find_preprocessed_pair_ids(preprocessed_dir))
    if skip_stage1_if_present and have_npy:
        print(f"[main] Skipping Stage 1; using existing .npy under '{preprocessed_dir}'")
    else:
        run_batch(
            input1_dir=input1_dir,
            input2_dir=input2_dir,
            output_dir=preprocessed_dir,
            max_pairs=max_pairs,
            save_npy=True,
        )

    pair_ids = find_preprocessed_pair_ids(preprocessed_dir)
    if not pair_ids:
        raise RuntimeError(
            f"No paired *_a_gray.npy / *_b_gray.npy in '{preprocessed_dir}'. "
            "Run Stage 1 or fix paths."
        )
    pair_ids = pair_ids[:max_pairs]

    print("\n" + "=" * 60)
    print("  STAGES 2–4 — Detect → Match → Stitch (per pair)")
    print("=" * 60)

    rows: List[Dict[str, Any]] = []
    saved_paths: List[str] = []

    for i, pair_id in enumerate(pair_ids):
        t0 = time.time()

        kp_a, _da, kp_b, _db, good, color_a, color_b = run_pair(
            preprocessed_dir,
            pair_id,
            nfeatures=nfeatures,
            ratio=match_ratio,
        )

        if color_a is None or color_b is None:
            rows.append(
                {
                    "pair_id": pair_id,
                    "kp_a": len(kp_a),
                    "kp_b": len(kp_b),
                    "good_matches": len(good),
                    "stitched": "NO_COLOR_NPY",
                    "time_sec": round(time.time() - t0, 2),
                    "output": "",
                    "shape": "",
                }
            )
            print(
                f"  [{i+1}/{len(pair_ids)}] {pair_id}: skip stitch (missing *_color.npy)"
            )
            continue

        pano = stitch_pair(
            color_a,
            color_b,
            good,
            kp_a,
            kp_b,
            ransac_reproj_threshold=ransac_thresh,
        )
        elapsed = time.time() - t0

        if pano is None:
            rows.append(
                {
                    "pair_id": pair_id,
                    "kp_a": len(kp_a),
                    "kp_b": len(kp_b),
                    "good_matches": len(good),
                    "stitched": "FAILED",
                    "time_sec": round(elapsed, 2),
                    "output": "",
                    "shape": "",
                }
            )
            print(
                f"  [{i+1}/{len(pair_ids)}] {pair_id}: stitch failed ({len(good)} matches)"
            )
            continue

        out_name = f"panorama_{i+1:03d}.jpg"
        out_path = os.path.join(panorama_dir, out_name)
        cv2.imwrite(out_path, pano)
        saved_paths.append(out_path)

        h, w = pano.shape[:2]
        rows.append(
            {
                "pair_id": pair_id,
                "kp_a": len(kp_a),
                "kp_b": len(kp_b),
                "good_matches": len(good),
                "stitched": "OK",
                "time_sec": round(elapsed, 2),
                "output": _output_path_for_csv(out_path),
                "shape": f"{w}x{h}",
            }
        )
        print(
            f"  [{i+1}/{len(pair_ids)}] {pair_id}: "
            f"{len(good)} matches → {out_name}  ({elapsed:.2f}s)"
        )

    df = pd.DataFrame(rows)
    metrics_path = os.path.join(panorama_dir, "stitching_metrics.csv")
    df.to_csv(metrics_path, index=False)
    print(f"\n[main] Metrics saved → {_output_path_for_csv(metrics_path)}")
    return df, saved_paths


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Smart Panorama pipeline Stages 1–4 (preprocess → stitch)"
    )
    p.add_argument("--input1", default="data/udis/test/input1", help="UDIS-D left images")
    p.add_argument("--input2", default="data/udis/test/input2", help="UDIS-D right images")
    p.add_argument(
        "--preprocessed",
        default="outputs/preprocessed",
        help="Stage 1 .npy output folder",
    )
    p.add_argument(
        "--panoramas",
        default="outputs",
        help="Where to write panorama_*.jpg and stitching_metrics.csv",
    )
    p.add_argument("--pairs", type=int, default=5, help="Max UDIS-D pairs to process")
    p.add_argument("--nfeatures", type=int, default=5000)
    p.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio (Stage 3)")
    p.add_argument("--ransac", type=float, default=5.0, help="RANSAC px threshold (Stage 4)")
    p.add_argument(
        "--skip-stage1-if-present",
        action="store_true",
        help="If preprocessed .npy already exists, skip Stage 1",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    t_start = time.time()

    df, _paths = run_pipeline_stages(
        input1_dir=os.path.join(_REPO_ROOT, args.input1),
        input2_dir=os.path.join(_REPO_ROOT, args.input2),
        preprocessed_dir=os.path.join(_REPO_ROOT, args.preprocessed),
        panorama_dir=os.path.join(_REPO_ROOT, args.panoramas),
        max_pairs=args.pairs,
        nfeatures=args.nfeatures,
        match_ratio=args.ratio,
        ransac_thresh=args.ransac,
        skip_stage1_if_present=args.skip_stage1_if_present,
    )

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\n[main] Total wall time: {time.time() - t_start:.2f}s")
