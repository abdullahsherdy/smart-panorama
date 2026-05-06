"""
Stage 2 — Feature Detection
Smart Panorama & Object Recognition System
==========================================
Dataset : Preprocessed output from Stage 1 (gray .npy or in-memory from preprocess)
Input   : gray_img (np.ndarray, single-channel grayscale)
Output  : (keypoints, descriptors) — OpenCV types: list/cv.KeyPoint, np.ndarray
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# CORE
# ──────────────────────────────────────────────────────────────────────────────


def detect_features(gray_img: np.ndarray,
                   nfeatures: int = 5000,
                   nOctaveLayers: int = 3,
                   contrastThreshold: float = 0.04,
                   edgeThreshold: float = 10.0,
                   sigma: float = 1.6) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """
    Extract SIFT keypoints and descriptors from a grayscale image.

    Parameters
    ----------
    gray_img : np.ndarray
        Grayscale image (H, W). uint8 recommended; uint16/float inputs are
        scaled to uint8 when needed for stable OpenCV behaviour.
    nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma
        SIFT parameters (see cv2.SIFT_create).

    Returns
    -------
    keypoints : list of cv2.KeyPoint
    descriptors : np.ndarray, shape (N, 128), dtype float32, or None if no features
    """
    if gray_img is None or gray_img.size == 0:
        raise ValueError("[detect_features] gray_img is empty or None")

    if gray_img.ndim != 2:
        raise ValueError(
            f"[detect_features] Expected 2D grayscale array, got shape {gray_img.shape}"
        )

    g = gray_img
    if g.dtype != np.uint8:
        g = _to_uint8_gray(g)

    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        nOctaveLayers=nOctaveLayers,
        contrastThreshold=contrastThreshold,
        edgeThreshold=edgeThreshold,
        sigma=sigma,
    )
    keypoints, descriptors = sift.detectAndCompute(g, None)
    return keypoints, descriptors


def _to_uint8_gray(arr: np.ndarray) -> np.ndarray:
    """Normalize float or wider integer grayscale to uint8."""
    x = arr.astype(np.float64, copy=False)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    if x_max <= x_min:
        return np.zeros_like(arr, dtype=np.uint8)
    x = (x - x_min) / (x_max - x_min)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def draw_keypoints_bgr(gray_img: np.ndarray,
                      keypoints: List[cv2.KeyPoint],
                      rich: bool = True) -> np.ndarray:
    """Draw keypoints on a 3-channel image for saving / display."""
    g = gray_img if gray_img.dtype == np.uint8 else _to_uint8_gray(gray_img)
    bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if rich else 0
    return cv2.drawKeypoints(bgr, keypoints, None, flags=flags)


# ──────────────────────────────────────────────────────────────────────────────
# BATCH — Stage 1 .npy outputs
# ──────────────────────────────────────────────────────────────────────────────


def load_gray_npy(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return arr


def run_on_preprocessed(preprocessed_dir: str,
                       max_images: Optional[int] = None,
                       **sift_kw) -> list:
    """
    Run SIFT on all `*_gray.npy` files under preprocessed_dir (Stage 1 outputs).

    Returns
    -------
    list of dict with keys: stem, path, keypoints, descriptors, n_kp
    """
    pattern = os.path.join(preprocessed_dir, "*_gray.npy")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"[run_on_preprocessed] No *_gray.npy under '{preprocessed_dir}'. "
            "Run Stage 1 first or set --preprocessed."
        )
    if max_images is not None:
        paths = paths[:max_images]

    out = []
    for p in paths:
        gray = load_gray_npy(p)
        kp, des = detect_features(gray, **sift_kw)
        stem = os.path.splitext(os.path.basename(p))[0]
        out.append({
            "stem": stem,
            "path": p,
            "keypoints": kp,
            "descriptors": des,
            "n_kp": len(kp),
        })
    return out


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2 — SIFT Feature Detection")
    p.add_argument(
        "--preprocessed",
        default="../outputs/preprocessed",
        help="Folder with Stage 1 *_gray.npy files",
    )
    p.add_argument(
        "--output",
        default="../outputs/kp_sift.png",
        help="Visualization path (first gray image in folder)",
    )
    p.add_argument("--nfeatures", type=int, default=5000)
    p.add_argument("--max-images", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 55)
    print("  Stage 2 — Feature Detection (SIFT)")
    print("=" * 55)

    sift_kw = {"nfeatures": args.nfeatures}

    results = run_on_preprocessed(
        args.preprocessed,
        max_images=args.max_images,
        **sift_kw,
    )

    print(f"[Stage 2] Processed {len(results)} grayscale image(s).")
    for r in results[:5]:
        n = r["n_kp"]
        dshape = r["descriptors"].shape if r["descriptors"] is not None else None
        print(f"  {r['stem']}: {n} keypoints, descriptors {dshape}")
    if len(results) > 5:
        print(f"  ... and {len(results) - 5} more")

    # Demo figure: first image keypoints (README: outputs/kp_sift.png)
    first = results[0]
    gray = load_gray_npy(first["path"])
    vis = draw_keypoints_bgr(gray, first["keypoints"])

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, vis)
    print(f"[Stage 2] Saved keypoint visualization → {out_path}")

    print("\n[Stage 2] Complete")
