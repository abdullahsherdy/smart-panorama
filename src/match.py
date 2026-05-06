"""
Stage 3 — Feature Matching
Smart Panorama & Object Recognition System
==========================================
Input   : (kp_a, des_a) and (kp_b, des_b) from Stage 2 — OpenCV types
Output  : good_matches — list of cv2.DMatch (query = A, train = B)

Uses BFMatcher + kNN (k=2) and Lowe's ratio test (default 0.75), suitable for SIFT.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import List, Optional, Tuple

import cv2
import numpy as np

from detect import detect_features, load_gray_npy

# ──────────────────────────────────────────────────────────────────────────────
# CORE
# ──────────────────────────────────────────────────────────────────────────────


def match_features(
    kp_a: List[cv2.KeyPoint],
    des_a: Optional[np.ndarray],
    kp_b: List[cv2.KeyPoint],
    des_b: Optional[np.ndarray],
    ratio: float = 0.75,
    norm: int = cv2.NORM_L2,
) -> List[cv2.DMatch]:
    """
    Match SIFT-like descriptors with brute-force kNN and Lowe's ratio test.

    Convention: image A is the *query* and B is the *train* for cv2.BFMatcher,
    so for each ``m`` in the returned list:
    ``kp_a[m.queryIdx]`` pairs with ``kp_b[m.trainIdx]``.

    Parameters
    ----------
    ratio : float
        Accept match m only if distance(m) < ratio * distance(second_best).
    norm  : int
        ``cv2.NORM_L2`` for SIFT; use ``cv2.NORM_HAMMING`` for binary descriptors.

    Returns
    -------
    good_matches : list of cv2.DMatch
    """
    if des_a is None or des_b is None:
        return []
    if des_a.size == 0 or des_b.size == 0:
        return []
    # kNN needs at least 2 train descriptors
    if len(kp_a) == 0 or len(kp_b) == 0 or des_b.shape[0] < 2:
        return []

    bf = cv2.BFMatcher(norm, crossCheck=False)
    raw = bf.knnMatch(des_a, des_b, k=2)

    good: List[cv2.DMatch] = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair[0], pair[1]
        if m.distance < ratio * n.distance:
            good.append(m)

    return good


def draw_matches_horizontal(
    img_a: np.ndarray,
    kp_a: List[cv2.KeyPoint],
    img_b: np.ndarray,
    kp_b: List[cv2.KeyPoint],
    good_matches: List[cv2.DMatch],
    max_draw: int = 80,
) -> np.ndarray:
    """
    Build a side-by-side visualization (BGR). ``img_a`` / ``img_b`` can be
    colour (H,W,3) or grayscale (H,W) encoded later as BGR.
    """
    def to_bgr(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    a = to_bgr(img_a)
    b = to_bgr(img_b)

    vis_matches = good_matches
    if len(vis_matches) > max_draw:
        # stable subsample for readability
        idx = np.linspace(0, len(vis_matches) - 1, max_draw, dtype=int)
        vis_matches = [good_matches[i] for i in idx]

    return cv2.drawMatches(
        a,
        kp_a,
        b,
        kp_b,
        vis_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


# ──────────────────────────────────────────────────────────────────────────────
# PAIRS — Stage 1 naming: ``{pair_id}_a_gray.npy`` / ``{pair_id}_b_gray.npy``
# ──────────────────────────────────────────────────────────────────────────────


def find_preprocessed_pair_ids(preprocessed_dir: str) -> List[str]:
    pattern = os.path.join(preprocessed_dir, "*_a_gray.npy")
    ids: List[str] = []
    for path in sorted(glob.glob(pattern)):
        base = os.path.basename(path)
        m = re.match(r"^(.+)_a_gray\.npy$", base, re.IGNORECASE)
        if not m:
            continue
        pair_id = m.group(1)
        b_path = os.path.join(preprocessed_dir, f"{pair_id}_b_gray.npy")
        if os.path.isfile(b_path):
            ids.append(pair_id)
    return ids


def run_pair(
    preprocessed_dir: str,
    pair_id: str,
    nfeatures: int = 5000,
    ratio: float = 0.75,
) -> Tuple[
    List[cv2.KeyPoint],
    Optional[np.ndarray],
    List[cv2.KeyPoint],
    Optional[np.ndarray],
    List[cv2.DMatch],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """
    Load one pair's grays from Stage 1 outputs, run Stage 2 + Stage 3.

    Returns
    -------
    kp_a, des_a, kp_b, des_b, good_matches,
    color_a, color_b  (or None if *_color.npy missing)
    """
    ga = os.path.join(preprocessed_dir, f"{pair_id}_a_gray.npy")
    gb = os.path.join(preprocessed_dir, f"{pair_id}_b_gray.npy")
    if not os.path.isfile(ga) or not os.path.isfile(gb):
        raise FileNotFoundError(f"[run_pair] Missing gray .npy for pair_id={pair_id}")

    gray_a = load_gray_npy(ga)
    gray_b = load_gray_npy(gb)
    kp_a, des_a = detect_features(gray_a, nfeatures=nfeatures)
    kp_b, des_b = detect_features(gray_b, nfeatures=nfeatures)
    good = match_features(kp_a, des_a, kp_b, des_b, ratio=ratio)

    ca = os.path.join(preprocessed_dir, f"{pair_id}_a_color.npy")
    cb = os.path.join(preprocessed_dir, f"{pair_id}_b_color.npy")
    color_a = np.load(ca) if os.path.isfile(ca) else None
    color_b = np.load(cb) if os.path.isfile(cb) else None

    return kp_a, des_a, kp_b, des_b, good, color_a, color_b


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Stage 3 — Feature Matching")
    p.add_argument(
        "--preprocessed",
        default="../outputs/preprocessed",
        help="Folder with Stage 1 *.npy pair outputs",
    )
    p.add_argument(
        "--output",
        default="../outputs/matches_pair1.png",
        help="Match visualization path",
    )
    p.add_argument("--pair-id", default=None, help="Specific pair id; default = first pair")
    p.add_argument("--nfeatures", type=int, default=5000)
    p.add_argument("--ratio", type=float, default=0.75)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 55)
    print("  Stage 3 — Feature Matching")
    print("=" * 55)

    pre = os.path.abspath(args.preprocessed)
    ids = find_preprocessed_pair_ids(pre)
    if not ids:
        raise SystemExit(
            f"[Stage 3] No A/B gray pairs in '{pre}'. Run Stage 1 first (saves *_a_gray.npy and *_b_gray.npy)."
        )

    pair_id = args.pair_id if args.pair_id else ids[0]
    if pair_id not in ids:
        raise SystemExit(f"[Stage 3] pair_id '{pair_id}' not found. Available: {ids[:10]}{'...' if len(ids) > 10 else ''}")

    kp_a, des_a, kp_b, des_b, good, color_a, color_b = run_pair(
        pre, pair_id, nfeatures=args.nfeatures, ratio=args.ratio
    )

    print(f"  Pair: {pair_id}")
    print(f"  Keypoints: A={len(kp_a)}  B={len(kp_b)}")
    print(f"  Good matches (ratio={args.ratio}): {len(good)}")

    ga_path = os.path.join(pre, f"{pair_id}_a_gray.npy")
    gb_path = os.path.join(pre, f"{pair_id}_b_gray.npy")
    gray_a = load_gray_npy(ga_path)
    gray_b = load_gray_npy(gb_path)

    img_a = color_a if color_a is not None else gray_a
    img_b = color_b if color_b is not None else gray_b

    canvas = draw_matches_horizontal(img_a, kp_a, img_b, kp_b, good)

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, canvas)
    print(f"[Stage 3] Saved match visualization → {out_path}")
    print("\n[Stage 3] Complete")
