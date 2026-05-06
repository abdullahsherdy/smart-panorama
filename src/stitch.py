"""
Stage 4 — Homography & Stitching
Smart Panorama & Object Recognition System
============================================
Input   : ``good_matches``, ``kp_a``, ``kp_b``, ``color_a``, ``color_b`` (BGR uint8)
          from Stages 2–3; matches use ``queryIdx`` → A, ``trainIdx`` → B (Stage 3).
Output  : Stitched panorama ``np.ndarray`` BGR uint8, or ``None`` if stitching fails.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from match import find_preprocessed_pair_ids, run_pair

# ──────────────────────────────────────────────────────────────────────────────
# CORE
# ──────────────────────────────────────────────────────────────────────────────


def compute_homography(
    good_matches: List[cv2.DMatch],
    kp_a: List[cv2.KeyPoint],
    kp_b: List[cv2.KeyPoint],
    ransac_reproj_threshold: float = 5.0,
    min_matches: int = 8,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estimate homography mapping points in image A into image B's coordinate frame.

    Returns
    -------
    H : 3x3 ``float64`` or ``None``
        Such that ``[x',y',w']^T = H [x,y,1]^T`` maps A → B (``cv2.warpPerspective``).
    mask : inlier mask from RANSAC, or ``None``.
    """
    if len(good_matches) < min_matches:
        return None, None

    src_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        ransac_reproj_threshold,
        maxIters=2000,
        confidence=0.995,
    )
    return H, mask


def _canvas_corners(
    h_a: int,
    w_a: int,
    h_b: int,
    w_b: int,
    H_a_to_b: np.ndarray,
) -> Tuple[int, int, np.ndarray]:
    """
    Build translation ``T`` and output size so both warped A (via ``T @ H``)
    and translated B (via ``T``) fit in a single canvas with origin at the
    top-left of the union bounding box (B stays in its own plane; only canvas shifts).
    """
    corners_a = np.float32([[0, 0], [w_a, 0], [w_a, h_a], [0, h_a]]).reshape(-1, 1, 2)
    warped_a = cv2.perspectiveTransform(corners_a, H_a_to_b.astype(np.float64))

    corners_b = np.float32([[0, 0], [w_b, 0], [w_b, h_b], [0, h_b]]).reshape(-1, 1, 2)

    pts = np.vstack((warped_a.reshape(-1, 2), corners_b.reshape(-1, 2)))
    xmin, ymin = np.floor(pts.min(axis=0)).astype(np.float64)
    xmax, ymax = np.ceil(pts.max(axis=0)).astype(np.float64)

    pad = 1.0
    xmin -= pad
    ymin -= pad
    xmax += pad
    ymax += pad

    out_w = int(max(1, xmax - xmin))
    out_h = int(max(1, ymax - ymin))

    T = np.array(
        [[1.0, 0.0, -xmin], [0.0, 1.0, -ymin], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return out_w, out_h, T


def _linear_blend(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Blend two BGR images of the same shape; prefer pixels with higher alpha weight."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    ma = ga > 1
    mb = gb > 1
    overlap = ma & mb
    only_a = ma & ~mb
    only_b = mb & ~ma

    out = np.zeros_like(a)
    out[only_a] = a[only_a]
    out[only_b] = b[only_b]

    if overlap.any():
        wa = a[overlap].astype(np.float32)
        wb = b[overlap].astype(np.float32)
        out[overlap] = (0.5 * wa + 0.5 * wb).astype(np.uint8)

    return out


def stitch_pair(
    color_a: np.ndarray,
    color_b: np.ndarray,
    good_matches: List[cv2.DMatch],
    kp_a: List[cv2.KeyPoint],
    kp_b: List[cv2.KeyPoint],
    ransac_reproj_threshold: float = 5.0,
    min_matches: int = 8,
) -> Optional[np.ndarray]:
    """
    Warp ``color_a`` into ``color_b``'s plane and blend with ``color_b``.

    Images must be BGR ``uint8``. Convention matches Stage 3: query = A, train = B.
    """
    if color_a.ndim != 3 or color_b.ndim != 3:
        raise ValueError("[stitch_pair] Expected BGR colour images with shape (H, W, 3)")

    H, _mask = compute_homography(
        good_matches,
        kp_a,
        kp_b,
        ransac_reproj_threshold=ransac_reproj_threshold,
        min_matches=min_matches,
    )
    if H is None:
        return None

    h_a, w_a = color_a.shape[:2]
    h_b, w_b = color_b.shape[:2]

    out_w, out_h, T = _canvas_corners(h_a, w_a, h_b, w_b, H)
    H_out = (T @ H).astype(np.float64)

    warped_a = cv2.warpPerspective(color_a, H_out, (out_w, out_h))
    warped_b = cv2.warpPerspective(color_b, T.astype(np.float32), (out_w, out_h))

    panorama = _linear_blend(warped_a, warped_b)
    return crop_black_border(panorama)


def crop_black_border(img: np.ndarray, thresh: int = 1) -> np.ndarray:
    """Trim near-black borders after warping."""
    if img.ndim != 3:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y : y + h, x : x + w]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Stage 4 — Homography & Stitching")
    p.add_argument(
        "--preprocessed",
        default="../outputs/preprocessed",
        help="Folder with Stage 1 *.npy outputs",
    )
    p.add_argument(
        "--output",
        default="../outputs/panorama_001.jpg",
        help="Output panorama path",
    )
    p.add_argument("--pair-id", default=None)
    p.add_argument("--nfeatures", type=int, default=5000)
    p.add_argument("--ratio", type=float, default=0.75)
    p.add_argument("--ransac", type=float, default=5.0, help="RANSAC reprojection threshold (px)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 55)
    print("  Stage 4 — Homography & Stitching")
    print("=" * 55)

    pre = os.path.abspath(args.preprocessed)
    ids = find_preprocessed_pair_ids(pre)
    if not ids:
        raise SystemExit(
            f"[Stage 4] No A/B gray pair .npy files in '{pre}'. Run Stage 1 first."
        )

    pair_id = args.pair_id if args.pair_id else ids[0]
    if pair_id not in ids:
        raise SystemExit(f"[Stage 4] Unknown pair_id '{pair_id}'. First available: {ids[0]!r}")

    kp_a, des_a, kp_b, des_b, good, color_a, color_b = run_pair(
        pre, pair_id, nfeatures=args.nfeatures, ratio=args.ratio
    )

    if color_a is None or color_b is None:
        raise SystemExit(
            "[Stage 4] Missing *_a_color.npy / *_b_color.npy. "
            "Re-run Stage 1 with .npy saving enabled."
        )

    pano = stitch_pair(
        color_a,
        color_b,
        good,
        kp_a,
        kp_b,
        ransac_reproj_threshold=args.ransac,
    )

    if pano is None:
        print(f"  Pair {pair_id}: homography failed ({len(good)} good matches).")
        raise SystemExit(1)

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, pano)
    print(f"  Pair {pair_id}: {len(good)} matches → saved {out_path}  shape={pano.shape}")
    print("\n[Stage 4] Complete")
