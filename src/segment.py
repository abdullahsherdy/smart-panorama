"""
Stage 5 - Panorama Segmentation + VOC IoU Evaluation
====================================================
Input   : Panorama from Stage 4 (any RGB/BGR image)
Output  : Segmented image, label mask, overlay, IoU scores
Dataset : Panorama output (UDIS-D) + PASCAL VOC 2012
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC_DIR)

# VOC snippet from `helpers/download_voc_v2.py` lives under repo `data/voc/…` (≤100 image IDs).
VOC_PROJECT_ROOT_REL = os.path.join("data", "voc", "VOC2012_subset_300")
VOC_EVAL_MAX_IMAGE_IDS = 300


def _resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)


def _list_images(folder: str) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


def _resize_for_compute(img: np.ndarray, max_side: int = 900) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return img, 1.0
    scale = max_side / float(side)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return small, scale


def _build_features(bgr: np.ndarray, spatial_weight: float = 0.35) -> np.ndarray:
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ys, xs = np.indices((h, w), dtype=np.float32)
    xs /= max(1.0, float(w - 1))
    ys /= max(1.0, float(h - 1))
    xy = np.stack([xs, ys], axis=-1) * (255.0 * spatial_weight)
    feats = np.concatenate([lab, xy], axis=-1)
    return feats.reshape(-1, 5).astype(np.float32)


def segment_image_kmeans(
    image_bgr: np.ndarray,
    n_segments: int = 6,
    attempts: int = 5,
    max_side: int = 900,
    spatial_weight: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment image with K-means over LAB+XY features.

    Returns
    -------
    segmented_bgr : (H, W, 3) uint8
    label_mask    : (H, W) uint8 cluster ids
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty input image.")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected image shape (H, W, 3).")

    small, _ = _resize_for_compute(image_bgr, max_side=max_side)
    hs, ws = small.shape[:2]

    data = _build_features(small, spatial_weight=spatial_weight)
    k = int(max(2, n_segments))

    # Tightened criteria for faster/cleaner convergence.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    _compact, labels, _centers = cv2.kmeans(
        data,
        k,
        None,
        criteria,
        int(max(1, attempts)),
        cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.reshape(hs, ws).astype(np.uint8)

    segmented_small = np.zeros_like(small)
    for cid in range(k):
        m = labels == cid
        if np.any(m):
            segmented_small[m] = np.mean(small[m], axis=0).astype(np.uint8)

    if (hs, ws) != image_bgr.shape[:2]:
        label_mask = cv2.resize(
            labels,
            (image_bgr.shape[1], image_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        segmented = cv2.resize(
            segmented_small,
            (image_bgr.shape[1], image_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        label_mask = labels
        segmented = segmented_small

    return segmented.astype(np.uint8), label_mask.astype(np.uint8)


def _save_segmentation_outputs(
    src_image: np.ndarray,
    segmented: np.ndarray,
    label_mask: np.ndarray,
    output_dir: str,
    base_name: str,
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    seg_path = os.path.join(output_dir, f"{base_name}_segmented.png")
    mask_path = os.path.join(output_dir, f"{base_name}_label_mask.png")
    overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")

    cv2.imwrite(seg_path, segmented)
    cv2.imwrite(mask_path, label_mask)

    edges = cv2.Canny(label_mask, 30, 90)
    overlay = src_image.copy()
    overlay[edges > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(src_image, 0.75, overlay, 0.25, 0)
    cv2.imwrite(overlay_path, overlay)

    return {"segmented": seg_path, "label_mask": mask_path, "overlay": overlay_path}


def segment_image_file(
    image_path: str,
    output_dir: str,
    n_segments: int = 6,
    spatial_weight: float = 0.35,
    prefix: Optional[str] = None,
) -> Dict[str, str]:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    stem = os.path.splitext(os.path.basename(image_path))[0]
    base = f"{prefix}_{stem}" if prefix else stem
    seg, mask = segment_image_kmeans(
        img,
        n_segments=n_segments,
        attempts=5,
        spatial_weight=spatial_weight,
    )
    out = _save_segmentation_outputs(img, seg, mask, output_dir, base)
    return {"image": image_path, **out}


def segment_panorama_file(
    panorama_path: str,
    output_dir: str,
    n_segments: int = 6,
    spatial_weight: float = 0.35,
) -> Dict[str, str]:
    rec = segment_image_file(
        image_path=panorama_path,
        output_dir=output_dir,
        n_segments=n_segments,
        spatial_weight=spatial_weight,
        prefix=None,
    )
    return {
        "panorama": rec["image"],
        "segmented": rec["segmented"],
        "label_mask": rec["label_mask"],
        "overlay": rec["overlay"],
    }


def _background_clusters_from_border(
    label_mask: np.ndarray, border_frac: float = 0.06, ratio: float = 0.5
) -> List[int]:
    """Return all clusters whose pixel share on the image border is >= ``ratio``."""
    h, w = label_mask.shape[:2]
    t = max(1, int(round(min(h, w) * border_frac)))
    border = np.zeros((h, w), dtype=bool)
    border[:t, :] = True
    border[-t:, :] = True
    border[:, :t] = True
    border[:, -t:] = True

    n_clusters = int(label_mask.max()) + 1
    border_counts = np.bincount(label_mask[border].reshape(-1), minlength=n_clusters)
    total_counts = np.bincount(label_mask.reshape(-1), minlength=n_clusters)

    bg = []
    for cid in range(n_clusters):
        if total_counts[cid] == 0:
            continue
        if border_counts[cid] / float(total_counts[cid]) >= ratio:
            bg.append(cid)
    if not bg:
        bg.append(int(np.argmax(border_counts)))
    return bg


def _cleanup_binary_mask(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    k = max(3, int(ksize) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned.astype(np.uint8)


def _keep_large_components(mask: np.ndarray, min_area_frac: float = 0.005) -> np.ndarray:
    """Drop tiny connected components (< ``min_area_frac`` of image area)."""
    h, w = mask.shape[:2]
    min_area = max(50, int(round(h * w * min_area_frac)))
    n_labels, lbls, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros_like(mask, dtype=np.uint8)
    for cid in range(1, n_labels):
        if stats[cid, cv2.CC_STAT_AREA] >= min_area:
            keep[lbls == cid] = 1
    return keep


def _foreground_from_clusters(label_mask: np.ndarray, background_mode: str = "border") -> np.ndarray:
    mode = background_mode.strip().lower()
    if mode == "border":
        bg_ids = _background_clusters_from_border(label_mask)
        fg = np.ones_like(label_mask, dtype=np.uint8)
        for bg in bg_ids:
            fg[label_mask == bg] = 0
    elif mode == "largest":
        flat = label_mask.reshape(-1)
        counts = np.bincount(flat, minlength=int(label_mask.max()) + 1)
        bg = int(np.argmax(counts))
        fg = (label_mask != bg).astype(np.uint8)
    else:
        raise ValueError(f"Unknown background_mode: {background_mode}")
    fg = _cleanup_binary_mask(fg, ksize=5)
    return _keep_large_components(fg, min_area_frac=0.005)


def _binary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def _read_voc_mask_indices(mask_path: str) -> np.ndarray:
    """
    Read VOC SegmentationClass palette PNG as class index map.
    PIL preserves palette indices directly (critical for VOC labels).
    """
    with Image.open(mask_path) as im:
        return np.array(im)


def evaluate_on_voc_binary_iou(
    voc_root: str,
    n_samples: int = 20,
    n_segments: int = 6,
    spatial_weight: float = 0.35,
    background_mode: str = "border",
) -> pd.DataFrame:
    jpg_dir = os.path.join(voc_root, "JPEGImages")
    seg_dir = os.path.join(voc_root, "SegmentationClass")
    if not os.path.isdir(jpg_dir) or not os.path.isdir(seg_dir):
        raise FileNotFoundError(f"VOC folders missing under: {voc_root}")

    jpgs = sorted(glob.glob(os.path.join(jpg_dir, "*.jpg")))
    pairs: List[Tuple[str, str, str]] = []
    for jpg in jpgs:
        stem = os.path.splitext(os.path.basename(jpg))[0]
        gt_path = os.path.join(seg_dir, f"{stem}.png")
        if os.path.isfile(gt_path):
            pairs.append((stem, jpg, gt_path))

    if not pairs:
        raise RuntimeError("No matched VOC JPEG/mask pairs found.")

    pairs = pairs[: max(1, int(n_samples))]
    rows: List[Dict[str, object]] = []

    for stem, jpg_path, gt_path in pairs:
        img = cv2.imread(jpg_path)
        if img is None:
            continue
        gt = _read_voc_mask_indices(gt_path)
        if gt.shape[:2] != img.shape[:2]:
            gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        _seg, lbl = segment_image_kmeans(
            img,
            n_segments=n_segments,
            attempts=5,
            spatial_weight=spatial_weight,
        )
        pred_fg = _foreground_from_clusters(lbl, background_mode=background_mode)
        gt_fg = np.logical_and(gt != 0, gt != 255).astype(np.uint8)
        iou = _binary_iou(pred_fg, gt_fg)
        rows.append({"id": stem, "iou": round(iou, 4)})

    return pd.DataFrame(rows)


def segment_dataset_folder(
    dataset_dir: str,
    output_dir: str,
    n_segments: int = 6,
    spatial_weight: float = 0.35,
    max_images: int = 20,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Missing folder: {dataset_dir}")
    images = _list_images(dataset_dir)[: max(1, int(max_images))]
    if not images:
        raise RuntimeError(f"No images found in: {dataset_dir}")

    rows: List[Dict[str, str]] = []
    for path in images:
        rows.append(
            segment_image_file(
                image_path=path,
                output_dir=output_dir,
                n_segments=n_segments,
                spatial_weight=spatial_weight,
                prefix=prefix,
            )
        )
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser(description="Stage 5 - Segmentation (Panorama + VOC)")
    p.add_argument("--panorama", default="outputs/panorama_001.jpg")
    p.add_argument("--panorama-dir", default="outputs")
    p.add_argument("--panorama-samples", type=int, default=5)
    p.add_argument("--output-dir", default="outputs/segmentation")
    p.add_argument("--segments", type=int, default=6)
    p.add_argument("--spatial-weight", type=float, default=0.35)
    p.add_argument("--bg-mode", default="border", choices=["border", "largest"])
    p.add_argument(
        "--voc-root",
        default=VOC_PROJECT_ROOT_REL,
        help="VOC2012 root (repo data/voc/VOC2012_subset_300).",
    )
    p.add_argument(
        "--voc-samples",
        type=int,
        default=min(20, VOC_EVAL_MAX_IMAGE_IDS),
        help=f"VOC images to segment / IoU (≤{VOC_EVAL_MAX_IMAGE_IDS}).",
    )
    p.add_argument("--run-two-datasets", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output_dir = _resolve_path(args.output_dir)
    voc_root = _resolve_path(args.voc_root)
    voc_samples = min(max(1, int(args.voc_samples)), VOC_EVAL_MAX_IMAGE_IDS)

    print("=" * 55)
    print("  Stage 5 - Panorama Segmentation")
    print("=" * 55)

    if args.run_two_datasets:
        panorama_dir = _resolve_path(args.panorama_dir)
        pano_files = sorted(glob.glob(os.path.join(panorama_dir, "panorama_*.jpg")))
        pano_files = pano_files[: max(1, int(args.panorama_samples))]

        if pano_files:
            pano_rows: List[Dict[str, str]] = []
            pano_out = os.path.join(output_dir, "panorama_dataset")
            for pth in pano_files:
                pano_rows.append(
                    segment_image_file(
                        image_path=pth,
                        output_dir=pano_out,
                        n_segments=args.segments,
                        spatial_weight=args.spatial_weight,
                        prefix="pano",
                    )
                )
            pano_csv = os.path.join(output_dir, "panorama_segmentation_report.csv")
            pd.DataFrame(pano_rows).to_csv(pano_csv, index=False)
            print(f"[Stage 5] Panorama segmentation done: {len(pano_rows)} images")
            print(f"[Stage 5] Saved panorama report -> {pano_csv}")
        else:
            print(f"[Stage 5] No panorama_*.jpg found in '{panorama_dir}'")

        try:
            voc_seg_df = segment_dataset_folder(
                dataset_dir=os.path.join(voc_root, "JPEGImages"),
                output_dir=os.path.join(output_dir, "voc_dataset"),
                n_segments=args.segments,
                spatial_weight=args.spatial_weight,
                max_images=voc_samples,
                prefix="voc",
            )
            voc_seg_csv = os.path.join(output_dir, "voc_segmentation_report.csv")
            voc_seg_df.to_csv(voc_seg_csv, index=False)
            print(f"[Stage 5] VOC segmentation done: {len(voc_seg_df)} images")
            print(f"[Stage 5] Saved VOC report      -> {voc_seg_csv}")
        except Exception as e:
            print(f"[Stage 5] VOC segmentation skipped: {e}")

    else:
        panorama_path = _resolve_path(args.panorama)
        if not os.path.isfile(panorama_path):
            pano_dir = _resolve_path(args.panorama_dir)
            candidates = sorted(glob.glob(os.path.join(pano_dir, "panorama_*.jpg")))
            if not candidates:
                raise FileNotFoundError(
                    f"Panorama not found: {panorama_path} (and no panorama_*.jpg in {pano_dir})"
                )
            print(
                f"[Stage 5] {os.path.basename(panorama_path)} missing; "
                f"using {os.path.basename(candidates[0])} instead."
            )
            panorama_path = candidates[0]

        out = segment_panorama_file(
            panorama_path=panorama_path,
            output_dir=output_dir,
            n_segments=args.segments,
            spatial_weight=args.spatial_weight,
        )
        print(f"[Stage 5] Saved segmented image -> {out['segmented']}")
        print(f"[Stage 5] Saved label mask      -> {out['label_mask']}")
        print(f"[Stage 5] Saved overlay         -> {out['overlay']}")

    try:
        iou_df = evaluate_on_voc_binary_iou(
            voc_root=voc_root,
            n_samples=voc_samples,
            n_segments=args.segments,
            spatial_weight=args.spatial_weight,
            background_mode=args.bg_mode,
        )
        if len(iou_df) > 0:
            mean_iou = float(iou_df["iou"].mean())
            iou_csv = os.path.join(output_dir, "voc_binary_iou.csv")
            iou_df.to_csv(iou_csv, index=False)
            print(f"[Stage 5] VOC binary IoU mean ({len(iou_df)} imgs): {mean_iou:.4f}")
            print(f"[Stage 5] Saved IoU report    -> {iou_csv}")
        else:
            print("[Stage 5] VOC IoU skipped: no valid samples")
    except Exception as e:
        print(f"[Stage 5] VOC IoU evaluation skipped: {e}")

    print("\n[Stage 5] Complete")