"""
Stage 1 — Image Preprocessing
Smart Panorama & Object Recognition System
==========================================
Dataset : UDIS-D test split (data/udis/test/input1/ + data/udis/test/input2/)
Input   : Raw image pair paths
Output  : (color_img, gray_img) as np.ndarray + saved .npy files

Preprocessing Pipeline
-----------------------
1. Load image
2. Gaussian Blur  → cv2.GaussianBlur(img, (5,5), 1.0)
3. Median Filter  → cv2.medianBlur(img, 3)
4. CLAHE          → lighting normalisation on L channel (LAB space)
5. Grayscale copy → for SIFT / Harris detector input
6. Colour copy    → for stitching
"""

import os
import glob
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# CORE FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(img_path: str) -> tuple:
    """
    Load and preprocess a single image.

    Parameters
    ----------
    img_path : str  →  path to raw input image (JPEG / PNG)

    Returns
    -------
    color_img : np.ndarray  (H, W, 3) uint8 BGR  →  used for stitching
    gray_img  : np.ndarray  (H, W)    uint8      →  used for feature detection
    """

    # Step 1 — Load
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"[preprocess] Cannot load: '{img_path}'")

    # Step 2 — Gaussian Blur (noise reduction)
    img = cv2.GaussianBlur(img, (5, 5), 1.0)

    # Step 3 — Median Filter (salt-and-pepper removal)
    img = cv2.medianBlur(img, 3)

    # Step 4 — CLAHE on Luminance channel only (lighting normalisation)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch  = clahe.apply(l_ch)
    lab   = cv2.merge([l_ch, a_ch, b_ch])
    color_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Step 5 — Grayscale copy
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    return color_img, gray_img


# ──────────────────────────────────────────────────────────────────────────────
# BATCH PROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def run_batch(input1_dir: str,
              input2_dir: str,
              output_dir: str,
              max_pairs:  int  = 100,
              save_npy:   bool = True) -> list:
    """
    Process up to max_pairs UDIS-D image pairs.

    Parameters
    ----------
    input1_dir : str   →  UDIS-D input1/ folder (left images)
    input2_dir : str   →  UDIS-D input2/ folder (right images)
    output_dir : str   →  where to save .npy files
    max_pairs  : int   →  maximum pairs to process (default 50)
    save_npy   : bool  →  save .npy arrays to disk

    Returns
    -------
    results : list of dict
        Keys: pair_id, path_a, path_b, color_a, gray_a, color_b, gray_b
    """
    os.makedirs(output_dir, exist_ok=True)

    exts   = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    files1 = sorted(f for ext in exts for f in glob.glob(os.path.join(input1_dir, ext)))
    files2 = sorted(f for ext in exts for f in glob.glob(os.path.join(input2_dir, ext)))

    if not files1 or not files2:
        raise RuntimeError(
            f"[run_batch] No images found!\n"
            f"  input1: {input1_dir} → {len(files1)} files\n"
            f"  input2: {input2_dir} → {len(files2)} files"
        )

    n_pairs = min(len(files1), len(files2), max_pairs)
    print(f"[run_batch] Processing {n_pairs} pairs ...")

    results = []
    for i in range(n_pairs):
        pair_id = os.path.splitext(os.path.basename(files1[i]))[0]

        color_a, gray_a = preprocess(files1[i])
        color_b, gray_b = preprocess(files2[i])

        results.append({
            "pair_id": pair_id,
            "path_a":  files1[i],
            "path_b":  files2[i],
            "color_a": color_a,
            "gray_a":  gray_a,
            "color_b": color_b,
            "gray_b":  gray_b,
        })

        if save_npy:
            np.save(os.path.join(output_dir, f"{pair_id}_a_color.npy"), color_a)
            np.save(os.path.join(output_dir, f"{pair_id}_a_gray.npy"),  gray_a)
            np.save(os.path.join(output_dir, f"{pair_id}_b_color.npy"), color_b)
            np.save(os.path.join(output_dir, f"{pair_id}_b_gray.npy"),  gray_b)

        if (i + 1) % 10 == 0 or (i + 1) == n_pairs:
            print(f"  Processed {i+1}/{n_pairs} pairs")

    print(f"[run_batch] Done. Saved to '{output_dir}'")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────

def visualise_samples(results:    list,
                      n_samples:  int = 5,
                      output_dir: str = "outputs/preprocessed") -> None:
    """
    Save before/after figures for n_samples pairs.

    Layout (per figure):
    Row 0: Raw A | Processed A | Raw B | Processed B   (colour)
    Row 1:        | Grayscale A |       | Grayscale B   (gray)
    """
    vis_dir = os.path.join(output_dir, "visualisations")
    os.makedirs(vis_dir, exist_ok=True)

    n_samples = min(n_samples, len(results))
    indices   = np.linspace(0, len(results) - 1, n_samples, dtype=int)

    for idx in indices:
        rec     = results[idx]
        pair_id = rec["pair_id"]
        raw_a   = cv2.imread(rec["path_a"])
        raw_b   = cv2.imread(rec["path_b"])

        def to_rgb(bgr):
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(2, 4, figsize=(20, 9))
        fig.suptitle(f"Pair {pair_id}  |  Before vs After Preprocessing",
                     fontsize=14, fontweight="bold")

        top_imgs   = [to_rgb(raw_a), to_rgb(rec["color_a"]),
                      to_rgb(raw_b), to_rgb(rec["color_b"])]
        top_titles = ["Raw (Image A)", "Preprocessed (Image A)",
                      "Raw (Image B)", "Preprocessed (Image B)"]
        bot_imgs   = [None, rec["gray_a"], None, rec["gray_b"]]
        bot_titles = ["", "Grayscale A (detector input)",
                      "", "Grayscale B (detector input)"]

        for col in range(4):
            axes[0, col].imshow(top_imgs[col])
            axes[0, col].set_title(top_titles[col], fontsize=10, fontweight="bold")
            axes[0, col].axis("off")

            if bot_imgs[col] is not None:
                axes[1, col].imshow(bot_imgs[col], cmap="gray")
                axes[1, col].set_title(bot_titles[col], fontsize=9)
            axes[1, col].axis("off")

        plt.tight_layout()
        save_path = os.path.join(vis_dir, f"pair_{pair_id}_before_after.png")
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [vis] Saved → {save_path}")

    print(f"[visualise_samples] {n_samples} figures saved to '{vis_dir}'")


# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM SCENES
# ──────────────────────────────────────────────────────────────────────────────

# def preprocess_custom_scenes(scenes_root: str, output_dir: str) -> None:
#     """
#     Preprocess all images inside scenes_root/<scene_name>/ sub-folders.

#     Expected layout
#     ---------------
#     data/custom_photos/
#         scene_01/   img_001.jpg  img_002.jpg  ...
#         scene_02/   ...
#         scene_03/   ...
#     """
#     scene_dirs = sorted(
#         d for d in glob.glob(os.path.join(scenes_root, "*"))
#         if os.path.isdir(d)
#     )

#     if not scene_dirs:
#         print(f"[custom_scenes] No sub-folders found in '{scenes_root}'. Skipping.")
#         return

#     print(f"[custom_scenes] Found {len(scene_dirs)} scene(s).")
#     for scene_dir in scene_dirs:
#         scene_name = os.path.basename(scene_dir)
#         out_scene  = os.path.join(output_dir, "custom", scene_name)
#         os.makedirs(out_scene, exist_ok=True)

#         imgs = sorted(
#             f for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
#             for f in glob.glob(os.path.join(scene_dir, ext))
#         )
#         if not imgs:
#             print(f"  [custom_scenes] '{scene_name}': no images. Skipping.")
#             continue

#         print(f"  [custom_scenes] '{scene_name}': {len(imgs)} image(s).")
#         for img_path in imgs:
#             stem = os.path.splitext(os.path.basename(img_path))[0]
#             color, gray = preprocess(img_path)
#             np.save(os.path.join(out_scene, f"{stem}_color.npy"), color)
#             np.save(os.path.join(out_scene, f"{stem}_gray.npy"),  gray)

#     print(f"[custom_scenes] Done. Saved to '{output_dir}/custom/'")


# ──────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Stage 1 — Image Preprocessing")
    p.add_argument("--input1",  default="data/udis/test/input1")
    p.add_argument("--input2",  default="data/udis/test/input2")
    p.add_argument("--output",  default="outputs/preprocessed")
    p.add_argument("--pairs",   type=int, default=50)
    p.add_argument("--vis",     type=int, default=5)
    p.add_argument("--custom",  default=None)
    p.add_argument("--no-npy",  action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 55)
    print("  Stage 1 — Image Preprocessing")
    print("=" * 55)

    results = run_batch(
        input1_dir = args.input1,
        input2_dir = args.input2,
        output_dir = args.output,
        max_pairs  = args.pairs,
        save_npy   = not args.no_npy
        )

    if args.vis > 0:
        visualise_samples(results, n_samples=args.vis, output_dir=args.output)

    # custom scenes preprocessing(cancelled)
    # if args.custom:
    #     preprocess_custom_scenes(args.custom, args.output)

    print("\n[Stage 1] Complete ")
