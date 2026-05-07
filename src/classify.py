"""
Stage 6 - Object Classification
================================
Input  : Object crop (np.ndarray BGR) from VOC bboxes or panorama segments
Output : Predicted class label + confidence
Dataset: PASCAL VOC 2012 (JPEGImages + Annotations XML)

Training uses LinearSVC (no Platt / probability) for speed.
Confidence during normal inference = softmax over decision_function scores.
Optional --demo-proba refits SVC(probability=True) on a small subset for true proba demo.
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC, SVC
from skimage.feature import hog

_SRC = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC)


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)


def extract_hog_from_bgr(bgr: np.ndarray, resize: int = 128) -> np.ndarray:
    """Fixed-size HOG descriptor from a BGR crop."""
    if bgr is None or bgr.size == 0:
        raise ValueError("Empty crop")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (resize, resize), interpolation=cv2.INTER_AREA)
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )


def parse_voc_xml(xml_path: str) -> Tuple[str, List[Tuple[str, int, int, int, int]]]:
    """
    Returns (image_filename, list of (class_name, xmin, ymin, xmax, ymax)) for non-difficult objects.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename_el = root.find("filename")
    if filename_el is None or not filename_el.text:
        raise ValueError(f"Missing filename in {xml_path}")
    filename = filename_el.text.strip()
    boxes: List[Tuple[str, int, int, int, int]] = []
    for obj in root.findall("object"):
        diff = obj.find("difficult")
        if diff is not None and diff.text == "1":
            continue
        name_el = obj.find("name")
        if name_el is None or not name_el.text:
            continue
        cls = name_el.text.strip()
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = int(float(box.findtext("xmin", "0")))
        ymin = int(float(box.findtext("ymin", "0")))
        xmax = int(float(box.findtext("xmax", "0")))
        ymax = int(float(box.findtext("ymax", "0")))
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append((cls, xmin, ymin, xmax, ymax))
    return filename, boxes


def read_image_list(list_path: str) -> List[str]:
    with open(list_path, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    return [ln.split()[0].strip() for ln in lines if ln.strip()]


def collect_voc_crops(
    voc_root: str,
    image_ids: Sequence[str],
    max_crops: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build feature matrix X and labels y (strings) from VOC crops.
    """
    jpg_dir = os.path.join(voc_root, "JPEGImages")
    ann_dir = os.path.join(voc_root, "Annotations")
    feats: List[np.ndarray] = []
    labels: List[str] = []
    meta: List[str] = []

    n = 0
    for stem in image_ids:
        if max_crops is not None and n >= max_crops:
            break
        xml_path = os.path.join(ann_dir, f"{stem}.xml")
        jpg_path = os.path.join(jpg_dir, f"{stem}.jpg")
        if not os.path.isfile(xml_path) or not os.path.isfile(jpg_path):
            continue
        try:
            fname, boxes = parse_voc_xml(xml_path)
        except Exception:
            continue
        img = cv2.imread(jpg_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        for cls, xmin, ymin, xmax, ymax in boxes:
            if max_crops is not None and n >= max_crops:
                break
            xmin = max(0, min(xmin, w - 1))
            xmax = max(0, min(xmax, w))
            ymin = max(0, min(ymin, h - 1))
            ymax = max(0, min(ymax, h))
            if xmax <= xmin + 2 or ymax <= ymin + 2:
                continue
            crop = img[ymin:ymax, xmin:xmax]
            if crop.size == 0 or crop.shape[0] * crop.shape[1] < 400:
                continue
            try:
                fvec = extract_hog_from_bgr(crop)
            except Exception:
                continue
            feats.append(fvec)
            labels.append(cls)
            meta.append(f"{stem}:{cls}")
            n += 1

    if not feats:
        raise RuntimeError("No VOC crops collected. Check voc_root and ImageSets.")
    return np.vstack(feats), np.array(labels), meta


@dataclass
class ClassifierBundle:
    clf: LinearSVC
    encoder: LabelEncoder
    demo_svc: Optional[SVC] = None


def train_linear_svc(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
) -> Tuple[LinearSVC, LabelEncoder]:
    enc = LabelEncoder()
    y_enc = enc.fit_transform(y)
    clf = LinearSVC(C=C, max_iter=5000, dual=False, random_state=42)
    clf.fit(X, y_enc)
    return clf, enc


def predict_label_confidence(
    clf: LinearSVC,
    encoder: LabelEncoder,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Label + confidence from decision scores (softmax), no Platt scaling."""
    scores = clf.decision_function(X)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    prob = softmax(scores, axis=1)
    idx = np.argmax(prob, axis=1)
    conf = prob[np.arange(len(idx)), idx]
    return encoder.inverse_transform(idx), conf.astype(np.float64)


def fit_demo_proba_svc(
    X: np.ndarray,
    y_enc: np.ndarray,
    max_samples: int = 2500,
    random_state: int = 42,
) -> SVC:
    """Small subset + RBF SVC with probability=True for demo only (slower)."""
    rng = np.random.default_rng(random_state)
    n = len(X)
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        Xs, ys = X[idx], y_enc[idx]
    else:
        Xs, ys = X, y_enc
    return SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=random_state,
    ).fit(Xs, ys)


def predict_demo_proba(
    demo_svc: SVC,
    encoder: LabelEncoder,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    prob = demo_svc.predict_proba(X)
    idx = np.argmax(prob, axis=1)
    conf = prob[np.arange(len(idx)), idx]
    return encoder.inverse_transform(demo_svc.classes_[idx]), conf.astype(np.float64)


def crops_from_panorama_segments(
    bgr: np.ndarray,
    label_mask: np.ndarray,
    min_area: int = 800,
    max_crops: int = 12,
) -> List[np.ndarray]:
    """Rough crops from K-means style label mask (one bbox per segment id)."""
    crops: List[np.ndarray] = []
    h, w = bgr.shape[:2]
    for cid in np.unique(label_mask):
        m = label_mask == cid
        if m.sum() < min_area:
            continue
        ys, xs = np.where(m)
        ymin, ymax = int(ys.min()), int(ys.max()) + 1
        xmin, xmax = int(xs.min()), int(xs.max()) + 1
        pad = 2
        ymin, ymax = max(0, ymin - pad), min(h, ymax + pad)
        xmin, xmax = max(0, xmin - pad), min(w, xmax + pad)
        crop = bgr[ymin:ymax, xmin:xmax]
        if crop.size > 0:
            crops.append(crop)
        if len(crops) >= max_crops:
            break
    return crops


def _find_train_val_lists(voc_root: str) -> Tuple[str, str]:
    """Prefer Main/train.txt; fall back to Segmentation splits if Main not extracted."""
    main_dir = os.path.join(voc_root, "ImageSets", "Main")
    seg_dir = os.path.join(voc_root, "ImageSets", "Segmentation")
    train_main = os.path.join(main_dir, "train.txt")
    val_main = os.path.join(main_dir, "val.txt")
    if os.path.isfile(train_main):
        return train_main, val_main if os.path.isfile(val_main) else train_main
    train_seg = os.path.join(seg_dir, "train.txt")
    val_seg = os.path.join(seg_dir, "val.txt")
    tv_seg = os.path.join(seg_dir, "trainval.txt")
    if os.path.isfile(train_seg) and os.path.isfile(val_seg):
        return train_seg, val_seg
    if os.path.isfile(tv_seg):
        return tv_seg, tv_seg
    raise FileNotFoundError(
        f"No VOC image lists found under {voc_root}/ImageSets (Main or Segmentation)."
    )


def run_stage6(
    voc_root: str,
    output_dir: str,
    max_train_crops: int = 8000,
    max_val_crops: int = 2000,
    C: float = 1.0,
    demo_proba: bool = False,
    demo_proba_samples: int = 2500,
    panorama_paths: Optional[List[str]] = None,
    pano_segments: int = 6,
    panorama_limit: int = 1,
) -> None:
    """
    Stage 6 pipeline (callable from main). ``voc_root`` and ``output_dir`` should be absolute paths.
    """
    voc_root = os.path.abspath(voc_root)
    output_dir = os.path.abspath(output_dir)
    ann_dir = os.path.join(voc_root, "Annotations")
    if not os.path.isdir(ann_dir):
        raise FileNotFoundError(
            f"Stage 6 needs VOC Annotations XML under '{ann_dir}'. "
            "Extract Annotations from the official VOC trainval tar."
        )

    train_list, val_list = _find_train_val_lists(voc_root)
    if train_list == val_list:
        print("[Stage 6] Using single list; splitting 80/20 by stem order for train/val.")
        all_ids = read_image_list(train_list)
        cut = int(len(all_ids) * 0.8)
        train_ids, val_ids = all_ids[:cut], all_ids[cut:]
    else:
        train_ids = read_image_list(train_list)
        val_ids = read_image_list(val_list) if os.path.isfile(val_list) else train_ids[:500]

    print(f"[Stage 6] Collecting up to {max_train_crops} train crops...")
    X_train, y_train, _ = collect_voc_crops(voc_root, train_ids, max_crops=max_train_crops)
    print(f"[Stage 6] Train: {X_train.shape[0]} crops, {len(np.unique(y_train))} classes")

    clf, enc = train_linear_svc(X_train, y_train, C=C)
    bundle = ClassifierBundle(clf=clf, encoder=enc, demo_svc=None)

    os.makedirs(output_dir, exist_ok=True)

    if demo_proba:
        print(f"[Stage 6] Fitting demo SVC(probability=True) on <= {demo_proba_samples} samples...")
        y_enc_train = enc.transform(y_train)
        bundle.demo_svc = fit_demo_proba_svc(
            X_train, y_enc_train, max_samples=demo_proba_samples
        )

    print(f"[Stage 6] Collecting up to {max_val_crops} val crops...")
    X_val, y_val, meta_val = collect_voc_crops(voc_root, val_ids, max_crops=max_val_crops)

    if bundle.demo_svc is not None:
        pred, conf = predict_demo_proba(bundle.demo_svc, enc, X_val)
    else:
        pred, conf = predict_label_confidence(clf, enc, X_val)

    acc = accuracy_score(y_val, pred)
    print(f"[Stage 6] Val accuracy: {acc:.4f}")
    print(classification_report(y_val, pred, zero_division=0))

    rows = []
    for i in range(len(y_val)):
        rows.append(
            {
                "true": y_val[i],
                "pred": pred[i],
                "confidence": round(float(conf[i]), 4),
                "meta": meta_val[i] if i < len(meta_val) else "",
            }
        )
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "classification_val.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Stage 6] Saved -> {csv_path}")

    model_path = os.path.join(output_dir, "classifier_stage6.joblib")
    joblib.dump(
        {"linear_svc": clf, "encoder": enc, "demo_svc": bundle.demo_svc},
        model_path,
    )
    print(f"[Stage 6] Model saved -> {model_path}")

    if panorama_paths:
        from segment import segment_image_kmeans

        for pano_path in panorama_paths[: max(1, panorama_limit)]:
            pano_path = os.path.abspath(pano_path)
            img = cv2.imread(pano_path)
            if img is None:
                print(f"[Stage 6] Could not load panorama: {pano_path}")
                continue
            _, lbl = segment_image_kmeans(img, n_segments=pano_segments)
            crops = crops_from_panorama_segments(img, lbl)
            if not crops:
                print(f"[Stage 6] No segment crops from: {pano_path}")
                continue
            print(f"[Stage 6] Panorama segment predictions ({os.path.basename(pano_path)}):")
            Xp = np.vstack([extract_hog_from_bgr(c) for c in crops])
            if bundle.demo_svc is not None:
                plab, pconf = predict_demo_proba(bundle.demo_svc, enc, Xp)
            else:
                plab, pconf = predict_label_confidence(clf, enc, Xp)
            for i, (lab, co) in enumerate(zip(plab, pconf)):
                print(f"  segment[{i}]: {lab}  (conf={co:.3f})")


def run_stage6_cli(args: argparse.Namespace) -> None:
    run_stage6(
        voc_root=_resolve(args.voc_root),
        output_dir=_resolve(args.output_dir),
        max_train_crops=args.max_train_crops,
        max_val_crops=args.max_val_crops,
        C=args.C,
        demo_proba=args.demo_proba,
        demo_proba_samples=args.demo_proba_samples,
        panorama_paths=[_resolve(args.panorama)] if args.panorama else None,
        pano_segments=args.pano_segments,
        panorama_limit=1,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Stage 6 - VOC object classification (HOG + SVM)")
    p.add_argument("--voc-root", default="../data/voc/VOCdevkit/VOC2012")
    p.add_argument("--output-dir", default="outputs/classification")
    p.add_argument("--max-train-crops", type=int, default=8000, help="Cap training crops for speed")
    p.add_argument("--max-val-crops", type=int, default=2000)
    p.add_argument("--C", type=float, default=1.0, help="LinearSVC regularization")
    p.add_argument(
        "--demo-proba",
        action="store_true",
        help="After fast LinearSVC, fit SVC(probability=True) on a small subset for demo",
    )
    p.add_argument("--demo-proba-samples", type=int, default=2500)
    p.add_argument("--panorama", default=None, help="Optional panorama path for segment-based demo")
    p.add_argument("--pano-segments", type=int, default=6)
    return p.parse_args()


if __name__ == "__main__":
    print("=" * 55)
    print("  Stage 6 - Object Classification")
    print("=" * 55)
    run_stage6_cli(parse_args())
    print("\n[Stage 6] Complete")