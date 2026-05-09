"""
Stage 6 - Object Classification
================================
Input  : Object crop (np.ndarray BGR) from VOC bboxes or panorama segments
Output : Predicted class label + confidence
Dataset: PASCAL VOC 2012 trainval (JPEGImages + Annotations XML)

Trains and benchmarks several classical classifiers on HOG + LAB-color
features (Gaussian Naive Bayes, Logistic Regression, LinearSVC, RBF-SVC,
Random Forest) and saves the best-scoring one as the final classifier.
Confidence at inference time = softmax over decision_function scores
(or predict_proba when natively available).
"""
from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC
from skimage.feature import hog

_SRC = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC)

# Full PASCAL VOC 2012 trainval extracted under repo data/voc/VOCdevkit/...
VOC_PROJECT_ROOT_REL = os.path.join(
    "data", "voc", "VOCdevkit", "VOC2012_train_val", "VOC2012_train_val"
)


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)


def _color_hist_lab(bgr: np.ndarray, bins: int = 8) -> np.ndarray:
    """Normalized 3D LAB color histogram (bins^3 dims)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [0, 1, 2], None, [bins, bins, bins],
                        [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)


def extract_hog_from_bgr(bgr: np.ndarray, resize: int = 128) -> np.ndarray:
    """HOG (gray) + LAB color histogram features from a BGR crop."""
    if bgr is None or bgr.size == 0:
        raise ValueError("Empty crop")
    resized = cv2.resize(bgr, (resize, resize), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)
    color_feat = _color_hist_lab(resized, bins=6)
    return np.concatenate([hog_feat, color_feat]).astype(np.float32)


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
    augment_flip: bool = False,
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
            pad_x = int(round((xmax - xmin) * 0.05))
            pad_y = int(round((ymax - ymin) * 0.05))
            xs0 = max(0, xmin - pad_x)
            ys0 = max(0, ymin - pad_y)
            xs1 = min(w, xmax + pad_x)
            ys1 = min(h, ymax + pad_y)
            crop = img[ys0:ys1, xs0:xs1]
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
            if augment_flip:
                try:
                    fvec_f = extract_hog_from_bgr(cv2.flip(crop, 1))
                    feats.append(fvec_f)
                    labels.append(cls)
                    meta.append(f"{stem}:{cls}#flip")
                except Exception:
                    pass

    if not feats:
        raise RuntimeError("No VOC crops collected. Check voc_root and ImageSets.")
    return np.vstack(feats), np.array(labels), meta


@dataclass
class ClassifierBundle:
    clf: Pipeline
    encoder: LabelEncoder
    name: str = ""
    val_accuracy: float = 0.0
    has_proba: bool = False


def filter_rare_classes(
    X: np.ndarray,
    y: np.ndarray,
    meta: Optional[List[str]] = None,
    min_count: int = 3,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Drop classes with < ``min_count`` samples (helps stratified split + SVM)."""
    classes, counts = np.unique(y, return_counts=True)
    keep = set(classes[counts >= min_count].tolist())
    mask = np.array([lbl in keep for lbl in y])
    meta_out = [m for m, k in zip(meta or [], mask) if k] if meta is not None else []
    return X[mask], y[mask], meta_out


def _candidate_pipelines(
    C: float,
    n_train: int,
    rbf_max_samples: int = 6000,
    skip: Optional[Sequence[str]] = None,
) -> List[Tuple[str, Pipeline, bool]]:
    """
    Return a list of (name, pipeline, has_proba) candidates to benchmark.

    RBF-SVC scales O(n^2) so we skip it past `rbf_max_samples` to keep
    training feasible on the full VOC trainval set.
    """
    skip_set = set(skip or [])
    cands: List[Tuple[str, Pipeline, bool]] = [
        (
            "gaussian_nb",
            Pipeline([("scaler", StandardScaler(with_mean=True)), ("clf", GaussianNB())]),
            True,
        ),
        (
            "logreg",
            Pipeline(
                [
                    ("scaler", StandardScaler(with_mean=True)),
                    (
                        "clf",
                        LogisticRegression(
                            C=float(C),
                            solver="lbfgs",
                            max_iter=2000,
                            n_jobs=-1,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            True,
        ),
        (
            "linear_svc",
            Pipeline(
                [
                    ("scaler", StandardScaler(with_mean=True)),
                    (
                        "clf",
                        LinearSVC(
                            C=float(C),
                            # Primal is dramatically faster when n_samples > n_features
                            # (true for VOC trainval: ~25k samples vs ~8.3k features).
                            # "auto" can still pick dual on some sklearn versions, so be explicit.
                            dual=False,
                            loss="squared_hinge",
                            tol=1e-3,
                            max_iter=2000,
                            verbose=1,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            False,
        ),
        (
            "random_forest",
            Pipeline(
                [
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=300,
                            n_jobs=-1,
                            random_state=42,
                        ),
                    )
                ]
            ),
            True,
        ),
    ]

    if n_train <= rbf_max_samples:
        cands.append(
            (
                "rbf_svc",
                Pipeline(
                    [
                        ("scaler", StandardScaler(with_mean=True)),
                        (
                            "clf",
                            SVC(
                                kernel="rbf",
                                C=max(1.0, float(C) * 8.0),
                                gamma="scale",
                                decision_function_shape="ovr",
                                random_state=42,
                            ),
                        ),
                    ]
                ),
                False,
            )
        )
    cands = [c for c in cands if c[0] not in skip_set]
    return cands


def _predict_with_pipeline(
    clf: Pipeline,
    encoder: LabelEncoder,
    X: np.ndarray,
    has_proba: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Label + confidence using predict_proba when available, else softmax(decision_function)."""
    if has_proba and hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(X)
        idx = np.argmax(prob, axis=1)
        conf = prob[np.arange(len(idx)), idx]
        classes = clf.classes_ if hasattr(clf, "classes_") else clf.named_steps["clf"].classes_
        return encoder.inverse_transform(classes[idx]), conf.astype(np.float64)
    scores = clf.decision_function(X)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    prob = softmax(scores, axis=1)
    idx = np.argmax(prob, axis=1)
    conf = prob[np.arange(len(idx)), idx]
    return encoder.inverse_transform(idx), conf.astype(np.float64)


def benchmark_and_select_best(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    encoder: LabelEncoder,
    C: float = 1.0,
    skip: Optional[Sequence[str]] = None,
) -> Tuple[ClassifierBundle, pd.DataFrame]:
    """Train every candidate, return the best (by val accuracy) plus a leaderboard."""
    y_train_enc = encoder.transform(y_train)
    leaderboard: List[Dict[str, object]] = []
    best: Optional[ClassifierBundle] = None

    for name, pipe, has_proba in _candidate_pipelines(C=C, n_train=len(X_train), skip=skip):
        
        try:
            print(f"[Stage 6]   - training '{name}' on {len(X_train)} samples...")
            pipe.fit(X_train, y_train_enc)
            pred, _ = _predict_with_pipeline(pipe, encoder, X_val, has_proba=has_proba)
            acc = float(accuracy_score(y_val, pred))
        except Exception as e:
            print(f"[Stage 6]     '{name}' failed: {e}")
            leaderboard.append({"classifier": name, "val_accuracy": float("nan"), "note": str(e)})
            continue

        leaderboard.append({"classifier": name, "val_accuracy": round(acc, 4), "note": ""})

        print(f"[Stage 6]     '{name}' val accuracy = {acc:.4f}")
        
        if best is None or acc > best.val_accuracy:
            best = ClassifierBundle(
                clf=pipe, encoder=encoder, name=name, val_accuracy=acc, has_proba=has_proba
            )

    if best is None:
        raise RuntimeError("All candidate classifiers failed to train.")
    return best, pd.DataFrame(leaderboard).sort_values("val_accuracy", ascending=False)


def predict_label_confidence(
    bundle: ClassifierBundle,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Label + confidence using the selected best classifier."""
    return _predict_with_pipeline(bundle.clf, bundle.encoder, X, has_proba=bundle.has_proba)


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
    max_train_crops: Optional[int] = None,
    max_val_crops: Optional[int] = None,
    C: float = 1.0,
    panorama_paths: Optional[List[str]] = None,
    pano_segments: int = 6,
    panorama_limit: int = 1,
    skip: Optional[Sequence[str]] = None,
) -> None:
    """
    Stage 6 pipeline (callable from main). Trains and benchmarks several
    classical classifiers on VOC trainval crops, picks the best by validation
    accuracy, and saves it to disk.
    """
    voc_root = os.path.abspath(voc_root)
    output_dir = os.path.abspath(output_dir)
    ann_dir = os.path.join(voc_root, "Annotations")
    if not os.path.isdir(ann_dir):
        raise FileNotFoundError(
            f"Stage 6 needs VOC Annotations XML under '{ann_dir}'. "
            "Extract Annotations from the official VOC trainval tar."
        )

    train_ids: List[str] = []
    val_ids: List[str] = []
    try:
        train_list, val_list = _find_train_val_lists(voc_root)
        train_ids = read_image_list(train_list)
        val_ids = read_image_list(val_list) if val_list != train_list else []
    except FileNotFoundError:
        pass

    if not train_ids and not val_ids:
        all_ids = sorted(
            os.path.splitext(f)[0] for f in os.listdir(ann_dir) if f.endswith(".xml")
        )
        print(f"[Stage 6] ImageSets missing; using {len(all_ids)} IDs from Annotations folder.")
        rng = np.random.default_rng(42)
        ids_arr = np.array(all_ids)
        rng.shuffle(ids_arr)
        cut = max(1, int(round(len(ids_arr) * 0.8)))
        train_ids = ids_arr[:cut].tolist()
        val_ids = ids_arr[cut:].tolist()
    elif not val_ids:
        rng = np.random.default_rng(42)
        ids_arr = np.array(train_ids)
        rng.shuffle(ids_arr)
        cut = max(1, int(round(len(ids_arr) * 0.8)))
        train_ids = ids_arr[:cut].tolist()
        val_ids = ids_arr[cut:].tolist()

    print(
        f"[Stage 6] VOC trainval IDs -> train: {len(train_ids)}, val: {len(val_ids)} "
        f"(from {voc_root})."
    )

    print(
        f"[Stage 6] Collecting train crops"
        f"{f' (cap {max_train_crops})' if max_train_crops else ''} with flip aug..."
    )
    X_train, y_train, meta_train = collect_voc_crops(
        voc_root, train_ids, max_crops=max_train_crops, augment_flip=True
    )

    print(
        f"[Stage 6] Collecting val crops"
        f"{f' (cap {max_val_crops})' if max_val_crops else ''}...")
    X_val, y_val, meta_val = collect_voc_crops(
        voc_root, val_ids, max_crops=max_val_crops, augment_flip=False
    )

    X_train, y_train, meta_train = filter_rare_classes(X_train, y_train, meta_train, min_count=3)

    keep_classes = set(np.unique(y_train).tolist())
    val_mask = np.array([lbl in keep_classes for lbl in y_val])
    X_val, y_val = X_val[val_mask], y_val[val_mask]
    meta_val = [m for m, k in zip(meta_val, val_mask) if k]
    print(
        f"[Stage 6] Train: {X_train.shape[0]} crops, {len(np.unique(y_train))} classes | "
        f"Val: {X_val.shape[0]} crops"
    )

    enc = LabelEncoder()
    enc.fit(np.concatenate([y_train, y_val]) if len(y_val) else y_train)

    print("[Stage 6] Benchmarking candidate classifiers (best for this domain wins)...")
    best, leaderboard = benchmark_and_select_best(
        X_train, y_train, X_val, y_val, encoder=enc, C=C, skip=skip
    )

    os.makedirs(output_dir, exist_ok=True)
    leaderboard_path = os.path.join(output_dir, "classifier_leaderboard.csv")
    leaderboard.to_csv(leaderboard_path, index=False)
    print(f"[Stage 6] Leaderboard saved -> {leaderboard_path}")
    print(leaderboard.to_string(index=False))
    print(
        f"[Stage 6] Best classifier: '{best.name}' "
        f"with val accuracy = {best.val_accuracy:.4f}"
    )

    pred, conf = predict_label_confidence(best, X_val)
    print("[Stage 6] Detailed report for best classifier:")
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
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "classification_val.csv"), index=False)
    print(f"[Stage 6] Per-sample predictions saved -> classification_val.csv")

    model_path = os.path.join(output_dir, "classifier_stage6.joblib")
    joblib.dump(
        {
            "best_name": best.name,
            "val_accuracy": best.val_accuracy,
            "has_proba": best.has_proba,
            "pipeline": best.clf,
            "encoder": best.encoder,
        },
        model_path,
    )
    print(f"[Stage 6] Final classifier saved -> {model_path}")

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
            plab, pconf = predict_label_confidence(best, Xp)
            for i, (lab, co) in enumerate(zip(plab, pconf)):
                print(f"  segment[{i}]: {lab}  (conf={co:.3f})")


def run_stage6_cli(args: argparse.Namespace) -> None:
    
    run_stage6(
        voc_root=_resolve(args.voc_root),
        output_dir=_resolve(args.output_dir),
        max_train_crops=args.max_train_crops if args.max_train_crops and args.max_train_crops > 0 else None,
        max_val_crops=args.max_val_crops if args.max_val_crops and args.max_val_crops > 0 else None,
        C=args.C,
        panorama_paths=[_resolve(args.panorama)] if args.panorama else None,
        pano_segments=args.pano_segments,
        panorama_limit=1,
        skip=args.skip or None,
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 6 - VOC object classification (HOG + color, classifier benchmark)"
    )
    p.add_argument("--voc-root", default=VOC_PROJECT_ROOT_REL)
    p.add_argument("--output-dir", default="outputs/classification")
    p.add_argument(
        "--max-train-crops",
        type=int,
        default=0,
        help="Cap training crops for memory/speed. 0 (default) = use all.",
    )
    p.add_argument(
        "--max-val-crops",
        type=int,
        default=0,
        help="Cap validation crops. 0 (default) = use all.",
    )
    p.add_argument("--C", type=float, default=1.0, help="Regularization for SVC/LogReg")
    p.add_argument("--panorama", default=None, help="Optional panorama path for segment-based demo")
    p.add_argument("--pano-segments", type=int, default=6)
    p.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["gaussian_nb", "logreg", "linear_svc", "random_forest", "rbf_svc"],
        help="Classifier names to skip (e.g. --skip linear_svc).",
    )
    return p.parse_args()


if __name__ == "__main__":
    print("=" * 55)
    print("  Stage 6 - Object Classification")
    print("=" * 55)
    run_stage6_cli(parse_args())
    print("\n[Stage 6] Complete")