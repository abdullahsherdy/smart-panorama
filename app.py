"""
Smart Panorama Studio — Streamlit UI
=====================================
End-to-end interactive playground for every stage of the project:

    Stage 1 — Preprocessing (Gaussian, Median, Bilateral, Box, CLAHE, Grayscale)
    Stage 2 — SIFT Feature Detection
    Stage 3 — Feature Matching (BFMatcher + Lowe's ratio)
    Stage 4 — Homography & Panorama Stitching
    Stage 5 — K-Means Segmentation + Overlay
    Stage 6 — Object Classification (uses ONLY the saved joblib from outputs/classification)
    🚀     — Full pipeline: 2 images → preprocess → SIFT → match → stitch → segment → classify

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
import os
import sys
import time
from typing import List, Optional, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.special import softmax

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Stage modules (reused, no duplication)
from detect import detect_features, draw_keypoints_bgr  # noqa: E402
from match import match_features, draw_matches_horizontal  # noqa: E402
from stitch import stitch_pair  # noqa: E402
from segment import segment_image_kmeans, _foreground_from_clusters  # noqa: E402
from classify import (  # noqa: E402
    extract_hog_from_bgr,
    crops_from_panorama_segments,
)

MODEL_PATH = os.path.join(_REPO_ROOT, "outputs", "classification", "classifier_stage6.joblib")


# ╭──────────────────────────────────────────────────────────────────────╮
# │  PAGE CONFIG + GLOBAL STYLES                                         │
# ╰──────────────────────────────────────────────────────────────────────╯
st.set_page_config(
    page_title="Smart Panorama Studio",
    page_icon="🌄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        h1, h2, h3 { letter-spacing: -0.5px; }
        .stage-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.8rem;
            background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
            margin-bottom: 0.6rem;
        }
        .metric-card {
            background: rgba(99,102,241,0.08);
            border: 1px solid rgba(99,102,241,0.25);
            padding: 14px 18px;
            border-radius: 12px;
        }
        .pred-card {
            background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(99,102,241,0.12));
            border: 1px solid rgba(99,102,241,0.3);
            padding: 18px 20px;
            border-radius: 14px;
            margin: 6px 0;
        }
        .small-muted { color: #6b7280; font-size: 0.85rem; }
        section[data-testid="stSidebar"] h2 { color: #6366f1; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ╭──────────────────────────────────────────────────────────────────────╮
# │  HELPERS                                                             │
# ╰──────────────────────────────────────────────────────────────────────╯
def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    if img is None:
        return img
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_uploaded_image(file) -> Optional[np.ndarray]:
    """Decode an uploaded file into a BGR uint8 ndarray."""
    if file is None:
        return None
    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def encode_png(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    return buf.tobytes() if ok else b""


def section_header(title: str, subtitle: str, badge: str) -> None:
    st.markdown(f'<div class="stage-badge">{badge}</div>', unsafe_allow_html=True)
    st.markdown(f"## {title}")
    st.caption(subtitle)


@st.cache_resource(show_spinner=False)
def load_classifier_bundle(path: str):
    """Load the trained classifier saved by Stage 6 (only artifact we use)."""
    if not os.path.isfile(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load classifier: {e}")
        return None


def predict_with_bundle(
    bundle: dict, X: np.ndarray, top_k: int = 3
) -> List[List[Tuple[str, float]]]:
    """Return top-k (label, prob) per sample using only the saved bundle."""
    pipe = bundle["pipeline"]
    encoder = bundle["encoder"]
    has_proba = bundle.get("has_proba", False)

    if has_proba and hasattr(pipe, "predict_proba"):
        prob = pipe.predict_proba(X)
        classes = pipe.classes_ if hasattr(pipe, "classes_") else pipe.named_steps["clf"].classes_
    else:
        scores = pipe.decision_function(X)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        prob = softmax(scores, axis=1)
        classes = (
            pipe.classes_
            if hasattr(pipe, "classes_")
            else pipe.named_steps["clf"].classes_
        )

    out: List[List[Tuple[str, float]]] = []
    k = min(top_k, prob.shape[1])
    for row in prob:
        idx = np.argsort(row)[::-1][:k]
        labels = encoder.inverse_transform(classes[idx])
        out.append([(str(labels[i]), float(row[idx[i]])) for i in range(k)])
    return out


# ╭──────────────────────────────────────────────────────────────────────╮
# │  PREPROCESSING FILTERS (sidebar-driven)                              │
# ╰──────────────────────────────────────────────────────────────────────╯
def apply_filters(img: np.ndarray, cfg: dict) -> np.ndarray:
    """Apply a chain of preprocessing filters described by `cfg`."""
    out = img.copy()
    for op in cfg.get("ops", []):
        kind = op["kind"]
        try:
            if kind == "gaussian":
                k = int(op.get("ksize", 5)) | 1
                out = cv2.GaussianBlur(out, (k, k), float(op.get("sigma", 1.0)))
            elif kind == "median":
                k = int(op.get("ksize", 3)) | 1
                out = cv2.medianBlur(out, k)
            elif kind == "bilateral":
                d = int(op.get("d", 9))
                out = cv2.bilateralFilter(
                    out, d, float(op.get("sigma_color", 75)), float(op.get("sigma_space", 75))
                )
            elif kind == "box":
                k = int(op.get("ksize", 5)) | 1
                out = cv2.blur(out, (k, k))
            elif kind == "clahe":
                lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
                L, A, B = cv2.split(lab)
                clahe = cv2.createCLAHE(
                    clipLimit=float(op.get("clip", 2.0)),
                    tileGridSize=(int(op.get("tile", 8)), int(op.get("tile", 8))),
                )
                L = clahe.apply(L)
                out = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)
            elif kind == "sharpen":
                amt = float(op.get("amount", 1.0))
                blur = cv2.GaussianBlur(out, (0, 0), 1.5)
                out = cv2.addWeighted(out, 1 + amt, blur, -amt, 0)
            elif kind == "canny":
                gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, int(op.get("low", 80)), int(op.get("high", 160)))
                out = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            elif kind == "grayscale":
                gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            st.warning(f"Filter '{kind}' failed: {e}")
    return out


def sidebar_filter_chain(key_prefix: str = "pre") -> dict:
    """Sidebar widget letting the user pick & chain preprocessing filters."""
    st.sidebar.markdown("### 🎛️ Preprocessing chain")
    available = {
        "Gaussian Blur": "gaussian",
        "Median Filter": "median",
        "Bilateral Filter": "bilateral",
        "Box Blur": "box",
        "CLAHE (LAB)": "clahe",
        "Unsharp Sharpen": "sharpen",
        "Canny Edges": "canny",
        "Grayscale": "grayscale",
    }
    default = ["Gaussian Blur", "Median Filter", "CLAHE (LAB)"]
    picks = st.sidebar.multiselect(
        "Pick filters (applied in order)",
        list(available.keys()),
        default=default,
        key=f"{key_prefix}_picks",
    )
    ops = []
    for label in picks:
        kind = available[label]
        with st.sidebar.expander(f"⚙️ {label}", expanded=False):
            op = {"kind": kind}
            if kind == "gaussian":
                op["ksize"] = st.slider("Kernel size", 3, 31, 5, step=2, key=f"{key_prefix}_g_k")
                op["sigma"] = st.slider("Sigma", 0.1, 5.0, 1.0, key=f"{key_prefix}_g_s")
            elif kind == "median":
                op["ksize"] = st.slider("Kernel size", 3, 15, 3, step=2, key=f"{key_prefix}_m_k")
            elif kind == "bilateral":
                op["d"] = st.slider("Diameter", 5, 25, 9, key=f"{key_prefix}_b_d")
                op["sigma_color"] = st.slider("Sigma color", 10, 200, 75, key=f"{key_prefix}_b_sc")
                op["sigma_space"] = st.slider("Sigma space", 10, 200, 75, key=f"{key_prefix}_b_ss")
            elif kind == "box":
                op["ksize"] = st.slider("Kernel size", 3, 31, 5, step=2, key=f"{key_prefix}_bx_k")
            elif kind == "clahe":
                op["clip"] = st.slider("Clip limit", 1.0, 8.0, 2.0, key=f"{key_prefix}_c_cl")
                op["tile"] = st.slider("Tile size", 2, 16, 8, key=f"{key_prefix}_c_t")
            elif kind == "sharpen":
                op["amount"] = st.slider("Amount", 0.1, 3.0, 1.0, key=f"{key_prefix}_sh")
            elif kind == "canny":
                op["low"] = st.slider("Low threshold", 0, 255, 80, key=f"{key_prefix}_cn_l")
                op["high"] = st.slider("High threshold", 0, 255, 160, key=f"{key_prefix}_cn_h")
            ops.append(op)
    return {"ops": ops}


# ╭──────────────────────────────────────────────────────────────────────╮
# │  PAGES                                                               │
# ╰──────────────────────────────────────────────────────────────────────╯
def page_home(bundle):
    st.markdown(
        """
        <h1 style="margin-bottom:0">🌄 Smart Panorama Studio</h1>
        <p class="small-muted">An end-to-end Computer Vision pipeline — preprocessing, SIFT,
        homography, K-means segmentation and a trained VOC classifier — wired into one playful UI.</p>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Stages", "6", "Preprocess → Classify")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Filters", "8", "Gaussian, Median, …")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if bundle:
            st.metric(
                "Classifier",
                bundle.get("best_name", "—"),
                f"val acc={bundle.get('val_accuracy', 0):.3f}",
            )
        else:
            st.metric("Classifier", "Not loaded", "Run Stage 6 first")
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Backend", "OpenCV + sklearn", "live")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧭 What's inside")
    cols = st.columns(3)
    items = [
        ("🎨 Preprocessing Studio", "Stack Gaussian / Median / Bilateral / CLAHE / Canny live."),
        ("🔑 Feature Detection", "Visualise SIFT keypoints with rich orientation circles."),
        ("🔗 Feature Matching", "Lowe's ratio test on SIFT descriptors, side-by-side view."),
        ("🌅 Panorama Stitching", "RANSAC homography + linear blending into one canvas."),
        ("🎭 Segmentation", "K-means over LAB+XY with foreground extraction overlay."),
        ("🏷️ Classification", "Crops → HOG+LAB features → trained classifier with top-k."),
    ]
    for i, (t, d) in enumerate(items):
        with cols[i % 3]:
            st.markdown(f"**{t}**\n\n{d}")
            st.markdown("&nbsp;")

    st.info(
        "Pick a stage from the **sidebar** to start. The classifier loaded here is the exact "
        "model saved by Stage 6 at `outputs/classification/classifier_stage6.joblib`.",
        icon="👈",
    )


def page_preprocess():
    section_header(
        "Preprocessing Studio",
        "Upload an image and chain filters from the sidebar — the pipeline updates live.",
        "STAGE 1",
    )
    cfg = sidebar_filter_chain("pre")
    upl = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="pre_upl")
    if not upl:
        st.info("👆 Upload a JPG/PNG to begin.")
        return
    img = read_uploaded_image(upl)
    if img is None:
        st.error("Could not decode image.")
        return

    out = apply_filters(img, cfg)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Original**")
        st.image(bgr_to_rgb(img), use_container_width=True)
    with c2:
        st.markdown(f"**Processed** ({len(cfg['ops'])} filter(s))")
        st.image(bgr_to_rgb(out), use_container_width=True)

    st.download_button(
        "⬇️ Download processed image (PNG)",
        data=encode_png(out),
        file_name="preprocessed.png",
        mime="image/png",
    )

    with st.expander("🔬 Filter chain JSON"):
        st.json(cfg)


def page_detect():
    section_header(
        "Feature Detection (SIFT)",
        "Detect keypoints with SIFT — slide knobs to see how parameters reshape the keypoint cloud.",
        "STAGE 2",
    )
    pre_cfg = sidebar_filter_chain("det")
    nfeatures = st.sidebar.slider("nfeatures", 200, 8000, 3000, step=200)
    contrast = st.sidebar.slider("contrastThreshold", 0.01, 0.20, 0.04)
    edge = st.sidebar.slider("edgeThreshold", 1.0, 30.0, 10.0)

    upl = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="det_upl")
    if not upl:
        st.info("👆 Upload a JPG/PNG to detect SIFT features.")
        return
    img = read_uploaded_image(upl)
    if img is None:
        st.error("Could not decode image.")
        return

    pre = apply_filters(img, pre_cfg)
    gray = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
    t0 = time.time()
    kp, des = detect_features(
        gray, nfeatures=nfeatures, contrastThreshold=contrast, edgeThreshold=edge
    )
    dt = (time.time() - t0) * 1000
    vis = draw_keypoints_bgr(gray, kp, rich=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Preprocessed (gray)**")
        st.image(gray, use_container_width=True, clamp=True)
    with c2:
        st.markdown("**SIFT keypoints**")
        st.image(bgr_to_rgb(vis), use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Keypoints", len(kp))
    m2.metric("Descriptor shape", f"{des.shape}" if des is not None else "—")
    m3.metric("Time", f"{dt:.0f} ms")


def page_match():
    section_header(
        "Feature Matching",
        "Brute-force kNN on SIFT descriptors with Lowe's ratio test — the inputs to homography.",
        "STAGE 3",
    )
    pre_cfg = sidebar_filter_chain("mat")
    nfeatures = st.sidebar.slider("nfeatures", 200, 8000, 3000, step=200, key="mat_nf")
    ratio = st.sidebar.slider("Lowe ratio", 0.5, 0.95, 0.75, key="mat_r")
    max_draw = st.sidebar.slider("Max matches drawn", 20, 300, 80, key="mat_md")

    c1, c2 = st.columns(2)
    with c1:
        upl_a = st.file_uploader("Image A (left)", type=["jpg", "jpeg", "png"], key="mat_a")
    with c2:
        upl_b = st.file_uploader("Image B (right)", type=["jpg", "jpeg", "png"], key="mat_b")
    if not (upl_a and upl_b):
        st.info("👆 Upload both images to compute matches.")
        return
    img_a = read_uploaded_image(upl_a)
    img_b = read_uploaded_image(upl_b)
    if img_a is None or img_b is None:
        st.error("Could not decode one of the images.")
        return

    pa = apply_filters(img_a, pre_cfg)
    pb = apply_filters(img_b, pre_cfg)
    ga, gb = cv2.cvtColor(pa, cv2.COLOR_BGR2GRAY), cv2.cvtColor(pb, cv2.COLOR_BGR2GRAY)

    kp_a, des_a = detect_features(ga, nfeatures=nfeatures)
    kp_b, des_b = detect_features(gb, nfeatures=nfeatures)
    good = match_features(kp_a, des_a, kp_b, des_b, ratio=ratio)
    vis = draw_matches_horizontal(pa, kp_a, pb, kp_b, good, max_draw=max_draw)

    st.image(bgr_to_rgb(vis), use_container_width=True, caption=f"Top {min(len(good), max_draw)} of {len(good)} good matches")
    m1, m2, m3 = st.columns(3)
    m1.metric("Keypoints A", len(kp_a))
    m2.metric("Keypoints B", len(kp_b))
    m3.metric("Good matches", len(good))

    st.session_state["match_state"] = {
        "img_a": pa, "img_b": pb, "kp_a": kp_a, "kp_b": kp_b, "good": good
    }


def page_stitch():
    section_header(
        "Panorama Stitching",
        "RANSAC homography + linear blending. Upload two overlapping photos to get a full panorama.",
        "STAGE 4",
    )
    pre_cfg = sidebar_filter_chain("stc")
    nfeatures = st.sidebar.slider("nfeatures", 500, 8000, 3000, step=200, key="stc_nf")
    ratio = st.sidebar.slider("Lowe ratio", 0.5, 0.95, 0.75, key="stc_r")
    ransac = st.sidebar.slider("RANSAC reproj. thresh (px)", 1.0, 15.0, 5.0, key="stc_rs")

    c1, c2 = st.columns(2)
    with c1:
        upl_a = st.file_uploader("Image A (left)", type=["jpg", "jpeg", "png"], key="stc_a")
    with c2:
        upl_b = st.file_uploader("Image B (right)", type=["jpg", "jpeg", "png"], key="stc_b")
    if not (upl_a and upl_b):
        st.info("👆 Upload two overlapping images.")
        return

    img_a = read_uploaded_image(upl_a)
    img_b = read_uploaded_image(upl_b)
    if img_a is None or img_b is None:
        st.error("Could not decode one of the images.")
        return

    with st.spinner("Detecting features and stitching..."):
        pa = apply_filters(img_a, pre_cfg)
        pb = apply_filters(img_b, pre_cfg)
        ga = cv2.cvtColor(pa, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(pb, cv2.COLOR_BGR2GRAY)
        kp_a, des_a = detect_features(ga, nfeatures=nfeatures)
        kp_b, des_b = detect_features(gb, nfeatures=nfeatures)
        good = match_features(kp_a, des_a, kp_b, des_b, ratio=ratio)
        pano = stitch_pair(pa, pb, good, kp_a, kp_b, ransac_reproj_threshold=ransac)

    if pano is None:
        st.error(f"Stitching failed (only {len(good)} good matches). Try lowering the ratio or use more overlap.")
        return

    st.image(bgr_to_rgb(pano), use_container_width=True, caption=f"Panorama — {pano.shape[1]}×{pano.shape[0]}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Good matches", len(good))
    m2.metric("Output size", f"{pano.shape[1]}×{pano.shape[0]}")
    m3.metric("Channels", pano.shape[2])

    st.session_state["last_panorama"] = pano
    st.download_button(
        "⬇️ Download panorama (JPG)",
        data=cv2.imencode(".jpg", pano)[1].tobytes(),
        file_name="panorama.jpg",
        mime="image/jpeg",
    )


def page_segment():
    section_header(
        "K-Means Segmentation",
        "Cluster pixels in LAB+XY space and extract a clean foreground mask.",
        "STAGE 5",
    )
    n_seg = st.sidebar.slider("Segments (K)", 2, 12, 6)
    spatial = st.sidebar.slider("Spatial weight", 0.0, 1.0, 0.35)
    bg_mode = st.sidebar.selectbox("Background mode", ["border", "largest"])

    upl = st.file_uploader("Upload an image (or panorama)", type=["jpg", "jpeg", "png"], key="seg_upl")
    img = read_uploaded_image(upl) if upl else None
    if img is None and "last_panorama" in st.session_state:
        if st.checkbox("Use last panorama from Stage 4", value=True):
            img = st.session_state["last_panorama"]
    if img is None:
        st.info("👆 Upload an image, or stitch one in Stage 4 first.")
        return

    with st.spinner("Running K-means..."):
        seg, mask = segment_image_kmeans(img, n_segments=n_seg, spatial_weight=spatial)
        fg = _foreground_from_clusters(mask, background_mode=bg_mode)

    edges = cv2.Canny(mask, 30, 90)
    overlay = img.copy()
    overlay[edges > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    fg_vis = img.copy()
    fg_vis[fg == 0] = (fg_vis[fg == 0] * 0.25).astype(np.uint8)

    tabs = st.tabs(["🎨 Segmented", "🧭 Overlay", "🌟 Foreground", "🗺️ Label mask"])
    with tabs[0]:
        st.image(bgr_to_rgb(seg), use_container_width=True)
    with tabs[1]:
        st.image(bgr_to_rgb(overlay), use_container_width=True)
    with tabs[2]:
        st.image(bgr_to_rgb(fg_vis), use_container_width=True)
    with tabs[3]:
        norm = (mask.astype(np.float32) / max(1, mask.max()) * 255).astype(np.uint8)
        st.image(norm, use_container_width=True, clamp=True)

    st.session_state["last_segmentation"] = {"img": img, "mask": mask, "fg": fg}

    m1, m2 = st.columns(2)
    m1.metric("Clusters used", int(mask.max()) + 1)
    m2.metric("Foreground %", f"{100 * fg.mean():.1f}%")


def page_classify(bundle):
    section_header(
        "Object Classification",
        "Predict a class for an uploaded crop, or auto-extract crops from a segmentation.",
        "STAGE 6",
    )

    if bundle is None:
        st.error(
            f"Classifier model not found at `outputs/classification/classifier_stage6.joblib`. "
            "Train Stage 6 first (`python src/classify.py --skip linear_svc rbf_svc`)."
        )
        return

    st.markdown(
        f"""
        <div class="pred-card">
            <b>Loaded model:</b> <code>{bundle.get('best_name', 'unknown')}</code> &nbsp;|&nbsp;
            <b>val accuracy:</b> {bundle.get('val_accuracy', 0):.4f} &nbsp;|&nbsp;
            <b>has_proba:</b> {bundle.get('has_proba', False)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Input mode",
        ["Single image / crop", "Auto-crops from segmentation"],
        horizontal=True,
    )

    top_k = st.slider("Top-K predictions", 1, 5, 3)

    if mode == "Single image / crop":
        upl = st.file_uploader(
            "Upload a tight crop of one object", type=["jpg", "jpeg", "png"], key="cls_upl"
        )
        if not upl:
            st.info("👆 Upload an object crop (e.g. a single dog, car, person...).")
            return
        img = read_uploaded_image(upl)
        if img is None:
            st.error("Could not decode image.")
            return

        feat = extract_hog_from_bgr(img).reshape(1, -1)
        preds = predict_with_bundle(bundle, feat, top_k=top_k)[0]

        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(bgr_to_rgb(img), use_container_width=True, caption="Input crop")
        with c2:
            st.markdown("### Predictions")
            for i, (lab, p) in enumerate(preds):
                emoji = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else "▫️"))
                st.markdown(
                    f'<div class="pred-card"><b>{emoji} {lab}</b> '
                    f'<span class="small-muted">— confidence {p:.3f}</span></div>',
                    unsafe_allow_html=True,
                )
                st.progress(min(max(p, 0.0), 1.0))

    else:
        st.caption(
            "We'll segment the image with K-means and run the classifier on each segment crop."
        )
        n_seg = st.slider("Segments (K)", 2, 12, 6, key="cls_k")
        upl = st.file_uploader(
            "Upload an image / panorama", type=["jpg", "jpeg", "png"], key="cls_pano"
        )
        img = read_uploaded_image(upl) if upl else None
        if img is None and "last_segmentation" in st.session_state:
            if st.checkbox("Use last segmented image (Stage 5)", value=True):
                img = st.session_state["last_segmentation"]["img"]
        if img is None and "last_panorama" in st.session_state:
            if st.checkbox("Use last panorama (Stage 4)", value=True, key="cls_pano_chk"):
                img = st.session_state["last_panorama"]
        if img is None:
            st.info("👆 Upload an image, or generate one in Stage 4 / 5 first.")
            return

        with st.spinner("Segmenting + classifying..."):
            _, mask = segment_image_kmeans(img, n_segments=n_seg)
            crops = crops_from_panorama_segments(img, mask, min_area=600, max_crops=12)
            if not crops:
                st.warning("No crops produced — try increasing K or using a busier image.")
                return
            X = np.vstack([extract_hog_from_bgr(c) for c in crops])
            preds_all = predict_with_bundle(bundle, X, top_k=top_k)

        st.image(bgr_to_rgb(img), use_container_width=True, caption=f"Source image ({len(crops)} crops)")
        cols = st.columns(min(4, len(crops)))
        for i, (crop, preds) in enumerate(zip(crops, preds_all)):
            with cols[i % len(cols)]:
                st.image(bgr_to_rgb(crop), use_container_width=True, caption=f"segment[{i}]")
                top_label, top_p = preds[0]
                st.markdown(
                    f'<div class="pred-card"><b>🥇 {top_label}</b><br>'
                    f'<span class="small-muted">conf {top_p:.3f}</span></div>',
                    unsafe_allow_html=True,
                )
                with st.expander("More predictions"):
                    for lab, p in preds[1:]:
                        st.write(f"• {lab} — {p:.3f}")


def page_full_pipeline(bundle):
    section_header(
        "Full Pipeline",
        "Two photos in → preprocessing → SIFT → matching → stitching → segmentation → classification.",
        "🚀 END-TO-END",
    )
    pre_cfg = sidebar_filter_chain("full")
    nfeatures = st.sidebar.slider("SIFT nfeatures", 500, 8000, 3000, step=200, key="full_nf")
    ratio = st.sidebar.slider("Lowe ratio", 0.5, 0.95, 0.75, key="full_r")
    n_seg = st.sidebar.slider("K-means segments", 2, 12, 6, key="full_k")

    c1, c2 = st.columns(2)
    with c1:
        upl_a = st.file_uploader("Image A (left)", type=["jpg", "jpeg", "png"], key="full_a")
    with c2:
        upl_b = st.file_uploader("Image B (right)", type=["jpg", "jpeg", "png"], key="full_b")
    if not (upl_a and upl_b):
        st.info("👆 Upload two overlapping images to run the full pipeline.")
        return
    img_a = read_uploaded_image(upl_a)
    img_b = read_uploaded_image(upl_b)
    if img_a is None or img_b is None:
        st.error("Could not decode images.")
        return

    progress = st.progress(0, text="Starting...")
    timings = {}

    progress.progress(10, text="Stage 1 — preprocessing...")
    t = time.time()
    pa = apply_filters(img_a, pre_cfg)
    pb = apply_filters(img_b, pre_cfg)
    timings["preprocess"] = time.time() - t

    progress.progress(25, text="Stage 2 — SIFT...")
    t = time.time()
    ga, gb = cv2.cvtColor(pa, cv2.COLOR_BGR2GRAY), cv2.cvtColor(pb, cv2.COLOR_BGR2GRAY)
    kp_a, des_a = detect_features(ga, nfeatures=nfeatures)
    kp_b, des_b = detect_features(gb, nfeatures=nfeatures)
    timings["detect"] = time.time() - t

    progress.progress(45, text="Stage 3 — matching...")
    t = time.time()
    good = match_features(kp_a, des_a, kp_b, des_b, ratio=ratio)
    timings["match"] = time.time() - t

    progress.progress(60, text="Stage 4 — stitching...")
    t = time.time()
    pano = stitch_pair(pa, pb, good, kp_a, kp_b, ransac_reproj_threshold=5.0)
    timings["stitch"] = time.time() - t
    if pano is None:
        progress.empty()
        st.error("Stitching failed — not enough matches.")
        return

    progress.progress(80, text="Stage 5 — segmentation...")
    t = time.time()
    _, mask = segment_image_kmeans(pano, n_segments=n_seg)
    crops = crops_from_panorama_segments(pano, mask, min_area=600, max_crops=12)
    timings["segment"] = time.time() - t

    preds_all: List[List[Tuple[str, float]]] = []
    if bundle and crops:
        progress.progress(95, text="Stage 6 — classification...")
        t = time.time()
        X = np.vstack([extract_hog_from_bgr(c) for c in crops])
        preds_all = predict_with_bundle(bundle, X, top_k=3)
        timings["classify"] = time.time() - t

    progress.progress(100, text="Done")
    progress.empty()

    st.success(f"Pipeline finished in {sum(timings.values()):.2f}s")
    st.image(bgr_to_rgb(pano), use_container_width=True, caption="Stitched panorama")

    cols = st.columns(len(timings))
    for col, (k, v) in zip(cols, timings.items()):
        col.metric(k, f"{v*1000:.0f} ms")

    if crops:
        st.markdown("### 🏷️ Detected segments")
        ncol = min(4, len(crops))
        cols = st.columns(ncol)
        for i, crop in enumerate(crops):
            with cols[i % ncol]:
                st.image(bgr_to_rgb(crop), use_container_width=True, caption=f"segment[{i}]")
                if preds_all:
                    top_label, top_p = preds_all[i][0]
                    st.markdown(
                        f'<div class="pred-card"><b>🥇 {top_label}</b><br>'
                        f'<span class="small-muted">conf {top_p:.3f}</span></div>',
                        unsafe_allow_html=True,
                    )


# ╭──────────────────────────────────────────────────────────────────────╮
# │  ROUTER                                                              │
# ╰──────────────────────────────────────────────────────────────────────╯
def main():
    bundle = load_classifier_bundle(MODEL_PATH)

    st.sidebar.markdown(
        '<h2 style="margin-bottom:0">🌄 Smart Panorama</h2>'
        '<p class="small-muted">Computer Vision Studio</p>',
        unsafe_allow_html=True,
    )

    pages = {
        "🏠 Home": lambda: page_home(bundle),
        "🎨 Preprocessing Studio": page_preprocess,
        "🔑 Feature Detection": page_detect,
        "🔗 Feature Matching": page_match,
        "🌅 Panorama Stitching": page_stitch,
        "🎭 Segmentation": page_segment,
        "🏷️ Classification": lambda: page_classify(bundle),
        "🚀 Full Pipeline": lambda: page_full_pipeline(bundle),
    }
    choice = st.sidebar.radio("Navigate", list(pages.keys()), label_visibility="collapsed")
    st.sidebar.markdown("---")

    if bundle:
        st.sidebar.success(
            f"Classifier loaded\n\n"
            f"**{bundle.get('best_name', '?')}** · acc={bundle.get('val_accuracy', 0):.3f}",
            icon="✅",
        )
    else:
        st.sidebar.warning("Classifier not loaded — Stage 6 features will be limited.", icon="⚠️")

    st.sidebar.markdown(
        '<p class="small-muted" style="margin-top:1.5rem">'
        'Built with OpenCV, scikit-learn & Streamlit.<br>'
        'Stages 1–6 reused directly from <code>src/</code>.</p>',
        unsafe_allow_html=True,
    )

    pages[choice]()


if __name__ == "__main__":
    main()
