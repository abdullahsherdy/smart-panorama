# Smart Panorama & Object Recognition System

## Full Implementation Plan — UDIS-D + PASCAL VOC 2012 Only

---

## Folder Structure

```
smart_panorama/
├── data/
│   ├── udis/test/input1/        # UDIS-D left images
│   ├── udis/test/input2/        # UDIS-D right images
│   └── voc/VOC2012/
│       ├── JPEGImages/
│       ├── Annotations/         # XML bounding boxes
│       └── SegmentationClass/   # PNG pixel masks
├── src/
│   ├── preprocess.py            # Stage 1
│   ├── detect.py                # Stage 2
│   ├── match.py                 # Stage 3
│   ├── stitch.py                # Stage 4
│   ├── segment.py               # Stage 5
│   ├── classify.py              # Stage 6
│   └── utils.py                 # Shared helpers
├── outputs/                     # Saved results
├── main.py                      # End-to-end runner
└── requirements.txt
```

**requirements.txt**

```
opencv-python>=4.9.0
numpy>=1.26.0
scikit-learn>=1.4.0
scikit-image>=0.22.0
matplotlib>=3.8.0
pandas>=2.0.0
Pillow>=10.0.0
```

---

## Stage 1 — Image Preprocessing (`src/preprocess.py`)

**Input:** Raw image pair paths from UDIS-D  
**Output:** `(color_img, gray_img)` both as `np.ndarray`  
**Dataset:** `data/udis/test/input1/` and `data/udis/test/input2/`

---

## Stage 2 — Feature Detection (`src/detect.py`)

**Input:** `gray_img` (np.ndarray, grayscale)  
**Output:** `(keypoints, descriptors)` — OpenCV format  
**Dataset:** Preprocessed output from Stage 1

---

## Stage 3 — Feature Matching (`src/match.py`)

**Input:** `(kp_a, des_a)` and `(kp_b, des_b)` from Stage 2  
**Output:** `good_matches` list of `cv2.DMatch`  
**Dataset:** UDIS-D pairs (via Stages 1+2)

---

## Stage 4 — Homography & Stitching (`src/stitch.py`)

**Input:** `good_matches`, `kp_a`, `kp_b`, `color_a`, `color_b` from Stages 2+3  
**Output:** Stitched panorama `np.ndarray`  
**Dataset:** UDIS-D pairs (colour images from Stage 1)

---

## Stage 5 — Segmentation (`src/segment.py`)

**Input:** Panorama from Stage 4 (any RGB image)  
**Output:** Segmented image, label mask, IoU scores  
**Dataset:** Panorama output (UDIS-D) + PASCAL VOC 2012 for IoU evaluation

---

## Stage 6 — Object Classification (`src/classify.py`)

**Input:** Object crop (from VOC bounding boxes or panorama segments)  
**Output:** Predicted class label + confidence  
**Dataset:** PASCAL VOC 2012 (JPEGImages + Annotations XML)

---

## Stage 7 — Integration (`main.py`)

**Full end-to-end pipeline runner.**

---

## Execution Order

```bash
# 1. Set up environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Download datasets
#    UDIS-D test split  → data/udis/test/input1/ and input2/
#    PASCAL VOC 2012    → data/voc/VOC2012/

# 3. Test each stage independently (per member)
cd src
python preprocess.py     # Stage 1 — saves outputs/preprocessing_demo.png
python detect.py         # Stage 2 — saves outputs/kp_sift.png
python match.py          # Stage 3 — saves outputs/matches_pair1.png
python stitch.py         # Stage 4 — saves outputs/panorama_001.jpg ...
python segment.py        # Stage 5 — saves outputs/segmentation_comparison.png
python classify.py       # Stage 6 — saves outputs/confusion_*.png

# 4. Run full pipeline
cd ..
python main.py
```

