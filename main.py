import cv2, os, time, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sktlearn.model_selection import train_test_split

if __name__ == "__main__":
    start_time = time.time()
    # calling every stage in the pipeline and saving the results
    # Stage 1: Image Preprocessing (preprocess.py)
    # Stage 2: Feature Detection (detect.py)
    # Stage 3: Feature Matching (match.py)
    # Stage 4: Homography & Stitching (stitch.py)
    # Stage 5: Segmentation (segment.py)
    # Stage 6: Object Classification (classify.py)

    