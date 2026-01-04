import cv2
import numpy as np

def detect_id_card(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return 0.0
        h, w = img.shape[:2]
        ratio = w / h if h else 0
        score_ratio = 1.0 if 1.4 < ratio < 1.8 else 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = (edges > 0).mean()
        score_edges = 1.0 if edge_density < 0.15 else 0.0
        return round((score_ratio + score_edges) / 2, 2)
    except Exception:
        return 0.0
