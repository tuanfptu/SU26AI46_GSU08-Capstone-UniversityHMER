"""Unicode-safe OpenCV image I/O for Windows paths."""

from pathlib import Path

import cv2
import numpy as np


def read_grayscale(path):
    path = Path(path)
    if not path.is_file():
        return None
    encoded = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)


def write_image(path, image) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True
