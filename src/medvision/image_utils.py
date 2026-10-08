from pathlib import Path
import numpy as np
import cv2
from PIL import Image

try:
    import pydicom
except Exception:
    pydicom = None


def _normalize_uint8(array: np.ndarray) -> np.ndarray:
    arr = array.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255).astype(np.uint8)


def load_medical_image(path: str | Path) -> Image.Image:
    """Load JPG/PNG or DICOM and return RGB PIL image.

    This preprocessing is for research/demo use only.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".dcm", ".dicom"}:
        if pydicom is None:
            raise RuntimeError("pydicom is required for DICOM images.")
        ds = pydicom.dcmread(str(path))
        arr = _normalize_uint8(ds.pixel_array)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = 255 - arr
    else:
        arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise ValueError(f"OpenCV could not read image: {path}")

    # Gentle contrast normalization for consistent model input.
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    arr = clahe.apply(arr)
    rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)
