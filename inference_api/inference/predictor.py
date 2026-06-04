from __future__ import annotations
from typing import List, Dict, Any
import io
import numpy as np
from PIL import Image
import onnxruntime as ort

def _preprocess(img_bytes: bytes, image_size: int, cfg: Dict[str, Any]) -> np.ndarray:
    # Decode to RGB, resize
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((image_size, image_size))
    arr = np.asarray(img, dtype=np.float32)

    # Scale to [0, 1] if requested (matches torchvision.ToTensor behavior)
    if cfg.get("preprocess", {}).get("scale_to_unit", True):
        arr /= 255.0

    # Optional mean/std normalization
    if cfg.get("preprocess", {}).get("normalize", False):
        mean = np.array(cfg["preprocess"].get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        std  = np.array(cfg["preprocess"].get("std",  [0.229, 0.224, 0.225]), dtype=np.float32)
        arr = (arr - mean) / std  # broadcasts over channel axis

    # RGB -> NCHW, add batch dim, ensure contiguous floa32
    x = np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)
    return np.ascontiguousarray(x)

def _softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max()
    exps = np.exp(logits - m)
    return exps / (exps.sum() + 1e-12)

def run_inference_ort(
    session: ort.InferenceSession,
    meta: Dict[str, Any],
    img_bytes: bytes
) -> List[Dict[str, Any]]:
    """Return 1D probs array for the uploaded image."""
    
    x = _preprocess(img_bytes, meta["image_size"], meta)
    input_name = meta.get("input_name") or session.get_inputs()[0].name
    outputs = session.run(None, {input_name: x})
    y = outputs[0].reshape(-1).astype(np.float32)
    
    return _softmax(y) if meta.get("output_is_logits", True) else y
