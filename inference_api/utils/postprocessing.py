from __future__ import annotations
from typing import List, Dict, Sequence, Optional
import numpy as np

def topk_indices(probs: np.ndarray, k: int) -> np.ndarray:
    """
    Return indices of the top-k probabilities, sorted descending by prob.
    Safe for any k (clamped to [1, len(probs)]).
    """
    p = probs.reshape(-1)
    k = int(max(1, min(k, p.size)))
    idx = np.argpartition(-p, range(k))[:k]
    return idx[np.argsort(-p[idx])]

def map_indices_to_labels(
    indices: Sequence[int],
    probs: np.ndarray,
    labels: Optional[Sequence[str]] = None
) -> List[Dict[str, float]]:
    """
    Map indices -> [{'class': <label or index>, 'prob': <float>}], preserving the
    order of `indices`. If labels is None/short, falls back to stringified index.
    """
    p = probs.reshape(-1)
    labels = list(labels) if labels is not None else []
    out: List[Dict[str, float]] = []
    for i in indices:
        name = labels[i] if i < len(labels) else str(int(i))
        out.append({"id": int(i), "class": name, "prob": float(p[i])})
    return out
