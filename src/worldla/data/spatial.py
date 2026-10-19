from __future__ import annotations
import numpy as np

def resize_preserving_mass(values: np.ndarray, size: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    mass = float(source.sum())
    if mass <= 0:
        return np.zeros((size, size), dtype=np.float32)
    import torch
    import torch.nn.functional as F
    tensor = torch.from_numpy(source)[None, None]
    output = F.interpolate(
        tensor, size=(size, size), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    output = np.maximum(output, 0.0)
    return (output * (mass / max(float(output.sum()), 1e-12))).astype(np.float32)


def to_pi_tokens(heatmap: np.ndarray, grid_size: int) -> np.ndarray:
    """Map Fast-WAM's composite grid to pi0.5's three SigLIP token grids."""
    values = np.asarray(heatmap, dtype=np.float32)
    if values.shape != (12, 10):
        raise ValueError(f"Expected Fast-WAM 12x10 heatmap, got {values.shape}")
    # The composite is high camera on top (8x10), then left/right wrists
    # side-by-side on the bottom (4x5 each).
    views = (
        values[:8, :],
        values[8:, :5],
        values[8:, 5:],
    )
    target = np.concatenate(
        [resize_preserving_mass(view, grid_size).reshape(-1) for view in views]
    )
    target = np.maximum(target, 0.0)
    total = float(target.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Route heatmap has no finite positive mass")
    return (target / total).astype(np.float32)
