"""Released fixed-weight tail fusion, independent of simulator installation."""
import numpy as np


def blend_plan(previous, candidate, *, k=10, old_weight=0.75):
    if not 0 <= old_weight <= 1 or k <= 0:
        raise ValueError("Invalid fusion coefficient or execution horizon")
    result = np.asarray(candidate, dtype=np.float64).copy()
    if previous is None:
        return result
    previous = np.asarray(previous)
    count = min(k, len(result), max(0, len(previous) - k))
    if count == 0:
        return result
    dim = result.shape[-1]
    grippers = [dim // 2 - 1, dim - 1] if dim >= 4 and dim % 2 == 0 else []
    arms = [i for i in range(dim) if i not in grippers]
    result[:count, arms] = old_weight * previous[k:k+count, arms] + (1-old_weight) * result[:count, arms]
    if grippers and (1-old_weight) < .5:
        result[:count, grippers] = previous[k:k+count, grippers]
    return result
