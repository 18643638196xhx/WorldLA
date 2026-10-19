"""Pure NumPy folding of this OpenPI implementation's scanned LoRA weights."""
from __future__ import annotations

import numpy as np


def assemble_original(flat_base: dict, flat_delta: dict) -> dict:
    """Package the deployed base and increments without modifying any leaf."""
    collisions = flat_base.keys() & flat_delta.keys()
    if collisions:
        raise ValueError(f"Base/increment parameter collision: {sorted(collisions)}")
    return {**flat_base, **flat_delta}


def lora_pair(key: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    if key[-1] == "lora_a":
        return key[:-1] + ("lora_b",), key[:-1] + ("w",), "attn"
    if key[-1].endswith("_lora_a"):
        stem = key[-1][:-len("_lora_a")]
        if stem not in {"gating_einsum", "linear"}:
            raise ValueError(f"Unsupported LoRA weight: {key}")
        return key[:-1] + (stem + "_lora_b",), key[:-1] + (stem,), "ffn"
    raise ValueError(f"Not a supported LoRA A leaf: {key}")


def fold_lora(flat_base: dict, flat_delta: dict, attention_scales: dict[str, float]) -> tuple[dict, list[dict]]:
    """Inputs already contain the bf16-rounded parameters used by deployment.

    Preserve float32 folded matrices to avoid an extra rounding during export.
    Attention follows LoRAConfig.scaling_value. OpenPI FeedForward._dot does
    NOT apply that scale, so its exact implemented scale is one.
    """
    result = dict(flat_base)
    consumed, reports = set(), []
    for key, a in flat_delta.items():
        if "lora" not in "/".join(key):
            continue
        if not (key[-1] == "lora_a" or key[-1].endswith("_lora_a")):
            continue
        pair, target, kind = lora_pair(key)
        if pair not in flat_delta or target not in result:
            raise ValueError(f"Missing LoRA pair or dense target: {key}")
        b, weight = flat_delta[pair], result[target]
        if a.shape[:-2] != b.shape[:-2] or a.shape[-1] != b.shape[-2]:
            raise ValueError(f"LoRA factor shape mismatch: {key}: {a.shape}, {b.shape}")
        if (*a.shape[:-2], a.shape[-2], b.shape[-1]) != weight.shape:
            raise ValueError(f"Dense target shape mismatch: {target}")
        branch = "action" if any(part.endswith("_1") for part in key) else "vision_language"
        scale = attention_scales[branch] if kind == "attn" else 1.0
        right = np.asarray(b, np.float32)
        contraction = "batched A @ B"
        if kind == "attn" and target[-2] in {"attn_vec_einsum", "attn_vec_einsum_1"}:
            # Exact source equations for BTNH,NHD->BTD are:
            # BTNH,NHL->BTL then BTL,NLD->BTD. N is already summed in
            # the first contraction, so B must be summed over heads too.
            # Headwise A_n @ B_n is NOT equivalent to this implementation.
            right = right.sum(axis=-3, keepdims=True)
            contraction = "A @ sum_heads(B); matches source BTNH,NHL->BTL; BTL,NLD->BTD"
        merged = np.asarray(weight, np.float32) + scale * np.matmul(np.asarray(a, np.float32), right)
        if not np.isfinite(merged).all():
            raise ValueError(f"Non-finite folded weight: {target}")
        result[target] = merged
        consumed.update((key, pair))
        reports.append({"target": "/".join(target), "kind": kind, "branch": branch,
                        "rank": a.shape[-1], "scale": scale, "shape": list(merged.shape), "contraction": contraction})
    expected_lora = {key for key in flat_delta if "lora" in "/".join(key)}
    if consumed != expected_lora:
        raise ValueError(f"Unconsumed LoRA leaves: {sorted(expected_lora - consumed)}")
    for key, value in flat_delta.items():
        if key in consumed:
            continue
        if key[0] not in {"stage_relation_router", "stage_message_adapter"}:
            raise ValueError(f"Unexpected non-LoRA incremental parameter: {key}")
        if key in result:
            raise ValueError(f"Unexpected collision at {key}")
        result[key] = value
    if any("lora" in "/".join(key) for key in result):
        raise ValueError("LoRA factors remain after folding")
    return result, reports


def validate_tree(expected: dict, actual: dict) -> None:
    if expected.keys() != actual.keys():
        raise ValueError(f"Parameter keys differ; missing={sorted(expected.keys()-actual.keys())}; extra={sorted(actual.keys()-expected.keys())}")
    for key, spec in expected.items():
        value = actual[key]
        if spec is None or value is None:
            if spec is not None or value is not None:
                raise ValueError(f"None/array mismatch at {key}")
        elif spec.shape != value.shape:
            raise ValueError(f"Parameter shape mismatch at {key}: {spec.shape} != {value.shape}")
