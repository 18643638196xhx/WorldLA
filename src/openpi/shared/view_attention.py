from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class InferenceStepSummary:
    step_idx: int
    time_value: float
    layer_view_ratios: list[dict[str, float]] = field(default_factory=list)

    def aggregated_view_ratios(self) -> dict[str, float]:
        if not self.layer_view_ratios:
            return {}
        keys = self.layer_view_ratios[0].keys()
        return {
            key: float(np.mean([layer_ratio[key] for layer_ratio in self.layer_view_ratios]))
            for key in keys
        }


@dataclass
class InferenceSummary:
    record_id: str
    step_summaries: list[InferenceStepSummary] = field(default_factory=list)

    def aggregated_view_ratios(self) -> dict[str, float]:
        per_step = [step.aggregated_view_ratios() for step in self.step_summaries if step.layer_view_ratios]
        if not per_step:
            return {}
        keys = per_step[0].keys()
        return {
            key: float(np.mean([step_ratio[key] for step_ratio in per_step]))
            for key in keys
        }


class ViewAttentionRecorder:
    """Collects per-layer visual attention ratios during eager baseline inference."""

    def __init__(self):
        self._view_ranges: list[tuple[str, int, int]] = []
        self._current_inference: InferenceSummary | None = None
        self._current_step: InferenceStepSummary | None = None
        self._summaries: list[InferenceSummary] = []

    def set_view_ranges(self, view_ranges: list[tuple[str, int, int]]) -> None:
        self._view_ranges = view_ranges

    def begin_inference(self, record_id: str) -> None:
        self._current_inference = InferenceSummary(record_id=record_id)
        self._current_step = None

    def end_inference(self) -> None:
        if self._current_inference is not None:
            self._summaries.append(self._current_inference)
        self._current_inference = None
        self._current_step = None

    def begin_step(self, step_idx: int, time_value: float) -> None:
        if self._current_inference is None:
            raise RuntimeError("begin_inference must be called before begin_step")
        step = InferenceStepSummary(step_idx=step_idx, time_value=float(time_value))
        self._current_inference.step_summaries.append(step)
        self._current_step = step

    def record(self, probs_np: Any, prefix_len: int) -> None:
        if self._current_step is None or not self._view_ranges:
            return

        probs_np = np.asarray(probs_np, dtype=np.float32)
        if probs_np.ndim != 5:
            return

        prefix_len = int(prefix_len)
        if prefix_len <= 0:
            return

        visual_mass = 0.0
        masses: dict[str, float] = {}
        for view_name, start, end in self._view_ranges:
            clipped_start = max(0, min(prefix_len, start))
            clipped_end = max(clipped_start, min(prefix_len, end))
            mass = float(probs_np[..., clipped_start:clipped_end].sum())
            masses[view_name] = mass
            visual_mass += mass

        if visual_mass <= 0:
            return

        ratios = {view_name: mass / visual_mass for view_name, mass in masses.items()}
        self._current_step.layer_view_ratios.append(ratios)

    def finalize(self) -> dict[str, Any]:
        summaries = []
        for inference in self._summaries:
            per_step = []
            for step in inference.step_summaries:
                step_ratios = step.aggregated_view_ratios()
                if step_ratios:
                    top_view = max(step_ratios, key=step_ratios.get)
                else:
                    top_view = None
                per_step.append(
                    {
                        "step_idx": step.step_idx,
                        "time_value": step.time_value,
                        "view_ratios": step_ratios,
                        "top_view": top_view,
                    }
                )

            inference_ratios = inference.aggregated_view_ratios()
            summaries.append(
                {
                    "record_id": inference.record_id,
                    "view_ratios": inference_ratios,
                    "top_view": max(inference_ratios, key=inference_ratios.get) if inference_ratios else None,
                    "steps": per_step,
                }
            )

        overall = {}
        valid = [entry["view_ratios"] for entry in summaries if entry["view_ratios"]]
        if valid:
            keys = valid[0].keys()
            overall = {
                key: float(np.mean([entry[key] for entry in valid]))
                for key in keys
            }

        return {
            "overall_view_ratios": overall,
            "overall_top_view": max(overall, key=overall.get) if overall else None,
            "records": summaries,
        }


_ACTIVE_RECORDER: ViewAttentionRecorder | None = None


def set_active_recorder(recorder: ViewAttentionRecorder | None) -> None:
    global _ACTIVE_RECORDER
    _ACTIVE_RECORDER = recorder


def get_active_recorder() -> ViewAttentionRecorder | None:
    return _ACTIVE_RECORDER


def host_record_attention(probs_np: Any, prefix_len: Any) -> None:
    recorder = get_active_recorder()
    if recorder is None:
        return
    recorder.record(probs_np, int(prefix_len))
