#!/usr/bin/env python3
"""Convert Fast-WAM per-(denoise, action) shared-K routes into four stages."""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

import numpy as np


from worldla.data.spatial import to_pi_tokens

SCHEMA = "multi_task4_stage_relation_route_v1"
NUM_STAGES = 4


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, path)


def normalize_last(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    return values / np.maximum(values.sum(axis=-1, keepdims=True), 1e-12)


def four_stage_route(arrays, grid_size: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    if "shared_k_route_abs_ta" in arrays:
        source = np.asarray(arrays["shared_k_route_abs_ta"], dtype=np.float32)
    elif "shared_k_route_probability_ta" in arrays:
        source = np.asarray(arrays["shared_k_route_probability_ta"], dtype=np.float32)
    else:
        raise RuntimeError(
            "Teacher cache has no per-(t,action) route; rerun Fast-WAM extraction"
        )
    if source.ndim != 3 or source.shape[-1] != 120:
        raise ValueError(f"Expected [T,A,120] route tensor, got {source.shape}")
    if source.shape[0] < 1 or source.shape[1] < NUM_STAGES or not np.isfinite(source).all():
        raise ValueError('Teacher routes require finite values and nonempty action stages')
    action_bins = np.array_split(np.arange(source.shape[1]), NUM_STAGES)
    stage_fastwam = []
    stage_strength = []
    for indices in action_bins:
        # Aggregate actual shared-K mass over denoising steps and action positions.
        mass = np.maximum(source[:, indices, :], 0.0).sum(axis=(0, 1))
        stage_strength.append(float(mass.sum()))
        mass /= max(float(mass.sum()), 1e-12)
        stage_fastwam.append(mass)
    stage_pi = np.stack(
        [to_pi_tokens(route.reshape(12, 10), grid_size) for route in stage_fastwam]
    ).astype(np.float32)
    return normalize_last(stage_pi), np.asarray(stage_strength, np.float32), source.shape[0], source.shape[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pi-grid-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    manifest_path = output / "stage_route_manifest.jsonl"
    if manifest_path.exists() and not args.overwrite:
        print(f"Stage route dataset already exists: {manifest_path}")
        return
    manifests = [path.expanduser().resolve() for path in args.teacher_manifest]
    rows = [{**row, "_manifest_dir": str(path.parent)} for path in manifests for row in read_jsonl(path)]
    rows.sort(key=lambda row: (int(row["episode_index"]), int(row["frame_index"])))
    if not rows:
        raise ValueError("No teacher rows")
    samples_dir = output / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    target_by_index = {}
    t_count = action_count = None
    for position, row in enumerate(rows, 1):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', str(row['sample_id'])):
            raise ValueError('sample_id must be a filename-safe identifier')
        teacher = Path(row["teacher_npz"]).expanduser()
        if not teacher.is_absolute():
            teacher = Path(row["_manifest_dir"]) / teacher
        teacher = teacher.resolve()
        with np.load(teacher, allow_pickle=False) as arrays:
            target, strength, current_t, current_a = four_stage_route(
                arrays, args.pi_grid_size
            )
        if t_count is None:
            t_count, action_count = current_t, current_a
        elif (current_t, current_a) != (t_count, action_count):
            raise ValueError("Teacher t/action shapes differ across samples")
        global_index = int(row["global_index"])
        if global_index in target_by_index:
            raise ValueError(f"Duplicate global index: {global_index}")
        target_by_index[global_index] = target
        target_path = samples_dir / f"{row['sample_id']}.npz"
        tmp = target_path.with_suffix(".npz.tmp")
        with tmp.open("wb") as handle:
            np.savez_compressed(
                handle,
                schema=np.asarray(SCHEMA),
                stage_route_target=target.astype(np.float16),
                stage_route_strength=strength,
            )
        os.replace(tmp, target_path)
        output_rows.append(
            {
                **{k: v for k, v in row.items() if k != "_manifest_dir"},
                "schema": SCHEMA,
                "stage_route_npz": str(target_path.relative_to(output)),
            }
        )
        if position % 100 == 0 or position == len(rows):
            print(f"[stage route dataset] {position}/{len(rows)}", flush=True)

    train_targets = np.stack(
        [
            target_by_index[int(row["global_index"])]
            for row in output_rows
            if row["split"] == "train"
        ]
    )
    global_stage_mean = normalize_last(train_targets.mean(axis=0))
    family_groups: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    for row in output_rows:
        if row["split"] == "train":
            family_groups[str(row["family"])].append(
                target_by_index[int(row["global_index"])]
            )
    family_names = sorted(family_groups)
    family_stage_means = np.stack(
        [normalize_last(np.stack(family_groups[name]).mean(axis=0)) for name in family_names]
    )
    with (output / "train_stage_priors.npz").open("wb") as handle:
        np.savez_compressed(
            handle,
            global_stage_mean=global_stage_mean.astype(np.float16),
            family_names=np.asarray(family_names),
            family_stage_means=family_stage_means.astype(np.float16),
        )
    atomic_jsonl(manifest_path, output_rows)
    atomic_json(
        output / "metadata.json",
        {
            "schema": SCHEMA,
            "teacher_manifests": [str(path) for path in manifests],
            "num_samples": len(output_rows),
            "num_stages": NUM_STAGES,
            "fastwam_denoise_steps": t_count,
            "fastwam_action_horizon": action_count,
            "fastwam_stage_bins": [list(map(int, x)) for x in np.array_split(np.arange(action_count), NUM_STAGES)],
            "pi05_action_horizon": 50,
            "pi05_stage_rule": "floor(action_index * 4 / 50)",
            "num_route_tokens": 3 * args.pi_grid_size**2,
            "route_target": (
                "Four independently normalized spatial routes. Denoising steps are "
                "aggregated, while Fast-WAM action positions are retained as four "
                "relative action-chunk stages."
            ),
            "controls": {
                "position_only": "global train-only four-stage spatial template; model ignores observation",
                "stage_shuffled": "same observation routes with stage labels permuted [2,0,3,1]",
                "static_average": "same observation mean route repeated across all four stages",
                "base_lora": "no route teacher and no stage-message injection",
            },
        },
    )
    print(f"Saved four-stage route dataset to {output}")


if __name__ == "__main__":
    main()
