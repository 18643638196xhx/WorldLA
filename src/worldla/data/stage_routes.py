from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from worldla.data.native import (
    create_batch_loader,
    create_native_dataset,
)


CONDITIONS = (
    "wam_stage_dynamic",
    "position_only",
    "stage_shuffled",
    "static_average",
    "stage_randomized",
    "global_feature_distill",
    "router_action_only",
    "base_lora",
)
ROUTE_CONDITIONS = tuple(x for x in CONDITIONS if x not in {"base_lora", "router_action_only"})
STAGE_PERMUTATION = np.asarray([2, 0, 3, 1], dtype=np.int64)


def randomize_stage_targets(targets, indices, *, seed: int, step: int):
    """Fresh per-sample derangements for training; step=0 fixes validation.

    A single global permutation can be learned back and is not this control.
    Every selected permutation moves all four stages, while preserving each
    sample's spatial maps, entropy, and total supervision mass.
    """
    import itertools
    permutations = np.asarray([
        p for p in itertools.permutations(range(4))
        if all(i != j for i, j in enumerate(p))
    ], dtype=np.int64)
    targets = np.asarray(targets)
    if targets.ndim != 3 or targets.shape[1] != 4:
        raise ValueError(f"Expected [batch,4,visual] targets, got {targets.shape}")
    indices = np.asarray(indices).reshape(-1)
    if len(indices) != len(targets):
        raise ValueError("One global index is required per route target")
    result = np.empty_like(targets)
    for i, index in enumerate(indices):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(step), int(index)]))
        result[i] = targets[i, permutations[rng.integers(len(permutations))]]
    return result


class GlobalFeatureTargetStore:
    """Load genuine pooled WAM hidden features, never shared-K route maps."""

    def __init__(self, manifest_path: Path, route_rows: list[dict]):
        manifest_path = manifest_path.expanduser().resolve()
        self.targets = {}
        self.rows = {}
        self.feature_dim = None
        self.provenance = None
        for row in read_jsonl(manifest_path):
            path = Path(row["global_feature_npz"])
            if not path.is_absolute():
                path = manifest_path.parent / path
            with np.load(path, allow_pickle=False) as arrays:
                if str(arrays["schema"].item()) != "worldla_global_wam_feature_v1":
                    raise ValueError(f"Not a global WAM hidden feature cache: {path}")
                if int(arrays["global_index"].item()) != int(row["global_index"]):
                    raise ValueError(f"Global feature anchor mismatch: {path}")
                provenance = {
                    "checkpoint": str(arrays["checkpoint"].item()),
                    "layer": int(arrays["layer"].item()),
                    "inference_seed": int(arrays["inference_seed"].item()),
                }
                value = np.asarray(arrays["global_visual_feature"], np.float32)
            if self.provenance is not None and provenance != self.provenance:
                raise ValueError(f"Mixed global feature teacher provenance: {path}")
            self.provenance = provenance
            if value.ndim != 1 or not np.isfinite(value).all() or np.linalg.norm(value) <= 0:
                raise ValueError(f"Invalid pooled visual feature in {path}")
            self.feature_dim = len(value) if self.feature_dim is None else self.feature_dim
            if len(value) != self.feature_dim:
                raise ValueError("Global teacher feature dimensions differ")
            index = int(row["global_index"])
            if index in self.targets:
                raise ValueError(f"Duplicate global feature index {index}")
            self.targets[index] = value / max(float(np.linalg.norm(value)), 1e-8)
            self.rows[index] = row
        if not self.targets:
            raise ValueError("Empty global feature manifest")
        expected_indices = {int(row["global_index"]) for row in route_rows}
        if set(self.targets) != expected_indices:
            raise ValueError("Global feature anchors must exactly match the route manifest")
        for row in route_rows:
            index = int(row["global_index"])
            if index not in self.targets:
                raise ValueError(f"Missing global feature for anchor {index}")
            teacher = self.rows[index]
            for key in ("sample_id", "episode_index", "frame_index", "split", "family"):
                if teacher[key] != row[key]:
                    raise ValueError(f"Global feature / route manifest mismatch: index={index}, field={key}")

    def target(self, index):
        return self.targets[int(index)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_last(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    return values / np.maximum(values.sum(axis=-1, keepdims=True), 1e-12)


class StageRouteTargetStore:
    def __init__(self, manifest_path: Path, *, load_routes: bool = True):
        self.manifest_path = manifest_path.expanduser().resolve()
        self.rows = read_jsonl(self.manifest_path)
        if not self.rows:
            raise ValueError(f"Empty manifest: {self.manifest_path}")
        seen = set()
        episode_split = {}
        for row in self.rows:
            index, episode = int(row['global_index']), int(row['episode_index'])
            if index in seen:
                raise ValueError(f'Duplicate global index: {index}')
            seen.add(index)
            if row['split'] not in {'train','val','test'}:
                raise ValueError(f'Invalid split: {row["split"]}')
            if episode in episode_split and episode_split[episode] != row['split']:
                raise ValueError(f'Episode {episode} appears in multiple splits')
            episode_split[episode] = row['split']
        self.row_by_index = {}
        self.target_by_index = {}
        if not load_routes:
            # R0/R1/R2 use this manifest ONLY for exact episode splits/anchors.
            metadata_path = self.manifest_path.parent / "metadata.json"
            metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            self.num_stages = int(metadata.get("num_stages", 4))
            self.num_route_tokens = int(metadata.get("num_route_tokens", 768))
            self.global_stage_mean = np.full(
                (self.num_stages, self.num_route_tokens), 1.0 / self.num_route_tokens, np.float32
            )
            self.row_by_index = {int(row["global_index"]): row for row in self.rows}
            return
        shape = None
        for row in self.rows:
            path = Path(row["stage_route_npz"])
            if not path.is_absolute():
                path = (self.manifest_path.parent / path).resolve()
            with np.load(path, allow_pickle=False) as arrays:
                target = normalize_last(arrays["stage_route_target"])
            shape = target.shape if shape is None else shape
            if target.shape != shape:
                raise ValueError(f"Stage target shape mismatch: {target.shape} != {shape}")
            index = int(row["global_index"])
            self.row_by_index[index] = row
            self.target_by_index[index] = target
        if shape is None or len(shape) != 2:
            raise ValueError(f"Expected [stage,visual_token] targets, got {shape}")
        self.num_stages, self.num_route_tokens = map(int, shape)
        with np.load(self.manifest_path.parent / "train_stage_priors.npz", allow_pickle=False) as arrays:
            self.global_stage_mean = normalize_last(arrays["global_stage_mean"])

    def target(self, global_index: int, condition: str) -> np.ndarray:
        if condition in {"base_lora", "router_action_only", "global_feature_distill"}:
            return np.full((self.num_stages, self.num_route_tokens), 1.0 / self.num_route_tokens, np.float32)
        original = self.target_by_index[int(global_index)]
        if condition in {"wam_stage_dynamic", "stage_randomized"}:
            return original
        if condition == "position_only":
            return self.global_stage_mean
        if condition == "stage_shuffled":
            return original[STAGE_PERMUTATION]
        if condition == "static_average":
            mean = normalize_last(original.mean(axis=0, keepdims=True))
            return np.repeat(mean, self.num_stages, axis=0)
        raise ValueError(f"Unknown condition: {condition}")


class StageRouteOpenPiDataset:
    def __init__(self, transformed, store: StageRouteTargetStore, *, split: str, condition: str, feature_store=None):
        if condition not in CONDITIONS:
            raise ValueError(condition)
        self.transformed = transformed
        self.store = store
        self.condition = condition
        self.feature_store = feature_store
        self.rows = [row for row in store.rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No {split} rows")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        row = self.rows[int(item)]
        index = int(row["global_index"])
        sample = dict(self.transformed[index])
        sample["route_target"] = (
            self.feature_store.target(index) if self.feature_store is not None
            else self.store.target(index, self.condition)
        ).astype(np.float32)
        sample["route_position_mean"] = self.store.global_stage_mean.astype(np.float32)
        sample["route_global_index"] = np.asarray(index, np.int64)
        sample["route_episode_index"] = np.asarray(int(row["episode_index"]), np.int64)
        return sample


class FullEpisodeActionDataset:
    def __init__(self, transformed, store: StageRouteTargetStore, *, split: str):
        split_by_episode = {}
        start_by_episode = {}
        for row in store.rows:
            episode = int(row["episode_index"])
            start = int(row["global_index"]) - int(row["frame_index"])
            if episode in split_by_episode and split_by_episode[episode] != row["split"]:
                raise ValueError(f"Episode {episode} appears in multiple splits")
            split_by_episode[episode] = str(row["split"])
            start_by_episode[episode] = start
        ordered = sorted(start_by_episode.items(), key=lambda item: item[1])
        self.indices = []
        for i, (episode, start) in enumerate(ordered):
            end = ordered[i + 1][1] if i + 1 < len(ordered) else len(transformed)
            if split_by_episode[episode] == split:
                self.indices.extend((index, episode) for index in range(start, end))
        if not self.indices:
            raise ValueError(f"No full-frame samples for split={split}")
        self.transformed = transformed
        self.dummy = np.full(
            (store.num_stages, store.num_route_tokens),
            1.0 / store.num_route_tokens,
            np.float32,
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index, episode = self.indices[int(item)]
        sample = dict(self.transformed[index])
        sample["route_target"] = self.dummy
        sample["route_position_mean"] = self.dummy
        sample["route_global_index"] = np.asarray(index, np.int64)
        sample["route_episode_index"] = np.asarray(episode, np.int64)
        return sample


def split_batch(batch: dict[str, Any], *, sharding=None):
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_api

    if sharding is None:
        batch = jax.tree.map(jnp.asarray, batch)
    else:
        batch = jax.tree.map(
            lambda value: jax.make_array_from_process_local_data(sharding, np.asarray(value)),
            batch,
        )
    route = batch.pop("route_target")
    position_mean = batch.pop("route_position_mean")
    global_index = batch.pop("route_global_index")
    episode_index = batch.pop("route_episode_index")
    actions = batch["actions"]
    return (
        model_api.Observation.from_dict(batch), actions, route, position_mean,
        global_index, episode_index,
    )
