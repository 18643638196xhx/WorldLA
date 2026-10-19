#!/usr/bin/env python3
"""Build task-stratified train/val/test route anchors for the 50-task dataset."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--block-tasks",
        default=None,
        help=(
            "Comma-separated task families for repositories stored as contiguous "
            "episode blocks. Used only when meta/official_episodes.jsonl is absent."
        ),
    )
    parser.add_argument("--frames-per-episode", type=int, default=8)
    parser.add_argument("--expected-tasks", type=int, default=50)
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--train-episodes", type=int, default=40)
    parser.add_argument("--val-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    episodes = {int(row["episode_index"]): row for row in read_jsonl(root / "meta/episodes.jsonl")}
    official_path = root / "meta/official_episodes.jsonl"
    if official_path.is_file():
        official = read_jsonl(official_path)
    else:
        block_tasks = [item.strip() for item in (args.block_tasks or "").split(",") if item.strip()]
        if len(block_tasks) != args.expected_tasks:
            raise ValueError(
                f"{official_path} is absent; --block-tasks must provide exactly "
                f"{args.expected_tasks} task names"
            )
        official = []
        for episode_index in sorted(episodes):
            task_position = episode_index // args.episodes_per_task
            if task_position >= len(block_tasks):
                raise ValueError(f"Episode {episode_index} exceeds the declared task blocks")
            task_values = episodes[episode_index].get("tasks", [])
            if len(task_values) != 1:
                raise ValueError(f"Expected one instruction for episode {episode_index}")
            official.append(
                {
                    "episode_index": episode_index,
                    "official_task": block_tasks[task_position],
                    "instruction": str(task_values[0]),
                }
            )
    expected_episodes = args.expected_tasks * args.episodes_per_task
    if len(official) != expected_episodes:
        raise ValueError(
            f"Expected {expected_episodes} official episodes, found {len(official)}"
        )
    grouped: dict[str, list[int]] = collections.defaultdict(list)
    for row in official:
        grouped[str(row["official_task"])].append(int(row["episode_index"]))
    if len(grouped) != args.expected_tasks or any(
        len(values) != args.episodes_per_task for values in grouped.values()
    ):
        raise ValueError(
            f"Expected exactly {args.episodes_per_task} episodes for each of "
            f"{args.expected_tasks} tasks"
        )

    rng = np.random.default_rng(args.seed)
    split_by_episode = {}
    split_counts = {}
    for task, values in grouped.items():
        values = np.asarray(values, dtype=np.int64)
        rng.shuffle(values)
        n_train, n_val = args.train_episodes, args.val_episodes
        if n_train + n_val >= len(values):
            raise ValueError("Train+val must leave at least one test episode per task")
        for split, subset in (
            ("train", values[:n_train]),
            ("val", values[n_train : n_train + n_val]),
            ("test", values[n_train + n_val :]),
        ):
            split_by_episode.update({int(index): split for index in subset})
        split_counts[task] = {"train": n_train, "val": n_val, "test": len(values) - n_train - n_val}

    rows = []
    chunks_size = int(info["chunks_size"])
    pattern = str(info["data_path"])
    global_offset = 0
    for metadata in official:
        episode_index = int(metadata["episode_index"])
        episode = episodes[episode_index]
        length = int(episode["length"])
        parquet = root / pattern.format(
            episode_chunk=episode_index // chunks_size, episode_index=episode_index
        )
        table = pq.read_table(parquet, columns=["index", "frame_index"])
        indices = np.asarray(table["index"].to_numpy(), dtype=np.int64)
        frames = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        if table.num_rows != length or indices[0] != global_offset:
            raise RuntimeError(f"Inconsistent episode/global index at {parquet}")
        if not np.array_equal(frames, np.arange(length)):
            raise RuntimeError(f"Non-contiguous frames at {parquet}")
        max_start = max(0, length - args.action_horizon)
        anchors = np.unique(
            np.rint(np.linspace(0, max_start, args.frames_per_episode)).astype(np.int64)
        )
        for frame in anchors:
            rows.append(
                {
                    "schema": "robotwin50_stage_route_anchor_v1",
                    "sample_id": f"ep{episode_index:06d}_f{int(frame):04d}",
                    "split": split_by_episode[episode_index],
                    "family": metadata["official_task"],
                    "phase": min(3, int(4 * frame / max(max_start + 1, 1))),
                    "episode_index": episode_index,
                    "frame_index": int(frame),
                    "global_index": global_offset + int(frame),
                    "task_index": episode_index // args.episodes_per_task,
                    "instruction": metadata["instruction"],
                    "parquet": str(parquet),
                }
            )
        global_offset += length
    output = args.output_dir.expanduser().resolve()
    write_jsonl(output / "sample_manifest.jsonl", rows)
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "schema": "robotwin50_stage_route_anchor_v1",
                "dataset_root": str(root),
                "num_tasks": len(grouped),
                "num_episodes": len(official),
                "num_samples": len(rows),
                "frames_per_episode": args.frames_per_episode,
                "split_counts_by_task": split_counts,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"Saved {len(rows)} anchors for {len(grouped)} tasks to {output}")


if __name__ == "__main__":
    main()
