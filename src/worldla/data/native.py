from __future__ import annotations
from pathlib import Path

def create_native_dataset(config, lerobot_root: Path):
    import os

    from worldla.training import datasets as data_loader

    os.environ["LEROBOT_DATA_ROOT"] = str(lerobot_root.expanduser().resolve())
    data_config = config.data.create(config.assets_dirs, config.model)
    raw = data_loader.create_torch_dataset(
        data_config, config.model.action_horizon, config.model
    )
    return data_loader.transform_dataset(raw, data_config), data_config


def create_batch_loader(dataset, *, batch_size: int, shuffle: bool, seed: int):
    import torch
    from worldla.training.datasets import _collate_fn

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=shuffle,
        collate_fn=_collate_fn,
        generator=generator,
    )
