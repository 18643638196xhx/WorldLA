"""WorldLA configuration independent of legacy OpenPI experiment names."""
from pathlib import Path


def create_training_config(args):
    from worldla.data import transforms
    from worldla.models.configuration import WorldLAConfig
    from worldla.training.settings import AssetConfig
    from worldla.training.settings import DatasetConfig
    from worldla.training.settings import AlohaDatasetConfig
    from worldla.training.settings import TrainingConfig
    model = WorldLAConfig(paligemma_variant='gemma_2b_lora', action_expert_variant='gemma_300m_lora', pi05=True)
    return TrainingConfig(
        name='worldla_pi05', project_name='worldla', model=model,
        assets_base_dir=str(Path(args.asset_dir).expanduser().resolve()),
        data=AlohaDatasetConfig(
            repo_id=str(args.dataset_repo_id), adapt_to_pi=False,
            assets=AssetConfig(assets_dir=str(Path(args.asset_dir).expanduser().resolve()), asset_id=args.asset_id),
            repack_transforms=transforms.Group(inputs=[transforms.RepackTransform({
                'images': {camera: f'observation.images.{camera}' for camera in
                           ('cam_high', 'cam_left_wrist', 'cam_right_wrist')},
                'state': 'observation.state', 'actions': 'action', 'prompt': 'prompt',
            })]),
            base_config=DatasetConfig(prompt_from_task=True,
                                   lerobot_data_root=str(Path(args.lerobot_root).expanduser().resolve())),
        ),
        freeze_filter=model.get_freeze_filter(), wandb_enabled=False,
    )
