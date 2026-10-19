"""WorldLA configuration independent of legacy OpenPI experiment names."""
from pathlib import Path


def create_training_config(args):
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.training.config import AssetsConfig, DataConfig, LeRobotAlohaDataConfig, TrainConfig
    model = Pi0Config(paligemma_variant='gemma_2b_lora', action_expert_variant='gemma_300m_lora', pi05=True)
    return TrainConfig(
        name='worldla_pi05', project_name='worldla', model=model,
        assets_base_dir=str(Path(args.asset_dir).expanduser().resolve()),
        data=LeRobotAlohaDataConfig(
            repo_id=str(args.dataset_repo_id), adapt_to_pi=False,
            assets=AssetsConfig(assets_dir=str(Path(args.asset_dir).expanduser().resolve()), asset_id=args.asset_id),
            repack_transforms=transforms.Group(inputs=[transforms.RepackTransform({
                'images': {camera: f'observation.images.{camera}' for camera in
                           ('cam_high', 'cam_left_wrist', 'cam_right_wrist')},
                'state': 'observation.state', 'actions': 'action', 'prompt': 'prompt',
            })]),
            base_config=DataConfig(prompt_from_task=True,
                                   lerobot_data_root=str(Path(args.lerobot_root).expanduser().resolve())),
        ),
        freeze_filter=model.get_freeze_filter(), wandb_enabled=False,
    )
