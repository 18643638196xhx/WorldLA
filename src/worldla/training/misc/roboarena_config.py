"""RoboArena baseline policy configs."""

from typing import TypeAlias

import worldla.models.interfaces as _model
import worldla.models.configuration as backbone_config
import worldla.models.token_backbone as token_backbone
import worldla.models.tokenization as _tokenizer
import worldla.policies.droid as droid_policy
import worldla.data.transforms as _transforms

ModelFamily: TypeAlias = _model.ModelFamily


def get_roboarena_configs():
    # Import here to avoid circular imports.
    from worldla.training.settings import AssetConfig
    from worldla.training.settings import DatasetConfig
    from worldla.training.settings import SimpleDatasetConfig
    from worldla.training.settings import TrainingConfig

    return [
        #
        # RoboArena DROID baseline inference configs.
        #
        TrainingConfig(
            # Trained from PaliGemma, using RT-2 / OpenVLA style binning tokenizer.
            name="paligemma_binning_droid",
            model=token_backbone.WorldLATokenConfig(
                action_dim=8,
                action_horizon=15,
                max_token_len=400,
                fast_model_tokenizer=_tokenizer.BinningTokenizer,
            ),
            data=SimpleDatasetConfig(
                assets=AssetConfig(asset_id="droid"),
                data_transforms=lambda model: _transforms.Group(
                    inputs=[droid_policy.DroidInputs(action_dim=model.action_dim, model_type=ModelFamily.PI0_FAST)],
                    outputs=[droid_policy.DroidOutputs()],
                ),
                base_config=DatasetConfig(
                    prompt_from_task=True,
                ),
            ),
        ),
        TrainingConfig(
            # Trained from PaliGemma, using FAST tokenizer (using universal FAST+ tokenizer).
            name="paligemma_fast_droid",
            model=token_backbone.WorldLATokenConfig(action_dim=8, action_horizon=15),
            data=SimpleDatasetConfig(
                assets=AssetConfig(asset_id="droid"),
                data_transforms=lambda model: _transforms.Group(
                    inputs=[droid_policy.DroidInputs(action_dim=model.action_dim, model_type=ModelFamily.PI0_FAST)],
                    outputs=[droid_policy.DroidOutputs()],
                ),
                base_config=DatasetConfig(
                    prompt_from_task=True,
                ),
            ),
        ),
        TrainingConfig(
            # Trained from PaliGemma, using FAST tokenizer (tokenizer trained on DROID dataset).
            name="paligemma_fast_specialist_droid",
            model=token_backbone.WorldLATokenConfig(
                action_dim=8,
                action_horizon=15,
                fast_model_tokenizer=_tokenizer.FASTTokenizer,
                fast_model_tokenizer_kwargs={"fast_tokenizer_path": "KarlP/fast_droid_specialist"},
            ),
            data=SimpleDatasetConfig(
                assets=AssetConfig(asset_id="droid"),
                data_transforms=lambda model: _transforms.Group(
                    inputs=[droid_policy.DroidInputs(action_dim=model.action_dim, model_type=ModelFamily.PI0_FAST)],
                    outputs=[droid_policy.DroidOutputs()],
                ),
                base_config=DatasetConfig(
                    prompt_from_task=True,
                ),
            ),
        ),
        TrainingConfig(
            # Trained from PaliGemma, using FSQ tokenizer.
            name="paligemma_vq_droid",
            model=token_backbone.WorldLATokenConfig(
                action_dim=8,
                action_horizon=15,
                fast_model_tokenizer=_tokenizer.FSQTokenizer,
                fast_model_tokenizer_kwargs={"fsq_tokenizer_path": "gs://openpi-assets/tokenizers/droid_fsq_tokenizer"},
            ),
            data=SimpleDatasetConfig(
                assets=AssetConfig(asset_id="droid"),
                data_transforms=lambda model: _transforms.Group(
                    inputs=[droid_policy.DroidInputs(action_dim=model.action_dim, model_type=ModelFamily.PI0_FAST)],
                    outputs=[droid_policy.DroidOutputs()],
                ),
                base_config=DatasetConfig(
                    prompt_from_task=True,
                ),
            ),
        ),
        TrainingConfig(
            # pi0-style diffusion / flow VLA, trained on DROID from PaliGemma.
            name="paligemma_diffusion_droid",
            model=backbone_config.WorldLAConfig(action_horizon=10, action_dim=8),
            data=SimpleDatasetConfig(
                assets=AssetConfig(asset_id="droid"),
                data_transforms=lambda model: _transforms.Group(
                    inputs=[droid_policy.DroidInputs(action_dim=model.action_dim)],
                    outputs=[droid_policy.DroidOutputs()],
                ),
                base_config=DatasetConfig(
                    prompt_from_task=True,
                ),
            ),
        ),
    ]
