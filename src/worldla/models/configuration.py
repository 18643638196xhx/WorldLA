import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from worldla.models import interfaces as _model
import worldla.models.language as _language
from worldla.utils import typing as at
import worldla.utils.nnx as nnx_utils

if TYPE_CHECKING:
    from worldla.models.backbone import WorldLABackbone


@dataclasses.dataclass(frozen=True)
class WorldLAConfig(_model.ModelConfigBase):
    dtype: str = "bfloat16"
    paligemma_variant: _language.Variant = "gemma_2b"
    action_expert_variant: _language.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # Optional compact motion bottleneck. A value of zero preserves the original
    # pi0/pi0.5 architecture and checkpoint structure exactly.
    motion_num_tokens: int = 0
    motion_latent_dim: int = 256
    motion_future_horizons: tuple[int, ...] = ()
    motion_loss_weight: float = 0.0
    motion_append_to_prefix: bool = True

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.motion_num_tokens < 0:
            raise ValueError("motion_num_tokens must be non-negative")
        if self.motion_num_tokens > 0 and self.motion_latent_dim <= 0:
            raise ValueError("motion_latent_dim must be positive when motion tokens are enabled")
        if any(horizon <= 0 for horizon in self.motion_future_horizons):
            raise ValueError("motion_future_horizons must contain positive frame offsets")
        if self.motion_loss_weight < 0:
            raise ValueError("motion_loss_weight must be non-negative")
        if self.motion_loss_weight > 0 and not self.motion_future_horizons:
            raise ValueError("motion_future_horizons are required when motion_loss_weight is positive")

    @property
    @override
    def model_type(self) -> _model.ModelFamily:
        if self.pi05:
            return _model.ModelFamily.PI05
        return _model.ModelFamily.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> 'WorldLABackbone':
        from worldla.models.backbone import WorldLABackbone

        return WorldLABackbone(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.RobotObservation, _model.ActionSequence]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.RobotObservation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
