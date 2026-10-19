import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model_multiview as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0_multiview import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 30
    # Number of future intent/routing steps predicted from the current
    # observation. Keep this separate from action_horizon so the reactive H=1
    # ablation still predicts a full action chunk.
    intent_prediction_horizon: int | None = None
    history_action_length: int = 30
    use_temporal_history_encoder: bool = True
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = True
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    intent_warmup_steps: int = 5000
    # Table-2 routing ablations. A zero intent loss disables explicit
    # future-intent supervision while leaving the continuous router trainable
    # from the action objective. ``time_state`` removes visual/language inputs
    # from the intent predictor. Random view masking is training-only.
    intent_loss_weight: float = 1.0
    intent_input_mode: str = "multimodal"
    # Non-intent control: future-step index plus robot state/history directly
    # predicts action-side global/local routing weights.
    use_direct_time_state_routing: bool = False
    random_view_masking_prob: float = 0.0
    # Intent-MoE routing knobs. Defaults are set to preserve the original behavior.
    # Match the intended global/local two-expert design.
    visual_moe_num_experts: int = 2
    alpha_prior_scale: float = 0.3
    alpha_entropy_loss_weight: float = 0.0
    alpha_clip_min: float = 0.0
    alpha_clip_max: float = 1.0
    detach_alpha_for_moe: bool = False
    use_teacher_forced_alpha_for_moe: bool = True
    intent_delta_scale: float = 1.0
    # Stage-1 global/local expert plan: alpha is the main intent-to-view control.
    # Keep intent_view_prior frozen so it does not compete with alpha.
    freeze_intent_view_prior: bool = True
    view_prior_scale: float = 0.0
    global_local_expert_bias: float = -4.0
    alpha_only_moe_routing: bool = True
    view_router_logit_scale: float = 0.0
    token_router_logit_scale: float = 0.0
    # Phase-driven alpha control. The raw intent_labels are short event pulses,
    # so we first expand them into chunk phases:
    # 0 = global/search, 1 = mixed/transition, 2 = local/contact.
    use_phase_alpha: bool = True
    phase_loss_weight: float = 0.0
    phase_event_before_window: int = 20
    phase_event_after_window: int = 8
    phase_event_local_window: int = 4
    phase_chunk_local_min_count: int = 4
    phase_chunk_mixed_min_count: int = 6
    phase_rule_second_half_min_count: int = 5
    phase_rule_total_min_count: int = 3
    phase_rule_topk_threshold: float = 0.5
    phase_softness: float = 4.0
    phase_alpha_global: float = 0.75
    phase_alpha_mixed: float = 0.50
    phase_alpha_local: float = 0.75
    use_action_alpha_seq: bool = True
    # Main FocusVLA routing switch. When disabled, the model still predicts
    # future intent / alpha_seq and trains the intent branch, but action
    # generation no longer uses alpha_seq to fuse global/local contexts.
    # This is the clean "intent-only" ablation.
    use_action_side_routing: bool = True
    # Chunk-level intent token injected into the action suffix. This is kept
    # independent from routing for the reviewer-requested 2x2 ablation.
    use_intent_token: bool = True
    # Static-routing ablation: keep the action-side global/local routing
    # structure, but replace dynamic alpha_seq with a fixed routing weight at
    # every future step.
    use_static_action_routing: bool = False
    static_action_routing_global: float = 0.5
    static_action_routing_local: float = 0.5
    # Learned-static-routing ablation: preserve the same action-side
    # global/local fusion structure, but replace future-intent-guided
    # step-wise routing with one globally shared learnable routing ratio.
    # The learned logits are optimized end-to-end and reused for every sample,
    # every task stage, and every future step.
    use_learned_static_action_routing: bool = False
    learned_static_action_routing_init_global: float = 0.5
    learned_static_action_routing_init_local: float = 0.5
    # Keep the pretrained visual prefix intact by default. Intent/alpha still
    # controls action-side global/local context, but visual MoE does not rewrite
    # visual tokens unless this switch is explicitly enabled for ablations.
    use_visual_moe_update: bool = False
    action_alpha_seq_pool_steps: int = 10
    action_alpha_seq_low_threshold: float = 0.3
    action_alpha_seq_high_threshold: float = 0.6
    # Scheduled sampling for intent token injection (train-time).
    # Starts from full teacher forcing and linearly decays to end prob.
    intent_teacher_forcing_start_prob: float = 1.0
    intent_teacher_forcing_end_prob: float = 0.3
    intent_teacher_forcing_decay_steps: int = 15000
    # Residual delta head controls.
    # delta_head_base_scale: float = 0.2
    # delta_head_min_late_scale: float = 0.2
    # delta_aux_loss_weight: float = 0.5
    delta_head_base_scale: float = 0.0
    delta_head_min_late_scale: float = 0.2
    delta_aux_loss_weight: float = 0.0
    action_visual_cross_attn_num_heads: int = 4
    action_visual_cross_attn_scale: float = 0.1
    action_adapter_bottleneck_dim: int = 256
    action_adapter_scale: float = 0.1
    action_context_residual_scale: float = 0.1
    # Inference-time plug-in: make continuous end-effector motions more
    # conservative when the future chunk looks interaction/local-heavy.
    # This protects gripper dimensions and scales only motion dimensions.
    use_action_stability_scaling: bool = False
    action_stability_scale: float = 0.2
    action_stability_source: str = "alpha_local"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.intent_prediction_horizon is None:
            object.__setattr__(self, "intent_prediction_horizon", self.action_horizon)
        if self.intent_prediction_horizon not in (1, self.action_horizon):
            raise ValueError(
                "intent_prediction_horizon must be 1 (reactive routing) or "
                f"action_horizon ({self.action_horizon}), got {self.intent_prediction_horizon}."
            )
        if self.intent_input_mode not in ("multimodal", "time_state"):
            raise ValueError(
                "intent_input_mode must be 'multimodal' or 'time_state', "
                f"got {self.intent_input_mode!r}."
            )
        if self.use_direct_time_state_routing:
            if self.intent_input_mode != "time_state":
                raise ValueError("Direct time/state routing requires intent_input_mode='time_state'.")
            if self.use_intent_token or self.intent_loss_weight != 0.0:
                raise ValueError("Direct time/state routing must disable intent token and intent loss.")
            if self.intent_warmup_steps != 0:
                raise ValueError("Direct time/state routing must disable intent warmup.")
            if self.use_teacher_forced_alpha_for_moe:
                raise ValueError("Direct time/state routing must disable teacher-forced alpha.")
            if (
                self.intent_teacher_forcing_start_prob != 0.0
                or self.intent_teacher_forcing_end_prob != 0.0
            ):
                raise ValueError("Direct time/state routing must disable teacher forcing.")
        if not 0.0 <= self.random_view_masking_prob <= 1.0:
            raise ValueError(
                "random_view_masking_prob must be in [0, 1], "
                f"got {self.random_view_masking_prob}."
            )

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0_multiview import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
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
                history_actions=jax.ShapeDtypeStruct(
                    [batch_size, self.history_action_length, self.action_dim], jnp.float32
                ),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                intent_label=jax.ShapeDtypeStruct([batch_size, self.intent_prediction_horizon], jnp.int32),
                phase_label=jax.ShapeDtypeStruct([batch_size], jnp.int32),
                task_index=jax.ShapeDtypeStruct([batch_size], jnp.int32),
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
