import logging

import einops,optax
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from worldla.models import multiview_interfaces as _model
from worldla.models import multiview_configuration as backbone_config
import worldla.models.language as _language
import worldla.models.vision_encoder as _vision
from worldla.utils import typing as at
import worldla.models.intention as _intent # 导入新模块
import worldla.models.sparse_layers as _moe  # 导入新模块
import os

logger = logging.getLogger('worldla')

# 定义日志文件路径
ALPHA_LOG_FILE = "alpha_log.txt"

# 这是一个纯 Python 函数
def _log_alpha_to_file(alpha_val):
    # 转换：确保是 float
    val = float(alpha_val)
    # 追加写入 ('a' mode)
    with open(ALPHA_LOG_FILE, "a") as f:
        f.write(f"{val}\n")

def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)

class TimeIntentMLP(nnx.Module):
    def __init__(self, time_dim: int, intent_dim: int, out_dim: int, rngs: nnx.Rngs):
        """
        Args:
            time_dim: 时间正弦编码的维度 (config.action_expert_config.width)
            intent_dim: 意图 Embedding 的维度
            out_dim: 最终 adarms_cond 的维度 (LLM 的 hidden_size)
        """
        self.mlp = nnx.Sequential(
            # 拼接时间步和意图特征
            nnx.Linear(time_dim + intent_dim, out_dim, rngs=rngs),
            nnx.swish,
            nnx.Linear(out_dim, out_dim, rngs=rngs),
            nnx.swish
        )

    def __call__(self, time_emb, intent_emb):
        # 拼接并投影
        x = jnp.concatenate([time_emb, intent_emb], axis=-1)
        return self.mlp(x)


class TemporalHistoryEncoder(nnx.Module):
    """Lightweight temporal encoder for recent action history.

    This is a small 1D-conv-style block implemented with shared Linear layers
    over [t-1, t, t+1] windows. It preserves action trend information that a
    plain mean pool would discard.
    """

    def __init__(self, action_dim: int, embed_dim: int, rngs: nnx.Rngs):
        self.action_proj = nnx.Linear(action_dim, embed_dim, rngs=rngs)
        self.delta_proj = nnx.Linear(action_dim, embed_dim, rngs=rngs)
        self.conv1 = nnx.Linear(3 * embed_dim, embed_dim, rngs=rngs)
        self.conv2 = nnx.Linear(3 * embed_dim, embed_dim, rngs=rngs)
        self.norm1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.pool_score = nnx.Linear(embed_dim, 1, rngs=rngs)

    def _window3(self, x: at.Float[at.Array, "b h emb"]) -> at.Float[at.Array, "b h d"]:
        left = jnp.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
        right = jnp.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
        return jnp.concatenate([left, x, right], axis=-1)

    def __call__(
        self,
        history_actions: at.Float[at.Array, "b h a"],
    ) -> at.Float[at.Array, "b 1 emb"]:
        prev_actions = jnp.concatenate([history_actions[:, :1, :], history_actions[:, :-1, :]], axis=1)
        action_delta = history_actions - prev_actions

        x = self.action_proj(history_actions) + self.delta_proj(action_delta)
        x = self.norm1(x + nnx.swish(self.conv1(self._window3(x))))
        x = self.norm2(x + nnx.swish(self.conv2(self._window3(x))))

        scores = self.pool_score(x).squeeze(-1)
        weights = jax.nn.softmax(scores, axis=1)
        history_token = jnp.sum(x * weights[..., None], axis=1, keepdims=True)
        return history_token


class TimeStateRoutingHead(nnx.Module):
    """Directly predict step-wise routing logits from time and robot state/history."""

    def __init__(self, embed_dim: int, output_steps: int, rngs: nnx.Rngs):
        self.embed_dim = embed_dim
        self.output_steps = output_steps
        self.norm = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.mlp_in = nnx.Linear(embed_dim, embed_dim, rngs=rngs)
        self.mlp_out = nnx.Linear(embed_dim, 1, rngs=rngs)

    def __call__(self, state_features: at.Float[at.Array, "b t emb"]):
        state_context = jnp.mean(state_features, axis=1, keepdims=True)
        step_positions = jnp.linspace(0.0, 1.0, self.output_steps, dtype=state_features.dtype)
        step_features = posemb_sincos(
            step_positions,
            self.embed_dim,
            min_period=1e-2,
            max_period=1e2,
        )[None, :, :]
        x = self.norm(state_context + step_features)
        x = nnx.gelu(self.mlp_in(x))
        return self.mlp_out(x).squeeze(-1)


class WorldLABackbone(_model.RobotModelBase):
    def __init__(self, config: backbone_config.WorldLAConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _language.get_config(config.paligemma_variant)
        action_expert_config = _language.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _language.LanguageTransformer(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _vision.VisionEncoder(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            #self.adarms_expert_global = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            #self.adarms_expert_local = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        self.pose_refine_head = nnx.Sequential(
            nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs),
            nnx.swish,
            nnx.Linear(
                action_expert_config.width,
                config.action_dim,
                rngs=rngs,
            ),
        )

        # 🔥 [修改] 使用独立的 IntentHead
        if config.intent_input_mode == "time_state":
            self.intent_predictor_head = TimeStateRoutingHead(
                embed_dim=paligemma_config.width,
                output_steps=config.intent_prediction_horizon,
                rngs=rngs,
            )
        else:
            self.intent_predictor_head = _intent.IntentHead(
                embed_dim=paligemma_config.width,
                output_steps=config.intent_prediction_horizon,
                num_heads=4,
                rngs=rngs
            )
        
        # 🔥 [修改] 状态编码器 (用于意图预测)
        self.intent_state_encoder = nnx.Linear(
            config.action_dim, 
            paligemma_config.width, 
            rngs=rngs
        )
        self.intent_history_encoder = nnx.Linear(
            config.action_dim,
            paligemma_config.width,
            rngs=rngs,
        )
        self.intent_temporal_history_encoder = TemporalHistoryEncoder(
            action_dim=config.action_dim,
            embed_dim=paligemma_config.width,
            rngs=rngs,
        )
        self.alpha_gate_in = nnx.Linear(3 * paligemma_config.width + 3, paligemma_config.width, rngs=rngs)
        self.alpha_gate_out = nnx.Linear(
            paligemma_config.width,
            config.visual_moe_num_experts,
            rngs=rngs,
        )
        self.intent_alpha_delta_in = nnx.Linear(3, paligemma_config.width // 4, rngs=rngs)
        self.intent_alpha_delta_out = nnx.Linear(
            paligemma_config.width // 4,
            config.visual_moe_num_experts,
            rngs=rngs,
        )
        self.intent_view_prior_in = nnx.Linear(3, paligemma_config.width // 4, rngs=rngs)
        self.intent_view_prior_out = nnx.Linear(paligemma_config.width // 4, 2, rngs=rngs)
        self.control_context_in = nnx.Linear(3 * paligemma_config.width, action_expert_config.width, rngs=rngs)
        self.control_context_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        # 🔥 修正初始化：显式传入 action_expert_width
        self.visual_moe = _moe.SoftMoERouter(
            paligemma_config,
            action_expert_width=action_expert_config.width, 
            num_experts=config.visual_moe_num_experts,
            alpha_prior_scale=config.alpha_prior_scale,
            view_prior_scale=config.view_prior_scale,
            global_local_expert_bias=config.global_local_expert_bias,
            alpha_only_moe_routing=config.alpha_only_moe_routing,
            view_router_logit_scale=config.view_router_logit_scale,
            token_router_logit_scale=config.token_router_logit_scale,
            rngs=rngs
        )
        # 🔥 [新增] 给 MoE 输出加一个“音量限制器”，防止绝对尺度偏移搞崩 AdaRMS
        self.moe_norm = nnx.RMSNorm(paligemma_config.width, rngs=rngs)

        self.intent_embedding = nnx.Embed(
            num_embeddings=2,  # 对应 0 (移动) 和 1 (交互) 两个标签
            features=action_expert_config.width, # 维度要和 action expert 匹配
            rngs=rngs
        )
        self.visual_to_action_proj = nnx.Linear(
            paligemma_config.width,
            action_expert_config.width,
            rngs=rngs,
        )
        self.action_visual_cross_attn = nnx.MultiHeadAttention(
            num_heads=config.action_visual_cross_attn_num_heads,
            in_features=action_expert_config.width,
            qkv_features=action_expert_config.width,
            decode=False,
            rngs=rngs,
        )
        self.action_visual_cross_attn_gate = nnx.Linear(
            action_expert_config.width,
            action_expert_config.width,
            rngs=rngs,
        )
        static_alpha_init = jnp.asarray(
            [
                config.learned_static_action_routing_init_global,
                config.learned_static_action_routing_init_local,
            ],
            dtype=jnp.float32,
        )
        static_alpha_init = jnp.clip(static_alpha_init, 1e-4, None)
        static_alpha_init = static_alpha_init / jnp.maximum(jnp.sum(static_alpha_init), 1e-6)
        self.learned_static_action_routing_logits = nnx.Param(jnp.log(static_alpha_init))
        self.action_adapter_ctx = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_adapter_gate = nnx.Linear(
            action_expert_config.width,
            action_expert_config.width,
            rngs=rngs,
        )
        self.action_adapter_down = nnx.Linear(
            action_expert_config.width,
            config.action_adapter_bottleneck_dim,
            rngs=rngs,
        )
        self.action_adapter_up = nnx.Linear(
            config.action_adapter_bottleneck_dim,
            action_expert_config.width,
            rngs=rngs,
        )
        self.action_context_residual_in = nnx.Linear(
            2 * action_expert_config.width,
            action_expert_config.width,
            rngs=rngs,
        )
        self.action_context_residual_out = nnx.Linear(
            action_expert_config.width,
            config.action_dim,
            rngs=rngs,
        )
        

        self.config = config 

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def _delta_blend_scale(self, time: at.Float[at.Array, " b"]) -> at.Float[at.Array, "b 1 1"]:
        """Late-stage-biased scaling for the residual delta head."""
        late_scale = self.config.delta_head_min_late_scale + (1.0 - self.config.delta_head_min_late_scale) * (1.0 - time)
        return (self.config.delta_head_base_scale * late_scale)[..., None, None]

    def _action_motion_mask(self, dtype=jnp.float32) -> at.Float[at.Array, "ad"]:
        """Mask continuous motion dims for stability scaling, leaving grippers intact.

        RoboTwin action vectors are often repeated 7D arm chunks:
        xyz + rotation + gripper. For 7D chunks, the first 6 dims are motion
        and the 7th is gripper, so only the first 6 are scaled. For other
        layouts, fall back to scaling the first min(6, action_dim) dims.
        """
        if self.action_dim % 7 == 0:
            chunk = jnp.array([1, 1, 1, 1, 1, 1, 0], dtype=dtype)
            return jnp.tile(chunk, self.action_dim // 7)
        mask = jnp.zeros((self.action_dim,), dtype=dtype)
        return mask.at[: min(6, self.action_dim)].set(1.0)

    def _apply_action_stability_scaling(
        self,
        v_t: at.Float[at.Array, "b h ad"],
        alpha_seq: at.Float[at.Array, "b h 2"] | None,
        intent_probs: at.Float[at.Array, "b h"] | None = None,
    ) -> at.Float[at.Array, "b h ad"]:
        """Inference-time plug-in that dampens motion during interaction-like steps."""
        if not self.config.use_action_stability_scaling:
            return v_t
        if self.config.action_stability_scale <= 0.0:
            return v_t

        if self.config.action_stability_source == "intent_mean" and intent_probs is not None:
            stability = jnp.broadcast_to(jnp.mean(intent_probs, axis=1, keepdims=True), v_t.shape[:2])
        elif alpha_seq is not None:
            action_alpha_seq = self._expand_alpha_seq_to_action_horizon(alpha_seq)
            stability = action_alpha_seq[:, : v_t.shape[1], 1]
        elif intent_probs is not None:
            if intent_probs.shape[1] == 1:
                stability = jnp.broadcast_to(intent_probs, v_t.shape[:2])
            else:
                stability = intent_probs[:, : v_t.shape[1]]
        else:
            return v_t

        stability = jnp.clip(stability, 0.0, 1.0)
        scale = 1.0 - self.config.action_stability_scale * stability
        motion_mask = self._action_motion_mask(v_t.dtype)[None, None, :]
        per_dim_scale = 1.0 - motion_mask + motion_mask * scale[..., None]
        return v_t * per_dim_scale

    def _stabilize_alpha(self, alpha: at.Float[at.Array, "b k"]) -> at.Float[at.Array, "b k"]:
        """Optionally clip routing weights and renormalize to reduce collapse."""
        if self.config.alpha_clip_min > 0.0 or self.config.alpha_clip_max < 1.0:
            alpha = jnp.clip(alpha, self.config.alpha_clip_min, self.config.alpha_clip_max)
            alpha = alpha / jnp.maximum(jnp.sum(alpha, axis=-1, keepdims=True), 1e-6)
        return alpha

    def _masked_mean_pool(
        self,
        features: at.Float[at.Array, "b s emb"],
        mask: at.Bool[at.Array, "b s"] | None = None,
    ) -> at.Float[at.Array, "b emb"]:
        if mask is None:
            return jnp.mean(features, axis=1)
        mask_f = mask.astype(features.dtype)[..., None]
        denom = jnp.maximum(jnp.sum(mask_f, axis=1), 1e-6)
        return jnp.sum(features * mask_f, axis=1) / denom

    def _compute_alpha_vec(
        self,
        visual_features: at.Float[at.Array, "b s emb"],
        text_features: at.Float[at.Array, "b s emb"],
        state_features: at.Float[at.Array, "b t emb"],
        intent_logits: at.Float[at.Array, "b h"],
        text_mask: at.Bool[at.Array, "b s"] | None = None,
        intent_summary: at.Float[at.Array, "b 3"] | None = None,
    ) -> at.Float[at.Array, "b k"]:
        return self._compute_alpha_vec_with_diagnostics(
            visual_features,
            text_features,
            state_features,
            intent_logits,
            text_mask,
            intent_summary,
        )[0]

    def _compute_alpha_vec_with_diagnostics(
        self,
        visual_features: at.Float[at.Array, "b s emb"],
        text_features: at.Float[at.Array, "b s emb"],
        state_features: at.Float[at.Array, "b t emb"],
        intent_logits: at.Float[at.Array, "b h"],
        text_mask: at.Bool[at.Array, "b s"] | None = None,
        intent_summary: at.Float[at.Array, "b 3"] | None = None,
    ) -> tuple[at.Float[at.Array, "b k"], dict[str, at.Array]]:
        visual_summary = self._masked_mean_pool(visual_features)
        text_summary = self._masked_mean_pool(text_features, text_mask)
        state_summary = self._masked_mean_pool(state_features)
        horizon_summary = self._compute_intent_horizon_summary(intent_logits) if intent_summary is None else intent_summary
        gate_input = jnp.concatenate([visual_summary, text_summary, state_summary, horizon_summary], axis=-1)
        gate_hidden = self.alpha_gate_in(gate_input)
        gate_hidden = nnx.swish(gate_hidden)
        base_logits = self.alpha_gate_out(gate_hidden)

        intent_delta_hidden = self.intent_alpha_delta_in(horizon_summary)
        intent_delta_hidden = nnx.swish(intent_delta_hidden)
        intent_delta_logits = self.intent_alpha_delta_out(intent_delta_hidden)
        # Intent-only alpha: keep base logits for diagnostics, but do not let the
        # learned dataset/task prior dominate the routing signal.
        gate_logits = self.config.intent_delta_scale * intent_delta_logits
        alpha = self._stabilize_alpha(jax.nn.softmax(gate_logits, axis=-1))
        diagnostics = {
            "base_logits": base_logits,
            "intent_delta_logits": intent_delta_logits,
            "combined_logits": gate_logits,
            "intent_summary": horizon_summary,
        }
        return alpha, diagnostics

    def _compute_intent_horizon_summary(
        self,
        intent_logits: at.Float[at.Array, "b h"],
    ) -> at.Float[at.Array, "b 3"]:
        intent_probs = nnx.sigmoid(intent_logits)
        return self._compute_intent_horizon_summary_from_probs(intent_probs)

    def _compute_intent_view_prior(
        self,
        intent_summary: at.Float[at.Array, "b 3"],
    ) -> at.Float[at.Array, "b 2"]:
        hidden = self.intent_view_prior_in(intent_summary)
        hidden = nnx.swish(hidden)
        logits = self.intent_view_prior_out(hidden)
        return jax.nn.softmax(logits, axis=-1)

    def _compute_intent_horizon_summary_from_probs(
        self,
        intent_probs: at.Float[at.Array, "b h"],
    ) -> at.Float[at.Array, "b 3"]:
        if intent_probs.shape[-1] == 1:
            return jnp.repeat(intent_probs, 3, axis=-1)
        half = max(1, intent_probs.shape[-1] // 2)
        intent_mean = jnp.mean(intent_probs, axis=-1, keepdims=True)
        second_half = jnp.mean(intent_probs[:, half:], axis=-1, keepdims=True)
        second_half_max = jnp.max(intent_probs[:, half:], axis=-1, keepdims=True)
        return jnp.concatenate([intent_mean, second_half, second_half_max], axis=-1)

    def _ensure_intent_label_sequence(
        self,
        intent_label: at.Array,
    ) -> at.Array:
        if intent_label.ndim == 1:
            return jnp.repeat(intent_label[:, None], self.config.intent_prediction_horizon, axis=1)
        return intent_label

    def _chunk_intent_label_from_seq(
        self,
        intent_seq: at.Float[at.Array, "b h"],
    ) -> at.Int[at.Array, "b"]:
        if intent_seq.shape[-1] == 1:
            return (intent_seq[:, 0] > 0.5).astype(jnp.int32)
        half = max(1, intent_seq.shape[-1] // 2)
        tail_binary = (intent_seq[:, half:] > 0.5).astype(jnp.int32)
        if tail_binary.shape[-1] < 3:
            return (jnp.sum(tail_binary, axis=-1) == tail_binary.shape[-1]).astype(jnp.int32)

        run3 = (
            (tail_binary[:, :-2] == 1)
            & (tail_binary[:, 1:-1] == 1)
            & (tail_binary[:, 2:] == 1)
        )
        return jnp.any(run3, axis=-1).astype(jnp.int32)

    def _chunk_phase_label_from_intent_seq(
        self,
        intent_seq: at.Float[at.Array, "b h"],
    ) -> at.Int[at.Array, "b"]:
        """Fallback chunk phase from a future binary intent sequence.

        The data loader normally supplies phase labels expanded from event-pulse
        intent labels. This fallback keeps fake/legacy data usable.
        """
        binary = (intent_seq > 0.5).astype(jnp.int32)
        if binary.shape[-1] == 1:
            return (2 * binary[:, 0]).astype(jnp.int32)
        half = max(1, binary.shape[-1] // 2)
        local_count = jnp.sum(binary[:, half:], axis=-1)
        active_count = jnp.sum(binary, axis=-1)
        return jnp.where(
            local_count >= self.config.phase_chunk_local_min_count,
            2,
            jnp.where(active_count >= self.config.phase_chunk_mixed_min_count, 1, 0),
        ).astype(jnp.int32)

    def _has_positive_run(
        self,
        binary_seq: at.Int[at.Array, "b h"],
        run_length: int,
    ) -> at.Bool[at.Array, "b"]:
        if binary_seq.shape[-1] < run_length:
            return jnp.sum(binary_seq, axis=-1) == binary_seq.shape[-1]
        run_hits = []
        for offset in range(binary_seq.shape[-1] - run_length + 1):
            window = binary_seq[:, offset : offset + run_length]
            run_hits.append(jnp.all(window == 1, axis=-1))
        return jnp.any(jnp.stack(run_hits, axis=-1), axis=-1)

    def _rule_phase_from_intent_probs(
        self,
        intent_probs: at.Float[at.Array, "b h"],
    ) -> at.Int[at.Array, "b"]:
        """Rule-based chunk phase from predicted future intent probabilities.

        phase 2: stable evidence in the second half -> local.
        phase 1: weak/early evidence in the chunk -> mixed.
        phase 0: no event evidence -> global.
        """
        if intent_probs.shape[-1] == 1:
            return jnp.where(intent_probs[:, 0] > 0.5, 2, 0).astype(jnp.int32)
        binary = (intent_probs > 0.5).astype(jnp.int32)
        half = max(1, intent_probs.shape[-1] // 2)
        second_half_binary = binary[:, half:]
        second_half_probs = intent_probs[:, half:]
        second_half_count = jnp.sum(second_half_binary, axis=-1)
        total_count = jnp.sum(binary, axis=-1)
        has_second_half_run = self._has_positive_run(
            second_half_binary,
            self.config.phase_chunk_local_min_count,
        )
        # Avoid jax.lax.top_k here: some JAX/XLA versions hit an HLO verifier
        # error in topk-decomposer during training compilation. This mass score
        # behaves like a stable "top-3 evidence" proxy: one isolated spike is
        # not enough, but several moderate/high probabilities can trigger mixed.
        second_half_topk_mean = jnp.sum(second_half_probs, axis=-1) / 3.0

        phase2 = (second_half_count >= self.config.phase_rule_second_half_min_count) | has_second_half_run
        phase1 = (total_count >= self.config.phase_rule_total_min_count) | (
            second_half_topk_mean > self.config.phase_rule_topk_threshold
        )
        return jnp.where(phase2, 2, jnp.where(phase1, 1, 0)).astype(jnp.int32)

    def _rule_phase_probs_from_intent_probs(
        self,
        intent_probs: at.Float[at.Array, "b h"],
    ) -> at.Float[at.Array, "b 3"]:
        """Soft rule phase probabilities from future intent evidence.

        This keeps phase interpretable while avoiding hard three-speed switching.
        Scores are evidence ratios around 1.0, then squashed with a sigmoid:
        below threshold -> low score, above threshold -> high score.
        """
        if intent_probs.shape[-1] == 1:
            local_score = jnp.clip(intent_probs[:, 0], 0.0, 1.0)
            return jnp.stack([1.0 - local_score, jnp.zeros_like(local_score), local_score], axis=-1)
        binary = (intent_probs > 0.5).astype(jnp.float32)
        half = max(1, intent_probs.shape[-1] // 2)
        second_half_binary = binary[:, half:]
        second_half_probs = intent_probs[:, half:]

        second_half_count = jnp.sum(second_half_binary, axis=-1)
        total_count = jnp.sum(binary, axis=-1)
        second_half_run = jnp.zeros_like(second_half_count)
        run = jnp.zeros_like(second_half_count)
        for offset in range(second_half_binary.shape[-1]):
            run = (run + 1.0) * second_half_binary[:, offset]
            second_half_run = jnp.maximum(second_half_run, run)

        local_count_evidence = second_half_count / jnp.maximum(
            float(self.config.phase_rule_second_half_min_count), 1.0
        )
        local_run_evidence = second_half_run / jnp.maximum(float(self.config.phase_chunk_local_min_count), 1.0)
        local_evidence = jnp.maximum(local_count_evidence, local_run_evidence)

        # Probability mass proxy for "several moderately high second-half probs".
        second_half_mass_evidence = (jnp.sum(second_half_probs, axis=-1) / 3.0) / jnp.maximum(
            self.config.phase_rule_topk_threshold, 1e-6
        )
        total_evidence = total_count / jnp.maximum(float(self.config.phase_rule_total_min_count), 1.0)
        mixed_evidence = jnp.maximum(total_evidence, second_half_mass_evidence)

        sharpness = self.config.phase_softness
        local_score = jax.nn.sigmoid(sharpness * (local_evidence - 1.0))
        mixed_raw = jax.nn.sigmoid(sharpness * (mixed_evidence - 1.0))
        mixed_score = mixed_raw * (1.0 - local_score)
        global_score = jnp.maximum(1.0 - local_score - mixed_score, 0.0)
        phase_scores = jnp.stack([global_score, mixed_score, local_score], axis=-1)
        return phase_scores / jnp.maximum(jnp.sum(phase_scores, axis=-1, keepdims=True), 1e-6)

    def _phase_alpha_table(self) -> at.Float[at.Array, "3 2"]:
        global_pref = self.config.phase_alpha_global
        mixed_pref = self.config.phase_alpha_mixed
        local_pref = self.config.phase_alpha_local
        return jnp.asarray(
            [
                [global_pref, 1.0 - global_pref],
                [mixed_pref, 1.0 - mixed_pref],
                [1.0 - local_pref, local_pref],
            ],
            dtype=jnp.float32,
        )

    def _alpha_from_phase_probs(
        self,
        phase_probs: at.Float[at.Array, "b 3"],
    ) -> at.Float[at.Array, "b 2"]:
        alpha = phase_probs @ self._phase_alpha_table()
        return self._stabilize_alpha(alpha)

    def _alpha_from_phase_labels(
        self,
        phase_labels: at.Int[at.Array, "b"],
    ) -> at.Float[at.Array, "b 2"]:
        """Discrete phase alpha: phase 0/1/2 maps to fixed global/mixed/local gates."""
        phase_one_hot = jax.nn.one_hot(phase_labels.astype(jnp.int32), 3, dtype=jnp.float32)
        return self._alpha_from_phase_probs(phase_one_hot)

    def _alpha_seq_from_intent_probs(
        self,
        intent_probs: at.Float[at.Array, "b h"],
    ) -> at.Float[at.Array, "b h 2"]:
        """Continuous step-wise alpha schedule from future intent probabilities.

        Low intent stays global-biased, high intent becomes local-biased, and
        uncertain intent gives a smooth mixed gate. This avoids hard threshold
        jumps while preserving the 30-step horizon structure for action-side
        routing. Phase labels are still computed separately for diagnostics.
        """
        intent_probs = jnp.clip(intent_probs, 0.0, 1.0)
        local_min = 1.0 - self.config.phase_alpha_global
        local_max = self.config.phase_alpha_local
        alpha_local = local_min + (local_max - local_min) * intent_probs
        alpha_seq = jnp.stack([1.0 - alpha_local, alpha_local], axis=-1)
        return self._stabilize_alpha(alpha_seq)

    def _alpha_seq_from_direct_route_logits(
        self,
        route_logits: at.Float[at.Array, "b h"],
    ) -> at.Float[at.Array, "b h 2"]:
        """Map direct time/state router logits to bounded global/local weights."""
        local_score = nnx.sigmoid(route_logits)
        local_min = 1.0 - self.config.phase_alpha_global
        local_max = self.config.phase_alpha_local
        alpha_local = local_min + (local_max - local_min) * local_score
        return self._stabilize_alpha(jnp.stack([1.0 - alpha_local, alpha_local], axis=-1))

    def _expand_alpha_seq_to_action_horizon(
        self,
        alpha_seq: at.Float[at.Array, "b h 2"] | None,
    ) -> at.Float[at.Array, "b ah 2"] | None:
        """Align routing predictions with action tokens.

        Full FocusVLA predicts one route per action step. The reactive H=1
        control predicts one route from the current observation and reuses it
        for the complete action chunk.
        """
        if alpha_seq is None or alpha_seq.shape[1] == self.action_horizon:
            return alpha_seq
        if alpha_seq.shape[1] == 1:
            return jnp.broadcast_to(
                alpha_seq,
                (alpha_seq.shape[0], self.action_horizon, alpha_seq.shape[2]),
            )
        raise ValueError(
            "Routing horizon must be 1 or match action_horizon; "
            f"got {alpha_seq.shape[1]} and {self.action_horizon}."
        )

    def _static_action_alpha_seq(
        self,
        batch_size: int,
        horizon: int,
        dtype=jnp.float32,
    ) -> at.Float[at.Array, "b h 2"]:
        """Fixed per-step routing schedule for the static-routing ablation."""
        alpha = jnp.asarray(
            [
                self.config.static_action_routing_global,
                self.config.static_action_routing_local,
            ],
            dtype=dtype,
        )
        alpha = alpha / jnp.maximum(jnp.sum(alpha), 1e-6)
        alpha = self._stabilize_alpha(alpha[None, :])
        return jnp.broadcast_to(alpha[:, None, :], (batch_size, horizon, 2))

    def _learned_static_action_alpha_seq(
        self,
        batch_size: int,
        horizon: int,
        dtype=jnp.float32,
    ) -> at.Float[at.Array, "b h 2"]:
        """Globally shared learned routing ratio for the learned-static ablation."""
        alpha = jax.nn.softmax(jnp.asarray(self.learned_static_action_routing_logits), axis=-1).astype(dtype)
        alpha = self._stabilize_alpha(alpha[None, :])
        return jnp.broadcast_to(alpha[:, None, :], (batch_size, horizon, 2))

    def _pool_chunk_alpha_from_alpha_seq(
        self,
        alpha_seq: at.Float[at.Array, "b h 2"],
    ) -> at.Float[at.Array, "b 2"]:
        """Pool alpha_seq into one visual-MoE chunk alpha.

        Visual tokens are reshaped once for the whole action chunk, so this
        pooling should answer: "will the future middle/late chunk need local
        visual detail?"  We deliberately ignore the early segment here; a few
        early local steps often mean the previous contact is ending, not that
        the upcoming chunk should be locally biased.
        """
        horizon = alpha_seq.shape[1]
        if horizon == 1:
            return self._stabilize_alpha(alpha_seq[:, 0, :])
        first_cut = horizon // 3
        second_cut = (2 * horizon) // 3
        mid_local = jnp.mean(alpha_seq[:, first_cut:second_cut, 1], axis=1)
        late_local = jnp.mean(alpha_seq[:, second_cut:, 1], axis=1)
        chunk_local = jnp.maximum(mid_local, late_local)
        chunk_alpha = jnp.stack([1.0 - chunk_local, chunk_local], axis=-1)
        return self._stabilize_alpha(chunk_alpha)

    def _compute_control_context(
        self,
        visual_features: at.Float[at.Array, "b s emb"],
        text_features: at.Float[at.Array, "b s emb"],
        state_features: at.Float[at.Array, "b t emb"],
        text_mask: at.Bool[at.Array, "b s"] | None = None,
    ) -> at.Float[at.Array, "b emb"]:
        visual_summary = self._masked_mean_pool(visual_features)
        text_summary = self._masked_mean_pool(text_features, text_mask)
        state_summary = self._masked_mean_pool(state_features)
        context_input = jnp.concatenate([visual_summary, text_summary, state_summary], axis=-1)
        context = self.control_context_in(context_input)
        context = nnx.swish(context)
        context = self.control_context_out(context)
        return nnx.swish(context)

    def _apply_action_adapter(
        self,
        action_features: at.Float[at.Array, "b s emb"],
        control_context: at.Float[at.Array, "b emb"],
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Float[at.Array, "b"]]:
        context_bias = self.action_adapter_ctx(control_context)[:, None, :]
        adapter_hidden = self.action_adapter_down(action_features + context_bias)
        adapter_hidden = nnx.swish(adapter_hidden)
        adapter_delta = self.action_adapter_up(adapter_hidden)
        adapter_gate = nnx.sigmoid(self.action_adapter_gate(control_context))[:, None, :]
        adapted_features = action_features + self.config.action_adapter_scale * adapter_gate * adapter_delta
        adapter_norm = jnp.mean(jnp.linalg.norm(adapter_delta, axis=-1), axis=-1)
        return adapted_features, adapter_norm

    def _apply_action_visual_cross_attention(
        self,
        action_features: at.Float[at.Array, "b s emb"],
        visual_features: at.Float[at.Array, "b v embv"],
        control_context: at.Float[at.Array, "b emb"],
        alpha_seq: at.Float[at.Array, "b s 2"] | None = None,
        global_token_len: int | None = None,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Float[at.Array, "b"],
        at.Float[at.Array, "b"],
        at.Float[at.Array, "b"],
    ]:
        global_norm = jnp.zeros((action_features.shape[0],), dtype=action_features.dtype)
        local_norm = jnp.zeros((action_features.shape[0],), dtype=action_features.dtype)
        if (
            alpha_seq is not None
            and global_token_len is not None
            and global_token_len > 0
            and global_token_len < visual_features.shape[1]
        ):
            global_kv = self.visual_to_action_proj(visual_features[:, :global_token_len, :])
            local_kv = self.visual_to_action_proj(visual_features[:, global_token_len:, :])
            global_out = self.action_visual_cross_attn(
                inputs_q=action_features,
                inputs_k=global_kv,
                inputs_v=global_kv,
                decode=False,
            )
            local_out = self.action_visual_cross_attn(
                inputs_q=action_features,
                inputs_k=local_kv,
                inputs_v=local_kv,
                decode=False,
            )
            step_alpha = self._expand_alpha_seq_to_action_horizon(alpha_seq)
            step_alpha = step_alpha[:, : action_features.shape[1], :]
            cross_attn_out = (
                step_alpha[..., 0:1] * global_out
                + step_alpha[..., 1:2] * local_out
            )
            global_norm = jnp.mean(jnp.linalg.norm(global_out, axis=-1), axis=-1)
            local_norm = jnp.mean(jnp.linalg.norm(local_out, axis=-1), axis=-1)
        else:
            visual_kv = self.visual_to_action_proj(visual_features)
            cross_attn_out = self.action_visual_cross_attn(
                inputs_q=action_features,
                inputs_k=visual_kv,
                inputs_v=visual_kv,
                decode=False,
            )
        cross_attn_gate = nnx.sigmoid(self.action_visual_cross_attn_gate(control_context))[:, None, :]
        enhanced_features = (
            action_features
            + self.config.action_visual_cross_attn_scale * cross_attn_gate * cross_attn_out
        )
        cross_attn_norm = jnp.mean(jnp.linalg.norm(cross_attn_out, axis=-1), axis=-1)
        return enhanced_features, cross_attn_norm, global_norm, local_norm

    def _compute_action_context_residual(
        self,
        action_features: at.Float[at.Array, "b s emb"],
        control_context: at.Float[at.Array, "b emb"],
    ) -> tuple[at.Float[at.Array, "b s ad"], at.Float[at.Array, "b"]]:
        repeated_context = einops.repeat(control_context, "b emb -> b s emb", s=action_features.shape[1])
        residual_input = jnp.concatenate([action_features, repeated_context], axis=-1)
        residual_hidden = self.action_context_residual_in(residual_input)
        residual_hidden = nnx.swish(residual_hidden)
        residual = self.action_context_residual_out(residual_hidden)
        residual_norm = jnp.mean(jnp.linalg.norm(residual, axis=-1), axis=-1)
        return residual, residual_norm

    def _intent_teacher_forcing_prob(self, step: int | None) -> at.Float[at.Array, ""]:
        """Linear schedule for train-time GT intent injection."""
        if step is None:
            return jnp.asarray(self.config.intent_teacher_forcing_end_prob, dtype=jnp.float32)
        decay_steps = max(1, int(self.config.intent_teacher_forcing_decay_steps))
        progress = jnp.clip(jnp.asarray(step, dtype=jnp.float32) / decay_steps, 0.0, 1.0)
        start = jnp.asarray(self.config.intent_teacher_forcing_start_prob, dtype=jnp.float32)
        end = jnp.asarray(self.config.intent_teacher_forcing_end_prob, dtype=jnp.float32)
        return start + (end - start) * progress

    # 🔥 [新增] 辅助函数：编码状态与历史动作
    def _encode_state_for_intent(
        self,
        state: at.Float[at.Array, "b s"],
        history_actions: at.Float[at.Array, "b h a"] | None = None,
    ) -> at.Float[at.Array, "b t emb"]:
        state_token = self.intent_state_encoder(state)[:, None, :]
        if history_actions is None:
            return state_token

        if self.config.use_temporal_history_encoder:
            history_token = self.intent_temporal_history_encoder(history_actions)
        else:
            history_tokens = self.intent_history_encoder(history_actions)
            history_token = jnp.mean(history_tokens, axis=1, keepdims=True)
        return jnp.concatenate([state_token, history_token], axis=1)

    def _predict_intent_logits(
        self,
        visual_features: at.Float[at.Array, "b vi emb"],
        text_features: at.Float[at.Array, "b tx emb"],
        state_features: at.Float[at.Array, "b st emb"],
        text_mask: at.Bool[at.Array, "b tx"],
    ) -> at.Float[at.Array, "b h"]:
        if self.config.intent_input_mode == "time_state":
            return self.intent_predictor_head(state_features)
        return self.intent_predictor_head(
            text_features=text_features,
            visual_features=visual_features,
            state_features=state_features,
            text_mask=text_mask,
        )

    def _apply_random_view_masking(
        self,
        observation: _model.RobotObservation,
        rng: at.KeyArrayLike,
    ) -> _model.RobotObservation:
        """Drop at most one complete camera view per sample during training."""
        probability = float(self.config.random_view_masking_prob)
        if probability <= 0.0:
            return observation

        view_names = tuple(observation.images)
        apply_rng, choice_rng = jax.random.split(rng)
        batch_shape = observation.state.shape[:-1]
        apply_mask = jax.random.bernoulli(apply_rng, probability, batch_shape)
        dropped_view = jax.random.randint(choice_rng, batch_shape, 0, len(view_names))
        image_masks = dict(observation.image_masks)
        for view_index, view_name in enumerate(view_names):
            keep_view = ~(apply_mask & (dropped_view == view_index))
            image_masks[view_name] = jnp.asarray(image_masks[view_name]) & keep_view
        return observation.replace(image_masks=image_masks)

    # @at.typecheck
    def embed_prefix(
        self, obs: _model.RobotObservation,
        # 🔥 [修改 1] 新增参数：接收意图门控权重
        gating_alpha: at.Float[at.Array, "b k"] | None = None,
        intent_view_prior: at.Float[at.Array, "b 2"] | None = None,
        # 🔥 修改 1：确保能够接收时间向量
        timestep_emb: at.Float[at.Array, "b emb"] | None = None,
        return_routing_stats: bool = False,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        int,
        int,
        dict[str, at.Array] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []

        #======新增=======#
        img_token_count = 0
        global_len = 0 # 🔥 [修改 2] 用于记录全局视角 Token 的总长度
        #======新增=======#
        

        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)

            #======新增=======#
            current_len = image_tokens.shape[1]
            img_token_count += current_len

            # 🔥 [修改 2] 判定是否为全局视角，累加 global_len
            # 逻辑：只要名字里带 'base' 或 'high' 就认为是全局，其余默认为局部(wrist)
            # 你需要根据你的实际数据集 key 名称调整这里的字符串
            if "base" in name or "high" in name or "global" in name:
                global_len += current_len
            #======新增=======#

            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        #======新增=======#
        # 先将所有视觉 Token 拼在一起
        visual_tokens = jnp.concatenate(tokens, axis=1) # (B, Total_Img_Len, Dim)

        # --- 2. 🔥 [核心修改] 应用 Soft MoE 路由 🔥 ---
        # 如果传入了 alpha (说明是在计算 Action Loss)，则通过专家处理
        # 如果 alpha 是 None (说明是在预测意图)，则跳过，保持原始特征
        routing_stats = None
        if self.config.use_visual_moe_update and gating_alpha is not None and timestep_emb is not None:
            moe_output = self.visual_moe(
                visual_tokens=visual_tokens,
                global_token_len=global_len,
                gating_alpha=gating_alpha,
                timestep_emb=timestep_emb, # 这里现在会根据 t 动态调整专家分配
                intent_view_prior=intent_view_prior,
                return_stats=return_routing_stats,
            )
            if return_routing_stats:
                visual_tokens, routing_stats = moe_output
            else:
                visual_tokens = moe_output
            visual_tokens = self.moe_norm(visual_tokens)

        # --- 3. 准备最终 Token 列表 ---
        tokens = [visual_tokens]
        #======新增=======#

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)

        return tokens, input_mask, ar_mask, img_token_count, global_len, routing_stats

    @at.typecheck
    def embed_suffix(
        self, obs: _model.RobotObservation, noisy_actions: _model.ActionSequence, timestep: at.Float[at.Array, " b"],
        # 🔥 [新增参数] 接收意图 Token
        # 🔥 修改 1：建议直接接收 intent_id (int)，在函数内部转换
        # intent_id: at.Int[at.Array, " b"] | None = None,
        intent_token: at.Float[at.Array, "b 1 emb"] | None = None 
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,

    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        #======新增=======#
        # intent_token = None
        # # 🔥 [新增逻辑] 注入意图 Token
        if intent_token is not None:
            #print("hello.........")
            tokens.append(intent_token)
            input_mask.append(jnp.ones((intent_token.shape[0], 1), dtype=jnp.bool_))
            # 意图 Token 和 状态 Token 属于同一个上下文 Block，可以相互关注
            ar_mask += [False] 
        #======新增=======#

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        
        if self.pi05:
            # print("hello..........................................")
            # time MLP (for adaRMS)
            # intent_emb = self.intent_embedding(intent_id)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.RobotObservation, actions: _model.ActionSequence, *, train: bool = False, step: int | None = None  # 🔥 [修改] 添加 step 参数
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng, teacher_force_rng, view_mask_rng = jax.random.split(rng, 5)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        if train:
            observation = self._apply_random_view_masking(observation, view_mask_rng)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        #======新增=======#
        # =========================================================================
        # 🚀 第一阶段：意图预测 (Stage 1: Intent Prediction)
        # =========================================================================

        # one big forward pass of prefix + suffix at once
        prefix_tokens_raw, prefix_mask, prefix_ar_mask, img_len, global_len, _ = self.embed_prefix(
            observation, gating_alpha=None, timestep_emb=None
        )
        # 🔥🔥🔥 [新增] 初始化 adarms_cond，防止报错 🔥🔥🔥
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions_prefix = jnp.cumsum(prefix_mask, axis=1) - 1
        
        # 1.2 运行 VLM (仅 Prefix 部分)
        (prefix_features_raw, _), _ = self.PaliGemma.llm(
            [prefix_tokens_raw, None],
            mask=prefix_attn_mask,
            positions=positions_prefix,
        )
        mixed_features = prefix_features_raw

        # 恢复维度
        batch_size = observation.state.shape[0]
        prefix_seq_len = prefix_tokens_raw.shape[1]
        if mixed_features.ndim == 2:
             if mixed_features.shape[0] != batch_size * prefix_seq_len:
                 mixed_features = mixed_features[:batch_size * prefix_seq_len]
             mixed_features = mixed_features.reshape(batch_size, prefix_seq_len, -1)

        # ------------------------------------
        # 🔥🔥🔥【Stop Gradient】🔥🔥🔥
        visual_features = jax.lax.stop_gradient(mixed_features[:, :img_len, :])
        text_features = jax.lax.stop_gradient(mixed_features[:, img_len:, :])
        state_features = self._encode_state_for_intent(observation.state, observation.history_actions)
        text_mask = prefix_mask[:, img_len:]
        # ------------------------------------

        # 1.4 运行 IntentHead
        intent_logits = self._predict_intent_logits(
            visual_features,
            text_features,
            state_features,
            text_mask,
        )
        
        # 1.5 计算 Intent Loss & 统计调试信息
        true_labels = self._ensure_intent_label_sequence(observation.intent_label).astype(jnp.float32)
        intent_probs = nnx.sigmoid(intent_logits)
        soft_phase_probs = self._rule_phase_probs_from_intent_probs(intent_probs)
        true_label_summary = self._compute_intent_horizon_summary_from_probs(true_labels)
        pred_label_summary = self._compute_intent_horizon_summary(intent_logits)
        true_chunk_labels = self._chunk_intent_label_from_seq(true_labels)
        predicted_chunk_intent = self._chunk_intent_label_from_seq(intent_probs)
        if observation.phase_label is not None:
            true_phase_labels = jnp.asarray(observation.phase_label).astype(jnp.int32).reshape((-1,))
        else:
            true_phase_labels = self._chunk_phase_label_from_intent_seq(true_labels)
        predicted_phase = self._rule_phase_from_intent_probs(intent_probs)
        phase_probs = jax.nn.one_hot(predicted_phase, 3, dtype=jnp.float32)
        alpha_seq = (
            self._alpha_seq_from_direct_route_logits(intent_logits)
            if self.config.use_direct_time_state_routing
            else self._alpha_seq_from_intent_probs(intent_probs)
        )
        action_debug_alpha_seq = self._expand_alpha_seq_to_action_horizon(alpha_seq)
        chunk_alpha_vec = self._pool_chunk_alpha_from_alpha_seq(alpha_seq)
        true_alpha_seq = self._alpha_seq_from_intent_probs(true_labels)
        true_chunk_alpha_vec = self._pool_chunk_alpha_from_alpha_seq(true_alpha_seq)
        phase_loss = jnp.asarray(0.0, dtype=jnp.float32)
        chunk_intent_probs = pred_label_summary[:, 0]
        intent_view_prior = self._compute_intent_view_prior(pred_label_summary)
        
        # --- 正样本统计 ---
        pos_mask = true_chunk_labels.astype(jnp.float32)
        num_pos = jnp.sum(pos_mask)
        avg_pos_prob = jnp.where(num_pos > 0, jnp.sum(chunk_intent_probs * pos_mask) / num_pos, -1.0)
        
        # --- 负样本统计 (补全) ---
        neg_mask = 1.0 - pos_mask
        num_neg = jnp.sum(neg_mask)
        avg_neg_prob = jnp.where(num_neg > 0, jnp.sum(chunk_intent_probs * neg_mask) / num_neg, -1.0)
        intent_label_ones_count = jnp.sum(true_labels.astype(jnp.float32))
        intent_label_zeros_count = jnp.sum((1.0 - true_labels).astype(jnp.float32))
        
        # Loss 计算
        pos_weight = 15.0 
        intent_loss_elementwise = optax.sigmoid_binary_cross_entropy(intent_logits, true_labels)
        weight_vector = true_labels * pos_weight + (1.0 - true_labels)
        intent_loss = (intent_loss_elementwise * weight_vector).mean()
        
        # 1.6 生成 Alpha。alpha_seq 保留 30-step 时间结构；
        # 中后段 pooling 成 chunk_alpha 控制 visual MoE，避免前段短脉冲误导整段视觉重塑。
        legacy_alpha_vec, alpha_diag = self._compute_alpha_vec_with_diagnostics(
            visual_features,
            text_features,
            state_features,
            intent_logits,
            text_mask,
        )
        phase_alpha_vec = chunk_alpha_vec
        alpha_vec = phase_alpha_vec if self.config.use_phase_alpha else legacy_alpha_vec
        base_alpha_vec = self._stabilize_alpha(jax.nn.softmax(alpha_diag["base_logits"], axis=-1))
        intent_delta_alpha_vec = self._stabilize_alpha(
            jax.nn.softmax(self.config.intent_delta_scale * alpha_diag["intent_delta_logits"], axis=-1)
        )
        intent_delta_norm = jnp.linalg.norm(alpha_diag["intent_delta_logits"], axis=-1)
        base_delta_logit_cosine = jnp.sum(
            alpha_diag["base_logits"] * alpha_diag["intent_delta_logits"], axis=-1
        ) / (
            jnp.linalg.norm(alpha_diag["base_logits"], axis=-1)
            * jnp.linalg.norm(alpha_diag["intent_delta_logits"], axis=-1)
            + 1e-6
        )
        alpha_entropy = -jnp.sum(alpha_vec * jnp.log(alpha_vec + 1e-6), axis=-1)


        # =========================================================================
        # 🚀 阶段判断逻辑 (Warmup Logic)
        # =========================================================================
        w_intent = self.config.intent_loss_weight
        w_diffusion = 0.0 
        if step is not None:
            warmup_steps = self.config.intent_warmup_steps
            is_joint_phase = step >= warmup_steps
            w_diffusion = jnp.where(is_joint_phase, 1.0, 0.0)
            w_intent = self.config.intent_loss_weight

        # =========================================================================
        # 🚀 第二阶段：动作生成 (Stage 2: Action Generation)
        # =========================================================================
        # 🔥 修改 3：先计算时间向量，供视觉模块使用
        time_emb = posemb_sincos(time, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb) # 此时 time_emb 是 1024 维

        gt_intent = true_chunk_labels
        predicted_intent = predicted_chunk_intent
        tf_prob = self._intent_teacher_forcing_prob(step) if train else jnp.asarray(0.0, dtype=jnp.float32)
        use_gt_mask = jax.random.bernoulli(teacher_force_rng, p=tf_prob, shape=gt_intent.shape) if train else jnp.zeros_like(gt_intent, dtype=jnp.bool_)
        gt_intent_signal = true_label_summary
        if self.config.use_teacher_forced_alpha_for_moe and train:
            if self.config.use_phase_alpha:
                alpha_vec_for_moe = jnp.where(use_gt_mask[:, None], true_chunk_alpha_vec, chunk_alpha_vec)
                alpha_seq_for_action = jnp.where(use_gt_mask[:, None, None], true_alpha_seq, alpha_seq)
            else:
                alpha_vec_for_moe = self._compute_alpha_vec(
                    visual_features,
                    text_features,
                    state_features,
                    intent_logits,
                    text_mask,
                    intent_summary=gt_intent_signal,
                )
                alpha_seq_for_action = alpha_seq
            intent_view_prior_for_moe = None if self.config.freeze_intent_view_prior else self._compute_intent_view_prior(gt_intent_signal)
        else:
            alpha_vec_for_moe = alpha_vec
            alpha_seq_for_action = alpha_seq
            intent_view_prior_for_moe = None if self.config.freeze_intent_view_prior else intent_view_prior
        alpha_for_moe = jax.lax.stop_gradient(alpha_vec_for_moe) if self.config.detach_alpha_for_moe else alpha_vec_for_moe
        alpha_for_action = jax.lax.stop_gradient(alpha_seq_for_action) if self.config.detach_alpha_for_moe else alpha_seq_for_action
        if (
            self.config.use_action_side_routing
            and self.config.use_action_alpha_seq
            and self.config.use_learned_static_action_routing
        ):
            action_alpha_seq = self._learned_static_action_alpha_seq(
                batch_size=alpha_for_action.shape[0],
                horizon=alpha_for_action.shape[1],
                dtype=alpha_for_action.dtype,
            )
        elif (
            self.config.use_action_side_routing
            and self.config.use_action_alpha_seq
            and self.config.use_static_action_routing
        ):
            action_alpha_seq = self._static_action_alpha_seq(
                batch_size=alpha_for_action.shape[0],
                horizon=alpha_for_action.shape[1],
                dtype=alpha_for_action.dtype,
            )
        else:
            action_alpha_seq = alpha_for_action if (
                self.config.use_phase_alpha
                and self.config.use_action_alpha_seq
                and self.config.use_action_side_routing
            ) else None
        action_alpha_seq = self._expand_alpha_seq_to_action_horizon(action_alpha_seq)
        alpha_for_action_metrics = self._expand_alpha_seq_to_action_horizon(alpha_for_action)

        # 🔥 修改 4：将计算好的 alpha 和 time_emb 注入感知模块
        prefix_tokens_gated, _, _, _, global_len_gated, routing_stats = self.embed_prefix(
            observation,
            gating_alpha=alpha_for_moe,
            intent_view_prior=intent_view_prior_for_moe,
            timestep_emb=time_emb,
            return_routing_stats=True,
        )
        # 🔥🔥🔥【恢复】取消注释并修改 🔥🔥🔥
        # 2.2 准备意图 Token (使用 GT Teacher Forcing)
        # 确保转为 int 类型以进行 Embedding 查表
        mixed_intent = jnp.where(use_gt_mask, gt_intent, predicted_intent)
        intent_token_input = (
            self.intent_embedding(mixed_intent)[:, None, :]
            if self.config.use_intent_token
            else None
        )



        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time,intent_token=intent_token_input)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1


        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens_gated, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )

        action_visual_features = prefix_out[:, :img_len, :]
        action_text_features = prefix_out[:, img_len:, :]
        control_context = self._compute_control_context(
            action_visual_features,
            action_text_features,
            state_features,
            text_mask,
        )
        action_features = suffix_out[:, -self.action_horizon :]
        (
            action_features,
            action_visual_cross_attn_norm,
            action_visual_global_cross_attn_norm,
            action_visual_local_cross_attn_norm,
        ) = self._apply_action_visual_cross_attention(
            action_features,
            action_visual_features,
            control_context,
            alpha_seq=action_alpha_seq,
            global_token_len=global_len_gated,
        )
        action_features, action_adapter_norm = self._apply_action_adapter(action_features, control_context)
        action_context_residual, action_context_residual_norm = self._compute_action_context_residual(
            action_features, control_context
        )
        main_v_t = self.action_out_proj(action_features)
        delta_v_t = self.pose_refine_head(action_features)

        delta_scale = self._delta_blend_scale(time)
        v_t = (
            main_v_t
            + delta_scale * delta_v_t
            + self.config.action_context_residual_scale * action_context_residual
        )


        # v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        diffusion_loss_raw = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        diffusion_loss = diffusion_loss_raw * w_diffusion
        # Give delta branch a direct residual target so it does not collapse to near-zero.
        delta_target = jax.lax.stop_gradient(u_t - main_v_t)
        delta_aux_loss_raw = jnp.mean(jnp.square(delta_v_t - delta_target), axis=-1)
        delta_aux_loss = self.config.delta_aux_loss_weight * delta_aux_loss_raw * w_diffusion
        alpha_entropy_loss = self.config.alpha_entropy_loss_weight * alpha_entropy.mean()

        # =========================================================================
        # 🚀 组合总损失
        # =========================================================================
        phase_loss_weighted = self.config.phase_loss_weight * phase_loss
        total_loss = diffusion_loss + delta_aux_loss + w_intent * intent_loss + phase_loss_weighted + alpha_entropy_loss

        # 🔥🔥🔥【修复】补全所有需要的 key 🔥🔥🔥
        metrics = {
            'total_loss':  total_loss,
            'diffusion_loss': diffusion_loss_raw.mean(), 
            'delta_aux_loss': delta_aux_loss_raw.mean(),
            'intent_loss': intent_loss,
            'intent_loss_weight': jnp.asarray(self.config.intent_loss_weight, dtype=jnp.float32),
            'random_view_masking_prob': jnp.asarray(
                self.config.random_view_masking_prob, dtype=jnp.float32
            ),
            'phase_loss': phase_loss,
            'phase_loss_weighted': phase_loss_weighted,
            'phase_pred_acc': jnp.mean((predicted_phase == true_phase_labels).astype(jnp.float32)),
            'phase_rule_acc': jnp.mean((predicted_phase == true_phase_labels).astype(jnp.float32)),
            'phase_pred_0_ratio': jnp.mean((predicted_phase == 0).astype(jnp.float32)),
            'phase_pred_1_ratio': jnp.mean((predicted_phase == 1).astype(jnp.float32)),
            'phase_pred_2_ratio': jnp.mean((predicted_phase == 2).astype(jnp.float32)),
            'phase_gt_0_ratio': jnp.mean((true_phase_labels == 0).astype(jnp.float32)),
            'phase_gt_1_ratio': jnp.mean((true_phase_labels == 1).astype(jnp.float32)),
            'phase_gt_2_ratio': jnp.mean((true_phase_labels == 2).astype(jnp.float32)),
            'phase_prob_0_mean': phase_probs[:, 0].mean(),
            'phase_prob_1_mean': phase_probs[:, 1].mean(),
            'phase_prob_2_mean': phase_probs[:, 2].mean(),
            'soft_phase_prob_0_mean': soft_phase_probs[:, 0].mean(),
            'soft_phase_prob_1_mean': soft_phase_probs[:, 1].mean(),
            'soft_phase_prob_2_mean': soft_phase_probs[:, 2].mean(),
            'use_phase_alpha': jnp.asarray(float(self.config.use_phase_alpha), dtype=jnp.float32),
            'alpha_mean': jnp.mean(alpha_vec),
            'alpha_global_mean': alpha_vec[:, 0].mean(),
            'alpha_local_mean': alpha_vec[:, 1].mean(),
            'alpha_moe_global_mean': alpha_for_moe[:, 0].mean(),
            'alpha_moe_local_mean': alpha_for_moe[:, 1].mean(),
            'alpha_seq_local_early_mean': alpha_for_action_metrics[:, :10, 1].mean(),
            'alpha_seq_local_mid_mean': alpha_for_action_metrics[:, 10:20, 1].mean(),
            'alpha_seq_local_late_mean': alpha_for_action_metrics[:, 20:, 1].mean(),
            'chunk_alpha_pool_mid_local_mean': alpha_for_action_metrics[:, 10:20, 1].mean(),
            'chunk_alpha_pool_late_local_mean': alpha_for_action_metrics[:, 20:, 1].mean(),
            'chunk_alpha_pool_selected_local_mean': jnp.maximum(
                alpha_for_action_metrics[:, 10:20, 1].mean(axis=1),
                alpha_for_action_metrics[:, 20:, 1].mean(axis=1),
            ).mean(),
            'alpha_seq_local_std': jnp.std(alpha_for_action_metrics[:, :, 1]),
            'intent_prediction_horizon': jnp.asarray(
                float(self.config.intent_prediction_horizon), dtype=jnp.float32
            ),
            'intent_token_enabled': jnp.asarray(float(self.config.use_intent_token), dtype=jnp.float32),
            'direct_time_state_routing_enabled': jnp.asarray(
                float(self.config.use_direct_time_state_routing), dtype=jnp.float32
            ),
            'action_alpha_seq_enabled': jnp.asarray(
                float(self.config.use_phase_alpha and self.config.use_action_alpha_seq), dtype=jnp.float32
            ),
            'action_side_routing_enabled': jnp.asarray(
                float(
                    self.config.use_phase_alpha
                    and self.config.use_action_alpha_seq
                    and self.config.use_action_side_routing
                ),
                dtype=jnp.float32,
            ),
            'static_action_routing_enabled': jnp.asarray(
                float(
                    self.config.use_action_side_routing
                    and self.config.use_action_alpha_seq
                    and self.config.use_static_action_routing
                ),
                dtype=jnp.float32,
            ),
            'static_action_routing_global': jnp.asarray(
                self.config.static_action_routing_global, dtype=jnp.float32
            ),
            'static_action_routing_local': jnp.asarray(
                self.config.static_action_routing_local, dtype=jnp.float32
            ),
            'learned_static_action_routing_enabled': jnp.asarray(
                float(
                    self.config.use_action_side_routing
                    and self.config.use_action_alpha_seq
                    and self.config.use_learned_static_action_routing
                ),
                dtype=jnp.float32,
            ),
            'learned_static_action_routing_global': jax.nn.softmax(
                jnp.asarray(self.learned_static_action_routing_logits), axis=-1
            )[0].astype(jnp.float32),
            'learned_static_action_routing_local': jax.nn.softmax(
                jnp.asarray(self.learned_static_action_routing_logits), axis=-1
            )[1].astype(jnp.float32),
            'visual_moe_update_enabled': jnp.asarray(
                float(self.config.use_visual_moe_update), dtype=jnp.float32
            ),
            'alpha_max_mean': jnp.mean(jnp.max(alpha_vec, axis=-1)),
            'alpha_entropy': alpha_entropy.mean(),
            'alpha_entropy_loss': alpha_entropy_loss,
            'base_alpha_max_mean': jnp.mean(jnp.max(base_alpha_vec, axis=-1)),
            'intent_delta_alpha_max_mean': jnp.mean(jnp.max(intent_delta_alpha_vec, axis=-1)),
            'intent_delta_norm': jnp.mean(intent_delta_norm),
            'base_delta_logit_cosine': jnp.mean(base_delta_logit_cosine),
            'intent_view_global_mean': jnp.mean(intent_view_prior[:, 0]),
            'intent_view_local_mean': jnp.mean(intent_view_prior[:, 1]),
            'intent_view_max_mean': jnp.mean(jnp.max(intent_view_prior, axis=-1)),
            'intent_view_entropy': jnp.mean(-jnp.sum(intent_view_prior * jnp.log(intent_view_prior + 1e-6), axis=-1)),
            'intent_view_frozen': jnp.asarray(float(self.config.freeze_intent_view_prior), dtype=jnp.float32),
            'view_prior_scale': jnp.asarray(self.config.view_prior_scale, dtype=jnp.float32),
            'global_local_expert_bias': jnp.asarray(self.config.global_local_expert_bias, dtype=jnp.float32),
            'alpha_only_moe_routing': jnp.asarray(float(self.config.alpha_only_moe_routing), dtype=jnp.float32),
            'view_router_logit_scale': jnp.asarray(self.config.view_router_logit_scale, dtype=jnp.float32),
            'token_router_logit_scale': jnp.asarray(self.config.token_router_logit_scale, dtype=jnp.float32),
            'action_visual_cross_attn_norm': action_visual_cross_attn_norm.mean(),
            'action_visual_global_cross_attn_norm': action_visual_global_cross_attn_norm.mean(),
            'action_visual_local_cross_attn_norm': action_visual_local_cross_attn_norm.mean(),
            'action_adapter_norm': action_adapter_norm.mean(),
            'action_context_residual_norm': action_context_residual_norm.mean(),
            'delta_scale_mean': jnp.mean(delta_scale),
            'w_diffusion': w_diffusion, 
            'tf_prob': tf_prob,
            'tf_ratio': jnp.mean(use_gt_mask.astype(jnp.float32)),
            'intent_pred_acc': jnp.mean((predicted_intent == gt_intent).astype(jnp.float32)),
            'intent_label_ones_count': intent_label_ones_count,
            'intent_label_zeros_count': intent_label_zeros_count,
            # Per-sample diagnostics for task-level aggregation in train_step.
            'per_sample_diffusion_loss': diffusion_loss_raw.mean(axis=-1),
            'per_sample_intent_acc': (predicted_intent == gt_intent).astype(jnp.float32),
            'per_sample_phase_acc': (predicted_phase == true_phase_labels).astype(jnp.float32),
            'per_sample_tf_ratio': use_gt_mask.astype(jnp.float32),
            'per_sample_alpha_entropy': alpha_entropy,
            # 补全训练脚本需要的 key
            'debug_num_pos': num_pos,
            'debug_pos_prob': avg_pos_prob,
            'debug_num_neg': num_neg,
            'debug_neg_prob': avg_neg_prob
        }
        for expert_idx in range(alpha_vec.shape[-1]):
            metrics[f'alpha_usage_{expert_idx}'] = alpha_vec[:, expert_idx].mean()
        if routing_stats is not None:
            expert_usage = routing_stats["expert_usage"]
            metrics["router_confidence"] = routing_stats["router_confidence"].mean()
            metrics["per_sample_router_confidence"] = routing_stats["router_confidence"]
            for expert_idx in range(expert_usage.shape[-1]):
                metrics[f"expert_usage_{expert_idx}"] = expert_usage[:, expert_idx].mean()
            metrics["per_sample_expert_usage"] = expert_usage
            if "view_patch_scale_global" in routing_stats:
                metrics["view_patch_scale_global"] = routing_stats["view_patch_scale_global"].mean()
                metrics["view_patch_scale_local"] = routing_stats["view_patch_scale_local"].mean()
                metrics["alpha_only_moe_routing"] = routing_stats["alpha_only_moe_routing"].mean()
                metrics["view_router_logit_scale"] = routing_stats["view_router_logit_scale"].mean()
                metrics["token_router_logit_scale"] = routing_stats["token_router_logit_scale"].mean()
                metrics["alpha_routing_error"] = routing_stats["alpha_routing_error"].mean()
                metrics["moe_delta_norm"] = routing_stats["moe_delta_norm"].mean()
                metrics["moe_patch_norm"] = routing_stats["moe_patch_norm"].mean()
                metrics["moe_gain_tanh"] = routing_stats["moe_gain_tanh"].mean()
                metrics["moe_dynamic_scale"] = routing_stats["moe_dynamic_scale"].mean()
            for key in (
                "global_token_global_expert_weight",
                "global_token_local_expert_weight",
                "local_token_global_expert_weight",
                "local_token_local_expert_weight",
                "cross_expert_leakage",
            ):
                if key in routing_stats:
                    metrics[key] = routing_stats[key].mean()

        return total_loss, metrics

    def infer_alpha_debug(
        self,
        observation: _model.RobotObservation,
    ) -> dict[str, at.Array]:
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]

        prefix_tokens_raw, prefix_mask, prefix_ar_mask, img_len, global_len, _ = self.embed_prefix(observation, gating_alpha=None)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions_prefix = jnp.cumsum(prefix_mask, axis=1) - 1

        (prefix_features_raw, _), _ = self.PaliGemma.llm(
            [prefix_tokens_raw, None],
            mask=prefix_attn_mask,
            positions=positions_prefix
        )
        mixed_features = prefix_features_raw
        if mixed_features.ndim == 2:
            mixed_features = mixed_features.reshape(batch_size, prefix_tokens_raw.shape[1], -1)

        visual_features = mixed_features[:, :img_len, :]
        text_features = mixed_features[:, img_len:, :]
        text_mask = prefix_mask[:, img_len:]
        state_features = self._encode_state_for_intent(observation.state, observation.history_actions)

        intent_logits = self._predict_intent_logits(
            visual_features,
            text_features,
            state_features,
            text_mask,
        )
        intent_probs = nnx.sigmoid(intent_logits)
        soft_phase_probs = self._rule_phase_probs_from_intent_probs(intent_probs)
        predicted_phase = self._rule_phase_from_intent_probs(intent_probs)
        phase_probs = jax.nn.one_hot(predicted_phase, 3, dtype=jnp.float32)
        alpha_seq = (
            self._alpha_seq_from_direct_route_logits(intent_logits)
            if self.config.use_direct_time_state_routing
            else self._alpha_seq_from_intent_probs(intent_probs)
        )
        action_debug_alpha_seq = self._expand_alpha_seq_to_action_horizon(alpha_seq)
        chunk_alpha_vec = self._pool_chunk_alpha_from_alpha_seq(alpha_seq)
        legacy_alpha_vec, alpha_diag = self._compute_alpha_vec_with_diagnostics(
            visual_features,
            text_features,
            state_features,
            intent_logits,
            text_mask,
        )
        phase_alpha_vec = chunk_alpha_vec
        alpha_vec = phase_alpha_vec if self.config.use_phase_alpha else legacy_alpha_vec
        base_alpha_vec = self._stabilize_alpha(jax.nn.softmax(alpha_diag["base_logits"], axis=-1))
        intent_delta_alpha_vec = self._stabilize_alpha(
            jax.nn.softmax(self.config.intent_delta_scale * alpha_diag["intent_delta_logits"], axis=-1)
        )
        intent_summary = self._compute_intent_horizon_summary(intent_logits)
        intent_view_prior = self._compute_intent_view_prior(intent_summary)
        intent_view_prior_for_moe = None if self.config.freeze_intent_view_prior else intent_view_prior
        neutral_intent_view_prior = jnp.ones_like(intent_view_prior) * 0.5
        action_stability = action_debug_alpha_seq[:, :, 1]
        if self.config.action_stability_source == "intent_mean":
            action_stability = jnp.broadcast_to(
                jnp.mean(intent_probs, axis=1, keepdims=True), action_debug_alpha_seq[:, :, 1].shape
            )
        action_stability_scale_seq = 1.0 - self.config.action_stability_scale * jnp.clip(action_stability, 0.0, 1.0)
        return {
            "alpha": alpha_vec,
            "base_alpha": base_alpha_vec,
            "intent_delta_alpha": intent_delta_alpha_vec,
            "base_logits": alpha_diag["base_logits"],
            "intent_delta_logits": alpha_diag["intent_delta_logits"],
            "intent_view_prior": intent_view_prior,
            "intent_view_prior_used": neutral_intent_view_prior if self.config.freeze_intent_view_prior else intent_view_prior,
            "intent_view_frozen": jnp.asarray(float(self.config.freeze_intent_view_prior), dtype=jnp.float32),
            "intent_logits": intent_logits,
            "intent_probs": intent_probs,
            "intent_summary": intent_summary,
            "predicted_intent": self._chunk_intent_label_from_seq(intent_probs),
            "predicted_intent_seq": (intent_logits > 0).astype(jnp.int32),
            "phase_probs": phase_probs,
            "soft_phase_probs": soft_phase_probs,
            "predicted_phase": predicted_phase,
            "alpha_seq": action_debug_alpha_seq,
            "predicted_alpha_seq": alpha_seq,
            "phase_alpha": phase_alpha_vec,
            "phase_alpha_table": self._phase_alpha_table(),
            "use_phase_alpha": jnp.asarray(float(self.config.use_phase_alpha), dtype=jnp.float32),
            "action_stability_enabled": jnp.asarray(float(self.config.use_action_stability_scaling), dtype=jnp.float32),
            "action_stability_strength": jnp.asarray(self.config.action_stability_scale, dtype=jnp.float32),
            "action_stability_scale_seq": action_stability_scale_seq,
        }

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.RobotObservation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        intent_corruption_rate: float = 0.0,
        intent_corruption_seed: int = 0,
    ) -> _model.ActionSequence:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # =========================================================================
        # 🚀 第一阶段：预测意图 & 准备加权 Prefix (Stage 1: Predict Intent & Gate Prefix)
        # =========================================================================


        # first fill KV cache with a forward pass of the prefix
        prefix_tokens_raw, prefix_mask, prefix_ar_mask, img_len, global_len, _ = self.embed_prefix(observation, gating_alpha=None)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions_prefix = jnp.cumsum(prefix_mask, axis=1) - 1

        # 1.2 运行 VLM 提取特征
        (prefix_features_raw, _), _ = self.PaliGemma.llm(
            [prefix_tokens_raw, None],
            mask=prefix_attn_mask,
            positions=positions_prefix
        )
        mixed_features = prefix_features_raw

        # Reshape (如果需要)
        if mixed_features.ndim == 2:
            mixed_features = mixed_features.reshape(batch_size, prefix_tokens_raw.shape[1], -1)

        # 1.3 拆分特征 & 准备输入
        visual_features = mixed_features[:, :img_len, :]
        text_features = mixed_features[:, img_len:, :]
        text_mask = prefix_mask[:, img_len:]
        state_features = self._encode_state_for_intent(observation.state, observation.history_actions)
        control_context = self._compute_control_context(
            visual_features,
            text_features,
            state_features,
            text_mask,
        )
        
        # 1.4 运行 IntentHead 预测意图
        intent_logits = self._predict_intent_logits(
            visual_features,
            text_features,
            state_features,
            text_mask,
        )
        # Flip a controlled fraction of intent predictions at the exact
        # action-facing interface. The dedicated key leaves diffusion noise
        # untouched and gives matched 0/10/20% runs reproducible masks.
        corruption_rate = jnp.clip(
            jnp.asarray(intent_corruption_rate, dtype=jnp.float32),
            0.0,
            1.0,
        )
        corruption_key = jax.random.key(jnp.asarray(intent_corruption_seed, dtype=jnp.uint32))
        corruption_mask = jax.random.bernoulli(
            corruption_key,
            p=corruption_rate,
            shape=intent_logits.shape,
        )
        intent_logits = jnp.where(corruption_mask, -intent_logits, intent_logits)
        intent_probs = nnx.sigmoid(intent_logits)
        predicted_phase = self._rule_phase_from_intent_probs(intent_probs)
        alpha_seq = (
            self._alpha_seq_from_direct_route_logits(intent_logits)
            if self.config.use_direct_time_state_routing
            else self._alpha_seq_from_intent_probs(intent_probs)
        )
        chunk_alpha_vec = self._pool_chunk_alpha_from_alpha_seq(alpha_seq)
        
        
        # # 1.5 生成门控权重 Alpha 和 意图 Token
        legacy_alpha_vec = self._compute_alpha_vec(
            visual_features,
            text_features,
            state_features,
            intent_logits,
            text_mask,
        )
        phase_alpha_vec = chunk_alpha_vec
        alpha_vec = phase_alpha_vec if self.config.use_phase_alpha else legacy_alpha_vec
        intent_summary = self._compute_intent_horizon_summary(intent_logits)
        intent_view_prior = self._compute_intent_view_prior(intent_summary)
        intent_view_prior_for_moe = None if self.config.freeze_intent_view_prior else intent_view_prior
        if (
            self.config.use_action_side_routing
            and self.config.use_action_alpha_seq
            and self.config.use_learned_static_action_routing
        ):
            action_alpha_seq = self._learned_static_action_alpha_seq(
                batch_size=batch_size,
                horizon=alpha_seq.shape[1],
                dtype=alpha_seq.dtype,
            )
        elif (
            self.config.use_action_side_routing
            and self.config.use_action_alpha_seq
            and self.config.use_static_action_routing
        ):
            action_alpha_seq = self._static_action_alpha_seq(
                batch_size=batch_size,
                horizon=alpha_seq.shape[1],
                dtype=alpha_seq.dtype,
            )
        else:
            action_alpha_seq = alpha_seq if (
                self.config.use_phase_alpha
                and self.config.use_action_alpha_seq
                and self.config.use_action_side_routing
            ) else None
        action_alpha_seq = self._expand_alpha_seq_to_action_horizon(action_alpha_seq)

        # # 🔥🔥🔥 [新增] 使用 jax.debug.callback 调用外部写入函数 🔥🔥🔥
        # # 这里的 alpha 是 GPU/TPU 上的数组，jax 会自动把它传回 CPU 给回调函数
        # # 我们取 mean() 以防 batch_size > 1，或者直接传 alpha[0,0,0]
        # jax.debug.callback(_log_alpha_to_file, alpha.mean())

        #  # 🔥🔥🔥【恢复】取消注释 🔥🔥🔥
        # 将 logits (Batch,) 转为 0 或 1 的索引
        predicted_intent = self._chunk_intent_label_from_seq(nnx.sigmoid(intent_logits))
        # 查表得到 Embedding 向量
        predicted_intent_token = (
            self.intent_embedding(predicted_intent)[:, None, :]
            if self.config.use_intent_token
            else None
        )

        # jax.debug.print("INFERENCE DEBUG: alpha mean={x}, intent_token={y}", 
        #         x=alpha.mean(), 
        #         y=predicted_intent.astype(jnp.int32))

        # # 1.6 🔥【关键】生成加权后的 Prefix，并填充 KV Cache
        # # 注意：这里的 prefix_mask 和 positions_prefix 都是复用的
        # 🔥 修改 3：先计算时间向量，供视觉模块使用
        inf_time = jnp.zeros((batch_size,))
        time_emb = posemb_sincos(inf_time, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb) # 此时 time_emb 是 1024 维

        # 🔥 修改 4：将计算好的 alpha 和 time_emb 注入感知模块
        prefix_tokens_gated, _, _, _, global_len_gated, _ = self.embed_prefix(
            observation,
            gating_alpha=alpha_vec,
            intent_view_prior=intent_view_prior_for_moe,
            timestep_emb=time_emb,
        )
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens_gated, None], 
            mask=prefix_attn_mask, 
            positions =positions_prefix
        )
        action_visual_features = prefix_out[:, :img_len, :]
        action_text_features = prefix_out[:, img_len:, :]
        control_context = self._compute_control_context(
            action_visual_features,
            action_text_features,
            state_features,
            text_mask,
        )
        # =========================================================================
        # 🚀 第二阶段：扩散去噪循环 (Stage 2: Diffusion Denoising Loop)
        # =========================================================================
        
        

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size),
                intent_token=predicted_intent_token
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask_step = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask_step, suffix_attn_mask], axis=-1)
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
    
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            
            action_features = suffix_out[:, -self.action_horizon :]
            action_features, _, _, _ = self._apply_action_visual_cross_attention(
                action_features,
                action_visual_features,
                control_context,
                alpha_seq=action_alpha_seq,
                global_token_len=global_len_gated,
            )
            action_features, _ = self._apply_action_adapter(action_features, control_context)
            action_context_residual, _ = self._compute_action_context_residual(action_features, control_context)
            main_v_t = self.action_out_proj(action_features)
            delta_v_t = self.pose_refine_head(action_features)

            step_time = jnp.broadcast_to(time, (batch_size,))
            delta_scale = self._delta_blend_scale(step_time)
            v_t = (
                main_v_t
                + delta_scale * delta_v_t
                + self.config.action_context_residual_scale * action_context_residual
            )
            v_t = self._apply_action_stability_scaling(
                v_t,
                action_alpha_seq,
                intent_probs=intent_probs,
            )

            
            # v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
