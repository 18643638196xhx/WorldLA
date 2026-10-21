import logging

import einops
import optax
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
import worldla.models.intention as _intent
import worldla.models.sparse_layers as _moe

logger = logging.getLogger('worldla')


def make_attn_mask(input_mask, mask_ar):
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
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


class WorldLABackbone(_model.RobotModelBase):
    def __init__(self, config: backbone_config.WorldLAConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _language.get_config(config.paligemma_variant)
        action_expert_config = _language.get_config(config.action_expert_variant)

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
                kernel_init=nnx.initializers.zeros,
                rngs=rngs,
            ),
        )

        self.intent_predictor_head = _intent.IntentHead(
            embed_dim=paligemma_config.width,
            num_heads=4,
            rngs=rngs,
        )
        self.intent_state_encoder = nnx.Linear(
            config.action_dim,
            paligemma_config.width,
            rngs=rngs,
        )
        self.visual_moe = _moe.SoftMoERouter(
            paligemma_config,
            action_expert_width=action_expert_config.width,
            rngs=rngs,
        )
        self.moe_norm = nnx.RMSNorm(paligemma_config.width, rngs=rngs)
        self.intent_embedding = nnx.Embed(
            num_embeddings=2,
            features=action_expert_config.width,
            rngs=rngs,
        )

        self.config = config
        self.deterministic = True

    def _delta_blend_scale(self, time: at.Float[at.Array, " b"]) -> at.Float[at.Array, "b 1 1"]:
        late_scale = self.config.delta_head_min_late_scale + (
            1.0 - self.config.delta_head_min_late_scale
        ) * (1.0 - time)
        return (self.config.delta_head_base_scale * late_scale)[..., None, None]

    def _stabilize_alpha(self, alpha: at.Float[at.Array, "b 1 1"]) -> at.Float[at.Array, "b 1 1"]:
        return jnp.clip(alpha, self.config.alpha_clip_min, self.config.alpha_clip_max)

    def _intent_teacher_forcing_prob(self, step: int | None) -> at.Float[at.Array, ""]:
        if step is None:
            return jnp.asarray(self.config.intent_teacher_forcing_end_prob, dtype=jnp.float32)
        decay_steps = max(1, int(self.config.intent_teacher_forcing_decay_steps))
        progress = jnp.clip(jnp.asarray(step, dtype=jnp.float32) / decay_steps, 0.0, 1.0)
        start = jnp.asarray(self.config.intent_teacher_forcing_start_prob, dtype=jnp.float32)
        end = jnp.asarray(self.config.intent_teacher_forcing_end_prob, dtype=jnp.float32)
        return start + (end - start) * progress

    def _encode_state_for_intent(self, state):
        return self.intent_state_encoder(state)[:, None, :]

    def embed_prefix(
        self,
        obs: _model.RobotObservation,
        gating_alpha: at.Float[at.Array, "b 1 1"] | None = None,
        timestep_emb: at.Float[at.Array, "b emb"] | None = None,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        int,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []

        img_token_count = 0
        global_len = 0

        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)

            current_len = image_tokens.shape[1]
            img_token_count += current_len
            if "base" in name or "high" in name or "global" in name:
                global_len += current_len

            input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=current_len))
            ar_mask += [False] * current_len

        visual_tokens = jnp.concatenate(tokens, axis=1)
        if gating_alpha is not None and timestep_emb is not None:
            visual_tokens = self.visual_moe(
                visual_tokens=visual_tokens,
                global_token_len=global_len,
                gating_alpha=gating_alpha,
                timestep_emb=timestep_emb,
            )
            visual_tokens = self.moe_norm(visual_tokens)
        tokens = [visual_tokens]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, img_token_count

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.RobotObservation,
        noisy_actions: _model.ActionSequence,
        timestep: at.Float[at.Array, " b"],
        intent_token: at.Float[at.Array, "b 1 emb"] | None = None,
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
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]

        if intent_token is not None:
            tokens.append(intent_token)
            input_mask.append(jnp.ones((intent_token.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [False]

        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)

        if self.pi05:
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.RobotObservation,
        actions: _model.ActionSequence,
        *,
        train: bool = False,
        step: int | None = None,
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng, teacher_force_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens_raw, prefix_mask, prefix_ar_mask, img_len = self.embed_prefix(
            observation, gating_alpha=None, timestep_emb=None
        )
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions_prefix = jnp.cumsum(prefix_mask, axis=1) - 1

        (prefix_features_raw, _), _ = self.PaliGemma.llm(
            [prefix_tokens_raw, None],
            mask=prefix_attn_mask,
            positions=positions_prefix,
        )
        mixed_features = prefix_features_raw

        batch_size = observation.state.shape[0]
        prefix_seq_len = prefix_tokens_raw.shape[1]
        if mixed_features.ndim == 2:
            if mixed_features.shape[0] != batch_size * prefix_seq_len:
                mixed_features = mixed_features[: batch_size * prefix_seq_len]
            mixed_features = mixed_features.reshape(batch_size, prefix_seq_len, -1)

        visual_features = jax.lax.stop_gradient(mixed_features[:, :img_len, :])
        text_features = jax.lax.stop_gradient(mixed_features[:, img_len:, :])
        state_features = self._encode_state_for_intent(observation.state)
        text_mask = prefix_mask[:, img_len:]

        intent_logits = self.intent_predictor_head(
            text_features=text_features,
            visual_features=visual_features,
            state_features=state_features,
            text_mask=text_mask,
        )

        true_labels = observation.intent_label.astype(jnp.float32)
        intent_probs = nnx.sigmoid(intent_logits)
        pos_mask = true_labels > 0.5
        num_pos = jnp.sum(pos_mask)
        avg_pos_prob = jnp.where(num_pos > 0, jnp.sum(intent_probs * pos_mask) / num_pos, -1.0)
        neg_mask = 1.0 - pos_mask
        num_neg = jnp.sum(neg_mask)
        avg_neg_prob = jnp.where(num_neg > 0, jnp.sum(intent_probs * neg_mask) / num_neg, -1.0)

        pos_weight = 15.0
        intent_loss_elementwise = optax.sigmoid_binary_cross_entropy(intent_logits, true_labels)
        weight_vector = true_labels * pos_weight + (1.0 - true_labels)
        intent_loss = (intent_loss_elementwise * weight_vector).mean()

        alpha = nnx.sigmoid(intent_logits)[:, None, None]
        alpha = self._stabilize_alpha(alpha)
        alpha_for_moe = jax.lax.stop_gradient(alpha) if self.config.detach_alpha_for_moe else alpha

        w_intent = 1.0
        w_diffusion = 0.0
        if step is not None:
            warmup_steps = self.config.intent_warmup_steps
            is_joint_phase = step >= warmup_steps
            w_diffusion = jnp.where(is_joint_phase, 1.0, 0.0)

        time_emb = posemb_sincos(time, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)

        prefix_tokens_gated, _, _, _ = self.embed_prefix(
            observation,
            gating_alpha=alpha_for_moe,
            timestep_emb=time_emb,
        )

        gt_intent = observation.intent_label.astype(jnp.int32)
        predicted_intent = (intent_logits > 0).astype(jnp.int32)
        tf_prob = self._intent_teacher_forcing_prob(step) if train else jnp.asarray(0.0, dtype=jnp.float32)
        use_gt_mask = (
            jax.random.bernoulli(teacher_force_rng, p=tf_prob, shape=gt_intent.shape)
            if train
            else jnp.zeros_like(gt_intent, dtype=jnp.bool_)
        )
        mixed_intent = jnp.where(use_gt_mask, gt_intent, predicted_intent)
        intent_token_input = self.intent_embedding(mixed_intent)[:, None, :]

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation,
            x_t,
            time,
            intent_token=intent_token_input,
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens_gated, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )

        main_v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        delta_v_t = self.pose_refine_head(suffix_out[:, -self.action_horizon :])
        delta_scale = self._delta_blend_scale(time)
        v_t = main_v_t + delta_scale * delta_v_t

        diffusion_loss_raw = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        diffusion_loss = diffusion_loss_raw * w_diffusion
        delta_target = jax.lax.stop_gradient(u_t - main_v_t)
        delta_aux_loss_raw = jnp.mean(jnp.square(delta_v_t - delta_target), axis=-1)
        delta_aux_loss = self.config.delta_aux_loss_weight * delta_aux_loss_raw * w_diffusion

        total_loss = diffusion_loss + delta_aux_loss + w_intent * intent_loss
        metrics = {
            "total_loss": total_loss,
            "diffusion_loss": diffusion_loss_raw.mean(),
            "delta_aux_loss": delta_aux_loss_raw.mean(),
            "intent_loss": intent_loss,
            "alpha_mean": jnp.mean(alpha),
            "delta_scale_mean": jnp.mean(delta_scale),
            "w_diffusion": w_diffusion,
            "tf_prob": tf_prob,
            "tf_ratio": jnp.mean(use_gt_mask.astype(jnp.float32)),
            "intent_pred_acc": jnp.mean((predicted_intent == gt_intent).astype(jnp.float32)),
            "per_sample_diffusion_loss": diffusion_loss_raw.mean(axis=-1),
            "per_sample_intent_acc": (predicted_intent == gt_intent).astype(jnp.float32),
            "per_sample_tf_ratio": use_gt_mask.astype(jnp.float32),
            "debug_num_pos": num_pos,
            "debug_pos_prob": avg_pos_prob,
            "debug_num_neg": num_neg,
            "debug_neg_prob": avg_neg_prob,
        }
        return total_loss, metrics

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.RobotObservation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.ActionSequence:
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens_raw, prefix_mask, prefix_ar_mask, img_len = self.embed_prefix(observation, gating_alpha=None)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions_prefix = jnp.cumsum(prefix_mask, axis=1) - 1

        (prefix_features_raw, _), _ = self.PaliGemma.llm(
            [prefix_tokens_raw, None],
            mask=prefix_attn_mask,
            positions=positions_prefix,
        )
        mixed_features = prefix_features_raw
        if mixed_features.ndim == 2:
            mixed_features = mixed_features.reshape(batch_size, prefix_tokens_raw.shape[1], -1)

        visual_features = mixed_features[:, :img_len, :]
        text_features = mixed_features[:, img_len:, :]
        text_mask = prefix_mask[:, img_len:]
        state_features = self._encode_state_for_intent(observation.state)

        intent_logits = self.intent_predictor_head(
            text_features=text_features,
            visual_features=visual_features,
            state_features=state_features,
            text_mask=text_mask,
        )

        alpha = nnx.sigmoid(intent_logits)[:, None, None]
        alpha = self._stabilize_alpha(alpha)
        predicted_intent = (intent_logits > 0).astype(jnp.int32)
        predicted_intent_token = self.intent_embedding(predicted_intent)[:, None, :]

        inf_time = jnp.zeros((batch_size,))
        time_emb = posemb_sincos(inf_time, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)

        prefix_tokens_gated, _, _, _ = self.embed_prefix(
            observation,
            gating_alpha=alpha,
            timestep_emb=time_emb,
        )
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens_gated, None],
            mask=prefix_attn_mask,
            positions=positions_prefix,
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation,
                x_t,
                jnp.broadcast_to(time, batch_size),
                intent_token=predicted_intent_token,
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_step = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_step, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None

            main_v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            delta_v_t = self.pose_refine_head(suffix_out[:, -self.action_horizon :])
            step_time = jnp.broadcast_to(time, (batch_size,))
            delta_scale = self._delta_blend_scale(step_time)
            v_t = main_v_t + delta_scale * delta_v_t
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
