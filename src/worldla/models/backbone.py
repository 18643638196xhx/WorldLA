import logging
import os

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from worldla.models import interfaces as _model
from worldla.models import configuration as backbone_config
import worldla.models.language as _language
import worldla.models.vision_encoder as _vision
from worldla.utils import typing as at

logger = logging.getLogger('worldla')
LOSS_DEBUG = os.environ.get("WORLDLA_DEBUG_LOSS", "0") == "1"


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


class CompactMotionBottleneck(nnx.Module):
    """Compress dense visual tokens into a small action-visible motion interface."""

    def __init__(
        self,
        *,
        input_dim: int,
        latent_dim: int,
        num_tokens: int,
        num_horizons: int,
        rngs: nnx.Rngs,
    ):
        query_key = rngs.params()
        self.queries = nnx.Param(
            nnx.initializers.normal(stddev=0.02)(query_key, (num_tokens, latent_dim), jnp.float32)
        )
        self.input_norm = nnx.LayerNorm(input_dim, rngs=rngs)
        self.key_proj = nnx.Linear(input_dim, latent_dim, rngs=rngs)
        self.value_proj = nnx.Linear(input_dim, latent_dim, rngs=rngs)
        self.latent_norm = nnx.LayerNorm(latent_dim, rngs=rngs)
        self.ffn_in = nnx.Linear(latent_dim, 4 * latent_dim, rngs=rngs)
        self.ffn_out = nnx.Linear(4 * latent_dim, latent_dim, rngs=rngs)
        self.ffn_norm = nnx.LayerNorm(latent_dim, rngs=rngs)
        self.prefix_out = nnx.Linear(latent_dim, input_dim, rngs=rngs)
        self.motion_decoder_in = nnx.Linear(latent_dim, latent_dim, rngs=rngs)
        self.motion_decoder_out = nnx.Linear(latent_dim, num_horizons * input_dim, rngs=rngs)
        self.num_horizons = num_horizons
        self.input_dim = input_dim

    def __call__(
        self,
        visual_tokens: at.Float[at.Array, "b v emb"],
        visual_mask: at.Bool[at.Array, "b v"],
        *,
        return_prediction: bool,
    ) -> tuple[at.Float[at.Array, "b m emb"], at.Float[at.Array, "b f emb"] | None]:
        source = self.input_norm(visual_tokens)
        keys = self.key_proj(source)
        values = self.value_proj(source)
        queries = jnp.broadcast_to(self.queries.value[None], (source.shape[0], *self.queries.value.shape))
        scores = jnp.einsum("bmd,bvd->bmv", queries, keys) * (keys.shape[-1] ** -0.5)
        scores = jnp.where(visual_mask[:, None, :], scores, jnp.finfo(scores.dtype).min)
        weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(values.dtype)
        latents = self.latent_norm(queries + jnp.einsum("bmv,bvd->bmd", weights, values))
        latents = self.ffn_norm(latents + self.ffn_out(nnx.swish(self.ffn_in(latents))))

        prefix_tokens = self.prefix_out(latents).astype(visual_tokens.dtype)
        if not return_prediction:
            return prefix_tokens, None
        summary = jnp.mean(latents, axis=1)
        prediction = self.motion_decoder_out(nnx.swish(self.motion_decoder_in(summary)))
        prediction = prediction.reshape(source.shape[0], self.num_horizons, self.input_dim)
        return prefix_tokens, prediction


class WorldLABackbone(_model.RobotModelBase):
    def __init__(self, config: backbone_config.WorldLAConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.motion_num_tokens = config.motion_num_tokens
        self.motion_future_horizons = config.motion_future_horizons
        self.motion_loss_weight = config.motion_loss_weight
        self.motion_append_to_prefix = config.motion_append_to_prefix
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
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        if self.motion_num_tokens > 0:
            self.motion_bottleneck = CompactMotionBottleneck(
                input_dim=paligemma_config.width,
                latent_dim=config.motion_latent_dim,
                num_tokens=self.motion_num_tokens,
                num_horizons=max(1, len(self.motion_future_horizons)),
                rngs=rngs,
            )

        # Inference-only causal intervention switches (no retraining required).
        self.infer_swap_wrist_views = os.getenv("WORLDLA_INTERVENE_SWAP_WRIST", "0") == "1"
        self.infer_block_cross_view = os.getenv("WORLDLA_INTERVENE_BLOCK_CROSS_VIEW", "0") == "1"
        if self.infer_swap_wrist_views:
            logger.info("Enabled inference intervention: swap left/right wrist view token blocks.")
        if self.infer_block_cross_view:
            logger.info("Enabled inference intervention: block cross-view prefix self-attention.")

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def _view_bucket_id(self, name: str) -> int:
        """Map camera name to a stable bucket id used for attention blocking."""
        lname = name.lower()
        if "left_wrist" in lname:
            return 1
        if "right_wrist" in lname:
            return 2
        if "base" in lname or "global" in lname or "high" in lname:
            return 0
        return 3

    def _apply_prefix_cross_view_block(
        self,
        prefix_attn_mask: at.Bool[at.Array, "b s s"],
        prefix_view_ids: at.Int[at.Array, "b s"] | None,
    ) -> at.Bool[at.Array, "b s s"]:
        """Block image-image attention across different view buckets."""
        if prefix_view_ids is None:
            return prefix_attn_mask

        ids_q = prefix_view_ids[:, :, None]
        ids_k = prefix_view_ids[:, None, :]
        # Image tokens are marked with non-negative ids, language tokens use -1.
        both_image = jnp.logical_and(ids_q >= 0, ids_k >= 0)
        same_view = ids_q == ids_k
        allow = jnp.where(both_image, same_view, jnp.ones_like(both_image))
        return jnp.logical_and(prefix_attn_mask, allow)

    @at.typecheck
    def embed_prefix(
        self,
        obs: _model.RobotObservation,
        *,
        swap_wrist_views: bool = False,
        return_view_ids: bool = False,
        return_motion_aux: bool = False,
    ) -> tuple:
        if return_view_ids and return_motion_aux:
            raise ValueError("return_view_ids and return_motion_aux cannot be enabled together")
        input_mask = []
        ar_mask = []
        token_blocks = []
        view_id_blocks = []
        motion_aux = {}
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            image_mask = einops.repeat(
                obs.image_masks[name],
                "b -> b s",
                s=image_tokens.shape[1],
            )
            token_blocks.append((name, image_tokens, image_mask))
            if return_view_ids:
                view_id = self._view_bucket_id(name)
                view_id_blocks.append(jnp.full(image_mask.shape, view_id, dtype=jnp.int32))
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # Swap wrist camera token blocks (content + position) for inference interventions.
        if swap_wrist_views:
            left_idx = next((i for i, (name, _, _) in enumerate(token_blocks) if "left_wrist" in name.lower()), None)
            right_idx = next((i for i, (name, _, _) in enumerate(token_blocks) if "right_wrist" in name.lower()), None)
            if left_idx is not None and right_idx is not None:
                token_blocks[left_idx], token_blocks[right_idx] = token_blocks[right_idx], token_blocks[left_idx]
                if return_view_ids:
                    view_id_blocks[left_idx], view_id_blocks[right_idx] = (
                        view_id_blocks[right_idx],
                        view_id_blocks[left_idx],
                    )

        tokens = [block[1] for block in token_blocks]
        input_mask.extend([block[2] for block in token_blocks])
        if self.motion_num_tokens > 0:
            visual_tokens = jnp.concatenate(tokens, axis=1)
            visual_mask = jnp.concatenate(input_mask, axis=1)
            motion_tokens, motion_prediction = self.motion_bottleneck(
                visual_tokens,
                visual_mask,
                return_prediction=return_motion_aux,
            )
            if motion_prediction is not None:
                motion_aux["future_feature_delta_prediction"] = motion_prediction
            if self.motion_append_to_prefix:
                motion_mask = jnp.ones(motion_tokens.shape[:2], dtype=jnp.bool_)
                tokens.append(motion_tokens)
                input_mask.append(motion_mask)
                ar_mask += [False] * motion_tokens.shape[1]
                if return_view_ids:
                    view_id_blocks.append(jnp.full(motion_mask.shape, -1, dtype=jnp.int32))

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            if return_view_ids:
                view_id_blocks.append(jnp.full(obs.tokenized_prompt_mask.shape, -1, dtype=jnp.int32))
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        if return_motion_aux:
            return tokens, input_mask, ar_mask, motion_aux
        if not return_view_ids:
            return tokens, input_mask, ar_mask
        view_ids = jnp.concatenate(view_id_blocks, axis=1)
        return tokens, input_mask, ar_mask, view_ids

    @at.typecheck
    def embed_suffix(
        self, obs: _model.RobotObservation, noisy_actions: _model.ActionSequence, timestep: at.Float[at.Array, " b"]
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

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
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

    def _future_feature_delta_loss(
        self,
        observation: _model.RobotObservation,
        prediction: at.Float[at.Array, "b f emb"],
    ) -> at.Float[at.Array, " b"]:
        if observation.future_images is None or "base_0_rgb" not in observation.future_images:
            raise ValueError(
                "Motion supervision is enabled, but future_image.base_0_rgb is missing from the training batch."
            )
        future_images = observation.future_images["base_0_rgb"]
        expected_horizons = len(self.motion_future_horizons)
        if future_images.shape[1] != expected_horizons:
            raise ValueError(
                f"Expected {expected_horizons} future base images, got shape {future_images.shape}"
            )
        if prediction.shape[1] != expected_horizons:
            raise ValueError(
                f"Motion decoder produced {prediction.shape[1]} horizons, expected {expected_horizons}"
            )

        current_image = observation.images["base_0_rgb"][:, None]
        target_images = jnp.concatenate([current_image, future_images], axis=1)
        batch_size, num_frames = target_images.shape[:2]
        flat_images = target_images.reshape((-1, *target_images.shape[2:]))
        target_tokens, _ = self.PaliGemma.img(flat_images, train=False)
        target_pooled = jnp.mean(target_tokens.astype(jnp.float32), axis=1)
        target_pooled = target_pooled.reshape(batch_size, num_frames, -1)
        target_delta = target_pooled[:, 1:] - target_pooled[:, :1]
        target_rms = jnp.sqrt(jnp.mean(jnp.square(target_delta), axis=-1, keepdims=True) + 1e-6)
        target_delta = jax.lax.stop_gradient(target_delta / target_rms)

        per_horizon_loss = jnp.mean(jnp.square(prediction.astype(jnp.float32) - target_delta), axis=-1)
        if observation.future_image_masks is None:
            future_mask = jnp.ones(per_horizon_loss.shape, dtype=jnp.bool_)
        else:
            future_mask = observation.future_image_masks["base_0_rgb"]
        future_mask = future_mask.astype(per_horizon_loss.dtype)
        return jnp.sum(per_horizon_loss * future_mask, axis=-1) / jnp.maximum(jnp.sum(future_mask, axis=-1), 1.0)

    def _compute_loss_and_metrics(
        self, rng: at.KeyArrayLike, observation: _model.RobotObservation, actions: _model.ActionSequence, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        motion_target_observation = observation
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask, motion_aux = self.embed_prefix(
            observation, return_motion_aux=True
        )
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        loss = action_loss
        metrics = {"action_loss": jnp.mean(action_loss)}
        if self.motion_loss_weight > 0:
            if "future_feature_delta_prediction" not in motion_aux:
                raise ValueError("motion_loss_weight is positive but the compact motion bottleneck is disabled")
            motion_loss = self._future_feature_delta_loss(
                motion_target_observation,
                motion_aux["future_feature_delta_prediction"],
            )
            loss = loss + self.motion_loss_weight * motion_loss[:, None]
            metrics["motion_loss"] = jnp.mean(motion_loss)
            metrics["weighted_motion_loss"] = self.motion_loss_weight * metrics["motion_loss"]

        if LOSS_DEBUG and train:
            jax.debug.print(
                (
                    "[loss_debug] actions_abs_max={a:.4f} x_t_abs_max={x:.4f} "
                    "u_t_abs_max={u:.4f} v_t_abs_max={v:.4f} suffix_abs_max={s:.4f} "
                    "loss_mean={l:.4f}"
                ),
                a=jnp.max(jnp.abs(actions)),
                x=jnp.max(jnp.abs(x_t)),
                u=jnp.max(jnp.abs(u_t)),
                v=jnp.max(jnp.abs(v_t)),
                s=jnp.max(jnp.abs(suffix_out[:, -self.action_horizon :])),
                l=jnp.mean(loss),
            )

        return loss, metrics

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.RobotObservation, actions: _model.ActionSequence, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        loss, _ = self._compute_loss_and_metrics(rng, observation, actions, train=train)
        return loss

    @override
    def compute_loss_with_metrics(
        self, rng: at.KeyArrayLike, observation: _model.RobotObservation, actions: _model.ActionSequence, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        return self._compute_loss_and_metrics(rng, observation, actions, train=train)

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
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_embed = self.embed_prefix(
            observation,
            swap_wrist_views=self.infer_swap_wrist_views,
            return_view_ids=self.infer_block_cross_view,
        )
        if self.infer_block_cross_view:
            prefix_tokens, prefix_mask, prefix_ar_mask, prefix_view_ids = prefix_embed
        else:
            prefix_tokens, prefix_mask, prefix_ar_mask = prefix_embed
            prefix_view_ids = None
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        if self.infer_block_cross_view:
            prefix_attn_mask = self._apply_prefix_cross_view_block(prefix_attn_mask, prefix_view_ids)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
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
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
