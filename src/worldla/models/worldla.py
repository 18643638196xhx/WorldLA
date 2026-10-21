from __future__ import annotations

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp


MESSAGE_STAGE_MODES = ("normal", "stage1_repeat", "stage_shuffled")
STAGE_SHUFFLE_PERMUTATION = (2, 0, 3, 1)


def message_source_stage_ids(horizon: int, num_stages: int, mode: str):
    """Map each action token to the stage message used by an inference ablation."""
    if mode not in MESSAGE_STAGE_MODES:
        raise ValueError(f"Unknown message stage mode {mode!r}; expected {MESSAGE_STAGE_MODES}")
    position_stage = (
        jnp.arange(horizon, dtype=jnp.int32) * int(num_stages)
    ) // int(horizon)
    if mode == "normal":
        return position_stage
    if mode == "stage1_repeat":
        return jnp.zeros_like(position_stage)
    if int(num_stages) != len(STAGE_SHUFFLE_PERMUTATION):
        raise ValueError(
            "stage_shuffled is defined for the trained four-stage Router, got "
            f"num_stages={num_stages}"
        )
    permutation = jnp.asarray(STAGE_SHUFFLE_PERMUTATION, dtype=jnp.int32)
    return permutation[position_stage]


class ContextAdaptiveStageRouter(nnx.Module):
    """Predict one visual-token route for each relative action-chunk stage."""

    def __init__(
        self,
        *,
        prefix_dim: int,
        hidden_dim: int,
        num_route_tokens: int,
        num_stages: int,
        position_only: bool,
        rngs: nnx.Rngs,
    ):
        self.source_norm = nnx.LayerNorm(prefix_dim, rngs=rngs)
        self.visual_key = nnx.Linear(prefix_dim, hidden_dim, rngs=rngs)
        self.context_in = nnx.Linear(2 * prefix_dim, hidden_dim, rngs=rngs)
        self.query_out = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        key_stage, _ = jax.random.split(rngs.params())
        self.stage_embedding = nnx.Param(
            0.02 * jax.random.normal(key_stage, (num_stages, hidden_dim))
        )
        # This explicit prior makes position_only a real observation-independent
        # control, rather than a visual-content router with its context removed.
        self.stage_position_logits = nnx.Param(
            jnp.zeros((num_stages, num_route_tokens), dtype=jnp.float32)
        )
        self.log_temperature = nnx.Param(jnp.asarray(0.0, dtype=jnp.float32))
        self.num_route_tokens = int(num_route_tokens)
        self.num_stages = int(num_stages)
        self.position_only = bool(position_only)

    def __call__(self, prefix_tokens, prefix_mask):
        if prefix_tokens.shape[1] < self.num_route_tokens:
            raise ValueError(
                f"Prefix has {prefix_tokens.shape[1]} tokens; expected at least "
                f"{self.num_route_tokens} visual tokens"
            )
        batch = prefix_tokens.shape[0]
        prior = jnp.broadcast_to(
            self.stage_position_logits.value[None],
            (batch, self.num_stages, self.num_route_tokens),
        )
        visual_mask = prefix_mask[:, : self.num_route_tokens]
        if self.position_only:
            logits = prior
        else:
            # Router supervision must not update the pi0.5 backbone.  Context is
            # split so the many visual tokens cannot drown out language/state.
            source = self.source_norm(jax.lax.stop_gradient(prefix_tokens))
            visual = source[:, : self.num_route_tokens]
            visual_weight = visual_mask.astype(source.dtype)
            visual_context = jnp.sum(
                visual * visual_weight[..., None], axis=1
            ) / jnp.maximum(jnp.sum(visual_weight, axis=1, keepdims=True), 1.0)

            other = source[:, self.num_route_tokens :]
            other_mask = prefix_mask[:, self.num_route_tokens :].astype(source.dtype)
            other_context = jnp.sum(
                other * other_mask[..., None], axis=1
            ) / jnp.maximum(jnp.sum(other_mask, axis=1, keepdims=True), 1.0)
            context = jnp.concatenate([visual_context, other_context], axis=-1)
            context = self.context_in(context)
            stage_context = nnx.gelu(
                context[:, None, :] + self.stage_embedding.value[None, :, :]
            )
            query = self.query_out(stage_context)
            keys = self.visual_key(visual)
            temperature = jnp.exp(jnp.clip(self.log_temperature.value, -2.0, 2.0))
            content_logits = (
                jnp.einsum("bvd,bsd->bsv", keys, query)
                * (keys.shape[-1] ** -0.5)
                / temperature
            )
            logits = prior + content_logits
        logits = jnp.where(
            visual_mask[:, None, :], logits, jnp.finfo(logits.dtype).min
        )
        return jax.nn.softmax(logits.astype(jnp.float32), axis=-1)


class StageMessageInjector(nnx.Module):
    """Read four pi0.5 visual messages and inject them into action suffix tokens."""

    def __init__(
        self,
        *,
        prefix_dim: int,
        suffix_dim: int,
        bottleneck_dim: int,
        num_stages: int,
        rngs: nnx.Rngs,
    ):
        self.norm = nnx.LayerNorm(prefix_dim, rngs=rngs)
        self.down = nnx.Linear(prefix_dim, bottleneck_dim, use_bias=False, rngs=rngs)
        self.up = nnx.Linear(
            bottleneck_dim,
            suffix_dim,
            use_bias=False,
            kernel_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        self.gate_logit = nnx.Param(
            jnp.full((num_stages,), -2.0, dtype=jnp.float32)
        )
        self.num_stages = int(num_stages)

    def __call__(
        self,
        suffix_tokens,
        visual_tokens,
        route,
        *,
        stage_mode="normal",
        window_stage_weights=None,
        window_size=0,
        window_strength=None,
    ):
        if route.shape[1] != self.num_stages:
            raise ValueError(f"Expected {self.num_stages} route stages, got {route.shape}")
        if route.shape[-1] != visual_tokens.shape[1]:
            raise ValueError(f"Route/visual mismatch: {route.shape} vs {visual_tokens.shape}")
        visual_value = self.up(nnx.gelu(self.down(self.norm(visual_tokens))))
        stage_message = jnp.einsum(
            "bsv,bvd->bsd", route.astype(visual_value.dtype), visual_value
        )
        horizon = suffix_tokens.shape[1]
        # Select the complete stage operation: both its routed message and gate.
        # Repeating stage 1 therefore does not leak later-stage gate strengths.
        source_stage_ids = message_source_stage_ids(
            horizon, self.num_stages, stage_mode
        )
        gate = jax.nn.sigmoid(self.gate_logit.value).astype(suffix_tokens.dtype)
        message_by_token = stage_message[:, source_stage_ids, :]
        gate_by_token = gate[source_stage_ids][None, :, None]
        native_message = gate_by_token * message_by_token.astype(suffix_tokens.dtype)
        if window_stage_weights is None:
            return suffix_tokens + native_message
        gated_message = gate[None, :, None] * stage_message.astype(suffix_tokens.dtype)
        message_by_token = native_message
        weights = window_stage_weights.astype(suffix_tokens.dtype)
        weights = weights / jnp.maximum(weights.sum(-1, keepdims=True), 1e-8)
        mixed = jnp.einsum("bs,bsd->bd", weights, gated_message)
        use_window = (jnp.arange(horizon) < int(window_size))[None, :, None]
        if window_strength is None:
            strength = jnp.ones((suffix_tokens.shape[0], 1, 1), suffix_tokens.dtype)
        else:
            strength = window_strength.astype(suffix_tokens.dtype).reshape(-1, 1, 1)
        replacement = (1.0 - strength) * message_by_token + strength * mixed[:, None, :]
        message_by_token = jnp.where(use_window, replacement, message_by_token)
        return suffix_tokens + message_by_token


class WorldLAModel(nnx.Module):
    """pi0.5 whose action suffix reads stage-conditioned visual messages."""

    def __init__(
        self,
        base,
        *,
        prefix_dim: int,
        suffix_dim: int,
        num_route_tokens: int,
        num_stages: int,
        router_hidden_dim: int,
        adapter_dim: int,
        router_active: bool,
        message_enabled: bool,
        position_only: bool,
        message_stage_mode: str = "normal",
        global_feature_dim: int = 0,
        rngs: nnx.Rngs,
    ):
        self.base = base
        router_rng, adapter_rng = jax.random.split(rngs.params())
        self.stage_relation_router = ContextAdaptiveStageRouter(
            prefix_dim=prefix_dim,
            hidden_dim=router_hidden_dim,
            num_route_tokens=num_route_tokens,
            num_stages=num_stages,
            position_only=position_only,
            rngs=nnx.Rngs(router_rng),
        )
        self.stage_message_adapter = StageMessageInjector(
            prefix_dim=prefix_dim,
            suffix_dim=suffix_dim,
            bottleneck_dim=adapter_dim,
            num_stages=num_stages,
            rngs=nnx.Rngs(adapter_rng),
        )
        self.num_route_tokens = int(num_route_tokens)
        self.num_stages = int(num_stages)
        self.global_feature_dim = int(global_feature_dim)
        if self.global_feature_dim:
            # Training-only R2 head. It matches a single pooled video-expert
            # representation and is not saved in router/deployment checkpoints.
            self.global_feature_head = nnx.Sequential(
                nnx.LayerNorm(prefix_dim, rngs=rngs),
                nnx.Linear(prefix_dim, adapter_dim, rngs=rngs),
                nnx.gelu,
                nnx.Linear(adapter_dim, self.global_feature_dim, rngs=rngs),
            )
        self.router_active = bool(router_active)
        self.message_enabled = bool(message_enabled)
        if message_stage_mode not in MESSAGE_STAGE_MODES:
            raise ValueError(
                f"Unknown message_stage_mode={message_stage_mode!r}; "
                f"expected {MESSAGE_STAGE_MODES}"
            )
        self.message_stage_mode = str(message_stage_mode)

    def _prefix_and_route(self, observation, route_override=None):
        tokens, mask, ar = self.base.embed_prefix(observation)
        if route_override is not None:
            route = route_override.astype(jnp.float32)
            expected = (tokens.shape[0], self.num_stages, self.num_route_tokens)
            if route.shape != expected:
                raise ValueError(f"Route override {route.shape} != {expected}")
        elif self.router_active:
            route = self.stage_relation_router(tokens, mask)
        else:
            route = jnp.full(
                (tokens.shape[0], self.num_stages, self.num_route_tokens),
                1.0 / self.num_route_tokens,
                dtype=jnp.float32,
            )
        return tokens, mask, ar, route

    def _inject_message(
        self,
        suffix_tokens,
        prefix_tokens,
        route,
        *,
        window_stage_weights=None,
        window_size=0,
        window_strength=None,
    ):
        if not self.message_enabled:
            return suffix_tokens
        return self.stage_message_adapter(
            suffix_tokens,
            prefix_tokens[:, : self.num_route_tokens],
            route,
            stage_mode=self.message_stage_mode,
            window_stage_weights=window_stage_weights,
            window_size=window_size,
            window_strength=window_strength,
        )

    def compute_route_loss(self, observation, route_target):
        from worldla.models import interfaces as model_api

        observation = model_api.preprocess_observation(None, observation, train=False)
        _, _, _, prediction = self._prefix_and_route(observation)
        target = route_target.astype(jnp.float32)
        target = target / jnp.maximum(jnp.sum(target, axis=-1, keepdims=True), 1e-8)
        if target.shape != prediction.shape:
            raise ValueError(f"Route target {target.shape} != prediction {prediction.shape}")
        eps = 1e-8
        kl_per_stage = jnp.sum(
            target * (jnp.log(target + eps) - jnp.log(prediction + eps)), axis=-1
        )
        cosine = jnp.sum(target * prediction, axis=-1) / jnp.maximum(
            jnp.linalg.vector_norm(target, axis=-1)
            * jnp.linalg.vector_norm(prediction, axis=-1),
            eps,
        )
        entropy = -jnp.sum(prediction * jnp.log(prediction + eps), axis=-1)
        centered = prediction - jnp.mean(prediction, axis=1, keepdims=True)
        return jnp.mean(kl_per_stage), {
            "loss": jnp.mean(kl_per_stage),
            "route_kl": jnp.mean(kl_per_stage),
            "route_cosine": jnp.mean(cosine),
            "route_entropy": jnp.mean(entropy),
            "route_peak": jnp.mean(jnp.max(prediction, axis=-1)),
            "predicted_stage_variation": 0.5 * jnp.mean(jnp.sum(jnp.abs(centered), axis=-1)),
        }

    def compute_global_feature_loss(self, observation, feature_target):
        """R2: align one pooled hidden vector, with no spatial/stage labels."""
        from worldla.models import interfaces as model_api

        if not self.global_feature_dim:
            raise ValueError("Global-feature distillation head is not enabled")
        observation = model_api.preprocess_observation(None, observation, train=False)
        tokens, _, _, route = self._prefix_and_route(observation)
        visual = jax.lax.stop_gradient(tokens[:, : self.num_route_tokens])
        # All stages contribute to ONE global vector; no stage-specific target
        # and no teacher action query/K/readout enters this objective.
        pooled = jnp.einsum("bsv,bvd->bsd", route.astype(visual.dtype), visual).mean(axis=1)
        prediction = self.global_feature_head(pooled).astype(jnp.float32)
        target = feature_target.astype(jnp.float32)
        if prediction.shape != target.shape:
            raise ValueError(f"Global feature mismatch: {prediction.shape} != {target.shape}")
        prediction = prediction / jnp.maximum(jnp.linalg.norm(prediction, axis=-1, keepdims=True), 1e-8)
        target = target / jnp.maximum(jnp.linalg.norm(target, axis=-1, keepdims=True), 1e-8)
        cosine = jnp.sum(prediction * target, axis=-1)
        loss = jnp.mean(1.0 - cosine)
        return loss, {"loss": loss, "feature_loss": loss, "feature_cosine": jnp.mean(cosine)}

    def compute_action_loss(
        self,
        rng,
        observation,
        actions,
        route_override=None,
        sample_weight=None,
        window_stage_weights=None,
        window_size=0,
        window_strength=None,
        loss_action_steps=None,
    ):
        from worldla.models import interfaces as model_api
        from worldla.models.backbone import make_attn_mask

        noise_rng, time_rng = jax.random.split(rng)
        observation = model_api.preprocess_observation(None, observation, train=False)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        expanded_time = time[..., None, None]
        x_t = expanded_time * noise + (1 - expanded_time) * actions
        target_velocity = noise - actions

        prefix_tokens, prefix_mask, prefix_ar, route = self._prefix_and_route(
            observation, route_override=route_override
        )
        suffix_tokens, suffix_mask, suffix_ar, adarms = self.base.embed_suffix(
            observation, x_t, time
        )
        suffix_tokens = self._inject_message(
            suffix_tokens,
            prefix_tokens,
            route,
            window_stage_weights=window_stage_weights,
            window_size=window_size,
            window_strength=window_strength,
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar, suffix_ar], axis=0)
        attention_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_output), _ = self.base.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attention_mask,
            positions=positions,
            adarms_cond=[None, adarms],
        )
        velocity = self.base.action_out_proj(suffix_output[:, -self.base.action_horizon :])
        error = jnp.square(velocity - target_velocity)
        if loss_action_steps is not None:
            error = error[:, : int(loss_action_steps)]
        per_sample = jnp.mean(error, axis=(-2, -1))
        if sample_weight is None:
            action_loss = jnp.mean(per_sample)
        else:
            sample_weight = sample_weight.astype(per_sample.dtype)
            action_loss = jnp.sum(per_sample * sample_weight) / jnp.maximum(
                jnp.sum(sample_weight), 1
            )
        gate = jax.nn.sigmoid(self.stage_message_adapter.gate_logit.value)
        return action_loss, {
            "loss": action_loss,
            "action_loss": action_loss,
            "adapter_gate": jnp.mean(gate),
            "adapter_gate_min": jnp.min(gate),
            "adapter_gate_max": jnp.max(gate),
            "route_entropy": -jnp.mean(jnp.sum(route * jnp.log(route + 1e-8), axis=-1)),
        }

    def predict_route(self, observation):
        """Return the native frozen Router output for diagnostic interventions."""
        from worldla.models import interfaces as model_api

        observation = model_api.preprocess_observation(None, observation, train=False)
        return self._prefix_and_route(observation)[-1]

    def sample_actions(
        self,
        rng,
        observation,
        *,
        num_steps: int = 10,
        noise=None,
        route_override=None,
        window_stage_weights=None,
        window_size=0,
        window_strength=None,
    ):
        from worldla.models import interfaces as model_api
        from worldla.models.backbone import make_attn_mask

        observation = model_api.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(
                rng, (batch_size, self.base.action_horizon, self.base.action_dim)
            )
        prefix_tokens, prefix_mask, prefix_ar, route = self._prefix_and_route(
            observation, route_override=route_override
        )
        visual_tokens = prefix_tokens[:, : self.num_route_tokens]
        prefix_attention = make_attn_mask(prefix_mask, prefix_ar)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, cache = self.base.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attention, positions=positions
        )
        dt = jnp.asarray(-1.0 / num_steps, dtype=jnp.float32)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar, adarms = self.base.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, (batch_size,))
            )
            if self.message_enabled:
                suffix_tokens = self.stage_message_adapter(
                    suffix_tokens,
                    visual_tokens,
                    route,
                    stage_mode=self.message_stage_mode,
                    window_stage_weights=window_stage_weights,
                    window_size=window_size,
                    window_strength=window_strength,
                )
            suffix_attention = make_attn_mask(suffix_mask, suffix_ar)
            prefix_to_suffix = einops.repeat(
                prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            full_attention = jnp.concatenate([prefix_to_suffix, suffix_attention], axis=-1)
            suffix_positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1)
                - 1
            )
            (_, suffix_output), _ = self.base.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attention,
                positions=suffix_positions,
                kv_cache=cache,
                adarms_cond=[None, adarms],
            )
            velocity = self.base.action_out_proj(suffix_output[:, -self.base.action_horizon :])
            return x_t + dt * velocity, time + dt

        result, _ = jax.lax.while_loop(
            lambda carry: carry[1] >= -dt / 2,
            step,
            (noise, jnp.asarray(1.0)),
        )
        return result
