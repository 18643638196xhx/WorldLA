import flax.nnx as nnx
import jax
import jax.numpy as jnp


class SoftVisualExpert(nnx.Module):
    """A lightweight residual expert that starts near-identity."""

    def __init__(self, width: int, num_heads: int, mlp_dim: int, rngs: nnx.Rngs):
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=width,
            qkv_features=width,
            decode=False,
            rngs=rngs,
        )
        self.mlp_in = nnx.Linear(width, mlp_dim, rngs=rngs)
        self.mlp_out = nnx.Linear(
            mlp_dim,
            width,
            kernel_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, x, bias_mask):
        attn_out = self.attn(inputs_q=x, inputs_k=x, inputs_v=x, mask=bias_mask, decode=False)
        x = x + attn_out
        mlp_h = nnx.gelu(self.mlp_in(x))
        x = x + self.mlp_out(mlp_h)
        return x


class SoftMoERouter(nnx.Module):
    """
    Initial multiview router:
    1. Align local tokens to global tokens.
    2. Blend two experts with a scalar intent-driven gate.
    """

    def __init__(self, config, action_expert_width: int, rngs: nnx.Rngs):
        embed_dim = config.width
        num_heads = config.num_heads
        mlp_dim = config.mlp_dim // 2

        self.expert_global = SoftVisualExpert(embed_dim, num_heads, mlp_dim, rngs)
        self.expert_local = SoftVisualExpert(embed_dim, num_heads, mlp_dim, rngs)
        self.cross_attn = nnx.MultiHeadAttention(
            num_heads=4,
            in_features=embed_dim,
            qkv_features=embed_dim,
            decode=False,
            rngs=rngs,
        )
        self.view_embeddings = nnx.Embed(2, 16, rngs=rngs)
        self.view_router = nnx.Linear(embed_dim, 2, rngs=rngs)
        self.token_router = nnx.Linear(embed_dim + 1 + 16, 2, rngs=rngs)
        self.time_gate_proj = nnx.Linear(
            action_expert_width,
            2,
            kernel_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.learnable_gain = nnx.Param(jnp.array(0.0, dtype=jnp.float32))
        self.suppress_value = -10.0

    def get_view_ids(self, batch_size, seq_len, global_token_len):
        ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
        ids = ids.at[:, global_token_len:].set(1)
        return ids

    def __call__(self, visual_tokens, global_token_len, gating_alpha, timestep_emb=None):
        shortcut = visual_tokens
        batch_size, seq_len, _ = visual_tokens.shape
        eps = 1e-6

        x_g = visual_tokens[:, :global_token_len]
        x_l = visual_tokens[:, global_token_len:]

        x_l_aligned = self.cross_attn(inputs_q=x_l, inputs_k=x_g, inputs_v=x_g, decode=False)
        x_l = x_l + x_l_aligned
        aligned_visual_tokens = jnp.concatenate([x_g, x_l], axis=1)

        view_mean = jnp.mean(aligned_visual_tokens, axis=1)
        view_weights = jax.nn.softmax(self.view_router(view_mean), axis=-1)[:, None, :]

        view_ids = self.get_view_ids(batch_size, seq_len, global_token_len)
        view_feats = self.view_embeddings(view_ids)
        alpha_expanded = jnp.broadcast_to(gating_alpha, (batch_size, seq_len, 1))
        router_input = jnp.concatenate([aligned_visual_tokens, alpha_expanded, view_feats], axis=-1)
        token_weights = jax.nn.softmax(self.token_router(router_input), axis=-1)

        combined_logits = jnp.log(view_weights + eps) + jnp.log(token_weights + eps)
        if timestep_emb is not None:
            time_bias = self.time_gate_proj(timestep_emb)[:, None, :]
            combined_logits = combined_logits + time_bias

        final_weights = jax.nn.softmax(combined_logits, axis=-1)
        w_g, w_l = final_weights[..., 0:1], final_weights[..., 1:2]

        bias_g = jnp.zeros((batch_size, 1, seq_len, seq_len)).at[:, :, :, global_token_len:].set(self.suppress_value)
        bias_l = jnp.zeros((batch_size, 1, seq_len, seq_len)).at[:, :, :, :global_token_len].set(self.suppress_value)

        y_g = self.expert_global(aligned_visual_tokens, bias_g)
        y_l = self.expert_local(aligned_visual_tokens, bias_l)
        fused_output = y_g * w_g + y_l * w_l

        gain = jnp.asarray(self.learnable_gain)
        dynamic_scale = 0.1 + jax.nn.softplus(gain)
        return shortcut + dynamic_scale * jnp.tanh(gain) * (fused_output - shortcut)
