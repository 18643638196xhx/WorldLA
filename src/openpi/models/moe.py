import flax.nnx as nnx
import jax
import jax.numpy as jnp

class SoftVisualExpert(nnx.Module):
    """针对 pi0.5 优化的无损专家：移除 LayerNorm，采用输出层零初始化。"""
    def __init__(self, width: int, num_heads: int, mlp_dim: int, rngs: nnx.Rngs):
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads, in_features=width, qkv_features=width,
            decode=False, rngs=rngs
        )
        self.mlp_in = nnx.Linear(width, mlp_dim, rngs=rngs)
        # 💡 零初始化：确保专家在训练初期“隐身”，保护预训练流形
        self.mlp_out = nnx.Linear(
            mlp_dim, width, 
            kernel_init=nnx.initializers.zeros, rngs=rngs
        )

    def __call__(self, x, bias_mask):
        # 纯残差结构：剔除 LayerNorm 以对齐 AdaRMSNorm
        attn_out = self.attn(inputs_q=x, inputs_k=x, inputs_v=x, mask=bias_mask, decode=False)
        x = x + attn_out
        
        mlp_h = nnx.gelu(self.mlp_in(x))
        x = x + self.mlp_out(mlp_h)
        return x

class SoftMoERouter(nnx.Module):
    """
    进阶版多视角 MoE 路由器：
    1. View-ID 注入：识别 Token 来源视角。
    2. 跨视角注意力对齐 (CVA)：让局部视角参考全局位置。
    3. 层次化路由：视角级主控 + Token 级微调。

    Note:
    The intended semantic split is two routing modes: global-oriented and local-oriented.
    """
    def __init__(
        self,
        config,
        action_expert_width: int,
        num_experts: int = 2,
        alpha_prior_scale: float = 1.0,
        view_prior_scale: float = 0.0,
        global_local_expert_bias: float = -2.0,
        alpha_only_moe_routing: bool = True,
        view_router_logit_scale: float = 0.0,
        token_router_logit_scale: float = 0.0,
        rngs: nnx.Rngs = None,
    ):
        embed_dim = config.width
        num_heads = config.num_heads
        mlp_dim = config.mlp_dim // 2
        self.num_experts = num_experts
        self.alpha_prior_scale = alpha_prior_scale
        self.view_prior_scale = view_prior_scale
        self.global_local_expert_bias = global_local_expert_bias
        self.alpha_only_moe_routing = alpha_only_moe_routing
        self.view_router_logit_scale = view_router_logit_scale
        self.token_router_logit_scale = token_router_logit_scale

        # 1. 专家定义
        self.experts = nnx.Dict(
            **{
                f"expert_{i}": SoftVisualExpert(embed_dim, num_heads, mlp_dim, rngs)
                for i in range(self.num_experts)
            }
        )
        
        # 2. 跨视角交互层 (Cross-View Alignment)
        # 让局部 Token 作为 Query 去查询全局 Token
        self.cross_attn = nnx.MultiHeadAttention(
            num_heads=4, in_features=embed_dim, qkv_features=embed_dim,
            decode=False, rngs=rngs
        )

        # 3. 视角身份与路由
        self.view_embeddings = nnx.Embed(2, 16, rngs=rngs) 
        
        # 💡 层次化路由设计
        # View-level: 决定视角的整体信赖度
        self.view_router = nnx.Linear(embed_dim, self.num_experts, rngs=rngs)
        # Token-level: 决定局部像素的细粒度偏置
        self.token_router = nnx.Linear(embed_dim + self.num_experts + 16, self.num_experts, rngs=rngs)
        
        # 时间步嵌入适配
        self.time_gate_proj = nnx.Linear(
            action_expert_width, self.num_experts,
            kernel_init=nnx.initializers.zeros, rngs=rngs
        )
        
        # 💡 可学习的缩放因子：取代硬编码的 0.1，允许模型自主释放信号强度
        self.learnable_gain = nnx.Param(jnp.array(0.0, dtype=jnp.float32))
        self.suppress_value = global_local_expert_bias

    def get_view_ids(self, B, L, global_token_len):
        ids = jnp.zeros((B, L), dtype=jnp.int32)
        ids = ids.at[:, global_token_len:].set(1)
        return ids

    def create_global_local_biases(self, B, L, global_token_len, dtype):
        """Create soft attention biases that give expert_0/1 global/local semantics."""
        full_bias = jnp.zeros((B, 1, L, L), dtype=dtype)
        global_bias = full_bias.at[:, :, :, global_token_len:].set(self.suppress_value)
        local_bias = full_bias.at[:, :, :, :global_token_len].set(self.suppress_value)
        return full_bias, global_bias, local_bias

    def __call__(
        self,
        visual_tokens,
        global_token_len,
        gating_alpha,
        timestep_emb=None,
        intent_view_prior=None,
        return_stats: bool = False,
    ):
        shortcut = visual_tokens
        B, L, D = visual_tokens.shape
        eps = 1e-6

        # --- 步骤 1：跨视角注意力校准 (CVA) ---
        # 提取两个视角的特征
        x_g = visual_tokens[:, :global_token_len]
        x_l = visual_tokens[:, global_token_len:]
        
        # 让局部视角 (Query) 检索全局视角 (Key/Value) 以对齐空间参考
        x_l_aligned = self.cross_attn(inputs_q=x_l, inputs_k=x_g, inputs_v=x_g, decode=False)
        x_l = x_l + x_l_aligned
        
        # 重新拼接校准后的特征
        aligned_visual_tokens = jnp.concatenate([x_g, x_l], axis=1)

        # --- 步骤 2：Alpha-only expert routing ---
        # Current mechanism ablation: alpha is the direct and only routing control.
        view_ids = self.get_view_ids(B, L, global_token_len)
        if self.alpha_only_moe_routing:
            final_weights = jnp.broadcast_to(gating_alpha[:, None, :], (B, L, self.num_experts))
        else:
            # Optional legacy path for later ablations.
            view_mean = jnp.mean(aligned_visual_tokens, axis=1) # (B, D)
            view_weights = jax.nn.softmax(self.view_router(view_mean), axis=-1)[:, None, :] # (B, 1, K)
            view_feats = self.view_embeddings(view_ids)
            alpha_expanded = jnp.broadcast_to(gating_alpha[:, None, :], (B, L, self.num_experts))
            router_input = jnp.concatenate([aligned_visual_tokens, alpha_expanded, view_feats], axis=-1)
            token_weights = jax.nn.softmax(self.token_router(router_input), axis=-1) # (B, L, K)
            alpha_prior = jnp.log(gating_alpha[:, None, :] + eps)
            combined_logits = (
                self.view_router_logit_scale * jnp.log(view_weights + eps)
                + self.token_router_logit_scale * jnp.log(token_weights + eps)
                + self.alpha_prior_scale * alpha_prior
            )
            if timestep_emb is not None:
                time_bias = self.time_gate_proj(timestep_emb)[:, None, :] # (B, 1, K)
                combined_logits = combined_logits + time_bias
            final_weights = jax.nn.softmax(combined_logits, axis=-1)
        
        # --- 步骤 3：并行专家处理 (保持 0 均值环境) ---
        # Stage-1 design: expert_0 is global-oriented, expert_1 is local-oriented.
        # The bias is intentionally soft, so the other view is discouraged rather than removed.
        full_bias, global_bias, local_bias = self.create_global_local_biases(
            B, L, global_token_len, aligned_visual_tokens.dtype
        )
        expert_outputs = []
        for i in range(self.num_experts):
            if i == 0:
                expert_bias = global_bias
            elif i == 1:
                expert_bias = local_bias
            else:
                expert_bias = full_bias
            expert_outputs.append(self.experts[f"expert_{i}"](aligned_visual_tokens, expert_bias))
        fused_stack = jnp.stack(expert_outputs, axis=2) # (B, L, K, D)

        # --- 步骤 4：专家融合输出 ---
        fused_output = jnp.sum(fused_stack * final_weights[..., None], axis=2)

        moe_delta = fused_output - shortcut
        # Current ablation: do not inject MoE as an outer residual patch.
        # The visual prefix is replaced by the alpha-routed expert mixture.
        output = fused_output
        if not return_stats:
            return output

        alpha_expanded = jnp.broadcast_to(gating_alpha[:, None, :], final_weights.shape)
        gain = jnp.asarray(self.learnable_gain)
        stats = {
            "expert_usage": jnp.mean(final_weights, axis=1),
            "router_confidence": jnp.max(final_weights, axis=-1).mean(axis=1),
            "alpha_routing_error": jnp.mean(jnp.abs(final_weights - alpha_expanded), axis=(1, 2)),
            "moe_delta_norm": jnp.mean(jnp.linalg.norm(moe_delta, axis=-1), axis=1),
            "moe_patch_norm": jnp.mean(jnp.linalg.norm(moe_delta, axis=-1), axis=1),
            "moe_gain_tanh": jnp.full((B,), jnp.tanh(gain)),
            "moe_dynamic_scale": jnp.ones((B,), dtype=aligned_visual_tokens.dtype),
            "view_patch_scale_global": jnp.ones((B,), dtype=aligned_visual_tokens.dtype),
            "view_patch_scale_local": jnp.ones((B,), dtype=aligned_visual_tokens.dtype),
            "alpha_only_moe_routing": jnp.full((B,), float(self.alpha_only_moe_routing)),
            "view_router_logit_scale": jnp.full((B,), self.view_router_logit_scale),
            "token_router_logit_scale": jnp.full((B,), self.token_router_logit_scale),
            "global_token_global_expert_weight": jnp.mean(final_weights[:, :global_token_len, 0], axis=1),
            "global_token_local_expert_weight": jnp.mean(final_weights[:, :global_token_len, 1], axis=1),
            "local_token_global_expert_weight": jnp.mean(final_weights[:, global_token_len:, 0], axis=1),
            "local_token_local_expert_weight": jnp.mean(final_weights[:, global_token_len:, 1], axis=1),
            "cross_expert_leakage": 0.5 * (
                jnp.mean(final_weights[:, :global_token_len, 1], axis=1)
                + jnp.mean(final_weights[:, global_token_len:, 0], axis=1)
            ),
        }
        return output, stats
