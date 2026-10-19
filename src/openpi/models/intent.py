# 文件: openpi/models/intent.py

import flax.nnx as nnx
import jax.numpy as jnp
from openpi.shared import array_typing as at

class IntentHead(nnx.Module):
    
    def __init__(self, embed_dim: int, output_steps: int = 1, num_heads: int = 4, rngs: nnx.Rngs = None):
        
        # 🔥【最终修复】使用 nnx.MultiHeadAttention 的正确初始化方式
        # 它只需要知道输入的特征维度即可，Q, K, V 的具体维度在调用时推断
        self.cross_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim, # Q, K, V 的输入维度都是 embed_dim
            rngs=rngs
        )
        
        self.norm = nnx.LayerNorm(embed_dim, rngs=rngs)
        
        # 输出层
        self.mlp = nnx.Sequential(
            nnx.Linear(embed_dim, embed_dim * 2, rngs=rngs),
            nnx.gelu,
            nnx.Linear(embed_dim * 2, output_steps, rngs=rngs)
        )

    def __call__(self, text_features, visual_features, state_features, text_mask=None):
        """
        Args:
            text_features: (Batch, Text_Len, Dim) -> Query (保留序列)
            visual_features: (Batch, Img_Len, Dim) -> Key/Value
            state_features: (Batch, State_Len, Dim) -> Key/Value
            text_mask: (Batch, Text_Len) -> Boolean mask, True for valid tokens
        """
        
        # 1. 构建 Query: 直接使用文本序列
        # Shape: (Batch, Text_Len, Dim)
        query = text_features
        
        # 2. 构建 Key/Value: 视觉 + 状态
        # Shape: (Batch, Img_Len + State_Len, Dim)
        kv = jnp.concatenate([visual_features, state_features], axis=1)
        
        # 3. 执行交叉注意力
        # 每个文本 token 都会去“查询”视觉和状态信息
        # Output: (Batch, Text_Len, Dim)
        # 🔥【关键修改 2】拆分 kv 并使用正确的参数名
        # 旧代码: attn_out = self.cross_attn(inputs_q=query, inputs_kv=kv)
        attn_out = self.cross_attn(
            inputs_q=query, 
            inputs_k=kv, 
            inputs_v=kv,
            decode=False   # ✅ 显式关闭自回归模式
        )

        
        # 4. 残差 + Norm
        x = self.norm(query + attn_out)
        
        # 5. 序列聚合 (Aggregation) -> Masked Mean Pooling
        # 我们需要把 (Batch, Text_Len, Dim) 变成 (Batch, Dim)
        # 必须使用 mask，否则 padding 部分的噪音会影响结果
        
        if text_mask is not None:
            # 扩展 mask 维度以匹配特征: (Batch, Text_Len) -> (Batch, Text_Len, 1)
            mask_expanded = text_mask[:, :, None].astype(jnp.float32)
            
            # 求和 (只加有效 token)
            sum_features = jnp.sum(x * mask_expanded, axis=1)
            
            # 有效 token 数量 (防止除以 0)
            valid_token_count = jnp.sum(mask_expanded, axis=1)
            valid_token_count = jnp.maximum(valid_token_count, 1e-9)
            
            pooled_x = sum_features / valid_token_count
        else:
            # 如果没有 mask (不推荐)，直接平均
            pooled_x = jnp.mean(x, axis=1)
            
        # 6. 最终预测
        logits = self.mlp(pooled_x)
        # Keep an explicit horizon axis for both Full (H=action_horizon) and
        # Reactive (H=1). Downstream routing consistently expects [batch, H].
        return logits
