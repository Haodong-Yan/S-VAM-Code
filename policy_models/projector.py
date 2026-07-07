import torch
import torch.nn as nn
import math

class PerceiverResampler(nn.Module):
    def __init__(self, input_dim=768, output_dim=384, num_queries=224):
        super().__init__()
        self.num_queries = num_queries
        self.output_dim = output_dim
        
        # 可学习的 Queries: [1, 224, 384]
        self.latents = nn.Parameter(torch.randn(1, num_queries, output_dim))
        
        # 为了计算 Attention，先把 Input 投影到和 Query 一样的维度
        self.input_proj = nn.Linear(input_dim, output_dim)
        
        # Cross Attention 层
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_dim, 
            num_heads=8, 
            batch_first=True
        )
        
        # 简单的 Feed Forward
        self.ln_q = nn.LayerNorm(output_dim)
        self.ln_kv = nn.LayerNorm(output_dim)
        self.ln_out = nn.LayerNorm(output_dim)

    def forward(self, x):
        # x: [bs, 16, 256, 768]
        b, t, s, c = x.shape
        x = x.view(b, t * s, c)  # [bs, 4096, 768]
        
        # Project input to match output dim: [bs, 4096, 384]
        kv = self.input_proj(x)
        
        # 扩展 learned queries 到 batch size: [bs, 224, 384]
        q = self.latents.repeat(b, 1, 1)
        
        # Cross Attention
        # Query = latents, Key/Value = visual features
        # attn_output: [bs, 224, 384]
        q_norm = self.ln_q(q)
        kv_norm = self.ln_kv(kv)
        
        attn_out, _ = self.cross_attn(query=q_norm, key=kv_norm, value=kv_norm)
        
        # Residual connection + LayerNorm
        x = self.ln_out(q + attn_out)
        
        return x

