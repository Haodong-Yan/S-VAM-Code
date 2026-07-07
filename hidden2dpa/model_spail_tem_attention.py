"""模型定义：将隐藏状态映射到 DINO patch 特征。"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpatialTemporalAttentionBlock(nn.Module):
    """先做空间注意力，再做时间注意力的轻量 Transformer block。"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.spatial_norm1 = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.spatial_dropout = nn.Dropout(dropout)

        self.spatial_norm2 = nn.LayerNorm(dim)
        self.spatial_mlp = self._build_mlp(dim, mlp_ratio, dropout)

        self.temporal_norm1 = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_dropout = nn.Dropout(dropout)

        self.temporal_norm2 = nn.LayerNorm(dim)
        self.temporal_mlp = self._build_mlp(dim, mlp_ratio, dropout)

    @staticmethod
    def _build_mlp(dim: int, mlp_ratio: float, dropout: float) -> nn.Sequential:
        hidden_mlp = int(dim * mlp_ratio)
        return nn.Sequential(
            nn.Linear(dim, hidden_mlp),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_mlp, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W, dim)
        if x.dim() != 5:
            raise ValueError(f"期望输入维度为 5 (B, T, H, W, dim)，得到 {x.shape}")

        B, T, H, W, D = x.shape

        # 1) 空间注意力：对每个时间步，在 H*W token 上做 self-attention。
        residual = x
        spatial_tokens = self.spatial_norm1(x).reshape(B * T, H * W, D)
        spatial_attn_out, _ = self.spatial_attn(
            spatial_tokens, spatial_tokens, spatial_tokens
        )
        spatial_attn_out = spatial_attn_out.reshape(B, T, H, W, D)
        x = residual + self.spatial_dropout(spatial_attn_out)

        residual = x
        x = residual + self.spatial_mlp(self.spatial_norm2(x))

        # 2) 时间注意力：对每个空间位置，在 T token 上做 self-attention。
        residual = x
        temporal_tokens = self.temporal_norm1(x).permute(0, 2, 3, 1, 4).reshape(
            B * H * W, T, D
        )
        temporal_attn_out, _ = self.temporal_attn(
            temporal_tokens, temporal_tokens, temporal_tokens
        )
        temporal_attn_out = (
            temporal_attn_out.reshape(B, H, W, T, D).permute(0, 3, 1, 2, 4).contiguous()
        )
        x = residual + self.temporal_dropout(temporal_attn_out)

        residual = x
        x = residual + self.temporal_mlp(self.temporal_norm2(x))
        return x


class HiddenToDA3Model(nn.Module):
    """将 (B, C_in, T, H, W) 的隐藏状态映射到 (B, C_out, T, H, W) 的 DINO 特征。"""

    def __init__(
        self,
        C_in: int = 1280,
        C_out: int = 768,
        T: int = 16,
        H: int = 16,
        W: int = 16,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.T = T
        self.H = H
        self.W = W

        self.input_norm = nn.LayerNorm(C_in)
        self.input_proj = nn.Linear(C_in, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                SpatialTemporalAttentionBlock(
                    hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, C_out)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                if module.out_proj.bias is not None:
                    nn.init.zeros_(module.out_proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            hidden_states: (B, C_in, T, H, W)

        Returns:
            (B, C_out, T, H, W)
        """

        if hidden_states.dim() != 5:
            raise ValueError(
                f"期望输入维度为 5 (B, C_in, T, H, W)，得到 {hidden_states.shape}"
            )

        B, C, T, H, W = hidden_states.shape
        if C != self.C_in:
            raise ValueError(f"C_in={self.C_in}，但输入通道数为 {C}")
        if T != self.T:
            raise ValueError(f"时间长度 T={self.T}，但输入为 {T}")
        if H != self.H or W != self.W:
            raise ValueError(f"空间尺寸应为 {(self.H, self.W)}，但输入为 {(H, W)}")

        x = hidden_states.permute(0, 2, 3, 4, 1).contiguous()  # (B, T, H, W, C)
        x = self.input_norm(x)
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        x = self.output_norm(x)
        x = self.output_proj(x)
        x = x.flatten(start_dim=2, end_dim=3).contiguous() 
        #x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x


class HiddenToDA3ModelWithRef(nn.Module):
    """在不改变原始 HiddenToDA3Model 结构的情况下提供第一帧 DINO 条件。"""

    def __init__(
        self,
        base_model: HiddenToDA3Model,
        ref_dim: int = 768,
    ) -> None:
        super().__init__()
        if not isinstance(base_model, HiddenToDA3Model):
            raise TypeError("base_model 必须是 HiddenToDA3Model 实例")
        self.base_model = base_model
        self.ref_dim = ref_dim
        self.adapter = nn.Conv3d(ref_dim, base_model.C_in, kernel_size=1)
        nn.init.xavier_uniform_(self.adapter.weight)
        if self.adapter.bias is not None:
            nn.init.zeros_(self.adapter.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        ref_dino: torch.Tensor,
    ) -> torch.Tensor:
        """将第一帧 DINO 特征映射到隐藏状态通道后注入模型。

        Args:
            hidden_states: (B, C_in, T, H, W)
            ref_dino: (B, C_out, 1, H, W)
        """
        B, one, num_patches, ref_dim = ref_dino.shape
        H = W = int(num_patches**0.5)
        ref_dino = ref_dino.view(B, 1, H, W, ref_dim).permute(0, 4, 1, 2, 3).contiguous() 
        if ref_dino.dim() != 5:
            raise ValueError(
                f"期望 ref_dino 维度为 5 (B, C_out, 1, H, W)，得到 {ref_dino.shape}"
            )
        if hidden_states.shape[0] != ref_dino.shape[0]:
            raise ValueError("hidden_states 与 ref_dino 的 batch size 不一致")
        if ref_dino.shape[2] != 1:
            raise ValueError(f"ref_dino 时间维度应为 1，得到 {ref_dino.shape[2]}")

        ref = ref_dino.expand(
            -1, -1, hidden_states.shape[2], -1, -1
        ).contiguous()
        ref = self.adapter(ref)
        conditioned_hidden = hidden_states + ref
        return self.base_model(conditioned_hidden)


__all__ = ["HiddenToDA3Model", "HiddenToDA3ModelWithRef"]

