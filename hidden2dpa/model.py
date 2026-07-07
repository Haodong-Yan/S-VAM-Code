"""Mapping hidden states to Depth Anything 3 / DualDPT patch tokens."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class TemporalSelfAttentionBlock(nn.Module):
    """Lightweight Transformer block on the temporal dimension."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_mlp = int(dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_mlp),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_mlp, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, dim)
        residual = x
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = residual + self.dropout(attn_out)

        residual = x
        x = self.mlp(self.norm2(x))
        return residual + x


class HiddenToDA3Model(nn.Module):
    """Produce DA3-compatible tokens for DualDPT from hidden states.

    Input:
        hidden_states: (B, C_in, T, H, W) where T is view/time and H, W are patch grid.
    Output:
        List of length num_stages. Each element is a tuple (tokens, cam_tokens):
            tokens:    (B, T, H*W, token_dim)
            cam_token: (B, T, token_dim) aggregated per view (mean or first token)

    These can be fed to DualDPT.forward(feats, H_img, W_img, patch_start_idx=0).
    Typically H_img = H * patch_size, W_img = W * patch_size, with patch_size=14 in DA3.
    """

    def __init__(
        self,
        C_in: int = 1280,
        token_dim: int = 1536,
        T: int = 16,
        H: int = 16,
        W: int = 16,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_stages: int = 4,
        camera_pool: str = "mean",
        use_stage_embed: bool = True,
    ) -> None:
        super().__init__()
        self.C_in = C_in
        self.token_dim = token_dim
        self.T = T
        self.H = H
        self.W = W
        self.num_stages = num_stages
        self.camera_pool = camera_pool.lower()

        self.input_norm = nn.LayerNorm(C_in)
        self.input_proj = nn.Linear(C_in, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                TemporalSelfAttentionBlock(
                    hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_norm = nn.LayerNorm(hidden_dim)

        self.stage_embed = (
            nn.Parameter(torch.zeros(num_stages, 1, 1, hidden_dim))
            if use_stage_embed
            else None
        )
        self.stage_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, token_dim) for _ in range(num_stages)]
        )

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
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _pool_camera_token(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T, H*W, token_dim) -> (B, T, token_dim)."""
        if self.camera_pool == "first":
            return tokens[:, :, 0]
        if self.camera_pool != "mean":
            raise ValueError(f"Unsupported camera_pool mode: {self.camera_pool}")
        return tokens.mean(dim=2)

    def forward(self, hidden_states: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        if hidden_states.dim() != 5:
            raise ValueError(
                f"Expected input shape (B, C_in, T, H, W), got {hidden_states.shape}"
            )

        B, C, T, H, W = hidden_states.shape
        if C != self.C_in:
            raise ValueError(f"C_in={self.C_in}, but got {C}")
        if T != self.T:
            raise ValueError(f"Temporal length T={self.T}, but got {T}")
        if H != self.H or W != self.W:
            raise ValueError(f"Spatial size should be {(self.H, self.W)}, but got {(H, W)}")

        # (B, C_in, T, H, W) -> (B*H*W, T, C_in)
        x = hidden_states.permute(0, 3, 4, 2, 1).contiguous()
        x = x.view(B * H * W, T, C)
        x = self.input_norm(x)
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x) + x  # residual

        x = self.output_norm(x)

        outputs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for idx, head in enumerate(self.stage_heads):
            stage_x = x
            if self.stage_embed is not None:
                stage_x = stage_x + self.stage_embed[idx]

            tokens = head(stage_x)  # (B*H*W, T, token_dim)
            tokens = tokens.view(B, H, W, T, self.token_dim)
            tokens = tokens.permute(0, 3, 1, 2, 4).contiguous()  # (B, T, H, W, token_dim)
            tokens = tokens.view(B, T, H * W, self.token_dim)  # (B, T, H*W, token_dim)

            cam_token = self._pool_camera_token(tokens)
            outputs.append((tokens, cam_token))

        return outputs


__all__ = ["HiddenToDA3Model", "TemporalSelfAttentionBlock"]


class HiddenToDA3ModelWithRef(nn.Module):
    """Inject first-frame DA3 tokens (stage-0) as condition into hidden states.

    Args:
        base_model: a HiddenToDA3Model instance
        ref_dim: token_dim of DA3 (default 1536)

    Forward:
        hidden_states: (B, C_in, T, H, W)
        ref_tokens: (B, 1, H*W, ref_dim)  # first frame tokens from backbone
    """

    def __init__(self, base_model: HiddenToDA3Model, ref_dim: int = 1536) -> None:
        super().__init__()
        if not isinstance(base_model, HiddenToDA3Model):
            raise TypeError("base_model must be HiddenToDA3Model")
        self.base_model = base_model
        self.ref_dim = ref_dim
        self.adapter = nn.Conv3d(ref_dim, base_model.C_in, kernel_size=1)
        nn.init.xavier_uniform_(self.adapter.weight)
        if self.adapter.bias is not None:
            nn.init.zeros_(self.adapter.bias)

    def forward(self, hidden_states: torch.Tensor, ref_tokens: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        if ref_tokens.dim() != 4:
            raise ValueError(
                f"Expected ref_tokens shape (B, 1, H*W, ref_dim), got {ref_tokens.shape}"
            )
        B, one, num_patches, ref_dim = ref_tokens.shape
        if one != 1:
            raise ValueError(f"ref_tokens time dim should be 1, got {one}")
        H = W = int(num_patches**0.5)
        if H * W != num_patches:
            raise ValueError(f"ref_tokens patch count {num_patches} not square")
        if ref_dim != self.ref_dim:
            raise ValueError(f"ref_dim expected {self.ref_dim}, got {ref_dim}")

        ref = ref_tokens.view(B, 1, H, W, ref_dim).permute(0, 4, 1, 2, 3).contiguous()  # (B, ref_dim, 1, H, W)
        ref = self.adapter(ref)  # (B, C_in, 1, H, W)
        # broadcast to T
        ref = ref.expand(-1, -1, self.base_model.T, -1, -1).contiguous()
        conditioned = hidden_states + ref
        return self.base_model(conditioned)


__all__ += ["HiddenToDA3ModelWithRef"]


class HiddenDepthProbe(nn.Module):
    """Lightweight probe: hidden states -> DA3-compatible tokens.

    This probe is intentionally simple (LayerNorm + linear projection), so it can
    be used to test whether aligned hidden features already encode depth cues.
    """

    def __init__(
        self,
        C_in: int,
        token_dim: int,
        num_stages: int = 4,
        camera_pool: str = "mean",
        use_norm: bool = True,
        share_projection: bool = False,
    ) -> None:
        super().__init__()
        if num_stages <= 0:
            raise ValueError(f"num_stages should be > 0, got {num_stages}")
        self.C_in = C_in
        self.token_dim = token_dim
        self.num_stages = num_stages
        self.camera_pool = camera_pool.lower()
        self.use_norm = use_norm
        self.share_projection = share_projection

        self.input_norm = nn.LayerNorm(C_in) if use_norm else nn.Identity()
        if share_projection:
            self.shared_proj = nn.Linear(C_in, token_dim)
            self.stage_projs = None
        else:
            self.shared_proj = None
            self.stage_projs = nn.ModuleList(
                [nn.Linear(C_in, token_dim) for _ in range(num_stages)]
            )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _pool_camera_token(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, T, H*W, token_dim)
        if self.camera_pool == "first":
            return tokens[:, :, 0]
        if self.camera_pool != "mean":
            raise ValueError(f"Unsupported camera_pool mode: {self.camera_pool}")
        return tokens.mean(dim=2)

    def forward(self, hidden_states: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        if hidden_states.dim() != 5:
            raise ValueError(
                f"Expected input shape (B, C_in, T, H, W), got {hidden_states.shape}"
            )
        B, C, T, H, W = hidden_states.shape
        if C != self.C_in:
            raise ValueError(f"C_in={self.C_in}, but got {C}")

        # (B, C, T, H, W) -> (B, T, H*W, C)
        x = hidden_states.permute(0, 2, 3, 4, 1).contiguous().view(B, T, H * W, C)
        x = self.input_norm(x)

        outputs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for stage_idx in range(self.num_stages):
            if self.share_projection:
                assert self.shared_proj is not None
                tokens = self.shared_proj(x)
            else:
                assert self.stage_projs is not None
                tokens = self.stage_projs[stage_idx](x)
            cam_token = self._pool_camera_token(tokens)
            outputs.append((tokens, cam_token))
        return outputs


__all__ += ["HiddenDepthProbe"]

