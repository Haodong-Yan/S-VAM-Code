from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModelWithProjection

from policy_models.module.clip import build_model, load_clip, tokenize


class LangClip(nn.Module):
    def __init__(
        self,
        freeze_backbone: bool = True,
        model_name: str = "RN50",
        pretrained_path: Optional[str] = None,
    ):
        super(LangClip, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_hf_backend = pretrained_path is not None

        if self.use_hf_backend:
            print(f"loading language CLIP model from local path: {pretrained_path}")
            self._load_hf_clip(pretrained_path)
            if freeze_backbone:
                for param in self.hf_text_encoder.parameters():
                    param.requires_grad = False
        else:
            print(f"loading language CLIP model with backbone: {model_name}")
            self._load_clip(model_name)
            if freeze_backbone:
                for param in self.clip_rn50.parameters():
                    param.requires_grad = False

    def _load_clip(self, model_name: str) -> None:
        model, _ = load_clip(model_name, device=self.device)
        self.clip_rn50 = build_model(model.state_dict()).to(self.device)

    def _load_hf_clip(self, pretrained_path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=False)
        self.hf_text_encoder = CLIPTextModelWithProjection.from_pretrained(
            pretrained_path
        ).to(self.device)

    def forward(self, x: List) -> torch.Tensor:
        with torch.no_grad():
            if self.use_hf_backend:
                tokenized = self.tokenizer(
                    x,
                    padding=True,
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                )
                tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
                emb = self.hf_text_encoder(**tokenized).text_embeds
            else:
                tokens = tokenize(x).to(self.device)
                emb = self.clip_rn50.encode_text(tokens)
        return torch.unsqueeze(emb, 1)
