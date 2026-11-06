from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .config import DESCRIPTOR_DIM


class CADTower(nn.Module):
    def __init__(
        self,
        input_dim: int = DESCRIPTOR_DIM,
        hidden_dim: int = 1024,
        output_dim: int = 512,
        blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(blocks)
            ]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.input(values)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output(hidden)


class TextEncoder(nn.Module):
    def __init__(
        self,
        model: str | Path,
        lora_rank: int,
        lora_alpha: int,
        hidden_dim: int = 1024,
        output_dim: int = 512,
        dropout: float = 0.1,
        adapter: str | Path | None = None,
    ):
        super().__init__()
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModel

        backbone = AutoModel.from_pretrained(
            model,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        backbone.config.use_cache = False
        if adapter is None:
            backbone.gradient_checkpointing_enable()
            backbone.enable_input_require_grads()
            config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                bias="none",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )
            self.backbone = get_peft_model(backbone, config)
        else:
            self.backbone = PeftModel.from_pretrained(backbone, adapter)
        hidden_size = backbone.config.hidden_size
        self.trunk = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        indexes = attention_mask.size(1) - 1 - attention_mask.flip(1).float().argmax(1)
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), indexes].float()
        return self.projection(self.trunk(pooled))
