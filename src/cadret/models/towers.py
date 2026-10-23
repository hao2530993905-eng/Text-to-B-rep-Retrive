from __future__ import annotations

import torch
from torch import nn

from ..config import AUX_CLS, AUX_REG
from ..geometry.descriptor import DESC_DIM


class CADTower(nn.Module):
    def __init__(self, dim=DESC_DIM, hidden=1024, out=512, n_blocks=3, p=0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU())
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                    nn.Dropout(p),
                    nn.Linear(hidden, hidden),
                )
                for _ in range(n_blocks)
            ]
        )
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out))

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)


class TextTower(nn.Module):
    def __init__(self, dim, hidden=1024, out=512, p=0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(p),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.proj = nn.Linear(hidden, out)
        self.cls = nn.ModuleDict({k: nn.Linear(hidden, n) for k, n in AUX_CLS})
        self.reg = nn.Linear(hidden, len(AUX_REG))

    def forward(self, x):
        h = self.trunk(x)
        return (self.proj(h), {k: m(h) for k, m in self.cls.items()}, self.reg(h))


class TextEncoder(nn.Module):
    def __init__(
        self, model, lora_r=32, lora_alpha=64, out=512, hidden=1024, p=0.1, adapter_dir=None
    ):
        super().__init__()
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModel

        bb = AutoModel.from_pretrained(model, dtype=torch.bfloat16, trust_remote_code=True)
        if adapter_dir is None:
            bb.gradient_checkpointing_enable()
            bb.enable_input_require_grads()
            cfg = LoraConfig(
                r=lora_r,
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
            self.bb = get_peft_model(bb, cfg)
        else:
            self.bb = PeftModel.from_pretrained(bb, adapter_dir)
        d = bb.config.hidden_size
        self.trunk = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(p),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.proj = nn.Linear(hidden, out)
        self.cls = nn.ModuleDict({k: nn.Linear(hidden, n) for k, n in AUX_CLS})
        self.reg = nn.Linear(hidden, len(AUX_REG))

    def forward(self, ids, mask):
        h = self.bb(input_ids=ids, attention_mask=mask).last_hidden_state
        idx = mask.size(1) - 1 - mask.flip(dims=[1]).float().argmax(dim=1)
        pooled = h[torch.arange(h.size(0), device=h.device), idx].float()
        t = self.trunk(pooled)
        return (self.proj(t), {k: m(t) for k, m in self.cls.items()}, self.reg(t))
