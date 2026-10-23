import os
from pathlib import Path

import torch

TEXT_HEADS = ("trunk", "proj", "cls", "reg")


def load_checkpoint(path, require_text=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"cadt", "mu", "sd"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("Checkpoint does not contain all required model parameters")
    if require_text and not {"text_heads", "config"}.issubset(checkpoint):
        raise ValueError(
            "Complete retrieval requires trained text heads and a LoRA adapter in addition to CAD parameters"
        )
    if require_text and not set(TEXT_HEADS).issubset(checkpoint["text_heads"]):
        raise ValueError("Checkpoint is missing trained text heads")
    return checkpoint


def restore_text_heads(text, checkpoint):
    for name in TEXT_HEADS:
        getattr(text, name).load_state_dict(checkpoint["text_heads"][name])


def save_checkpoint(cad, text, tokenizer, mu, sd, config, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / "lora_text"
    text.bb.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    def cpu_state(module):
        return {name: value.detach().cpu() for name, value in module.state_dict().items()}

    checkpoint = {
        "cadt": cpu_state(cad),
        "text_heads": {name: cpu_state(getattr(text, name)) for name in TEXT_HEADS},
        "mu": mu.detach().cpu(),
        "sd": sd.detach().cpu(),
        "config": dict(config),
    }
    temporary = output_dir / "lora_cad.pt.part"
    torch.save(checkpoint, temporary)
    os.replace(temporary, output_dir / "lora_cad.pt")
