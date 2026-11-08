from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


class GatherWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        outputs = [torch.zeros_like(value) for _ in range(dist.get_world_size())]
        dist.all_gather(outputs, value.contiguous())
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *gradients):
        stacked = torch.stack(gradients)
        dist.all_reduce(stacked)
        return stacked[dist.get_rank()]


def gather_with_gradient(value):
    return torch.cat(GatherWithGradient.apply(value), dim=0)


def gather_variable(value):
    size = torch.tensor([len(value)], dtype=torch.long, device=value.device)
    sizes = [torch.zeros_like(size) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, size)
    lengths = [int(item.item()) for item in sizes]
    padded = value.new_zeros((max(1, max(lengths)), *value.shape[1:]))
    padded[: len(value)] = value
    buffers = [torch.empty_like(padded) for _ in lengths]
    dist.all_gather(buffers, padded)
    return torch.cat([buffer[:length] for buffer, length in zip(buffers, lengths)], dim=0)


def symmetric_info_nce(text, cad, temperature):
    logits = text @ cad.T / temperature
    target = torch.arange(text.shape[0], device=text.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def pooled_recall(queries, gallery, rows, levels, pool_size, trials):
    if pool_size < 1 or pool_size > len(gallery):
        raise ValueError("invalid pool size")
    generator = random.Random(1)
    digest = hashlib.sha256()
    results = {}
    for level in levels:
        indices = [index for index, row in enumerate(rows) if row[1] == level]
        if not indices:
            continue
        hits = {1: 0, 5: 0, 10: 0}
        ranks = []
        for _ in range(trials):
            query_index = indices[generator.randrange(len(indices))]
            target = rows[query_index][0]
            candidates = [target]
            candidates.extend(
                generator.sample(
                    [index for index in range(len(gallery)) if index != target],
                    pool_size - 1,
                )
            )
            digest.update(np.asarray([query_index, *candidates], dtype=np.int64).tobytes())
            scores = (gallery[candidates] @ queries[query_index]).numpy()
            rank = int((scores[1:] > scores[0]).sum() + 0.5 * (scores[1:] == scores[0]).sum())
            ranks.append(rank + 1)
            for cutoff in hits:
                hits[cutoff] += rank < cutoff
        results[level] = {
            "R@1": hits[1] / trials,
            "R@5": hits[5] / trials,
            "R@10": hits[10] / trials,
            "MedR": float(np.median(ranks)),
            "queries": len(indices),
        }
    return results, digest.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    output_directory: Path,
    cad_tower,
    text_model,
    tokenizer,
    mean,
    std,
    config,
):
    output_directory.mkdir(parents=True, exist_ok=True)
    if config["tuning"] == "lora":
        adapter_directory = output_directory / "adapter"
        text_model.backbone.save_pretrained(adapter_directory)
        tokenizer.save_pretrained(adapter_directory)
    checkpoint = {
        "cad_tower": {name: value.detach().cpu() for name, value in cad_tower.state_dict().items()},
        "text_heads": {
            name: {
                key: value.detach().cpu()
                for key, value in getattr(text_model, name).state_dict().items()
            }
            for name in ("trunk", "projection")
        },
        "mean": mean.detach().cpu(),
        "std": std.detach().cpu(),
        "config": config,
    }
    temporary = output_directory / "retrieval.pt.part"
    torch.save(checkpoint, temporary)
    temporary.replace(output_directory / "retrieval.pt")


def load_checkpoint(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"cad_tower", "text_heads", "mean", "std", "config"}
    if not required.issubset(checkpoint):
        raise ValueError("checkpoint is incomplete")
    return checkpoint


def restore_text_heads(text_encoder, checkpoint):
    for name in ("trunk", "projection"):
        getattr(text_encoder, name).load_state_dict(checkpoint["text_heads"][name])
