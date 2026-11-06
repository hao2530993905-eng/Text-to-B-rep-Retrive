from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from .config import DATASET_LEVELS, INSTRUCTION
from .data import load_split
from .model import CADTower, TextEncoder
from .runtime import (
    gather_with_gradient,
    save_checkpoint,
    seed_everything,
    symmetric_info_nce,
)


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def positive_float(value):
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def write_curve(directory, history):
    with (directory / "training_curve.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    if not history:
        return
    with (directory / "training_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def build_parser():
    parser = argparse.ArgumentParser(prog="cadret train")
    parser.add_argument("--dataset", choices=tuple(DATASET_LEVELS), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--levels")
    parser.add_argument("--epochs", type=positive_int, default=3)
    parser.add_argument("--per-device-batch", type=positive_int, required=True)
    parser.add_argument("--backbone-lr", type=positive_float, default=1e-4)
    parser.add_argument("--head-lr", type=positive_float, default=5e-4)
    parser.add_argument("--temperature", type=positive_float, default=0.05)
    parser.add_argument("--max-length", type=positive_int, default=320)
    parser.add_argument("--lora-rank", type=positive_int, default=32)
    parser.add_argument("--lora-alpha", type=positive_int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=positive_int)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.lora_alpha < args.lora_rank:
        raise ValueError("lora alpha must be at least lora rank")
    if not args.cache.is_file():
        raise FileNotFoundError(f"cache not found: {args.cache}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    levels = tuple(
        value.strip()
        for value in (args.levels or ",".join(DATASET_LEVELS[args.dataset])).split(",")
        if value.strip()
    )
    data = load_split(args.cache, "train", levels)
    if data["dataset"] != args.dataset:
        raise ValueError("cache dataset does not match --dataset")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        primary = rank == 0
        seed_everything(args.seed)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        tokenizer.padding_side = "left"
        global_batch = args.per_device_batch * world_size
        steps_per_epoch = len(data["rows"]) // global_batch
        if steps_per_epoch < 1:
            raise ValueError("global batch is larger than the training data")
        planned_steps = steps_per_epoch * args.epochs
        total_steps = min(planned_steps, args.max_steps or planned_steps)
        if total_steps < 2:
            raise ValueError("at least two optimizer steps are required")
        mean = data["vectors"].mean(0)
        std = data["vectors"].std(0).clamp(min=1e-4)
        vectors = ((data["vectors"] - mean) / std).to(device)
        if primary:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            config = {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            }
            config["levels"] = ",".join(levels)
            config["world_size"] = world_size
            config["global_batch"] = global_batch
            config["steps_per_epoch"] = steps_per_epoch
            config["total_steps"] = total_steps
            with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2)
            print(
                f"dataset={args.dataset} train_models={len(data['uids'])} "
                f"train_rows={len(data['rows'])} world={world_size} "
                f"global_batch={global_batch} total_steps={total_steps}",
                flush=True,
            )
        dist.barrier()
        cad_tower = CADTower().to(device)
        text_encoder = TextEncoder(
            args.model,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
        ).to(device)
        invalid = [
            name
            for name, parameter in text_encoder.backbone.named_parameters()
            if parameter.requires_grad and "lora_" not in name
        ]
        if invalid:
            raise RuntimeError("non-adapter backbone parameters are trainable")
        cad_tower = DistributedDataParallel(cad_tower, device_ids=[local_rank])
        text_encoder = DistributedDataParallel(text_encoder, device_ids=[local_rank])
        backbone_parameters = [
            parameter
            for name, parameter in text_encoder.named_parameters()
            if parameter.requires_grad and ".backbone." in name
        ]
        head_parameters = [
            parameter
            for name, parameter in text_encoder.named_parameters()
            if parameter.requires_grad and ".backbone." not in name
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_parameters, "lr": args.backbone_lr},
                {"params": head_parameters, "lr": args.head_lr},
                {"params": list(cad_tower.parameters()), "lr": args.head_lr},
            ],
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            [args.backbone_lr, args.head_lr, args.head_lr],
            total_steps=total_steps,
            pct_start=0.05,
        )
        trainable = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        history = []
        completed = 0
        for epoch in range(args.epochs):
            text_encoder.train()
            cad_tower.train()
            generator = torch.Generator().manual_seed(args.seed + epoch)
            order = torch.randperm(len(data["rows"]), generator=generator).tolist()
            epoch_loss = 0.0
            epoch_steps = 0
            iterator = range(steps_per_epoch)
            if primary:
                iterator = tqdm(
                    iterator,
                    desc=f"{args.dataset} epoch {epoch + 1}/{args.epochs}",
                    dynamic_ncols=True,
                )
            for step in iterator:
                if completed >= total_steps:
                    break
                start = step * global_batch + rank * args.per_device_batch
                selected = order[start : start + args.per_device_batch]
                tokenized = tokenizer(
                    [INSTRUCTION + data["rows"][index][2] for index in selected],
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                ).to(device)
                model_indices = torch.tensor(
                    [data["rows"][index][0] for index in selected],
                    dtype=torch.long,
                    device=device,
                )
                text_embeddings = text_encoder(
                    tokenized.input_ids,
                    tokenized.attention_mask,
                )
                cad_embeddings = cad_tower(vectors[model_indices])
                all_text = gather_with_gradient(F.normalize(text_embeddings, dim=-1))
                all_cad = gather_with_gradient(F.normalize(cad_embeddings, dim=-1))
                loss = symmetric_info_nce(all_text, all_cad, args.temperature)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                completed += 1
                epoch_loss += float(loss.detach())
                epoch_steps += 1
                if primary:
                    iterator.set_postfix(
                        loss=f"{float(loss.detach()):.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )
            if epoch_steps:
                row = {
                    "epoch": epoch,
                    "steps": epoch_steps,
                    "loss": epoch_loss / epoch_steps,
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
                history.append(row)
                if primary:
                    write_curve(args.output_dir, history)
                    save_checkpoint(
                        args.output_dir,
                        cad_tower.module,
                        text_encoder.module,
                        tokenizer,
                        mean,
                        std,
                        config,
                    )
            dist.barrier()
            if completed >= total_steps:
                break
        if primary:
            print(
                f"training complete: {args.output_dir / 'retrieval.pt'}",
                flush=True,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()
