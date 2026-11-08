from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "prepare-text2cad": ("cadret.prepare", "text2cad_command", "Build Text2CAD data"),
    "prepare-mmcad": ("cadret.prepare", "mmcad_command", "Build MMCAD-B data"),
    "encode-text": ("cadret.text_features", "main", "Cache frozen text features"),
    "train": ("cadret.train", "main", "Train a frozen or LoRA model"),
    "evaluate": ("cadret.evaluate", "main", "Evaluate a checkpoint"),
}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print("usage: cadret <command> [options]")
        for command, (_, _, description) in COMMANDS.items():
            print(f"  {command:18s} {description}")
        return
    command = arguments[0]
    if command not in COMMANDS:
        raise SystemExit(f"unknown command: {command}")
    module_name, function_name, _ = COMMANDS[command]
    module = importlib.import_module(module_name)
    getattr(module, function_name)(arguments[1:])
