import argparse
import importlib
import sys

COMMANDS = {
    "prepare": ("cadret.data.prepare", "Build the CAD descriptor cache"),
    "train": ("cadret.training.lora", "Train the LoRA retrieval model with torchrun"),
    "train-frozen": ("cadret.training.frozen", "Train with cached text features"),
    "evaluate": ("cadret.evaluation.retrieval", "Evaluate a complete retrieval checkpoint"),
    "encode-text": ("cadret.data.text_features", "Cache frozen text features"),
    "validate-descriptors": ("cadret.evaluation.descriptors", "Evaluate geometric descriptors"),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="cadret")
    parser.add_argument("command", choices=COMMANDS)
    parser.epilog = "\n".join(
        f"{name}: {description}" for name, (_, description) in COMMANDS.items()
    )
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return
    args = parser.parse_args(argv[:1])
    module = importlib.import_module(COMMANDS[args.command][0])
    module.main(argv[1:])
