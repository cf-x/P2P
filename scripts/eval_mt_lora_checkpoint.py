import os
import argparse
import yaml
from types import SimpleNamespace

from hyper_llm_modulator.utils.eval_hypermod import eval_lora


def find_args_yaml(start_path: str) -> str | None:
    """Best-effort search for args.yaml starting from a LoRA directory.

    Tries: path, parent, grandparent.
    """
    candidates = [
        os.path.join(start_path, "args.yaml"),
        os.path.join(os.path.dirname(start_path), "args.yaml"),
        os.path.join(os.path.dirname(os.path.dirname(start_path)), "args.yaml"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def load_args_from_yaml(args_yaml: str) -> SimpleNamespace:
    with open(args_yaml, "r") as f:
        raw = yaml.safe_load(f)
    # Only keep keys eval_lora relies on
    keep_keys = {
        "model_dir",
        "eval_ds_info",
        "use_per_task_emb",
    }
    filtered = {k: raw.get(k, None) for k in keep_keys}
    return SimpleNamespace(**filtered)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_dir", type=str, required=True, help="Path to mt_lora checkpoint directory (contains adapter_model.safetensors)")
    parser.add_argument("--full_eval", action="store_true", help="Evaluate full test suites from args.yaml; otherwise benchmark set")
    parser.add_argument("--use-icl", action="store_true", help="Enable ICL eval variants when available")
    parser.add_argument("--tasks", nargs="+", default=None, help="Optional override list of tasks to evaluate")
    parser.add_argument("--model-dir", type=str, default=None, help="Optional override for base model directory")

    args_cli = parser.parse_args()

    # Normalize lora_dir to directory if a file was provided
    lora_dir = args_cli.lora_dir
    if os.path.isfile(lora_dir):
        lora_dir = os.path.dirname(lora_dir)

    # Locate and load training args for model_dir and eval_ds_info
    args_yaml = find_args_yaml(lora_dir)
    if args_yaml is None:
        raise FileNotFoundError(
            f"Could not find args.yaml near {args_cli.lora_dir}. Looked in the directory and its parents."
        )

    ns = load_args_from_yaml(args_yaml)

    # Optional overrides from CLI
    if args_cli.model_dir is not None:
        ns.model_dir = args_cli.model_dir
    if args_cli.tasks is not None:
        ns.eval_ds_info = list(args_cli.tasks)

    # Basic validations
    if not ns.model_dir:
        raise ValueError("model_dir missing; specify in args.yaml or via --model-dir")
    if not ns.eval_ds_info:
        raise ValueError("eval_ds_info missing; specify in args.yaml or via --tasks override")

    # Kick off evaluation (saves results under lora_dir/eval_results)
    eval_lora(ns, lora_dir, curstep=None, full_eval=args_cli.full_eval, use_icl=args_cli.use_icl)

