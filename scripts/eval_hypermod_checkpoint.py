import sys
from argparse import ArgumentParser
from hyper_llm_modulator.sft_trainer import eval_hypermod_checkpoint

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--full_eval", action="store_true")
    parser.add_argument("--use-icl", action="store_true")
    parser.add_argument("--tasks", type=str, nargs="+", help="List of task names to evaluate (e.g., openbookqa hellaswag boolq). If not specified, evaluates all available tasks.")
    parser.add_argument(
        "--random-profile-embs",
        action="store_true",
        help="Use random user profile embeddings instead of the true profiles for ablation studies.",
    )
    parser.add_argument(
        "--random-profile-strings",
        action="store_true",
        help="Replace user profiles with random strings before embedding and evaluation.",
    )
    eval_hypermod_checkpoint(**vars(parser.parse_args()), curstep=None)
