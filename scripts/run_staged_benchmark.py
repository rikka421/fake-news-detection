from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tiny first, then a larger dataset benchmark")
    parser.add_argument(
        "--target-dataset",
        choices=["small", "medium", "large", "balanced_small", "balanced_medium", "balanced_large"],
        default="medium",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--text-mode", default="content_only", choices=["all_fields", "title_content", "content_only"])
    parser.add_argument("--split-mode", default="random", choices=["random", "grouped_content", "grouped_title_content"])
    parser.add_argument("--mask-label-tokens", action="store_true")
    parser.add_argument("--models", nargs="+", default=["logistic_regression", "cnn", "transformer"])
    args = parser.parse_args()

    commands = [
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_news_benchmark.py"),
            "--dataset",
            "balanced_tiny" if args.target_dataset.startswith("balanced_") else "tiny",
            "--epochs",
            "2",
            "--text-mode",
            args.text_mode,
            "--split-mode",
            args.split_mode,
            "--output",
            "artifacts/results/benchmark_tiny_smoke.json",
            "--models",
            *args.models,
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_news_benchmark.py"),
            "--dataset",
            args.target_dataset,
            "--epochs",
            str(args.epochs),
            "--text-mode",
            args.text_mode,
            "--split-mode",
            args.split_mode,
            "--output",
            f"artifacts/results/benchmark_{args.target_dataset}_staged.json",
            "--models",
            *args.models,
        ],
    ]

    if args.mask_label_tokens:
        commands[0].append("--mask-label-tokens")
        commands[1].append("--mask-label-tokens")

    for command in commands:
        result = subprocess.run(command, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
