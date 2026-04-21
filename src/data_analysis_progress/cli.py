from __future__ import annotations

import argparse

from .training import BenchmarkConfig, run_benchmark, write_benchmark_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fake-news benchmark runner")
    parser.add_argument("--dataset", default="medium", help="Dataset alias or relative CSV path")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "cnn", "transformer"],
        choices=["logistic_regression", "cnn", "transformer"],
        help="Models to benchmark",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for deep models")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--max-length", type=int, default=64, help="Token cap per sample")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding width")
    parser.add_argument("--min-frequency", type=int, default=2, help="Minimum token frequency")
    parser.add_argument(
        "--text-mode",
        type=str,
        default="all_fields",
        choices=["all_fields", "title_content", "content_only"],
        help="Which text columns to use for modeling",
    )
    parser.add_argument(
        "--mask-label-tokens",
        action="store_true",
        help="Mask class-name tokens from text to reduce leakage",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="random",
        choices=["random", "grouped_content", "grouped_title_content"],
        help="How to split data into train/validation/test",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = BenchmarkConfig(
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        embedding_dim=args.embedding_dim,
        min_frequency=args.min_frequency,
        seed=args.seed,
        models=tuple(args.models),
        text_mode=args.text_mode,
        mask_label_tokens=args.mask_label_tokens,
        split_mode=args.split_mode,
    )
    report = run_benchmark(config)
    output_path = write_benchmark_report(report, args.output)

    print(f"Dataset size: {report['dataset_size']}")
    print(f"Label count: {report['label_count']}")
    for model_name, metrics in report["results"].items():
        print(f"{model_name}: test_accuracy={metrics['test_accuracy']:.4f}, test_macro_f1={metrics['test_macro_f1']:.4f}")
    print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
