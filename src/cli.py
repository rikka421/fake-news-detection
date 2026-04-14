"""
Command-line interface for News Classification Analysis
"""

import argparse
import sys
from pathlib import Path

from .data.loader import DataLoader


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="News Classification Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s explore --sample small
  %(prog)s analyze --sample medium
  %(prog)s split --sample small --test-size 0.2
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Explore command
    explore_parser = subparsers.add_parser("explore", help="Explore dataset")
    explore_parser.add_argument(
        "--sample", 
        choices=["test", "tiny", "small", "medium", "stratified"],
        default="small",
        help="Sample size to load"
    )
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze dataset")
    analyze_parser.add_argument(
        "--sample",
        choices=["test", "tiny", "small", "medium", "stratified"],
        default="small",
        help="Sample size to load"
    )
    
    # Split command
    split_parser = subparsers.add_parser("split", help="Create train/test split")
    split_parser.add_argument(
        "--sample",
        choices=["test", "tiny", "small", "medium", "stratified"],
        default="small",
        help="Sample size to load"
    )
    split_parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion for test set (default: 0.2)"
    )
    split_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize data loader
    loader = DataLoader()
    
    try:
        if args.command == "explore":
            explore_dataset(loader, args.sample)
        elif args.command == "analyze":
            analyze_dataset(loader, args.sample)
        elif args.command == "split":
            create_split(loader, args.sample, args.test_size, args.random_state)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def explore_dataset(loader, sample_size):
    """Explore dataset"""
    print(f"Loading {sample_size} sample...")
    df = loader.load_fakenews_sample(sample_size)
    
    print(f"\nDataset Overview:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    
    print(f"\nFirst 5 rows:")
    print(df.head().to_string())
    
    print(f"\nLabel Distribution:")
    label_counts = loader.get_label_distribution(df)
    print(label_counts.to_string())


def analyze_dataset(loader, sample_size):
    """Analyze dataset"""
    print(f"Loading {sample_size} sample...")
    df = loader.load_fakenews_sample(sample_size)
    
    analysis = loader.analyze_dataset(df)
    
    print(f"\nDataset Analysis:")
    print(f"  Total articles: {analysis['total_articles']}")
    print(f"  Number of columns: {len(analysis['columns'])}")
    
    print(f"\nMissing Values:")
    for col, count in analysis['missing_values'].items():
        if count > 0:
            print(f"  {col}: {count}")
    
    print(f"\nLabel Distribution:")
    for label, count in analysis['label_distribution'].items():
        print(f"  {label}: {count}")
    
    if 'text_length_stats' in analysis and 'content' in analysis['text_length_stats']:
        stats = analysis['text_length_stats']['content']
        print(f"\nContent Length Statistics:")
        print(f"  Mean: {stats['mean']:.0f} characters")
        print(f"  Std: {stats['std']:.0f} characters")
        print(f"  Min: {stats['min']} characters")
        print(f"  Max: {stats['max']} characters")


def create_split(loader, sample_size, test_size, random_state):
    """Create train/test split"""
    print(f"Loading {sample_size} sample...")
    df = loader.load_fakenews_sample(sample_size)
    
    splits = loader.create_train_test_split(df, test_size, random_state)
    
    print(f"\nTrain/Test Split:")
    print(f"  Train set: {len(splits['train'])} articles")
    print(f"  Test set: {len(splits['test'])} articles")
    print(f"  Test size: {test_size * 100:.0f}%")
    
    print(f"\nTrain Label Distribution:")
    train_counts = loader.get_label_distribution(splits['train'])
    print(train_counts.to_string())
    
    print(f"\nTest Label Distribution:")
    test_counts = loader.get_label_distribution(splits['test'])
    print(test_counts.to_string())
    
    # Save splits to files
    output_dir = Path("data/splits")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / f"train_{sample_size}.csv"
    test_path = output_dir / f"test_{sample_size}.csv"
    
    splits['train'].to_csv(train_path, index=False)
    splits['test'].to_csv(test_path, index=False)
    
    print(f"\nSplits saved to:")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")


if __name__ == "__main__":
    main()