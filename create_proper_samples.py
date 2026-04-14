#!/usr/bin/env python3
"""
Create proper samples from FakeNewsDataset with correct CSV parsing
Handles quoted fields with commas properly
"""

import csv
import os
import random
from collections import defaultdict

def read_sample_with_proper_csv(filepath, sample_size=1000):
    """Read sample from CSV with proper handling of quoted fields"""
    samples = []
    header = None
    
    print(f"Reading {sample_size} samples with proper CSV parsing...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        # Read header
        reader = csv.reader(f)
        header = next(reader)
        
        # Read sample rows
        for i, row in enumerate(reader):
            if i >= sample_size:
                break
            samples.append(row)
    
    print(f"Read {len(samples)} samples")
    return header, samples

def analyze_types(header, samples):
    """Analyze the distribution of types in the samples"""
    # Find type column index
    try:
        type_idx = header.index('type')
    except ValueError:
        print("'type' column not found in header")
        return None
    
    type_counts = defaultdict(int)
    for row in samples:
        if len(row) > type_idx:
            type_counts[row[type_idx]] += 1
    
    print(f"Found {len(type_counts)} different types:")
    for type_name, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {type_name}: {count} rows")
    
    return type_idx, type_counts

def create_balanced_sample(header, samples, type_idx, target_size=1000):
    """Create a balanced sample with equal representation of each type"""
    print(f"\nCreating balanced sample with {target_size} total rows...")
    
    # Group samples by type
    samples_by_type = defaultdict(list)
    for row in samples:
        if len(row) > type_idx:
            samples_by_type[row[type_idx]].append(row)
    
    # Calculate how many of each type to include
    types = list(samples_by_type.keys())
    per_type = target_size // len(types)
    
    print(f"Will include {per_type} rows from each of {len(types)} types")
    
    # Collect balanced samples
    balanced_samples = []
    for type_name in types:
        type_samples = samples_by_type[type_name]
        # Take random samples if we have enough, otherwise take all
        if len(type_samples) >= per_type:
            balanced_samples.extend(random.sample(type_samples, per_type))
        else:
            balanced_samples.extend(type_samples)
            print(f"  Note: Type '{type_name}' only has {len(type_samples)} samples")
    
    print(f"Created balanced sample with {len(balanced_samples)} rows")
    return balanced_samples

def write_samples(header, samples, output_dir, prefix="fakenews"):
    """Write samples to CSV files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Write different sample sizes
    sample_sizes = [10, 100, 1000, 5000]
    
    for size in sample_sizes:
        if len(samples) >= size:
            output_file = os.path.join(output_dir, f"{prefix}_proper_{size}.csv")
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(samples[:size])
            
            file_size_mb = os.path.getsize(output_file) / (1024**2)
            print(f"  Created: {os.path.basename(output_file)} - {size} rows, {file_size_mb:.2f} MB")

def main():
    """Main function"""
    print("=" * 60)
    print("FakeNewsDataset Proper Sampler")
    print("=" * 60)
    
    # File paths
    source_file = r"C:\Users\22130\Downloads\news.csv\news_cleaned_2018_02_13.csv"
    output_dir = r"C:\Users\22130\.openclaw\workspace\lessons\data-analysis-progress\data\FakeNewsCorpus\proper_samples"
    
    # Check if source file exists
    if not os.path.exists(source_file):
        print(f"Error: Source file not found: {source_file}")
        return
    
    # Read a larger sample for analysis
    print(f"Reading samples from {source_file}...")
    header, samples = read_sample_with_proper_csv(source_file, sample_size=10000)
    
    if not samples:
        print("No samples read. Exiting.")
        return
    
    print(f"\nHeader: {header}")
    
    # Analyze types
    type_info = analyze_types(header, samples)
    if type_info:
        type_idx, type_counts = type_info
        
        # Create balanced sample
        balanced_samples = create_balanced_sample(header, samples, type_idx, target_size=1000)
        
        # Write balanced sample
        balanced_dir = os.path.join(output_dir, "balanced")
        write_samples(header, balanced_samples, balanced_dir, prefix="fakenews_balanced")
    
    # Write regular samples
    print(f"\nCreating regular samples...")
    write_samples(header, samples, output_dir)
    
    print(f"\n✅ All samples created successfully!")
    print(f"Location: {output_dir}")

if __name__ == "__main__":
    main()