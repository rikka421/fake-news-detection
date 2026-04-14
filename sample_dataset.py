#!/usr/bin/env python3
"""
Sample FakeNewsDataset from 29GB CSV to manageable sizes
Creates samples of 100, 1000, and 10000 rows for training and testing
"""

import pandas as pd
import os
import sys
from pathlib import Path

def check_file_exists(filepath):
    """Check if the source file exists"""
    if not os.path.exists(filepath):
        print(f"❌ Error: File not found: {filepath}")
        return False
    
    file_size_gb = os.path.getsize(filepath) / (1024**3)
    print(f"✅ Found file: {filepath}")
    print(f"   Size: {file_size_gb:.2f} GB")
    return True

def get_csv_info(filepath, nrows=5):
    """Get basic information about the CSV file"""
    print("\n📊 Analyzing CSV structure...")
    
    try:
        # Read just the header to get column names
        df_sample = pd.read_csv(filepath, nrows=nrows)
        
        print(f"   Shape: {df_sample.shape}")
        print(f"   Columns: {list(df_sample.columns)}")
        print(f"   Data types:\n{df_sample.dtypes}")
        
        # Show first few rows
        print(f"\n   First {nrows} rows:")
        print(df_sample.head(nrows).to_string())
        
        return df_sample.columns.tolist()
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return None

def create_samples(filepath, output_dir, sample_sizes=[100, 1000, 10000]):
    """Create samples of different sizes"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🎯 Creating samples in: {output_dir}")
    
    for sample_size in sample_sizes:
        output_file = os.path.join(output_dir, f"fakenews_sample_{sample_size}.csv")
        print(f"\n   Creating {sample_size} row sample...")
        
        try:
            # Read specified number of rows (including header)
            df_sample = pd.read_csv(filepath, nrows=sample_size)
            
            # Save to CSV
            df_sample.to_csv(output_file, index=False)
            
            # Get file size
            file_size_mb = os.path.getsize(output_file) / (1024**2)
            print(f"   ✅ Saved: {output_file}")
            print(f"      Rows: {len(df_sample)}")
            print(f"      Size: {file_size_mb:.2f} MB")
            
            # Show sample statistics
            print(f"      Columns: {len(df_sample.columns)}")
            if 'label' in df_sample.columns:
                label_counts = df_sample['label'].value_counts()
                print(f"      Label distribution:\n{label_counts}")
            
        except Exception as e:
            print(f"❌ Error creating {sample_size} sample: {e}")

def create_balanced_sample(filepath, output_dir, target_size=1000, label_column='label'):
    """Create a balanced sample with equal class distribution"""
    
    print(f"\n⚖️ Creating balanced sample ({target_size} rows)...")
    
    try:
        # Read in chunks to find class distribution
        chunk_size = 10000
        class_counts = {}
        chunks = []
        
        print("   Analyzing class distribution...")
        for chunk in pd.read_csv(filepath, chunksize=chunk_size):
            if label_column in chunk.columns:
                # Count classes in this chunk
                chunk_counts = chunk[label_column].value_counts().to_dict()
                for cls, count in chunk_counts.items():
                    class_counts[cls] = class_counts.get(cls, 0) + count
            
            chunks.append(chunk)
            if sum(class_counts.values()) > target_size * 10:  # Read enough to get good distribution
                break
        
        print(f"   Class distribution: {class_counts}")
        
        # Calculate target per class
        classes = list(class_counts.keys())
        target_per_class = target_size // len(classes)
        print(f"   Target per class: {target_per_class}")
        
        # Collect balanced samples
        balanced_samples = []
        samples_per_class = {cls: 0 for cls in classes}
        
        # Reset and read again to collect balanced samples
        for chunk in pd.read_csv(filepath, chunksize=chunk_size):
            for cls in classes:
                if samples_per_class[cls] < target_per_class:
                    class_samples = chunk[chunk[label_column] == cls]
                    needed = target_per_class - samples_per_class[cls]
                    
                    if len(class_samples) > 0:
                        take = min(needed, len(class_samples))
                        balanced_samples.append(class_samples.head(take))
                        samples_per_class[cls] += take
            
            # Check if we have enough samples
            if all(count >= target_per_class for count in samples_per_class.values()):
                break
        
        # Combine all samples
        if balanced_samples:
            df_balanced = pd.concat(balanced_samples, ignore_index=True)
            output_file = os.path.join(output_dir, "fakenews_balanced_1000.csv")
            df_balanced.to_csv(output_file, index=False)
            
            file_size_mb = os.path.getsize(output_file) / (1024**2)
            print(f"   ✅ Saved balanced sample: {output_file}")
            print(f"      Rows: {len(df_balanced)}")
            print(f"      Size: {file_size_mb:.2f} MB")
            print(f"      Final distribution:\n{df_balanced[label_column].value_counts()}")
        else:
            print("❌ Could not create balanced sample")
            
    except Exception as e:
        print(f"❌ Error creating balanced sample: {e}")

def main():
    """Main function"""
    print("=" * 60)
    print("FakeNewsDataset Sampler")
    print("=" * 60)
    
    # File paths
    source_file = r"C:\Users\22130\Downloads\news.csv\news_cleaned_2018_02_13.csv"
    output_dir = r"C:\Users\22130\.openclaw\workspace\lessons\data-analysis-progress\data\FakeNewsCorpus"
    
    # Check if source file exists
    if not check_file_exists(source_file):
        sys.exit(1)
    
    # Get CSV info
    columns = get_csv_info(source_file)
    if not columns:
        print("❌ Could not read CSV information")
        sys.exit(1)
    
    # Create samples
    create_samples(source_file, output_dir)
    
    # Try to create balanced sample if we can identify label column
    label_candidates = ['label', 'Label', 'LABEL', 'type', 'Type', 'TYPE', 'category', 'Category', 'CATEGORY']
    label_column = None
    
    for candidate in label_candidates:
        if candidate in columns:
            label_column = candidate
            break
    
    if label_column:
        create_balanced_sample(source_file, output_dir, label_column=label_column)
    else:
        print(f"\n⚠️ Could not identify label column from: {columns}")
        print("   Skipping balanced sample creation")
    
    print("\n" + "=" * 60)
    print("✅ Sampling complete!")
    print("=" * 60)
    
    # List created files
    print(f"\n📁 Created files in {output_dir}:")
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith('.csv'):
                filepath = os.path.join(output_dir, file)
                file_size_mb = os.path.getsize(filepath) / (1024**2)
                # Count rows in the file
                df = pd.read_csv(filepath)
                print(f"   {file} - {len(df)} rows, {file_size_mb:.2f} MB")

if __name__ == "__main__":
    main()