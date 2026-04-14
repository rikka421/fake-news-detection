#!/usr/bin/env python3
"""
Repair corrupted CSV files
"""

import pandas as pd
import os

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

def repair_csv(input_path, output_path):
    """Attempt to repair a CSV file"""
    print(f"Attempting to repair {os.path.basename(input_path)}")
    
    try:
        # Try reading with error handling
        df = pd.read_csv(input_path, on_bad_lines='skip', engine='python')
        print(f"Successfully read {len(df)} rows")
        
        # Save cleaned version
        df.to_csv(output_path, index=False)
        print(f"Cleaned CSV saved to {os.path.basename(output_path)}")
        
        return df
    except Exception as e:
        print(f"Error reading CSV: {e}")
        
        # Try manual repair
        print("Attempting manual repair...")
        try:
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except:
            with open(input_path, 'r', encoding='latin-1', errors='ignore') as f:
                lines = f.readlines()
        
        print(f"Original file has {len(lines)} lines")
        
        # Keep only lines that look like CSV (have reasonable number of commas)
        cleaned_lines = []
        header = None
        
        for i, line in enumerate(lines):
            if i == 0 and ',' in line:
                # This is the header
                header = line
                cleaned_lines.append(line)
                print(f"Found header: {line[:100]}...")
            elif ',' in line:
                # Count commas
                comma_count = line.count(',')
                if 10 <= comma_count <= 30:  # Reasonable range for this dataset
                    cleaned_lines.append(line)
        
        print(f"Kept {len(cleaned_lines)} lines after cleaning")
        
        if header and len(cleaned_lines) > 1:
            # Write cleaned file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.writelines(cleaned_lines)
            
            print(f"Manually cleaned CSV saved to {os.path.basename(output_path)}")
            
            # Try reading it back
            try:
                df = pd.read_csv(output_path)
                print(f"Successfully read cleaned CSV: {len(df)} rows")
                return df
            except Exception as e2:
                print(f"Still can't read cleaned CSV: {e2}")
        
        return None

# Test with different files
files_to_repair = [
    os.path.join(script_dir, "data/FakeNewsCorpus/fakenews_test_10.csv"),
    os.path.join(script_dir, "data/FakeNewsCorpus/fakenews_tiny_100.csv"),
    os.path.join(script_dir, "data/FakeNewsCorpus/fakenews_small_1000.csv"),
]

for input_file in files_to_repair:
    if os.path.exists(input_file):
        print("\n" + "="*60)
        print(f"Processing: {os.path.basename(input_file)}")
        print(f"File size: {os.path.getsize(input_file)} bytes")
        
        output_file = input_file.replace('.csv', '_repaired.csv')
        df = repair_csv(input_file, output_file)
        
        if df is not None:
            print(f"\nSuccessfully repaired {os.path.basename(input_file)}")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {df.columns.tolist()}")
            
            # Show sample
            if len(df) > 0:
                print(f"\nSample data:")
                # Try to show type and title if they exist
                cols_to_show = []
                if 'type' in df.columns:
                    cols_to_show.append('type')
                if 'title' in df.columns:
                    cols_to_show.append('title')
                if cols_to_show:
                    print(df[cols_to_show].head(3))
        else:
            print(f"\nCould not repair {os.path.basename(input_file)}")
    else:
        print(f"\nFile not found: {os.path.basename(input_file)}")