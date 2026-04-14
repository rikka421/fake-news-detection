#!/usr/bin/env python3
"""
Fix CSV parsing issues by reading with different methods
"""

import pandas as pd
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

def try_read_csv(filepath):
    """Try different methods to read CSV"""
    print(f"Trying to read: {filepath}")
    print(f"File size: {os.path.getsize(filepath)} bytes")
    
    methods = [
        ("Standard read", lambda: pd.read_csv(filepath)),
        ("Skip bad lines", lambda: pd.read_csv(filepath, on_bad_lines='skip')),
        ("Engine python", lambda: pd.read_csv(filepath, engine='python')),
        ("Quoting", lambda: pd.read_csv(filepath, quoting=3)),  # QUOTE_NONE
        ("Escape char", lambda: pd.read_csv(filepath, escapechar='\\')),
    ]
    
    for name, func in methods:
        try:
            print(f"\nTrying method: {name}")
            df = func()
            print(f"  Success! Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            if len(df) > 0:
                print(f"  First few rows:")
                print(df.head(2).to_string())
            return df
        except Exception as e:
            print(f"  Failed: {str(e)[:100]}...")
    
    return None

# Test with different files
files = [
    os.path.join(script_dir, "data/FakeNewsCorpus/fakenews_test_10.csv"),
    os.path.join(script_dir, "data/FakeNewsCorpus/fakenews_tiny_100.csv"),
    os.path.join(script_dir, "data/FakeNewsCorpus/fakenews_small_1000.csv"),
]

for file in files:
    if os.path.exists(file):
        print("\n" + "="*60)
        print(f"Testing file: {os.path.basename(file)}")
        print("="*60)
        
        # Try to read
        df = try_read_csv(file)
        
        if df is not None:
            print(f"\n✅ Successfully loaded {os.path.basename(file)}")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {df.columns.tolist()}")
            
            # Save a clean version
            clean_file = file.replace('.csv', '_clean.csv')
            df.to_csv(clean_file, index=False)
            print(f"   Clean version saved to: {clean_file}")
        else:
            print(f"\n❌ Could not load {os.path.basename(file)}")
    else:
        print(f"\nFile not found: {os.path.basename(file)}")