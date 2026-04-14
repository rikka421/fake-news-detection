#!/usr/bin/env python3
import pandas as pd
import os

data_path = os.path.join(os.path.dirname(__file__), "data/FakeNewsCorpus/fakenews_small_1000_repaired.csv")
df = pd.read_csv(data_path)

print("Data shape:", df.shape)
print("\nColumn 'type' values:")
print(df['type'].value_counts())
print("\nFirst 10 'type' values:")
print(df['type'].head(10).tolist())

print("\nChecking for mixed data...")
# Look for values that don't look like labels
valid_labels = ['fake', 'bias', 'conspiracy', 'hate', 'rumor', 'satire', 'unreliable', 'clickbait', 'reliable', 'political', 'junksci', 'state']
for label in df['type'].unique():
    if label not in valid_labels:
        print(f"  Suspicious 'type' value: '{label}'")
        # Show the corresponding content
        row = df[df['type'] == label].iloc[0]
        print(f"    Corresponding content preview: {str(row['content'])[:100]}...")