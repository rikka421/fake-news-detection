#!/usr/bin/env python3
"""
Regenerate sample datasets with proper CSV formatting
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def create_sample_dataset(n_samples, output_path):
    """Create a properly formatted sample dataset"""
    
    print(f"Creating sample dataset with {n_samples} rows...")
    
    # News categories from FakeNewsCorpus
    categories = [
        'fake', 'bias', 'conspiracy', 'hate', 'rumor', 
        'satire', 'unreliable', 'clickbait', 'reliable', 'political'
    ]
    
    # Sample domains
    domains = [
        'express.co.uk', 'barenakedislam.com', 'theguardian.com',
        'foxnews.com', 'cnn.com', 'bbc.com', 'nytimes.com',
        'breitbart.com', 'infowars.com', 'snopes.com'
    ]
    
    # Generate data
    data = []
    
    for i in range(n_samples):
        # Random category
        category = np.random.choice(categories, p=[0.15, 0.12, 0.10, 0.08, 0.10, 0.08, 0.12, 0.10, 0.10, 0.05])
        domain = np.random.choice(domains)
        
        # Generate sample content based on category
        if category == 'fake':
            title = f"Breaking: {np.random.choice(['Alien', 'Government', 'Scientific', 'Medical'])} discovery shocks experts"
            content = f"Researchers have made an astonishing discovery that challenges everything we know. {np.random.choice(['The implications are profound.', 'This changes everything.', 'Experts are baffled.'])}"
        elif category == 'bias':
            title = f"{np.random.choice(['Study shows', 'Report indicates', 'Analysis reveals'])} clear pattern in {np.random.choice(['political', 'economic', 'social'])} trends"
            content = f"The evidence clearly points to a specific conclusion that supports established viewpoints. {np.random.choice(['The data speaks for itself.', 'Patterns are unmistakable.', 'Correlations are strong.'])}"
        elif category == 'conspiracy':
            title = f"{np.random.choice(['Hidden truth about', 'Secret documents reveal', 'Whistleblower exposes'])} {np.random.choice(['major event', 'government program', 'corporate cover-up'])}"
            content = f"What they don't want you to know could change everything. {np.random.choice(['The pieces are coming together.', 'Evidence is mounting.', 'The truth will emerge.'])}"
        elif category == 'reliable':
            title = f"{np.random.choice(['New research published in', 'Peer-reviewed study from', 'Scientific paper in'])} {np.random.choice(['Nature', 'Science', 'The Lancet'])}"
            content = f"A comprehensive study using rigorous methodology has produced significant findings. {np.random.choice(['Results are statistically significant.', 'Methodology was robust.', 'Findings are reproducible.'])}"
        else:
            title = f"{np.random.choice(['News report on', 'Update about', 'Coverage of'])} {np.random.choice(['current events', 'recent developments', 'ongoing situation'])}"
            content = f"Standard news reporting with balanced coverage of the facts. {np.random.choice(['Multiple perspectives included.', 'Sources are verified.', 'Reporting is factual.'])}"
        
        # Create row
        row = {
            'id': i + 1,
            'domain': domain,
            'type': category,
            'url': f"https://{domain}/article/{i}",
            'content': content,
            'scraped_at': datetime.now().isoformat(),
            'inserted_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'title': title,
            'authors': 'Reporter Name',
            'keywords': 'news,article',
            'meta_keywords': category,
            'meta_description': f"Article about {category} news",
            'tags': category,
            'summary': f"Summary of {category} article",
            'source': domain
        }
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")
    
    return df

def main():
    """Regenerate all sample datasets"""
    
    output_dir = "data/FakeNewsCorpus/regenerated"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create samples of different sizes
    samples = [
        (10, "test_10"),
        (100, "tiny_100"),
        (1000, "small_1000"),
        (5000, "medium_5000")
    ]
    
    for n_samples, name in samples:
        output_path = os.path.join(output_dir, f"fakenews_{name}.csv")
        df = create_sample_dataset(n_samples, output_path)
        
        # Verify
        print(f"  Verified: {len(df)} rows, {len(df.columns)} columns")
        print(f"  Label distribution:")
        print(df['type'].value_counts())
        print()

if __name__ == "__main__":
    print("=" * 60)
    print("REGENERATING SAMPLE DATASETS")
    print("=" * 60)
    print("\nNote: This creates synthetic data for testing.")
    print("For real experiments, use the original FakeNewsCorpus dataset.")
    print()
    
    main()
    
    print("=" * 60)
    print("REGENERATION COMPLETE")
    print("=" * 60)
    print("\nNew datasets available in: data/FakeNewsCorpus/regenerated/")
    print("\nTo use these datasets:")
    print("1. Update the DataLoader to point to the new directory")
    print("2. Or modify your scripts to use the regenerated files")
    print("\nExample:")
    print("  data_path = 'data/FakeNewsCorpus/regenerated/fakenews_small_1000.csv'")