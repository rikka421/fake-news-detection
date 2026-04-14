"""
Data loading module for news classification project
Handles loading of FakeNewsCorpus and other datasets
"""

import pandas as pd
import os
from pathlib import Path
from typing import Union, Optional, Dict, List


class DataLoader:
    """Loader for news classification datasets"""
    
    def __init__(self, data_dir: Union[str, Path] = "../data"):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing dataset files
        """
        self.data_dir = Path(data_dir)
        
    def load_fakenews_sample(self, sample_size: str = "small") -> pd.DataFrame:
        """
        Load a FakeNewsCorpus sample
        
        Args:
            sample_size: One of 'test' (10), 'tiny' (100), 
                        'small' (1000), 'medium' (5000)
                        
        Returns:
            DataFrame with news articles
        """
        sample_files = {
            "test": "fakenews_test_10.csv",
            "tiny": "fakenews_tiny_100.csv", 
            "small": "fakenews_small_1000.csv",
            "medium": "fakenews_medium_5000.csv",
            "stratified": "fakenews_stratified_1000.csv"
        }
        
        if sample_size not in sample_files:
            raise ValueError(f"sample_size must be one of {list(sample_files.keys())}")
        
        file_path = self.data_dir / "FakeNewsCorpus" / sample_files[sample_size]
        
        if not file_path.exists():
            raise FileNotFoundError(f"Sample file not found: {file_path}")
        
        print(f"Loading {sample_size} sample from {file_path}")
        # Use error_bad_lines=False to skip problematic lines
        df = pd.read_csv(file_path, on_bad_lines='skip')
        print(f"Loaded {len(df)} articles with {len(df.columns)} columns")
        
        return df
    
    def get_label_distribution(self, df: pd.DataFrame) -> pd.Series:
        """
        Get distribution of labels in dataset
        
        Args:
            df: DataFrame with 'type' column
            
        Returns:
            Series with label counts
        """
        if 'type' not in df.columns:
            raise ValueError("DataFrame must have 'type' column")
        
        return df['type'].value_counts()
    
    def get_text_columns(self) -> List[str]:
        """
        Get list of text columns available for analysis
        
        Returns:
            List of column names containing text
        """
        return ['content', 'title', 'summary', 'meta_description']
    
    def create_train_test_split(self, df: pd.DataFrame, test_size: float = 0.2, 
                               random_state: int = 42) -> Dict[str, pd.DataFrame]:
        """
        Create train/test split for classification
        
        Args:
            df: Input DataFrame
            test_size: Proportion for test set
            random_state: Random seed
            
        Returns:
            Dictionary with 'train' and 'test' DataFrames
        """
        from sklearn.model_selection import train_test_split
        
        # For simplicity, using content as features and type as labels
        X = df['content'].fillna('')
        y = df['type'].fillna('unknown')
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Reconstruct DataFrames
        train_df = df.loc[X_train.index].copy()
        test_df = df.loc[X_test.index].copy()
        
        return {
            'train': train_df,
            'test': test_df,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
    
    def analyze_dataset(self, df: pd.DataFrame) -> Dict:
        """
        Perform basic analysis on dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'total_articles': len(df),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'label_distribution': self.get_label_distribution(df).to_dict(),
            'text_length_stats': {}
        }
        
        # Analyze text length
        if 'content' in df.columns:
            content_lengths = df['content'].fillna('').str.len()
            analysis['text_length_stats']['content'] = {
                'mean': content_lengths.mean(),
                'std': content_lengths.std(),
                'min': content_lengths.min(),
                'max': content_lengths.max()
            }
        
        return analysis


def load_sample_data(sample_size: str = "small") -> pd.DataFrame:
    """
    Convenience function to load sample data
    
    Args:
        sample_size: Size of sample to load
        
    Returns:
        DataFrame with news articles
    """
    loader = DataLoader()
    return loader.load_fakenews_sample(sample_size)


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    
    # Load small sample
    df = loader.load_fakenews_sample("small")
    
    # Analyze dataset
    analysis = loader.analyze_dataset(df)
    print(f"Dataset analysis:")
    print(f"  Total articles: {analysis['total_articles']}")
    print(f"  Labels: {analysis['label_distribution']}")
    
    # Create train/test split
    splits = loader.create_train_test_split(df)
    print(f"\nTrain/test split:")
    print(f"  Train: {len(splits['train'])} articles")
    print(f"  Test: {len(splits['test'])} articles")