#!/usr/bin/env python3
"""
Consolidated Experiment Runner for News Classification Project

This script combines functionality from:
- baseline_model.py
- final_test_fixed.py  
- test_real_data.py
- regenerate_samples.py
- sample_dataset.py

Provides a unified interface for running experiments with different datasets and models.
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Load and preprocess datasets"""
    
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent / "data" / "FakeNewsCorpus"
        
    def list_available_datasets(self):
        """List all available dataset files"""
        datasets = []
        if self.data_dir.exists():
            for file in self.data_dir.glob("*.csv"):
                datasets.append({
                    'name': file.stem,
                    'path': file,
                    'size': file.stat().st_size
                })
        return datasets
    
    def load_dataset(self, dataset_name="fakenews_small_1000"):
        """Load a specific dataset"""
        file_path = self.data_dir / f"{dataset_name}.csv"
        
        if not file_path.exists():
            # Try with _repaired suffix
            file_path = self.data_dir / f"{dataset_name}_repaired.csv"
            
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_name}")
        
        print(f"Loading dataset: {dataset_name}")
        df = pd.read_csv(file_path)
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        return df
    
    def preprocess_data(self, df, text_column='content', label_column='type'):
        """Preprocess data for training"""
        # Handle missing values
        df = df.dropna(subset=[text_column, label_column])
        
        # Clean text
        df[text_column] = df[text_column].astype(str).str.lower()
        
        # Get features and labels
        X = df[text_column].values
        y = df[label_column].values
        
        return X, y

class ModelTrainer:
    """Train and evaluate models"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'svm': SVC(kernel='linear', random_state=42),
            'naive_bayes': MultinomialNB()
        }
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
    
    def train_models(self, X_train, y_train):
        """Train all models"""
        print("Vectorizing text data...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        print(f"  Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        trained_models = {}
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_vec, y_train)
            trained_models[name] = model
            
        return trained_models
    
    def evaluate_models(self, trained_models, X_test, y_test):
        """Evaluate all models"""
        X_test_vec = self.vectorizer.transform(X_test)
        
        results = {}
        for name, model in trained_models.items():
            print(f"\nEvaluating {name}...")
            y_pred = model.predict(X_test_vec)
            
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            results[name] = {
                'accuracy': accuracy,
                'report': report,
                'predictions': y_pred
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            
        return results
    
    def save_models(self, trained_models, save_dir="models"):
        """Save trained models"""
        save_path = Path(__file__).parent.parent / save_dir
        save_path.mkdir(exist_ok=True)
        
        # Save vectorizer
        joblib.dump(self.vectorizer, save_path / "vectorizer.pkl")
        
        # Save models
        for name, model in trained_models.items():
            joblib.dump(model, save_path / f"{name}_model.pkl")
        
        print(f"\nModels saved to: {save_path}")

class ExperimentRunner:
    """Main experiment runner"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.model_trainer = ModelTrainer()
    
    def run_experiment(self, dataset_name="fakenews_small_1000", test_size=0.2):
        """Run complete experiment pipeline"""
        print("=" * 60)
        print(f"Running experiment with dataset: {dataset_name}")
        print("=" * 60)
        
        # Load data
        df = self.data_loader.load_dataset(dataset_name)
        X, y = self.data_loader.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\nData split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")
        print(f"  Unique labels: {len(np.unique(y))}")
        
        # Train models
        trained_models = self.model_trainer.train_models(X_train, y_train)
        
        # Evaluate models
        results = self.model_trainer.evaluate_models(trained_models, X_test, y_test)
        
        # Save models
        self.model_trainer.save_models(trained_models)
        
        return results
    
    def compare_datasets(self, dataset_names=None):
        """Compare performance across different datasets"""
        if dataset_names is None:
            dataset_names = ["fakenews_tiny_100", "fakenews_small_1000", "fakenews_medium_5000"]
        
        comparison_results = {}
        
        for dataset_name in dataset_names:
            print(f"\n{'='*60}")
            print(f"Testing dataset: {dataset_name}")
            print('='*60)
            
            try:
                results = self.run_experiment(dataset_name)
                comparison_results[dataset_name] = results
            except Exception as e:
                print(f"  Error with {dataset_name}: {e}")
        
        return comparison_results

def main():
    parser = argparse.ArgumentParser(description="News Classification Experiment Runner")
    parser.add_argument("--dataset", type=str, default="fakenews_small_1000",
                       help="Dataset name to use (default: fakenews_small_1000)")
    parser.add_argument("--compare", action="store_true",
                       help="Compare performance across multiple datasets")
    parser.add_argument("--list-datasets", action="store_true",
                       help="List available datasets")
    
    args = parser.parse_args()
    
    runner = ExperimentRunner()
    
    if args.list_datasets:
        datasets = runner.data_loader.list_available_datasets()
        print("\nAvailable datasets:")
        for dataset in datasets:
            print(f"  {dataset['name']}: {dataset['size']:,} bytes")
        return
    
    if args.compare:
        runner.compare_datasets()
    else:
        runner.run_experiment(args.dataset)

if __name__ == "__main__":
    main()