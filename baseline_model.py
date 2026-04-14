#!/usr/bin/env python3
"""
Baseline model for news classification
This implements traditional ML models for text classification
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from datetime import datetime

class NewsClassifier:
    """Baseline news classifier using traditional ML"""
    
    def __init__(self, model_type='logistic_regression'):
        """
        Initialize classifier
        
        Args:
            model_type: Type of model to use
                - 'logistic_regression'
                - 'random_forest'
                - 'svm'
                - 'naive_bayes'
        """
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)  # Use unigrams and bigrams
        )
        self.model = None
        self.label_encoder = None
        
    def create_synthetic_data(self, n_samples=1000):
        """
        Create synthetic data for testing when real data is unavailable
        
        Args:
            n_samples: Number of samples to create
            
        Returns:
            DataFrame with text and labels
        """
        print("Creating synthetic dataset for testing...")
        
        # News categories (simplified)
        categories = ['fake', 'bias', 'conspiracy', 'reliable', 'political']
        
        # Sample texts for each category
        category_texts = {
            'fake': [
                "Breaking: Alien invasion confirmed by government sources",
                "Miracle cure discovered for all diseases",
                "Celebrity reveals secret government conspiracy"
            ],
            'bias': [
                "Study shows clear advantage for one political party",
                "Experts agree on controversial policy direction",
                "New evidence supports established viewpoint"
            ],
            'conspiracy': [
                "Hidden truth about major historical event",
                "Secret society controls world events",
                "Unexplained phenomenon points to cover-up"
            ],
            'reliable': [
                "New study published in peer-reviewed journal",
                "Government releases official statistics report",
                "Experts provide balanced analysis of current events"
            ],
            'political': [
                "Candidate announces policy platform",
                "Debate highlights differences between parties",
                "Poll shows changing voter preferences"
            ]
        }
        
        # Generate synthetic data
        texts = []
        labels = []
        
        for _ in range(n_samples):
            category = np.random.choice(categories)
            base_text = np.random.choice(category_texts[category])
            
            # Add some variation
            variations = [
                "In recent developments,",
                "According to sources,",
                "Reports indicate that",
                "New information reveals",
                "Analysis shows"
            ]
            
            text = f"{np.random.choice(variations)} {base_text}"
            texts.append(text)
            labels.append(category)
        
        return pd.DataFrame({
            'text': texts,
            'label': labels
        })
    
    def load_data(self, data_path=None):
        """
        Load data from CSV or create synthetic data
        
        Args:
            data_path: Path to CSV file (if None, use synthetic)
            
        Returns:
            DataFrame with text and label columns
        """
        if data_path and os.path.exists(data_path):
            print(f"Loading data from {data_path}")
            try:
                df = pd.read_csv(data_path)
                # Assume text column is 'content' and label column is 'type'
                if 'content' in df.columns and 'type' in df.columns:
                    df = df[['content', 'type']].copy()
                    df.columns = ['text', 'label']
                    df = df.dropna()
                    print(f"Loaded {len(df)} samples")
                    return df
                else:
                    print("Required columns not found, using synthetic data")
            except Exception as e:
                print(f"Error loading data: {e}, using synthetic data")
        
        # Use synthetic data as fallback
        return self.create_synthetic_data(n_samples=1000)
    
    def preprocess_data(self, df):
        """
        Preprocess text data
        
        Args:
            df: DataFrame with 'text' and 'label' columns
            
        Returns:
            X (features), y (labels), and label mapping
        """
        # Clean text
        df['text_clean'] = df['text'].fillna('').astype(str).str.lower()
        
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['label'])
        
        # Get text features
        X = self.vectorizer.fit_transform(df['text_clean'])
        
        return X, y
    
    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train the model
        
        Args:
            X: Feature matrix
            y: Labels
            test_size: Proportion for test set
            random_state: Random seed
            
        Returns:
            Trained model and evaluation metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Initialize model
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                class_weight='balanced'
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                class_weight='balanced'
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='linear',
                random_state=random_state,
                class_weight='balanced'
            )
        elif self.model_type == 'naive_bayes':
            self.model = MultinomialNB()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"Training {self.model_type}...")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, 
            y_pred, 
            target_names=self.label_encoder.classes_
        )
        
        print(f"\nModel: {self.model_type}")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        return {
            'accuracy': accuracy,
            'report': report,
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def predict(self, texts):
        """
        Predict labels for new texts
        
        Args:
            texts: List of text strings
            
        Returns:
            Predicted labels
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained yet")
        
        # Transform texts
        X = self.vectorizer.transform(texts)
        
        # Predict
        y_pred = self.model.predict(X)
        
        # Decode labels
        labels = self.label_encoder.inverse_transform(y_pred)
        
        return labels
    
    def save_model(self, path='models'):
        """
        Save trained model
        
        Args:
            path: Directory to save model
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        os.makedirs(path, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(path, f'{self.model_type}_{timestamp}.joblib')
        
        # Save all components
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
        
        return model_path
    
    @staticmethod
    def load_model(model_path):
        """
        Load trained model
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded NewsClassifier instance
        """
        model_data = joblib.load(model_path)
        
        # Create classifier
        classifier = NewsClassifier(model_type=model_data['model_type'])
        classifier.model = model_data['model']
        classifier.vectorizer = model_data['vectorizer']
        classifier.label_encoder = model_data['label_encoder']
        
        print(f"Model loaded from {model_path}")
        return classifier


def run_experiment(data_path=None):
    """
    Run complete experiment with different models
    
    Args:
        data_path: Path to data file (optional)
    """
    print("=" * 60)
    print("NEWS CLASSIFICATION BASELINE EXPERIMENT")
    print("=" * 60)
    
    # Test different models
    models = ['logistic_regression', 'random_forest', 'svm', 'naive_bayes']
    
    results = {}
    
    for model_type in models:
        print(f"\n{'='*40}")
        print(f"Training {model_type}")
        print('='*40)
        
        # Create and train classifier
        classifier = NewsClassifier(model_type=model_type)
        
        # Load data
        df = classifier.load_data(data_path)
        print(f"Dataset size: {len(df)} samples")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        
        # Preprocess
        X, y = classifier.preprocess_data(df)
        print(f"Feature matrix shape: {X.shape}")
        
        # Train and evaluate
        result = classifier.train(X, y)
        results[model_type] = result['accuracy']
        
        # Save model
        classifier.save_model('models')
    
    # Compare results
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    for model_type, accuracy in results.items():
        print(f"{model_type:25} Accuracy: {accuracy:.4f}")
    
    # Find best model
    best_model = max(results, key=results.get)
    print(f"\nBest model: {best_model} (Accuracy: {results[best_model]:.4f})")
    
    return results


if __name__ == "__main__":
    # Run experiment
    # You can provide a path to your data file, or it will use synthetic data
    # data_path = "data/FakeNewsCorpus/fakenews_small_1000.csv"
    data_path = None  # Use synthetic data for testing
    
    results = run_experiment(data_path)
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Fix CSV data loading issues")
    print("2. Try with real dataset (1000+ samples)")
    print("3. Experiment with different feature extraction methods")
    print("4. Try deep learning models (CNN, BERT, etc.)")