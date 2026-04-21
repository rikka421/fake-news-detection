from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import load_news_dataset, project_root, simple_tokenize, split_dataset
from .models import CNNTextClassifier, TransformerTextClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Vocabulary:
    def __init__(self, stoi: Dict[str, int]) -> None:
        self.stoi = stoi
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"

    @property
    def pad_index(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def unk_index(self) -> int:
        return self.stoi[self.unk_token]

    def encode(self, text: str, max_length: int) -> List[int]:
        tokens = list(simple_tokenize(text))[:max_length]
        token_ids = [self.stoi.get(token, self.unk_index) for token in tokens]
        if len(token_ids) < max_length:
            token_ids.extend([self.pad_index] * (max_length - len(token_ids)))
        return token_ids

    def __len__(self) -> int:
        return len(self.stoi)


def build_vocabulary(texts: Iterable[str], min_frequency: int = 2, max_tokens: int = 20000) -> Vocabulary:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(simple_tokenize(text))

    stoi = {"<pad>": 0, "<unk>": 1}
    for token, count in counter.most_common(max_tokens - len(stoi)):
        if count < min_frequency:
            continue
        stoi[token] = len(stoi)
    return Vocabulary(stoi)


class EncodedNewsDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocabulary: Vocabulary, max_length: int) -> None:
        self.texts = texts
        self.labels = labels
        self.vocabulary = vocabulary
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor(self.vocabulary.encode(self.texts[index], self.max_length), dtype=torch.long)
        attention_mask = (input_ids != self.vocabulary.pad_index).long()
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
        }


@dataclass
class BenchmarkConfig:
    dataset: str = "medium"
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    max_length: int = 64
    embedding_dim: int = 128
    min_frequency: int = 2
    seed: int = 42
    models: tuple[str, ...] = (
        "logistic_regression",
        "svm",
        "random_forest",
        "naive_bayes",
        "cnn",
        "transformer",
    )
    text_mode: str = "all_fields"
    mask_label_tokens: bool = False
    split_mode: str = "random"


def _frame_to_lists(frame) -> tuple[List[str], List[str]]:
    return frame["text"].tolist(), frame["label"].tolist()


def _encode_labels(*label_groups: List[str]) -> tuple[Dict[str, int], List[List[int]]]:
    unique_labels = sorted({label for group in label_groups for label in group})
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    encoded = [[label_to_index[label] for label in group] for group in label_groups]
    return label_to_index, encoded


def _class_weight_tensor(encoded_labels: List[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(encoded_labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_logistic_regression(train_frame, validation_frame, test_frame, seed: int) -> Dict[str, float]:
    train_texts, train_labels = _frame_to_lists(train_frame)
    validation_texts, validation_labels = _frame_to_lists(validation_frame)
    test_texts, test_labels = _frame_to_lists(test_frame)
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
    classifier = LogisticRegression(max_iter=1500, random_state=seed, class_weight="balanced")
    train_features = vectorizer.fit_transform(train_texts)
    classifier.fit(train_features, train_labels)

    validation_predictions = classifier.predict(vectorizer.transform(validation_texts))
    test_predictions = classifier.predict(vectorizer.transform(test_texts))
    return {
        "validation_accuracy": accuracy_score(validation_labels, validation_predictions),
        "validation_macro_f1": f1_score(validation_labels, validation_predictions, average="macro"),
        "test_accuracy": accuracy_score(test_labels, test_predictions),
        "test_macro_f1": f1_score(test_labels, test_predictions, average="macro"),
    }


def _build_classical_features(train_frame, validation_frame, test_frame):
    train_texts, train_labels = _frame_to_lists(train_frame)
    validation_texts, validation_labels = _frame_to_lists(validation_frame)
    test_texts, test_labels = _frame_to_lists(test_frame)

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
    train_features = vectorizer.fit_transform(train_texts)
    validation_features = vectorizer.transform(validation_texts)
    test_features = vectorizer.transform(test_texts)
    return {
        "train_features": train_features,
        "validation_features": validation_features,
        "test_features": test_features,
        "train_labels": train_labels,
        "validation_labels": validation_labels,
        "test_labels": test_labels,
    }


def _summarize_predictions(validation_labels, validation_predictions, test_labels, test_predictions) -> Dict[str, float]:
    return {
        "validation_accuracy": accuracy_score(validation_labels, validation_predictions),
        "validation_macro_f1": f1_score(validation_labels, validation_predictions, average="macro"),
        "test_accuracy": accuracy_score(test_labels, test_predictions),
        "test_macro_f1": f1_score(test_labels, test_predictions, average="macro"),
    }


def run_svm(train_frame, validation_frame, test_frame, seed: int) -> Dict[str, float]:
    features = _build_classical_features(train_frame, validation_frame, test_frame)
    classifier = LinearSVC(random_state=seed, class_weight="balanced")
    classifier.fit(features["train_features"], features["train_labels"])
    validation_predictions = classifier.predict(features["validation_features"])
    test_predictions = classifier.predict(features["test_features"])
    return _summarize_predictions(
        features["validation_labels"],
        validation_predictions,
        features["test_labels"],
        test_predictions,
    )


def run_random_forest(train_frame, validation_frame, test_frame, seed: int) -> Dict[str, float]:
    features = _build_classical_features(train_frame, validation_frame, test_frame)
    classifier = RandomForestClassifier(
        n_estimators=300,
        random_state=seed,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    classifier.fit(features["train_features"], features["train_labels"])
    validation_predictions = classifier.predict(features["validation_features"])
    test_predictions = classifier.predict(features["test_features"])
    return _summarize_predictions(
        features["validation_labels"],
        validation_predictions,
        features["test_labels"],
        test_predictions,
    )


def run_naive_bayes(train_frame, validation_frame, test_frame) -> Dict[str, float]:
    features = _build_classical_features(train_frame, validation_frame, test_frame)
    classifier = MultinomialNB()
    classifier.fit(features["train_features"], features["train_labels"])
    validation_predictions = classifier.predict(features["validation_features"])
    test_predictions = classifier.predict(features["test_features"])
    return _summarize_predictions(
        features["validation_labels"],
        validation_predictions,
        features["test_labels"],
        test_predictions,
    )


def _train_epoch(model, data_loader, optimizer, criterion, device: torch.device) -> float:
    model.train()
    losses = []
    for batch in data_loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        loss = criterion(logits, batch["labels"].to(device))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


def _evaluate(model, data_loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    predictions: List[int] = []
    labels: List[int] = []
    with torch.no_grad():
        for batch in data_loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }


def _build_model(model_name: str, vocabulary: Vocabulary, num_classes: int, config: BenchmarkConfig):
    if model_name == "cnn":
        return CNNTextClassifier(
            vocab_size=len(vocabulary),
            num_classes=num_classes,
            pad_index=vocabulary.pad_index,
            embedding_dim=config.embedding_dim,
        )
    if model_name == "transformer":
        return TransformerTextClassifier(
            vocab_size=len(vocabulary),
            num_classes=num_classes,
            pad_index=vocabulary.pad_index,
            embedding_dim=config.embedding_dim,
            max_length=config.max_length,
        )
    raise ValueError(f"Unsupported deep model: {model_name}")


def run_deep_model(model_name: str, train_frame, validation_frame, test_frame, config: BenchmarkConfig) -> Dict[str, float]:
    train_texts, train_labels = _frame_to_lists(train_frame)
    validation_texts, validation_labels = _frame_to_lists(validation_frame)
    test_texts, test_labels = _frame_to_lists(test_frame)
    label_to_index, encoded_groups = _encode_labels(train_labels, validation_labels, test_labels)
    encoded_train, encoded_validation, encoded_test = encoded_groups
    vocabulary = build_vocabulary(train_texts, min_frequency=config.min_frequency)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = EncodedNewsDataset(train_texts, encoded_train, vocabulary, config.max_length)
    validation_dataset = EncodedNewsDataset(validation_texts, encoded_validation, vocabulary, config.max_length)
    test_dataset = EncodedNewsDataset(test_texts, encoded_test, vocabulary, config.max_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = _build_model(model_name, vocabulary, len(label_to_index), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    class_weights = _class_weight_tensor(encoded_train, len(label_to_index)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_validation_score = -1.0
    best_state = None
    train_loss = 0.0
    for _ in range(config.epochs):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        validation_metrics = _evaluate(model, validation_loader, device)
        if validation_metrics["macro_f1"] > best_validation_score:
            best_validation_score = validation_metrics["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    validation_metrics = _evaluate(model, validation_loader, device)
    test_metrics = _evaluate(model, test_loader, device)
    return {
        "train_loss": train_loss,
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_macro_f1": validation_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "device": str(device),
        "vocab_size": len(vocabulary),
    }


def run_benchmark(config: BenchmarkConfig) -> Dict[str, object]:
    set_seed(config.seed)
    frame = load_news_dataset(
        config.dataset,
        text_mode=config.text_mode,
        mask_label_tokens=config.mask_label_tokens,
    )
    train_frame, validation_frame, test_frame = split_dataset(
        frame,
        random_state=config.seed,
        split_mode=config.split_mode,
    )
    results: Dict[str, Dict[str, float]] = {}

    for model_name in config.models:
        if model_name == "logistic_regression":
            results[model_name] = run_logistic_regression(
                train_frame,
                validation_frame,
                test_frame,
                seed=config.seed,
            )
        elif model_name == "svm":
            results[model_name] = run_svm(
                train_frame,
                validation_frame,
                test_frame,
                seed=config.seed,
            )
        elif model_name == "random_forest":
            results[model_name] = run_random_forest(
                train_frame,
                validation_frame,
                test_frame,
                seed=config.seed,
            )
        elif model_name == "naive_bayes":
            results[model_name] = run_naive_bayes(
                train_frame,
                validation_frame,
                test_frame,
            )
        else:
            results[model_name] = run_deep_model(
                model_name,
                train_frame,
                validation_frame,
                test_frame,
                config,
            )

    return {
        "config": asdict(config),
        "dataset_size": len(frame),
        "label_count": int(frame["label"].nunique()),
        "splits": {
            "train": len(train_frame),
            "validation": len(validation_frame),
            "test": len(test_frame),
            "unique_group_content": {
                "train": int(train_frame["group_content"].nunique()),
                "validation": int(validation_frame["group_content"].nunique()),
                "test": int(test_frame["group_content"].nunique()),
            },
            "unique_group_title_content": {
                "train": int(train_frame["group_title_content"].nunique()),
                "validation": int(validation_frame["group_title_content"].nunique()),
                "test": int(test_frame["group_title_content"].nunique()),
            },
        },
        "results": results,
    }


def write_benchmark_report(report: Dict[str, object], output_path: str | Path | None = None) -> Path:
    destination = Path(output_path) if output_path else project_root() / "artifacts" / "results" / "benchmark_results.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return destination
