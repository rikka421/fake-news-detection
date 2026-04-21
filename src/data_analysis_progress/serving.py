from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from .data import load_news_dataset, project_root, split_dataset


@dataclass
class ServingArtifacts:
    vectorizer: TfidfVectorizer
    classifier: LinearSVC
    metrics: dict[str, float]
    metadata: dict[str, Any]


def default_model_path() -> Path:
    return project_root() / "models" / "serving" / "fakenews_svm_pipeline.joblib"


def train_and_export_serving_model(
    dataset: str = "balanced_medium",
    text_mode: str = "title_content",
    mask_label_tokens: bool = True,
    split_mode: str = "grouped_title_content",
    random_state: int = 42,
    model_path: Path | None = None,
) -> Path:
    frame = load_news_dataset(dataset, text_mode=text_mode, mask_label_tokens=mask_label_tokens)
    train_frame, validation_frame, test_frame = split_dataset(
        frame,
        random_state=random_state,
        split_mode=split_mode,
    )

    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), stop_words="english")
    classifier = LinearSVC(random_state=random_state, class_weight="balanced")

    train_x = vectorizer.fit_transform(train_frame["text"].tolist())
    classifier.fit(train_x, train_frame["label"].tolist())

    validation_pred = classifier.predict(vectorizer.transform(validation_frame["text"].tolist()))
    test_pred = classifier.predict(vectorizer.transform(test_frame["text"].tolist()))

    from sklearn.metrics import accuracy_score, f1_score

    metrics = {
        "validation_accuracy": float(accuracy_score(validation_frame["label"].tolist(), validation_pred)),
        "validation_macro_f1": float(f1_score(validation_frame["label"].tolist(), validation_pred, average="macro")),
        "test_accuracy": float(accuracy_score(test_frame["label"].tolist(), test_pred)),
        "test_macro_f1": float(f1_score(test_frame["label"].tolist(), test_pred, average="macro")),
    }

    destination = model_path or default_model_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "metrics": metrics,
        "metadata": {
            "dataset": dataset,
            "text_mode": text_mode,
            "mask_label_tokens": mask_label_tokens,
            "split_mode": split_mode,
            "random_state": random_state,
            "train_rows": int(len(train_frame)),
            "validation_rows": int(len(validation_frame)),
            "test_rows": int(len(test_frame)),
            "label_count": int(frame["label"].nunique()),
        },
    }
    joblib.dump(payload, destination)
    return destination


def load_serving_artifacts(model_path: Path | None = None) -> ServingArtifacts:
    source = model_path or default_model_path()
    if not source.exists():
        raise FileNotFoundError(
            f"Serving model not found at {source}. Run scripts/train_serving_model.py first."
        )

    payload = joblib.load(source)
    return ServingArtifacts(
        vectorizer=payload["vectorizer"],
        classifier=payload["classifier"],
        metrics=payload["metrics"],
        metadata=payload["metadata"],
    )


def predict_labels(artifacts: ServingArtifacts, texts: list[str]) -> list[str]:
    clean_texts = [text.strip() for text in texts if text and text.strip()]
    if not clean_texts:
        return []
    features = artifacts.vectorizer.transform(clean_texts)
    return artifacts.classifier.predict(features).tolist()


def predict_from_news_fields(
    artifacts: ServingArtifacts,
    title: str,
    content: str,
    meta_description: str = "",
    summary: str = "",
) -> str:
    merged = " ".join([title or "", content or "", meta_description or "", summary or ""]).strip()
    predictions = predict_labels(artifacts, [merged])
    if not predictions:
        raise ValueError("Empty text input after normalization")
    return predictions[0]


def predict_batch_from_dataframe(artifacts: ServingArtifacts, frame: pd.DataFrame) -> list[str]:
    text_series = (
        frame.get("title", "").fillna("").astype(str).str.strip()
        + " "
        + frame.get("content", "").fillna("").astype(str).str.strip()
        + " "
        + frame.get("meta_description", "").fillna("").astype(str).str.strip()
        + " "
        + frame.get("summary", "").fillna("").astype(str).str.strip()
    )
    return predict_labels(artifacts, text_series.tolist())
