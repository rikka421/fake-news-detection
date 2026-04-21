from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split


DATASET_ALIASES: Dict[str, str] = {
    "tiny": "fakenews_tiny_10_raw.csv",
    "small": "fakenews_small_100_raw.csv",
    "medium": "fakenews_medium_1000_raw.csv",
    "large": "fakenews_large_10000_raw.csv",
    "balanced_tiny": "fakenews_balanced_tiny_10_raw.csv",
    "balanced_small": "fakenews_balanced_small_100_raw.csv",
    "balanced_medium": "fakenews_balanced_medium_1000_raw.csv",
    "balanced_large": "fakenews_balanced_large_10000_raw.csv",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_dir() -> Path:
    return project_root() / "data" / "FakeNewsCorpus"


def resolve_dataset_path(dataset_name: str, data_dir: Path | None = None) -> Path:
    base_dir = data_dir or default_data_dir()
    relative_path = DATASET_ALIASES.get(dataset_name, dataset_name)
    dataset_path = base_dir / relative_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    return dataset_path


def _combine_text_columns(frame: pd.DataFrame) -> pd.Series:
    parts = []
    for column_name in ("title", "content", "meta_description", "summary"):
        if column_name in frame.columns:
            parts.append(frame[column_name].fillna("").astype(str).str.strip())

    if not parts:
        raise ValueError("Expected at least one text column among title/content/meta_description/summary")

    merged = parts[0]
    for series in parts[1:]:
        merged = (merged + " " + series).str.strip()
    return merged


def _mask_label_tokens(text: pd.Series, labels: pd.Series) -> pd.Series:
    masked = text.str.lower()
    for label in sorted(labels.dropna().astype(str).str.lower().unique()):
        masked = masked.str.replace(fr"\b{re.escape(label)}\b", " ", regex=True)
    return masked.str.replace(r"\s+", " ", regex=True).str.strip()


def normalize_dataset(frame: pd.DataFrame, text_mode: str = "all_fields", mask_label_tokens: bool = False) -> pd.DataFrame:
    if "type" not in frame.columns:
        raise ValueError("Expected a 'type' column in the dataset")

    if text_mode == "content_only":
        if "content" not in frame.columns:
            raise ValueError("Expected 'content' column for content_only mode")
        text_series = frame["content"].fillna("").astype(str)
    elif text_mode == "title_content":
        title = frame["title"].fillna("").astype(str) if "title" in frame.columns else ""
        content = frame["content"].fillna("").astype(str) if "content" in frame.columns else ""
        if isinstance(title, str):
            text_series = pd.Series(content)
        elif isinstance(content, str):
            text_series = pd.Series(title)
        else:
            text_series = (title.str.strip() + " " + content.str.strip()).str.strip()
    elif text_mode == "all_fields":
        text_series = _combine_text_columns(frame).fillna("").astype(str)
    else:
        raise ValueError("text_mode must be one of: all_fields, title_content, content_only")

    labels = frame["type"].fillna("unknown").astype(str).str.strip().str.lower()
    if mask_label_tokens:
        text_series = _mask_label_tokens(text_series, labels)

    content_group = frame["content"].fillna("").astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    title_series = frame["title"].fillna("").astype(str) if "title" in frame.columns else ""
    if isinstance(title_series, str):
        title_content_group = content_group
    else:
        title_content_group = (
            title_series.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
            + " "
            + content_group
        ).str.strip()

    normalized = pd.DataFrame(
        {
            "text": text_series,
            "label": labels,
            "group_content": content_group,
            "group_title_content": title_content_group,
        }
    )
    normalized = normalized.loc[normalized["text"].str.len() > 0].copy()
    normalized = normalized.loc[normalized["label"].str.len() > 0].copy()
    normalized.loc[:, "text"] = normalized["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    return normalized.reset_index(drop=True)


def load_news_dataset(
    dataset_name: str = "medium",
    data_dir: Path | None = None,
    text_mode: str = "all_fields",
    mask_label_tokens: bool = False,
) -> pd.DataFrame:
    dataset_path = resolve_dataset_path(dataset_name, data_dir)
    frame = pd.read_csv(dataset_path, engine="python", on_bad_lines="skip")
    return normalize_dataset(frame, text_mode=text_mode, mask_label_tokens=mask_label_tokens)


def split_dataset(
    frame: pd.DataFrame,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_state: int = 42,
    split_mode: str = "random",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be smaller than 1")

    can_stratify = frame["label"].value_counts().min() >= 3 and len(frame) >= 20

    if split_mode == "random":
        train_frame, test_frame = train_test_split(
            frame,
            test_size=test_size,
            random_state=random_state,
            stratify=frame["label"] if can_stratify else None,
        )
        validation_ratio = validation_size / (1 - test_size)
        train_can_stratify = train_frame["label"].value_counts().min() >= 2 and len(train_frame) >= 15
        train_frame, validation_frame = train_test_split(
            train_frame,
            test_size=validation_ratio,
            random_state=random_state,
            stratify=train_frame["label"] if train_can_stratify else None,
        )
    elif split_mode in {"grouped_content", "grouped_title_content"}:
        group_column = "group_content" if split_mode == "grouped_content" else "group_title_content"

        outer_splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        outer_train_idx, test_idx = next(outer_splitter.split(frame, groups=frame[group_column]))
        train_val_frame = frame.iloc[outer_train_idx].copy()
        test_frame = frame.iloc[test_idx].copy()

        validation_ratio = validation_size / (1 - test_size)
        inner_splitter = GroupShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=random_state)
        inner_train_idx, val_idx = next(
            inner_splitter.split(train_val_frame, groups=train_val_frame[group_column])
        )
        train_frame = train_val_frame.iloc[inner_train_idx].copy()
        validation_frame = train_val_frame.iloc[val_idx].copy()
    else:
        raise ValueError("split_mode must be one of: random, grouped_content, grouped_title_content")

    keep_columns = ["text", "label", "group_content", "group_title_content"]
    return (
        train_frame[keep_columns].reset_index(drop=True),
        validation_frame[keep_columns].reset_index(drop=True),
        test_frame[keep_columns].reset_index(drop=True),
    )


TOKEN_PATTERN = re.compile(r"[A-Za-z']+")


def simple_tokenize(text: str) -> Iterable[str]:
    return TOKEN_PATTERN.findall(text.lower())
