from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from scipy.sparse import hstack


KNOWN_LABELS = [
    "fake",
    "satire",
    "bias",
    "conspiracy",
    "state",
    "junksci",
    "hate",
    "clickbait",
    "unreliable",
    "political",
    "reliable",
]

CLASS_WEIGHT = {
    "fake": 1.0,
    "satire": 8.0,
    "bias": 2.0,
    "conspiracy": 2.5,
    "state": 10.0,
    "junksci": 4.0,
    "hate": 8.0,
    "clickbait": 2.0,
    "unreliable": 3.0,
    "political": 1.0,
    "reliable": 6.0,
}


def stable_bucket(value: str, modulo: int = 100) -> int:
    digest = hashlib.md5(value.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:8], 16) % modulo


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_text(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "content_only":
        return df["content"].fillna("").astype(str)
    if mode == "title_content":
        return (df["title"].fillna("") + " " + df["content"].fillna("")).fillna("").astype(str)
    raise ValueError("mode must be one of: content_only, title_content")


def mask_label_tokens(texts: pd.Series) -> pd.Series:
    masked = texts.str.lower()
    for label in KNOWN_LABELS:
        masked = masked.str.replace(fr"\b{re.escape(label)}\b", " ", regex=True)
    masked = masked.str.replace(r"\s+", " ", regex=True).str.strip()
    return masked


def iter_clean_chunks(
    dataset_path: Path,
    chunksize: int,
    text_mode: str,
    mask_labels: bool,
    max_rows: int | None,
):
    usecols = ["id", "url", "type", "title", "content"]
    rows_seen = 0
    chunk_index = 0

    reader = pd.read_csv(
        dataset_path,
        usecols=usecols,
        chunksize=chunksize,
        on_bad_lines="skip",
        engine="python",
    )
    for chunk in reader:
        chunk_index += 1
        chunk = chunk.dropna(subset=["type", "content"]).copy()
        chunk.loc[:, "type"] = chunk["type"].astype(str).str.lower().str.strip()
        chunk = chunk[chunk["type"].isin(KNOWN_LABELS)].copy()
        if chunk.empty:
            continue

        text = build_text(chunk, text_mode).map(normalize_text)
        if mask_labels:
            text = mask_label_tokens(text)

        key = chunk["url"].fillna("").astype(str).str.strip()
        fallback = chunk["id"].fillna("").astype(str)
        key = np.where(key == "", fallback, key)
        key = pd.Series(key, index=chunk.index)

        cleaned = pd.DataFrame(
            {
                "key": key,
                "label": chunk["type"].astype(str),
                "text": text,
            }
        )
        cleaned = cleaned[cleaned["text"].str.len() > 0].copy()
        if cleaned.empty:
            continue

        if max_rows is not None:
            remaining = max_rows - rows_seen
            if remaining <= 0:
                break
            if len(cleaned) > remaining:
                cleaned = cleaned.head(remaining)

        rows_seen += len(cleaned)
        if chunk_index % 5 == 0:
            print(f"processed_rows={rows_seen}")
        yield cleaned

        if max_rows is not None and rows_seen >= max_rows:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on the full real FakeNewsCorpus CSV with streaming")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(r"C:\Users\22130\Downloads\news.csv\news_cleaned_2018_02_13.csv"),
    )
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=300000)
    parser.add_argument(
        "--text-mode",
        type=str,
        default="title_content",
        choices=["content_only", "title_content"],
    )
    parser.add_argument("--mask-label-tokens", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/results/real_dataset_sgd_report.json"),
    )
    args = parser.parse_args()

    if not args.dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    rng = np.random.default_rng(args.seed)
    del rng  # Seed is passed directly to SGDClassifier.

    word_vectorizer = HashingVectorizer(
        n_features=2**20,
        alternate_sign=False,
        ngram_range=(1, 2),
        norm="l2",
        lowercase=False,
    )
    char_vectorizer = HashingVectorizer(
        n_features=2**18,
        alternate_sign=False,
        analyzer="char_wb",
        ngram_range=(3, 5),
        norm="l2",
        lowercase=False,
    )
    classifier = SGDClassifier(
        loss="log_loss",
        alpha=5e-7,
        learning_rate="optimal",
        random_state=args.seed,
        average=True,
    )

    label_to_idx = {label: idx for idx, label in enumerate(KNOWN_LABELS)}

    train_rows = 0
    test_rows = 0
    holdout_texts: list[str] = []
    holdout_labels: list[str] = []

    for cleaned in iter_clean_chunks(
        args.dataset_path,
        args.chunksize,
        args.text_mode,
        args.mask_label_tokens,
        args.max_rows,
    ):
        buckets = cleaned["key"].map(lambda x: stable_bucket(str(x), 100))
        train_mask = buckets >= int(args.test_ratio * 100)

        train_chunk = cleaned[train_mask]
        if not train_chunk.empty:
            word_x = word_vectorizer.transform(train_chunk["text"].tolist())
            char_x = char_vectorizer.transform(train_chunk["text"].tolist())
            x_train = hstack([word_x, char_x], format="csr")
            y_labels = train_chunk["label"].tolist()
            y_train = np.array([label_to_idx[x] for x in y_labels], dtype=np.int64)
            sample_weight = np.array([CLASS_WEIGHT[label] for label in y_labels], dtype=np.float64)
            classifier.partial_fit(
                x_train,
                y_train,
                classes=np.arange(len(KNOWN_LABELS), dtype=np.int64),
                sample_weight=sample_weight,
            )
            train_rows += len(train_chunk)

        test_chunk = cleaned[~train_mask]
        if not test_chunk.empty:
            holdout_texts.extend(test_chunk["text"].tolist())
            holdout_labels.extend(test_chunk["label"].tolist())
        test_rows += len(test_chunk)

    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    if holdout_texts:
        word_x = word_vectorizer.transform(holdout_texts)
        char_x = char_vectorizer.transform(holdout_texts)
        x_test = hstack([word_x, char_x], format="csr")
        y_true = np.array([label_to_idx[x] for x in holdout_labels], dtype=np.int64)
        y_pred = classifier.predict(x_test)
        y_true_all = y_true.tolist()
        y_pred_all = y_pred.tolist()
    else:
        raise ValueError("No holdout rows were collected; adjust test_ratio or max_rows")

    accuracy = accuracy_score(y_true_all, y_pred_all)
    macro_f1 = f1_score(y_true_all, y_pred_all, average="macro")

    report = {
        "dataset_path": str(args.dataset_path),
        "max_rows": args.max_rows,
        "chunksize": args.chunksize,
        "text_mode": args.text_mode,
        "mask_label_tokens": bool(args.mask_label_tokens),
        "train_rows": train_rows,
        "test_rows": test_rows,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "labels": KNOWN_LABELS,
        "classification_report": classification_report(
            y_true_all,
            y_pred_all,
            labels=list(range(len(KNOWN_LABELS))),
            target_names=KNOWN_LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }

    output_path = args.output
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parents[1] / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"train_rows={train_rows}, test_rows={test_rows}")
    print(f"accuracy={accuracy:.4f}, macro_f1={macro_f1:.4f}")
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
