from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score


KNOWN_LABELS = {
    "fake",
    "satire",
    "bias",
    "conspiracy",
    "junksci",
    "hate",
    "clickbait",
    "unreliable",
    "political",
    "reliable",
}


def stable_bucket(value: str, modulo: int = 100) -> int:
    digest = hashlib.md5(value.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:8], 16) % modulo


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_100k(dataset_path: Path, max_rows: int = 100000, chunksize: int = 50000) -> pd.DataFrame:
    chunks = []
    rows = 0
    for chunk in pd.read_csv(
        dataset_path,
        usecols=["id", "url", "type", "title", "content"],
        chunksize=chunksize,
        on_bad_lines="skip",
        engine="python",
    ):
        chunk = chunk.dropna(subset=["type", "content"]).copy()
        chunk.loc[:, "type"] = chunk["type"].astype(str).str.lower().str.strip()
        chunk = chunk[chunk["type"].isin(KNOWN_LABELS)].copy()
        if chunk.empty:
            continue

        text = (chunk["title"].fillna("").astype(str) + " " + chunk["content"].fillna("").astype(str)).map(normalize_text)
        key = chunk["url"].fillna("").astype(str).str.strip()
        key = key.where(key != "", chunk["id"].fillna("").astype(str))

        cleaned = pd.DataFrame({"key": key, "text": text, "label": chunk["type"].astype(str)})
        cleaned = cleaned[cleaned["text"].str.len() > 0].copy()
        if cleaned.empty:
            continue

        remaining = max_rows - rows
        if remaining <= 0:
            break
        if len(cleaned) > remaining:
            cleaned = cleaned.head(remaining)

        chunks.append(cleaned)
        rows += len(cleaned)
        if rows >= max_rows:
            break

    if not chunks:
        raise ValueError("No valid rows loaded from dataset")
    return pd.concat(chunks, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LogisticRegression directly on 100k real rows")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(r"C:\Users\22130\Downloads\news.csv\news_cleaned_2018_02_13.csv"),
    )
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/results/real_dataset_logreg_report_100k.json"),
    )
    args = parser.parse_args()

    data = load_100k(args.dataset_path, max_rows=args.max_rows, chunksize=args.chunksize)
    buckets = data["key"].map(lambda x: stable_bucket(str(x), 100))
    train = data[buckets >= int(args.test_ratio * 100)].copy()
    test = data[buckets < int(args.test_ratio * 100)].copy()

    vectorizer = TfidfVectorizer(max_features=120000, ngram_range=(1, 2), min_df=2, stop_words="english")
    clf = LogisticRegression(max_iter=1500, class_weight="balanced", n_jobs=1)

    x_train = vectorizer.fit_transform(train["text"].tolist())
    x_test = vectorizer.transform(test["text"].tolist())
    y_train = train["label"].tolist()
    y_test = test["label"].tolist()

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    report = {
        "dataset_path": str(args.dataset_path),
        "max_rows": args.max_rows,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }

    output_path = args.output
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parents[1] / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"train_rows={len(train)}, test_rows={len(test)}")
    print(f"accuracy={report['accuracy']:.4f}, macro_f1={report['macro_f1']:.4f}")
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
