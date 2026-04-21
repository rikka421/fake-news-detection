from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path

import pandas as pd


KNOWN_LABELS = {
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
}


TARGETS = [
    ("tiny", 10),
    ("small", 100),
    ("medium", 1000),
    ("large", 10000),
]


def stable_hash(value: str) -> int:
    digest = hashlib.md5(value.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest, 16)


def canonical_key(frame: pd.DataFrame) -> pd.Series:
    url = frame["url"].fillna("").astype(str).str.strip()
    identifier = frame["id"].fillna("").astype(str)
    return url.where(url != "", identifier)


def normalize_subset(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.dropna(subset=["type", "content"]).copy()
    frame.loc[:, "type"] = frame["type"].astype(str).str.lower().str.strip()
    frame = frame[frame["type"].isin(KNOWN_LABELS)].copy()
    text_columns = [column for column in ["title", "content", "meta_description", "summary"] if column in frame.columns]
    for column in text_columns:
        normalized_column = frame[column].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        frame = frame.assign(**{column: normalized_column})
    frame = frame[frame["content"].astype(str).str.len() > 0].copy()
    return frame



def build_top_k_subset(dataset_path: Path, k: int, chunksize: int, max_clean_rows: int | None) -> list[dict]:
    usecols = [
        "id",
        "domain",
        "type",
        "url",
        "content",
        "scraped_at",
        "inserted_at",
        "updated_at",
        "title",
        "authors",
        "keywords",
        "meta_keywords",
        "meta_description",
        "tags",
        "summary",
        "source",
    ]
    heap: list[tuple[int, int, dict]] = []
    counter = 0
    rows_processed = 0

    reader = pd.read_csv(
        dataset_path,
        usecols=usecols,
        chunksize=chunksize,
        on_bad_lines="skip",
        engine="python",
    )
    for chunk_index, chunk in enumerate(reader, start=1):
        cleaned = normalize_subset(chunk)
        if cleaned.empty:
            continue

        if max_clean_rows is not None and rows_processed >= max_clean_rows:
            break

        keys = canonical_key(cleaned)
        for row_index, (_, row) in enumerate(cleaned.iterrows()):
            if max_clean_rows is not None and rows_processed >= max_clean_rows:
                break
            key = str(keys.iloc[row_index])
            score = stable_hash(key)
            row_dict = row.to_dict()
            entry = (-score, counter, row_dict)
            if len(heap) < k:
                heapq.heappush(heap, entry)
            elif score < -heap[0][0]:
                heapq.heapreplace(heap, entry)
            counter += 1
            rows_processed += 1

        if chunk_index % 20 == 0:
            print(f"processed_clean_rows={rows_processed}")

    rows = [item[2] for item in heap]
    rows.sort(key=lambda row: stable_hash(str(row.get("url") or row.get("id") or "")))
    return rows


def write_subsets(rows: list[dict], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    frame = pd.DataFrame(rows)
    for alias, size in TARGETS:
        subset = frame.head(size).copy()
        file_name = f"fakenews_{alias}_{size}_raw.csv"
        path = output_dir / file_name
        subset.to_csv(path, index=False)
        manifest[alias] = {
            "path": str(path),
            "rows": len(subset),
            "labels": subset["type"].value_counts().to_dict() if not subset.empty else {},
        }
    manifest_path = output_dir / "raw_subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic raw-data subsets from the full FakeNewsCorpus CSV")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(r"C:\Users\22130\Downloads\news.csv\news_cleaned_2018_02_13.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "FakeNewsCorpus",
    )
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument("--max-clean-rows", type=int, default=250000)
    args = parser.parse_args()

    if not args.dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    rows = build_top_k_subset(args.dataset_path, k=10000, chunksize=args.chunksize, max_clean_rows=args.max_clean_rows)
    manifest = write_subsets(rows, args.output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
