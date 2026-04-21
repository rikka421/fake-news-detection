from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections import Counter
from pathlib import Path

import pandas as pd


TARGET_LABELS = [
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
]

TARGETS = [
    ("balanced_tiny", 1),
    ("balanced_small", 10),
    ("balanced_medium", 100),
    ("balanced_large", 1000),
]


def stable_hash(value: str) -> int:
    digest = hashlib.md5(value.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest, 16)


def normalize_chunk(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.dropna(subset=["type", "content"]).copy()
    frame.loc[:, "type"] = frame["type"].astype(str).str.lower().str.strip()
    frame = frame[frame["type"].isin(TARGET_LABELS)].copy()
    if frame.empty:
        return frame

    for column in ["title", "content", "meta_description", "summary"]:
        if column in frame.columns:
            normalized_column = frame[column].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
            frame = frame.assign(**{column: normalized_column})

    url = frame["url"].fillna("").astype(str).str.strip()
    identifier = frame["id"].fillna("").astype(str)
    frame = frame.assign(sample_key=url.where(url != "", identifier))
    frame = frame[frame["content"].astype(str).str.len() > 0].copy()
    return frame


def collect_balanced_rows(dataset_path: Path, chunksize: int) -> tuple[dict[str, list[dict]], Counter[str], int]:
    max_per_label = max(quota for _, quota in TARGETS)
    heaps: dict[str, list[tuple[int, int, dict]]] = {label: [] for label in TARGET_LABELS}
    seen_counts: Counter[str] = Counter()
    counter = 0
    processed_clean_rows = 0

    reader = pd.read_csv(
        dataset_path,
        usecols=[
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
        ],
        chunksize=chunksize,
        on_bad_lines="skip",
        engine="python",
    )

    for chunk_index, chunk in enumerate(reader, start=1):
        cleaned = normalize_chunk(chunk)
        if cleaned.empty:
            continue

        for _, row in cleaned.iterrows():
            label = str(row["type"])
            seen_counts[label] += 1
            processed_clean_rows += 1
            key = str(row.get("sample_key") or row.get("url") or row.get("id") or "")
            score = stable_hash(key)
            entry = (-score, counter, row.to_dict())
            heap = heaps[label]
            if len(heap) < max_per_label:
                heapq.heappush(heap, entry)
            elif score < -heap[0][0]:
                heapq.heapreplace(heap, entry)
            counter += 1

        if chunk_index % 10 == 0:
            print(f"processed_clean_rows={processed_clean_rows}")

        if min(seen_counts.get(label, 0) for label in TARGET_LABELS) >= max_per_label:
            break

    selected_rows: dict[str, list[dict]] = {}
    for label, heap in heaps.items():
        rows = [item[2] for item in heap]
        rows.sort(key=lambda row: stable_hash(str(row.get("sample_key") or row.get("url") or row.get("id") or "")))
        selected_rows[label] = rows[:max_per_label]

    return selected_rows, seen_counts, processed_clean_rows


def write_subsets(selected_rows: dict[str, list[dict]], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {}

    for alias, quota in TARGETS:
        rows: list[dict] = []
        for label in TARGET_LABELS:
            rows.extend(selected_rows[label][:quota])
        frame = pd.DataFrame(rows).sort_values(["type", "sample_key", "id"]).reset_index(drop=True)
        frame = frame.drop(columns=["sample_key"], errors="ignore")
        file_name = f"fakenews_{alias}_{len(frame)}_raw.csv"
        path = output_dir / file_name
        frame.to_csv(path, index=False)
        manifest[alias] = {
            "path": str(path),
            "rows": int(len(frame)),
            "per_label_quota": quota,
            "labels": frame["type"].value_counts().sort_index().to_dict(),
        }

    manifest_path = output_dir / "balanced_raw_subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build balanced raw-data subsets from the full FakeNewsCorpus CSV")
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
    args = parser.parse_args()

    if not args.dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    selected_rows, seen_counts, processed_clean_rows = collect_balanced_rows(args.dataset_path, args.chunksize)
    manifest = write_subsets(selected_rows, args.output_dir)
    print(f"processed_clean_rows={processed_clean_rows}")
    print(json.dumps({"seen_counts": seen_counts, "subsets": manifest}, indent=2, default=int))


if __name__ == "__main__":
    main()
