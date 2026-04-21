# Fake News Detection Experiments

## Overview

This repository contains a clean, real-data fake-news classification workflow using FakeNewsCorpus subsets. It supports classical ML and PyTorch neural baselines and follows a tiny-first validation flow before larger runs.

## Project Organization

```
data-analysis-progress/
├── data/FakeNewsCorpus/         # curated raw and balanced subset CSVs
├── scripts/                     # data build and training entry scripts
├── src/data_analysis_progress/  # reusable package modules
├── notebooks/                   # optional exploration notebook
├── docs/                        # notes
├── artifacts/                   # generated result JSON (ignored by git)
└── README.md
```

## Dataset Aliases

- raw datasets:
	- `tiny`: 10 rows
	- `small`: 100 rows
	- `medium`: 1000 rows
	- `large`: 10000 rows
- balanced datasets (10 labels):
	- `balanced_tiny`: 10 rows
	- `balanced_small`: 100 rows
	- `balanced_medium`: 1000 rows
	- `balanced_large`: 10000 rows

## Models

- `logistic_regression` (TF-IDF)
- `cnn` (PyTorch)
- `transformer` (lightweight PyTorch encoder)

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a tiny smoke test:

```bash
python scripts/run_news_benchmark.py --dataset tiny --models logistic_regression cnn transformer --epochs 2
```

Run staged tiny-first benchmark up to a target dataset:

```bash
python scripts/run_staged_benchmark.py --target-dataset balanced_medium --text-mode title_content --mask-label-tokens
```

## Rebuild Subsets

Build deterministic raw subsets:

```bash
python scripts/build_raw_subsets.py
```

Build balanced raw subsets:

```bash
python scripts/build_balanced_raw_subsets.py
```

## 100k Direct Training

For a direct 100k logistic baseline:

```bash
python scripts/train_100k_direct.py --max-rows 100000 --chunksize 50000 --output artifacts/results/real_dataset_logreg_report_100k.json
```

## Serving API (For Frontend Integration)

Train and export the serving model:

```bash
python scripts/train_serving_model.py
```

Start the API service:

```bash
python -m data_analysis_progress.api
```

API endpoints:

- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /predict-batch`

See frontend integration doc:

- `docs/前端接口对接说明.md`

Project report:

- `docs/小组项目技术报告_简版.md`

## Notes

- The full 29.3GB source corpus file is not tracked by git.
- Generated outputs under `artifacts/` are ignored by default.