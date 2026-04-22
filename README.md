# Fake News Detection Experiments

## Overview

This repository contains a clean, real-data fake-news classification workflow using balanced FakeNewsCorpus subsets. It supports classical ML, lightweight neural baselines, and an optional <=0.5B LLM experiment track.

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

- balanced datasets (10 labels):
	- `balanced_tiny`: 10 rows
	- `balanced_small`: 100 rows
	- `balanced_medium`: 1000 rows
	- `balanced_large`: 10000 rows

## Models

- `logistic_regression` (TF-IDF)
- `svm`
- `random_forest`
- `naive_bayes`
- `xgboost`
- `lightgbm`
- `cnn` (PyTorch)
- `transformer` (lightweight PyTorch encoder)

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a tiny smoke test:

```bash
python scripts/run_news_benchmark.py --dataset balanced_tiny --models logistic_regression cnn transformer --epochs 2
```

Run all models on `balanced_large`:

```bash
python scripts/run_news_benchmark.py --dataset balanced_large --models logistic_regression svm random_forest naive_bayes xgboost lightgbm cnn transformer --epochs 3 --text-mode title_content --mask-label-tokens --split-mode grouped_title_content --output artifacts/results/benchmark_balanced_large_all_models.json
```

## Rebuild Subsets

Build balanced raw subsets:

```bash
python scripts/build_balanced_raw_subsets.py
```

## Optional <=0.5B LLM Evaluation

Run Qwen2.5-0.5B generation-mode test (GPU preferred):

```bash
python scripts/eval_llm_zero_shot.py --model Qwen/Qwen2.5-0.5B-Instruct --dataset balanced_large --method generation --max-samples 1000 --sample-mode random --precision auto --mask-label-tokens --split-mode grouped_title_content --output artifacts/results/llm_qwen_0p5b_generation_1000_gpu.json
```

Note: This script reports startup time, average sample time, throughput, and projected 10k runtime.

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
- `docs/项目技术报告_中文版.md`

## Notes

- The full 29.3GB source corpus file is not tracked by git.
- Generated outputs under `artifacts/` are ignored by default.

## Current Status (2026-04-22)

- End-to-end pipeline is ready for delivery (training, benchmark, serving, docs).
- `balanced_large` full-model benchmark has been completed.
- Frontend deployment docs and Docker deployment flow are finalized.
- <=0.5B LLM track is kept as an offline research path, not the default production path.