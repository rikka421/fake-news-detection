# Project Setup & Development Guide

## Project Overview
**Course:** Data Analysis and Progress  
**Project:** News Classification Analysis  
**Duration:** 14 days (April 13-26, 2026)  
**Location:** `lessons/data-analysis-progress/`

## Quick Start

### 1. Environment Setup
```bash
# Clone the primary dataset
git clone https://github.com/dolphinwesting248/news_classification.git data/news_classification

# Create virtual environment
python -m venv venv_fakenews

# Activate on Windows
venv_fakenews\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation
```bash
# Check available datasets
python -m src.experiment_runner --list-datasets

# Run basic experiment
python -m src.experiment_runner --dataset fakenews_small_1000
```

### 3. Project Structure
```
lessons/data-analysis-progress/
├── README.md                 # Project overview
├── PROJECT_SETUP.md         # This file - setup guide
├── project-plan.md          # Detailed 14-day plan
├── DATASET_RESOURCES.md     # Dataset information
├── data/                    # Dataset directory
│   ├── FakeNewsCorpus/      # Fake News Corpus samples
│   └── news_classification/ # Primary dataset (cloned)
├── src/                     # Source code
│   ├── experiment_runner.py # Consolidated experiment runner
│   ├── data/               # Data loading utilities
│   ├── models/             # Model implementations
│   └── cli.py              # Command-line interface
├── notebooks/              # Jupyter notebooks
├── scripts/               # Python scripts
├── models/                # Saved models
├── reports/               # Analysis reports
├── archive/               # Temporary/old files
└── venv_fakenews/         # Virtual environment
```

## Development Workflow

### Phase 1: Data Exploration (Days 1-2)
```bash
# Explore dataset structure
python -m src.cli explore --sample small

# Analyze dataset statistics
python -m src.cli analyze --sample medium
```

### Phase 2: Baseline Models (Days 3-5)
```bash
# Run traditional ML baselines
python -m src.experiment_runner --dataset fakenews_small_1000

# Compare across datasets
python -m src.experiment_runner --compare
```

### Phase 3: Advanced Models (Days 6-10)
```python
# Implement in notebooks/03_transformer_models.ipynb
# - BERT, DistilBERT, RoBERTa fine-tuning
# - Transformer-based classification
```

### Phase 4: Analysis & Reporting (Days 11-12)
```bash
# Generate reports
python scripts/generate_report.py

# Create visualizations
python scripts/create_visualizations.py
```

## Available Commands

### Experiment Runner
```bash
# List available datasets
python -m src.experiment_runner --list-datasets

# Run single experiment
python -m src.experiment_runner --dataset fakenews_small_1000

# Compare multiple datasets
python -m src.experiment_runner --compare
```

### CLI Interface
```bash
# Explore dataset
python -m src.cli explore --sample small

# Analyze dataset
python -m src.cli analyze --sample medium

# Create train/test split
python -m src.cli split --sample small --test-size 0.2
```

## Time Management Guidelines

### Experiment Time Estimates
| Dataset Size | Traditional ML | Deep Learning |
|--------------|----------------|---------------|
| 100 rows     | 1-3 minutes    | 5-10 minutes  |
| 1000 rows    | 5-15 minutes   | 15-30 minutes |
| 5000 rows    | 15-30 minutes  | 30-60 minutes |

### Best Practices
1. **Start small** - Test with 100 rows first
2. **Estimate time** before running experiments
3. **Monitor progress** - Check intermediate results
4. **Save checkpoints** - Save models periodically
5. **Document results** - Keep detailed logs

## Troubleshooting

### Common Issues

#### 1. Dataset Loading Errors
```
FileNotFoundError: Dataset not found
```
**Solution:**
```bash
# Check available datasets
python -m src.experiment_runner --list-datasets

# Regenerate samples if needed
python scripts/regenerate_samples.py
```

#### 2. Memory Issues
```
MemoryError: Unable to allocate array
```
**Solution:**
- Use smaller dataset samples
- Reduce `max_features` in TF-IDF vectorizer
- Use sparse matrices

#### 3. Dependency Issues
```
ModuleNotFoundError: No module named 'sklearn'
```
**Solution:**
```bash
# Activate virtual environment
venv_fakenews\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## Development Tips

### 1. Virtual Environment Management
```bash
# Create new environment
python -m venv venv_fakenews

# Activate
venv_fakenews\Scripts\activate

# Freeze dependencies
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt
```

### 2. Code Organization
- Keep **experiment code** in `src/experiment_runner.py`
- Store **data processing** in `src/data/` modules
- Put **model definitions** in `src/models/`
- Use **notebooks** for exploration and visualization
- Save **reports** in `reports/` folder

### 3. Version Control
```bash
# Track code changes
git add src/ notebooks/ scripts/

# Commit with descriptive messages
git commit -m "Add transformer model implementation"

# Push to remote repository
git push origin main
```

## Performance Optimization

### For Large Datasets:
1. **Use sampling** - Start with 1000 rows
2. **Limit features** - Set `max_features=5000` in TF-IDF
3. **Use sparse matrices** - For memory efficiency
4. **Batch processing** - Process data in chunks
5. **Parallel processing** - Use joblib for parallel training

### For Deep Learning:
1. **Use GPU** if available
2. **Mixed precision** for faster training
3. **Gradient accumulation** for large batches
4. **Model checkpointing** to save progress
5. **Early stopping** to prevent overfitting

## Documentation

### Key Documentation Files:
1. **`README.md`** - Project overview and structure
2. **`PROJECT_SETUP.md`** - This setup guide
3. **`project-plan.md`** - Detailed timeline and tasks
4. **`DATASET_RESOURCES.md`** - Dataset information
5. **Code docstrings** - Inline documentation

### Adding Documentation:
```python
def train_model(X, y, model_name='random_forest'):
    """
    Train a machine learning model.
    
    Args:
        X: Feature matrix
        y: Target labels
        model_name: Name of model to train
        
    Returns:
        Trained model object
        
    Example:
        >>> model = train_model(X_train, y_train, 'random_forest')
    """
    # Implementation...
```

## Next Steps

### Immediate (Next Session):
1. Test the consolidated experiment runner
2. Run baseline experiments on real data
3. Document initial results

### Short-term (This Week):
1. Implement deep learning models
2. Compare traditional ML vs. DL performance
3. Create comprehensive evaluation report

### Long-term (Project Completion):
1. Optimize best-performing model
2. Create deployment pipeline
3. Write final project report

## Support

For issues or questions:
1. Check the **README.md** for project overview
2. Review **DATASET_RESOURCES.md** for dataset information
3. Use `--help` flag with any command
4. Check error logs in console output
5. Review notebook examples for usage patterns

---

**Last Updated:** 2026-04-14  
**Maintained by:** OpenClaw AI Assistant