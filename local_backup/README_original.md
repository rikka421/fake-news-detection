# News Classification Analysis

**Course:** Data Analysis and Progress  
**Project Duration:** 14 days (April 13-26, 2026)  
**Primary Dataset:** https://github.com/dolphinwesting248/news_classification  
**Additional Dataset:** Fake News Corpus (9.4 million articles)

A comprehensive system for news classification and fake news detection using machine learning and deep learning techniques.

## 📋 Project Overview

This project analyzes news articles to classify them into different categories (fake, bias, conspiracy, hate, etc.) and detect fake news. The system uses both traditional machine learning models and modern transformer-based approaches.

### Objectives
1. Download and preprocess the news classification dataset
2. Implement baseline models (Random Forest, CNN, etc.)
3. Experiment with transformer models and small LLMs
4. Perform feature engineering and comparative analysis
5. Document findings and create a comprehensive report

### Features
- **Multi-class classification** of news articles (11 categories)
- **Fake news detection** using state-of-the-art models
- **Comparative analysis** of different ML approaches
- **Comprehensive evaluation** with multiple metrics
- **Scalable architecture** for handling large datasets

## 🗂️ Dataset

### Primary Dataset: News Classification
**Source:** https://github.com/dolphinwesting248/news_classification  
**Task:** Multi-class text classification of news articles  
**Expected Format:** News articles with classification labels

### Additional Dataset: Fake News Corpus
The primary dataset is the [Fake News Corpus](https://github.com/several27/FakeNewsCorpus) containing 9.4 million news articles labeled by type:

| Type | Tag | Count | Description |
|------|-----|-------|-------------|
| Fake News | fake | 928,083 | Fabricate information or grossly distort news |
| Satire | satire | 146,080 | Humor, irony, exaggeration for commentary |
| Extreme Bias | bias | 1,300,444 | Particular point of view, may use propaganda |
| Conspiracy Theory | conspiracy | 905,981 | Promoters of conspiracy theories |
| Junk Science | junksci | 144,939 | Promote pseudoscience, metaphysics |
| Hate News | hate | 117,374 | Promote discrimination |
| Clickbait | clickbait | 292,201 | Credible content with misleading headlines |
| Proceed With Caution | unreliable | 319,830 | May be reliable but require verification |
| Political | political | 2,435,471 | Support certain political views |
| Credible | reliable | 1,920,139 | Traditional ethical journalism |

**Dataset Information:**
- **Creation:** Scraped from 1001 domains + NYTimes + WebHose for balance
- **Labels:** Based on source domain credibility from OpenSources.co
- **Sampled Versions:** Created 10, 100, 1000, and 5000 row samples for practical use
- **Recommended:** `fakenews_small_1000.csv` (1000 rows) for initial training
- **Use case:** Fake news detection, multi-class news classification, comparative analysis

**Note:** For practical use, we've created sampled versions (10, 100, 1000, 5000 rows) included in this repository.

### Dataset Resources
See `DATASET_RESOURCES.md` for detailed information about available datasets and integration strategies.
See `data/FakeNewsCorpus/README.md` for complete dataset documentation and sample information.

## 📊 Project Structure

```
lessons/data-analysis-progress/
├── project-plan.md          # Detailed 14-day plan
├── README.md               # This file - project overview
├── PROJECT_SETUP.md        # Setup and development guide
├── DATASET_RESOURCES.md    # Dataset information
├── data/                   # Dataset directory
│   ├── FakeNewsCorpus/     # Fake News Corpus samples
│   │   ├── README.md       # Dataset documentation
│   │   ├── OFFICIAL_README.md # Original dataset README
│   │   └── *.csv          # Sample files (10, 100, 1000, 5000 rows)
│   └── news_classification/ # Cloned repository (primary dataset)
├── src/                    # Source code
│   ├── experiment_runner.py # Consolidated experiment runner
│   ├── data/              # Data loading and processing
│   │   └── loader.py
│   ├── models/            # Model implementations
│   ├── utils/             # Utility functions
│   └── cli.py             # Command-line interface
├── notebooks/              # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_transformer_models.ipynb
│   └── 04_analysis_report.ipynb
├── scripts/               # Python scripts
├── models/                # Saved models
├── reports/               # Analysis reports and visualizations
├── archive/               # Temporary/old files
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
├── .gitignore            # Git ignore rules
└── venv_fakenews/        # Virtual environment
```

## 🚀 Quick Start

For detailed setup instructions, see [PROJECT_SETUP.md](PROJECT_SETUP.md).

### 1. Clone the Dataset
```bash
git clone https://github.com/dolphinwesting248/news_classification.git data/news_classification
```

### 2. Set Up Environment
```bash
# Create virtual environment
python -m venv venv_fakenews

# Activate on Windows
venv_fakenews\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Explore the Data
Start with `notebooks/01_data_exploration.ipynb` to understand the dataset structure.

### Basic Usage

```python
from src.data.loader import DataLoader

# Load sample data
loader = DataLoader()
df = loader.load_fakenews_sample("small")  # 1000 articles

# Analyze dataset
analysis = loader.analyze_dataset(df)
print(f"Total articles: {analysis['total_articles']}")
print(f"Label distribution: {analysis['label_distribution']}")

# Create train/test split
splits = loader.create_train_test_split(df, test_size=0.2)
print(f"Train: {len(splits['train'])} articles")
print(f"Test: {len(splits['test'])} articles")
```

### Command Line Interface

```bash
# Explore dataset
python -m src.cli explore --sample small

# Analyze dataset statistics
python -m src.cli analyze --sample medium

# Create train/test split
python -m src.cli split --sample small --test-size 0.2
```

## Requirements

Create `requirements.txt` with:
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0
torch>=2.0.0
transformers>=4.25.0
nltk>=3.8.0
spacy>=3.5.0
gensim>=4.3.0
```

## 🧪 Model Approaches

### Phase 1: Baseline Models
- **Random Forest** with TF-IDF features
- **Logistic Regression** with different feature representations
- **Support Vector Machine (SVM)** with linear kernel
- **CNN** for text classification

### Phase 2: Advanced Models
- **BERT** (pretrained, fine-tuned)
- **DistilBERT** (lighter version)
- **RoBERTa** (optimized BERT)
- **ALBERT** (parameter-efficient)

### Phase 3: Ensemble Methods
- Voting classifiers
- Stacking ensembles
- Feature combination approaches

## 📈 Evaluation Metrics

- **Accuracy**
- **Precision, Recall, F1-score** (per class and weighted)
- **Confusion matrix** analysis
- **Training/inference time**
- **Model size** (for deployment considerations)

## 🗓️ Project Timeline

| Phase | Duration | Tasks |
|-------|----------|-------|
| Phase 1 | Days 1-2 | Data acquisition & preprocessing |
| Phase 2 | Days 3-5 | Baseline models implementation |
| Phase 3 | Days 6-10 | Advanced models (transformers, LLMs) |
| Phase 4 | Days 11-12 | Analysis & reporting |
| Phase 5 | Days 13-14 | Optimization & finalization |

## 🛠️ Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/

# Lint code
flake8 src/
```

### Adding New Features

1. Create feature branch: `git checkout -b feature/new-model`
2. Implement changes
3. Add tests
4. Run tests: `pytest tests/`
5. Submit pull request

## 📚 Documentation

- [Dataset Resources](DATASET_RESOURCES.md) - Detailed dataset information
- [Project Plan](project-plan.md) - Detailed project timeline and tasks
- [FakeNewsCorpus Documentation](data/FakeNewsCorpus/README.md) - Dataset-specific documentation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Fake News Corpus](https://github.com/several27/FakeNewsCorpus) for the dataset
- [OpenSources.co](http://www.opensources.co/) for domain categorization
- [Hugging Face](https://huggingface.co/) for transformer models
- All open-source libraries used in this project

## 📧 Contact

- **Author:** rikka421
- **Email:** 3550124064@qq.com
- **GitHub:** [rikka421](https://github.com/rikka421)

---

**Notes:**
- This project is part of the "Data Analysis and Progress" course at Nanjing University
- Focus on both technical implementation and analytical insights
- Document all experiments and results thoroughly
- Consider practical aspects like model efficiency and deployment

## Expected Deliverables

1. **Code:** Complete implementation with documentation
2. **Report:** Comparative analysis of different approaches
3. **Visualizations:** Performance charts and error analysis
4. **Presentation:** Summary of findings and recommendations