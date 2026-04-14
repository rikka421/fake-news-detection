# FakeNewsCorpus Samples

## Overview
This directory contains sampled versions of the Fake News Corpus dataset for use in training and testing fake news detection models. The original dataset is 29.3GB, too large for typical workflows, so these samples provide manageable subsets.

## Dataset Information (From Official README)

### Source
- **Official Repository:** https://github.com/several27/FakeNewsCorpus
- **Release:** v1.0 (https://github.com/several27/FakeNewsCorpus/releases/tag/v1.0)
- **Local File:** `C:\Users\22130\Downloads\news.csv\news_cleaned_2018_02_13.csv` (29.3 GB)

### Creation Method
The corpus was created by:
1. Scraping 1001 domains from http://www.opensources.co/
2. Adding NYTimes and WebHose English News Articles for class balance
3. Processing HTML content using the `newspaper` library
4. Labeling articles with the same label as their source domain

### Dataset Statistics
- **Total Articles:** 9,408,908 (from 745 out of 1001 planned domains)
- **Format:** CSV with 16 fields
- **Source Types:** opensources, nytimes, webhose

### Label Types and Counts
| Type | Tag | Count | Description |
|------|-----|-------|-------------|
| Fake News | fake | 928,083 | Entirely fabricate information or grossly distort news |
| Satire | satire | 146,080 | Use humor, irony, exaggeration for commentary |
| Extreme Bias | bias | 1,300,444 | Particular point of view, may use propaganda |
| Conspiracy Theory | conspiracy | 905,981 | Promoters of conspiracy theories |
| State News | state | 0 | Government-sanctioned sources in repressive states |
| Junk Science | junksci | 144,939 | Promote pseudoscience, metaphysics |
| Hate News | hate | 117,374 | Promote racism, misogyny, homophobia, discrimination |
| Clickbait | clickbait | 292,201 | Credible content with exaggerated/misleading headlines |
| Proceed With Caution | unreliable | 319,830 | May be reliable but require verification |
| Political | political | 2,435,471 | Verifiable information supporting certain political views |
| Credible | reliable | 1,920,139 | Traditional and ethical journalism practices |

### Original Dataset
- **Size:** 29.3 GB
- **Rows:** 9,408,908 articles
- **Columns:** id, domain, type, url, content, scraped_at, inserted_at, updated_at, title, authors, keywords, meta_keywords, meta_description, tags, summary, source

## Sample Files Created

### Quick Test Samples
1. **`fakenews_test_10.csv`** - 10 rows, 0.00 MB
   - For quick code testing and data structure verification

2. **`fakenews_tiny_100.csv`** - 100 rows, 0.02 MB
   - For rapid prototyping and debugging

### Training Samples
3. **`fakenews_small_1000.csv`** - 1000 rows, 0.11 MB
   - Recommended for initial model training and hyperparameter tuning

4. **`fakenews_medium_5000.csv`** - 5000 rows, 0.65 MB
   - For more robust training and validation

### Special Samples
5. **`fakenews_stratified_1000.csv`** - 162 rows, 0.08 MB
   - Attempt at balanced sampling by type (limited success due to parsing issues)

## Data Structure
Each CSV file contains the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `id` | Unique identifier | 2 |
| `domain` | Website domain | express.co.uk |
| `type` | News type/label | rumor, hate, conspiracy, etc. |
| `url` | Article URL | https://www.express.co.uk/... |
| `content` | Full article text | "Life is an illusion, at least on a quantum level..." |
| `scraped_at` | When article was scraped | 2018-01-25 16:17:44.789555 |
| `inserted_at` | When added to database | 2018-02-02 01:19:41.756632 |
| `updated_at` | Last update time | 2018-02-02 01:19:41.756664 |
| `title` | Article title | "Is life an ILLUSION? Researchers prove..." |
| `authors` | Author names | "Sean Martin" |
| `keywords` | Article keywords | [] |
| `meta_keywords` | Meta keywords | [''] |
| `meta_description` | Meta description | "THE UNIVERSE ceases to exist..." |
| `tags` | Article tags |  |
| `summary` | Article summary |  |
| `source` | Data source |  |

## Usage Recommendations

### For Model Development:
1. **Start with `fakenews_test_10.csv`** to verify your data loading pipeline
2. **Use `fakenews_small_1000.csv`** for initial model training and hyperparameter tuning
3. **Validate with `fakenews_medium_5000.csv`** for more reliable performance metrics

### For Analysis:
- The `type` column contains labels like "rumor", "hate", "conspiracy", etc.
- The `content` column contains the full article text for NLP tasks
- Multiple articles may come from the same `domain`

## Notes on Data Quality

### Issues Found:
1. **Parsing complexity**: Some fields contain commas within quotes, requiring proper CSV parsing
2. **Label variety**: The `type` column shows many different labels, some very specific
3. **Data completeness**: Some columns (tags, summary, source) appear empty in samples

### Recommendations:
- Use Python's `csv` module or pandas with proper quoting for parsing
- Consider grouping similar `type` labels for classification tasks
- Focus on `content` and `type` columns for most ML tasks

## Integration with Project

### In your code:
```python
import pandas as pd

# Load a sample
df = pd.read_csv('data/FakeNewsCorpus/fakenews_small_1000.csv')

# For text classification
X = df['content']  # Features
y = df['type']     # Labels
```

### Project structure:
```
lessons/data-analysis-progress/
├── data/
│   └── FakeNewsCorpus/
│       ├── README.md              # This file
│       ├── fakenews_test_10.csv   # 10 rows for testing
│       ├── fakenews_tiny_100.csv  # 100 rows
│       ├── fakenews_small_1000.csv # 1000 rows (recommended)
│       ├── fakenews_medium_5000.csv # 5000 rows
│       └── fakenews_stratified_1000.csv # Attempt at balanced
└── ... other project files
```

## Next Steps

1. **Analyze label distribution** in the samples
2. **Preprocess text data** (cleaning, tokenization)
3. **Train classification models** using the samples
4. **Compare performance** across different sample sizes
5. **Consider creating a balanced dataset** if needed for your specific task

## Documentation
- **Official README:** `OFFICIAL_README.md` (moved from Downloads)
- **Project Documentation:** This file (`README.md`)
- **Dataset Resources:** `../../DATASET_RESOURCES.md`

## Created
- **Date:** 2026-04-14
- **Purpose:** Data Analysis and Progress course project
- **Original dataset:** FakeNewsCorpus v1.0 (29.3GB, 9.4M articles)
- **Sampling method:** First N rows (sequential sampling)
- **Official Source:** https://github.com/several27/FakeNewsCorpus