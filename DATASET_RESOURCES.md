# Dataset Resources for Data Analysis Project

## Project Context
- **Course:** Data Analysis and Progress
- **Project:** News Classification Analysis
- **Primary Dataset:** News Classification from GitHub
- **Additional Datasets:** Fake news detection datasets for comparison/extension

## Primary Dataset

### News Classification Dataset
- **URL:** https://github.com/dolphinwesting248/news_classification
- **Description:** News articles with classification labels
- **Task:** Multi-class text classification
- **Repository:** Clone for access to dataset files
- **Usage:** Main dataset for the project

## Additional Datasets for Fake News Detection

### FakeNewsCorpus Dataset
- **URL:** https://github.com/several27/FakeNewsCorpus/releases/tag/v1.0
- **Official Repository:** https://github.com/several27/FakeNewsCorpus
- **Description:** Large collection of 9.4 million news articles labeled by type for fake news detection
- **Release:** v1.0 (specific version)
- **Format:** CSV file with news articles and labels
- **Original Size:** 29.3 GB (news_cleaned_2018_02_13.csv)
- **Total Articles:** 9,408,908 (from 745 domains)
- **Local Location:** `C:\Users\22130\Downloads\news.csv\news_cleaned_2018_02_13.csv`

#### Creation Methodology
1. **Source Domains:** 1001 domains from http://www.opensources.co/
2. **Additional Sources:** NYTimes and WebHose English News Articles for class balance
3. **Scraping:** Using Scrapy framework
4. **Processing:** HTML content extracted using `newspaper` library
5. **Labeling:** Articles inherit labels from their source domain

#### Label Types (11 categories)
| Type | Tag | Count | Description |
|------|-----|-------|-------------|
| Fake News | fake | 928,083 | Fabricate information or grossly distort news |
| Satire | satire | 146,080 | Humor, irony, exaggeration for commentary |
| Extreme Bias | bias | 1,300,444 | Particular point of view, may use propaganda |
| Conspiracy Theory | conspiracy | 905,981 | Promoters of conspiracy theories |
| State News | state | 0 | Government-sanctioned sources |
| Junk Science | junksci | 144,939 | Promote pseudoscience, metaphysics |
| Hate News | hate | 117,374 | Promote discrimination |
| Clickbait | clickbait | 292,201 | Credible content with misleading headlines |
| Proceed With Caution | unreliable | 319,830 | May be reliable but require verification |
| Political | political | 2,435,471 | Support certain political views |
| Credible | reliable | 1,920,139 | Traditional ethical journalism |

#### Sampled Versions (Created for Practical Use)
1. **`fakenews_test_10.csv`** - 10 rows (testing)
2. **`fakenews_tiny_100.csv`** - 100 rows (prototyping)
3. **`fakenews_small_1000.csv`** - 1000 rows (⭐ recommended for training)
4. **`fakenews_medium_5000.csv`** - 5000 rows (robust training)

#### Columns
id, domain, type, url, content, scraped_at, inserted_at, updated_at, title, authors, keywords, meta_keywords, meta_description, tags, summary, source

#### Use Case
- Fake news detection specific task
- Multi-class news classification (11 categories)
- Comparison with general news classification
- Data augmentation or transfer learning
- Studying different types of misinformation (bias, conspiracy, hate, etc.)

#### Limitations (From Official README)
- Not manually filtered (some labels may be incorrect)
- Some URLs may not point to actual articles
- Intended for ML training where these issues are less critical
- Dataset may become outdated for purposes other than content-based algorithms

### Other Fake News Datasets (Potential)
1. **LIAR Dataset** - Political fact-checking dataset
2. **FakeNewsNet** - Comprehensive fake news dataset with social context
3. **COVID-19 Fake News Dataset** - Pandemic-related misinformation
4. **Twitter Fake News** - Social media fake news detection

## Dataset Integration Strategy

### Option 1: Primary Analysis
- Use only the primary news classification dataset
- Focus on general news categorization

### Option 2: Comparative Analysis
- Use both primary dataset and FakeNewsCorpus
- Compare performance on different news types
- Analyze feature differences between real and fake news

### Option 3: Multi-task Learning
- Train models on combined datasets
- Learn shared representations for news analysis
- Specialize for specific tasks (classification vs. fake detection)

## Download Instructions

### Primary Dataset
```bash
git clone https://github.com/dolphinwesting248/news_classification.git
```

### FakeNewsCorpus Dataset
```bash
# Check the GitHub releases page for download options
# Likely a compressed file (zip, tar.gz) to download
# Or use direct download link from releases
```

## Data Preprocessing Considerations

### For News Classification Dataset:
- Text cleaning and normalization
- Label encoding for categories
- Train/validation/test split

### For FakeNewsCorpus:
- May require additional cleaning (HTML, special formats)
- Binary classification (real vs. fake) or multi-class
- Potential class imbalance issues

## Project Extension Ideas

### 1. Fake News Detection Module
- Add fake news detection as additional task
- Compare models on both classification and detection
- Analyze what features indicate fake news

### 2. Cross-Dataset Evaluation
- Train on one dataset, test on another
- Measure generalization across news domains
- Identify dataset-specific patterns

### 3. Ensemble Approach
- Combine predictions from models trained on different datasets
- Improve robustness through diversity
- Handle different types of news content

## Implementation Notes

### File Structure Update
```
lessons/data-analysis-progress/
├── data/
│   ├── news_classification/    # Primary dataset
│   └── FakeNewsCorpus/         # Additional dataset (if downloaded)
├── DATASET_RESOURCES.md        # This file
└── ... other project files
```

### Code Modifications
- Update data loading to handle multiple datasets
- Add configuration for dataset selection
- Implement comparative evaluation scripts

## Next Steps

1. **Download FakeNewsCorpus** to examine its structure
2. **Compare dataset characteristics** (size, format, labels)
3. **Decide on project scope** (single dataset or comparative)
4. **Update project plan** based on dataset decisions
5. **Implement data loading pipeline** for chosen datasets

## References

- FakeNewsCorpus GitHub: https://github.com/several27/FakeNewsCorpus
- Primary Dataset: https://github.com/dolphinwesting248/news_classification
- Created: 2026-04-14 (added FakeNewsCorpus resource)