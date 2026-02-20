# Student Score Analysis

> **Dataset:** This project uses the [Students Exam Scores dataset](https://www.kaggle.com/datasets/desalegngeb/students-exam-scores) from Kaggle, originally uploaded by Desalegn Geb. The CSV files are not included in this repository — download the dataset from Kaggle and place the raw file at `StudentScoresData_Expanded.csv` before running the notebooks.

End-to-end data science project analysing factors that influence student academic performance, building from exploratory analysis through predictive modeling. The dataset contains 30,641 student records with demographic, behavioural, and socioeconomic features alongside math, reading, and writing scores.

## Project Structure

```
StudentScores/
├── 01_data_exploration.ipynb
├── 02_single_factor_analysis.ipynb
├── 03_multi_factor_pivot_analysis.ipynb
├── 04_correlation_and_feature_engineering.ipynb
├── 05_linear_regression_modeling.ipynb
├── 06_tree_based_modeling.ipynb
├── StudentScoresData_Expanded.csv       # raw data
├── ExpandedScoresData_Cleaned.csv       # after imputation
└── ExpandedScoresData_Engineered.csv    # with engineered features
```

## Notebooks

| # | Notebook | Description |
|---|---|---|
| 01 | Data Exploration | Load raw data, rename columns to snake_case, impute missing values (`unknown` for categoricals, median for `num_siblings`), inspect score distributions and value counts |
| 02 | Single-Factor Analysis | Test each categorical variable independently against avg_score; rank factors by gap between best and worst categories |
| 03 | Multi-Factor Pivot Analysis | Examine pairwise interactions between the top factors using pivot tables and heatmaps; quantify combined disadvantages |
| 04 | Correlation & Feature Engineering | Create ordinal encodings (`parent_ed_ordinal`, `study_hours_ordinal`), composite proxies (`ses_proxy`, `effort_level`), subject-gap features (`math_writing_gap`, `math_reading_gap`), and performance tiers (`perf_tier`) |
| 05 | Linear Regression Modeling | Baseline regression (R² = 0.263, RMSE = 12.32) and binary at-risk classification (75% accuracy, AUC = 0.737) using logistic regression |
| 06 | Tree-Based Modeling | Random Forest, Gradient Boosting, XGBoost, and LightGBM; cross-validation; GridSearchCV hyperparameter tuning; feature importance analysis |

## Key Findings

**Strongest single predictors (by avg_score gap):**
1. Parental education — 10.9 pt gap (master's vs some high school)
2. Lunch type (SES proxy) — 9.6 pt gap (standard vs free/reduced)
3. Ethnic group — 9.1 pt gap (group E vs group A)
4. Test preparation — 7.0 pt gap (completed vs none)

**Modeling results:**

| Model | Test R² | Test RMSE |
|---|---|---|
| Linear Regression | **0.263** | 12.32 |
| XGBoost (tuned) | 0.258 | 12.37 |
| Gradient Boosting | 0.257 | 12.37 |
| LightGBM | 0.257 | 12.38 |
| Random Forest | 0.221 | 12.67 |

All models converge on ~26% explained variance. The ceiling is a property of the feature set — demographic snapshots alone cannot capture the remaining ~74% of variance (prior knowledge, instruction quality, individual motivation). Linear regression is the practical deployment choice: best test R², simplest to interpret.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm jupyter
```

Then run notebooks in order (01 → 06); each notebook reads the CSV produced by the previous one.

## Data

The raw dataset (`StudentScoresData_Expanded.csv`) contains 30,641 rows and 14 columns: three score columns (`MathScore`, `ReadingScore`, `WritingScore`) and 11 demographic/behavioural features. 37% of rows have at least one missing value, handled via semantic imputation rather than row deletion.
