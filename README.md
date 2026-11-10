# Credit Risk Pipeline

Production-grade machine learning pipeline for automated loan default risk prediction, tabular feature preprocessing, and threshold optimization.

## Overview

Underwriting workflows often struggle with manual risk assessment bottlenecks and data leakage during data preprocessing. When feature scaling or imputation happens outside cross-validation splits, models generalize poorly on unseen applicant pools. This tool solves that operational friction by encapsulating the entire data transformation, hyperparameter tuning, and evaluation workflow inside unified scikit-learn pipelines.

## How It Works

The execution pipeline processes financial tabular data through a rigorous, leakage-free architectural design:
1. **Feature Engineering:** Upstream transformations inject domain-specific metrics—specifically debt-to-income and loan-to-age ratios—directly into the DataFrame.
2. **Preprocessing Pipeline:** A `ColumnTransformer` routes numeric features through median imputation and categorical features through most-frequent imputation followed by one-hot encoding with unknown category handling.
3. **Ensemble Modeling:** A `RandomForestClassifier` with balanced class weights handles imbalanced credit distributions.
4. **Validation & Tuning:** `StratifiedKFold` cross-validation combined with `GridSearchCV` isolates optimal hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`) using ROC-AUC scoring.
5. **Threshold Optimization:** Rather than defaulting to a rigid 0.5 probability cutoff, the system computes Youden's J statistic across receiver operating characteristic curves to derive an empirically optimal decision boundary.
6. **Persistence:** Trained pipeline states and optimal thresholds are serialized using `joblib` to disk, enabling instant reloading for inference.

## Key Features

- Synthetic dataset generator simulating realistic financial distributions and class imbalance.
- Unified preprocessing and modeling pipeline preventing data leakage.
- Automated missing value handling for both numerical and categorical features.
- Dynamic classification threshold tuning via ROC curve analysis.
- Command-line evaluation interface for real-time applicant risk scoring.

## Tech Stack & Core Dependencies Breakdown

- **Python 3.10+:** Core runtime environment.
- **scikit-learn:** Provides `Pipeline`, `ColumnTransformer`, `RandomForestClassifier`, `GridSearchCV`, and metric utilities.
- **pandas:** Manages tabular DataFrame manipulation and feature engineering.
- **numpy:** Handles numerical array operations and synthetic data distributions.
- **joblib:** Manages model serialization and disk caching.
- **os / sys:** Standard library modules handling file existence checks and exit codes.

## Environment & Web-Based Quick Start

### GitHub Codespaces
1. Open the repository in a browser-based GitHub Codespace.
2. Ensure Python 3.10+ is active in the terminal environment.

### Local or Virtual Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install scikit-learn pandas numpy joblib
```

3. Execution
Run the script directly to trigger synthetic dataset generation, hyperparameter optimization, and the command-line evaluation prompt:
```bash
python main.py
```

## Repository Structure

```bash
credit-risk-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── README.md
├── main.py
└── requirements.txt
```

## Roadmap
**[ ] Type Stubs & Static Analysis:** Integrate strict type annotations and validate codebases using mypy.

**[ ] Asynchronous Batch Inference:** Migrate evaluation routines to support concurrent batch processing over asynchronous queues.

**[ ] Structured Audit Logging:** Replace standard print statements with structured JSON logging to improve traceability in production environments.