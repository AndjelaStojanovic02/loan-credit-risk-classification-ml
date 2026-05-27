# Loan Credit Policy Classification

This project applies machine learning techniques to predict whether a loan applicant satisfies a predefined credit policy. It was developed as part of an Artificial Intelligence course project and later refactored and cleaned for portfolio use.

## Project overview

The dataset contains loan records with financial and credit-related attributes such as interest rate, installment amount, annual income, debt-to-income ratio, FICO score, credit history length, revolving balance, delinquencies, public records, and loan purpose.

The target variable is:

- `credit.policy` — `1` if the applicant meets the credit policy, `0` otherwise.

The goal is to compare different classification models and evaluate their performance using standard metrics and ROC-AUC.

## Objectives

- Perform exploratory data analysis (EDA) and visualize key patterns.
- Detect anomalies using Isolation Forest.
- Address class imbalance using SMOTE-Tomek.
- Train and compare multiple classification models.
- Apply cross-validation and basic hyperparameter tuning.
- Evaluate models using accuracy, precision, recall, F1-score, MCC, log loss, confusion matrix, and ROC-AUC.

## Models used

The project includes both basic and ensemble models:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest — bagging-based ensemble model
- Gradient Boosting — boosting-based ensemble model
- Stacking Classifier — ensemble model that combines Random Forest and Gradient Boosting with Logistic Regression as the final estimator

## Project structure

```text
loan-credit-policy-classification/
├── data/
│   └── loan_data.csv
├── docs/
│   └── project_documentation_en.md
├── reports/
│   └── figures/
├── src/
│   └── credit_policy_modeling.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Dataset columns

| Column | Description |
|---|---|
| `credit.policy` | Target variable. Shows whether the applicant meets the credit policy. |
| `purpose` | Purpose of the loan. |
| `int.rate` | Interest rate of the loan. |
| `installment` | Monthly installment amount. |
| `log.annual.inc` | Natural log of the applicant's annual income. |
| `dti` | Debt-to-income ratio. |
| `fico` | FICO credit score. |
| `days.with.cr.line` | Number of days the applicant has had a credit line. |
| `revol.bal` | Revolving balance. |
| `revol.util` | Revolving line utilization rate. |
| `inq.last.6mths` | Number of credit inquiries in the last 6 months. |
| `delinq.2yrs` | Number of delinquencies in the last 2 years. |
| `pub.rec` | Number of derogatory public records. |
| `not.fully.paid` | Shows whether the loan was not fully paid. |

## Setup

Clone the repository and install the dependencies:

```bash
git clone https://github.com/AndjelaStojanovic02/loan-credit-risk-classification-ml.git
cd loan-credit-policy-classification
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to run

Run the full pipeline:
```bash
python src/credit_policy_modeling.py 
```

Faster execution (skip cross-validation):

```bash
python src/credit_policy_modeling.py --skip-cv
```

Quick test run:
```bash
python src/credit_policy_modeling.py --sample-size 1000 --quick
```

For optional hyperparameter tuning, run:

```bash
python src/credit_policy_modeling.py --tune
```

The script will:

1. Load the dataset from `data/loan_data.csv`.
2. Print dataset information.
3. Create exploratory analysis plots in `reports/figures/`.
4. Detect potential anomalies with Isolation Forest.
5. Train and evaluate all models.
6. Save ROC curve plots to `reports/figures/`.

Use `--skip-cv` for a faster portfolio demo run. Use `--sample-size 1000 --quick` when you only want to test the script quickly.


### Fast smoke test

To quickly check that the project runs end-to-end, use:

```bash
python src/credit_policy_modeling.py --skip-cv --sample-size 1000 --quick
```

This project is intended for educational and portfolio purposes.
The results may vary depending on preprocessing and hyperparameter settings.
