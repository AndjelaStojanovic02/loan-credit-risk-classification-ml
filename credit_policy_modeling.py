"""Loan credit policy classification project.

This script loads a loan dataset, performs basic exploratory analysis,
detects possible anomalies, trains several classification models, and
compares their performance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "loan_data.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the loan dataset from a CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def print_dataset_summary(dataframe: pd.DataFrame) -> None:
    """Print basic information about the dataset."""
    print("Dataset shape:", dataframe.shape)
    print("\nColumns:")
    print(dataframe.dtypes)
    print("\nMissing values:")
    print(dataframe.isna().sum())
    print("\nTarget distribution:")
    print(dataframe["credit.policy"].value_counts(normalize=True).round(3))


def save_exploratory_plots(dataframe: pd.DataFrame) -> None:
    """Create and save basic exploratory analysis plots."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    dataframe.hist(figsize=(12, 10), bins=20)
    plt.suptitle("Histograms of numerical attributes")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "numeric_histograms.png", dpi=150)
    plt.close()

    purpose_distribution = dataframe.groupby(["purpose", "credit.policy"]).size().unstack()
    purpose_distribution.plot(kind="bar", stacked=True, figsize=(12, 6))
    plt.title("Credit policy distribution by loan purpose")
    plt.xlabel("Loan purpose")
    plt.ylabel("Number of records")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_by_purpose.png", dpi=150)
    plt.close()

    numeric_data = dataframe.select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.matshow(correlation_matrix)
    fig.colorbar(image)
    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_yticks(range(len(correlation_matrix.columns)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=90)
    ax.set_yticklabels(correlation_matrix.columns)
    plt.title("Correlation matrix of numerical attributes", pad=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=150)
    plt.close()

    dataframe["credit.policy"].value_counts().plot(
        kind="pie", autopct="%1.1f%%", startangle=90, figsize=(7, 7)
    )
    plt.title("Target variable distribution")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=150)
    plt.close()


def detect_anomalies(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Detect possible anomalies using Isolation Forest."""
    numeric_data = dataframe.select_dtypes(include=[np.number])
    isolation_forest = IsolationForest(contamination=0.10, random_state=RANDOM_STATE)
    outlier_labels = isolation_forest.fit_predict(numeric_data)
    anomalies = dataframe.loc[outlier_labels == -1].copy()
    print(f"\nPossible anomalies detected with Isolation Forest: {len(anomalies)}")
    return anomalies


def split_features_and_target(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split the dataframe into feature matrix X and target vector y."""
    X = dataframe.drop(columns=["credit.policy"])
    y = dataframe["credit.policy"]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing for numerical and categorical columns."""
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def build_models() -> Dict[str, object]:
    """Define all classification models used in the project."""
    base_models = [
        ("random_forest", RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)),
        ("gradient_boosting", GradientBoostingClassifier(n_estimators=50, random_state=RANDOM_STATE)),
    ]

    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=RANDOM_STATE),
        "Stacking Classifier": StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        ),
    }


def build_pipeline(model: object, preprocessor: ColumnTransformer) -> ImbPipeline:
    """Create a modeling pipeline with preprocessing, balancing, and model training."""
    return ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote_tomek", SMOTETomek(random_state=RANDOM_STATE)),
            ("model", model),
        ]
    )


def evaluate_model(model_name: str, pipeline: ImbPipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate a fitted model and return its metrics."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "log_loss": log_loss(y_test, y_prob),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    print(f"\nModel: {model_name}")
    for metric_name, metric_value in metrics.items():
        if metric_name != "model":
            print(f"{metric_name}: {metric_value:.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return metrics


def save_confusion_matrix(model_name: str, pipeline: ImbPipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Save the confusion matrix plot for one model."""
    y_pred = pipeline.predict(X_test)
    display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    display.ax_.set_title(f"Confusion Matrix - {model_name}")
    filename = model_name.lower().replace(" ", "_").replace("-", "")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"confusion_matrix_{filename}.png", dpi=150)
    plt.close()


def save_roc_curves(fitted_pipelines: Dict[str, ImbPipeline], X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Save one ROC curve chart with all trained models."""
    plt.figure(figsize=(9, 7))

    for model_name, pipeline in fitted_pipelines.items():
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curves.png", dpi=150)
    plt.close()


def tune_selected_models(preprocessor: ColumnTransformer, X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Run lightweight hyperparameter tuning for selected models."""
    search_spaces = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"model__max_depth": [None, 5, 10, 20]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {"model__n_estimators": [50], "model__max_depth": [None, 10]},
        ),
    }

    print("\nHyperparameter tuning results:")
    for model_name, (model, parameters) in search_spaces.items():
        pipeline = build_pipeline(model, preprocessor)
        grid_search = GridSearchCV(pipeline, parameters, cv=2, scoring="f1", n_jobs=1)
        grid_search.fit(X_train, y_train)
        print(f"{model_name}: best parameters = {grid_search.best_params_}, best F1 = {grid_search.best_score_:.3f}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate credit policy classification models.")
    parser.add_argument("--tune", action="store_true", help="Run optional hyperparameter tuning.")
    parser.add_argument("--skip-cv", action="store_true", help="Skip cross-validation for a faster run.")
    parser.add_argument("--sample-size", type=int, default=None, help="Use a random sample for a faster demo run.")
    parser.add_argument("--quick", action="store_true", help="Run only a small subset of models for a quick smoke test.")
    return parser.parse_args()


def main() -> None:
    """Run the full project workflow."""
    args = parse_arguments()
    dataframe = load_data()
    if args.sample_size is not None and args.sample_size < len(dataframe):
        dataframe = dataframe.sample(n=args.sample_size, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"Using a random sample with {args.sample_size} rows.")
    print_dataset_summary(dataframe)
    save_exploratory_plots(dataframe)
    detect_anomalies(dataframe)

    X, y = split_features_and_target(dataframe)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)
    models = build_models()
    if args.quick:
        models = {name: model for name, model in models.items() if name in ["Logistic Regression", "Decision Tree"]}
    fitted_pipelines = {}
    results = []

    for model_name, model in models.items():
        pipeline = build_pipeline(model, preprocessor)
        if not args.skip_cv:
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=2, scoring="f1", n_jobs=1)
            print(f"\n{model_name} cross-validation F1 scores: {np.round(cv_scores, 3)}")
            print(f"{model_name} mean cross-validation F1: {cv_scores.mean():.3f}")

        pipeline.fit(X_train, y_train)
        fitted_pipelines[model_name] = pipeline
        results.append(evaluate_model(model_name, pipeline, X_test, y_test))
        save_confusion_matrix(model_name, pipeline, X_test, y_test)

    results_dataframe = pd.DataFrame(results).sort_values(by="f1", ascending=False)
    print("\nModel comparison:")
    print(results_dataframe.to_string(index=False))

    save_roc_curves(fitted_pipelines, X_test, y_test)

    if args.tune:
        tune_selected_models(preprocessor, X_train, y_train)


if __name__ == "__main__":
    main()
