# Project Documentation

## 1. Data loading and basic setup

The original dataset was provided as an Excel file. It was converted to CSV so it could be loaded with `pandas.read_csv()`.

The target variable is `credit.policy`. The original version removed the `purpose` column, but the cleaned version keeps it and processes it with one-hot encoding inside a preprocessing pipeline.

## 2. Anomaly detection

Potential anomalies are detected with the Isolation Forest algorithm. The algorithm is applied to numerical columns only. Rows predicted as `-1` are treated as possible anomalies.

```python
isolation_forest = IsolationForest(contamination=0.10, random_state=42)
isolation_forest.fit(numeric_data)
outlier_labels = isolation_forest.predict(numeric_data)
anomalies = dataframe.loc[outlier_labels == -1]
```

This step is used for inspection and reporting. The cleaned version does not automatically remove these rows because anomaly removal should be based on business context, not only on an algorithmic flag.

## 3. Exploratory data analysis

The project includes several exploratory charts:

- histograms for numerical attributes,
- target distribution by loan purpose,
- correlation matrix for numerical attributes,
- pie chart for the target variable distribution.

These charts help understand the dataset before model training.

## 4. Feature and target selection

The target variable is:

```python
y = dataframe["credit.policy"]
```

The feature matrix contains all remaining columns:

```python
X = dataframe.drop(columns=["credit.policy"])
```

The cleaned version uses:

- `StandardScaler` for numerical columns,
- `OneHotEncoder` for categorical columns,
- `SMOTETomek` for class balancing inside the training pipeline.

## 5. Class balancing

The target variable is imbalanced, so the project uses SMOTE-Tomek:

- SMOTE creates synthetic examples of the minority class.
- Tomek Links remove borderline or overlapping examples.

This helps the models learn patterns from the minority class without simply duplicating rows.

In the cleaned version, class balancing is applied only during training. This is important because applying SMOTE before the train-test split can leak information from the test set into the training process.

## 6. Models

The project compares the following models:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Gradient Boosting
- Stacking Classifier

Random Forest represents a bagging approach, Gradient Boosting represents a boosting approach, and Stacking combines multiple base models through a final meta-model.

## 7. Evaluation metrics

The models are evaluated with:

- accuracy,
- precision,
- recall,
- F1-score,
- Matthews Correlation Coefficient,
- log loss,
- confusion matrix,
- ROC-AUC.

Accuracy alone is not enough for imbalanced classification problems, so precision, recall, F1-score, MCC, and ROC-AUC give a better view of model quality.

## 8. ROC curve

The ROC curve shows the relationship between the true positive rate and the false positive rate at different classification thresholds. The AUC score summarizes the curve into one value. A higher AUC usually means the model separates the two classes better.

## 9. Feature selection

The original documentation mentioned feature selection as a planned task, but it was not implemented. This remains a good future improvement for the project.
