import pandas as pd


# ============================================================
# CLASSIFICATION RESULTS
# ============================================================

classification_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        0.7809,
        0.7697,
        0.7978
    ],
    "Precision": [
        0.7544,
        0.7143,
        0.7759
    ],
    "Recall": [
        0.6324,
        0.6618,
        0.6618
    ],
    "F1": [
        0.6880,
        0.6870,
        0.7143
    ],
    "ROC-AUC": [
        0.8265,
        0.7412,
        0.8211
    ]
})


# ============================================================
# TUNED RANDOM FOREST
# ============================================================

tuned_random_forest = {
    "Best Parameters": {
        "max_depth": 5,
        "max_features": "sqrt",
        "n_estimators": 100
    },
    "Best CV F1": 0.7436,
    "OOB Score": 0.8073,
    "Test Accuracy": 0.8034
}


# ============================================================
# REGRESSION RESULTS
# ============================================================

regression_results = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R²",
        "Adjusted R²"
    ],
    "Value": [
        21.0986,
        41.7021,
        0.3482,
        0.3173
    ]
})


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("=" * 70)
print("FINAL CLASSIFICATION COMPARISON")
print("=" * 70)

print(classification_results.to_string(index=False))


print("\n" + "=" * 70)
print("TUNED RANDOM FOREST")
print("=" * 70)

print("Best Parameters:")
print(tuned_random_forest["Best Parameters"])

print(
    "Best CV F1:",
    tuned_random_forest["Best CV F1"]
)

print(
    "OOB Score:",
    tuned_random_forest["OOB Score"]
)

print(
    "Test Accuracy:",
    tuned_random_forest["Test Accuracy"]
)


print("\n" + "=" * 70)
print("REGRESSION RESULTS")
print("=" * 70)

print(regression_results.to_string(index=False))


# ============================================================
# SAVE TABLES
# ============================================================

classification_results.to_csv(
    "classification_comparison.csv",
    index=False
)

regression_results.to_csv(
    "regression_results.csv",
    index=False
)

print("\nComparison tables saved successfully.")