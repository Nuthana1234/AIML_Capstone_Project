import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATA
# ============================================================

# Get the folder containing this Python file
BASE_DIR = Path(__file__).resolve().parent

# Dataset is in the same analytics folder as this script
DATA_PATH = BASE_DIR / "titanic_cleaned.csv"

df = pd.read_csv(DATA_PATH)

# Target
y = df["fare"]

# Use all other available features
X = df.drop("fare", axis=1)


# ============================================================
# 2. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("=" * 70)
print("FARE REGRESSION")
print("=" * 70)

print("\nTraining shape:", X_train.shape)
print("Testing shape :", X_test.shape)


# ============================================================
# 3. DEFINE FEATURES
# ============================================================

numeric_features = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch"
]

categorical_features = [
    "sex",
    "embarked"
]


# ============================================================
# 4. PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ============================================================
# 5. LINEAR REGRESSION PIPELINE
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# ============================================================
# 6. TRAIN MODEL
# ============================================================

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# ============================================================
# 7. REGRESSION METRICS
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


# Number of test observations
n = len(y_test)

# Number of predictors after preprocessing
X_test_transformed = model.named_steps[
    "preprocessor"
].transform(X_test)

p = X_test_transformed.shape[1]

adjusted_r2 = 1 - (
    (1 - r2) * (n - 1) / (n - p - 1)
)


# ============================================================
# 8. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("REGRESSION RESULTS")
print("=" * 70)

print("MAE        :", round(mae, 4))
print("RMSE       :", round(rmse, 4))
print("R²         :", round(r2, 4))
print("Adjusted R²:", round(adjusted_r2, 4))

print("\nNumber of observations (n):", n)
print("Number of predictors (p):", p)


# ============================================================
# 9. RESIDUALS
# ============================================================

residuals = y_test - y_pred

print("\nResidual statistics:")
print(pd.Series(residuals).describe())


# ============================================================
# 10. RESIDUAL PLOT
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Fare")
plt.ylabel("Residuals")
plt.title("Residual Plot - Fare Regression")

plt.tight_layout()

# Save plot in the analytics folder
PLOT_PATH = BASE_DIR / "fare_regression_residuals.png"

plt.savefig(
    PLOT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 11. HETEROSCEDASTICITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("HETEROSCEDASTICITY CHECK")
print("=" * 70)

print(
    "Inspect the residual plot for a funnel-shaped pattern."
)

print(
    "If residual spread increases as predicted fare increases, "
    "this indicates heteroscedasticity."
)

print(
    "If the residuals have approximately constant spread around "
    "zero, there is no strong visual evidence of heteroscedasticity."
)

print("\nResidual plot saved to:")
print(PLOT_PATH)

print("\nFare regression completed successfully.")