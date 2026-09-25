import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("titanic_cleaned.csv")

X = df.drop("survived", axis=1)
y = df["survived"]


# ============================================================
# 2. STRATIFIED TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 3. PREPROCESSING
# ============================================================

numeric_features = [
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]

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
# 4. RANDOM FOREST WITH OOB SCORE
# ============================================================

rf = RandomForestClassifier(
    random_state=42,
    oob_score=True,
    bootstrap=True
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", rf)
    ]
)


# ============================================================
# 5. HYPERPARAMETER GRID
# ============================================================

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__max_features": ["sqrt", "log2"]
}


# ============================================================
# 6. GRID SEARCH
# ============================================================

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

print("=" * 70)
print("STARTING RANDOM FOREST GRID SEARCH")
print("=" * 70)

grid_search.fit(X_train, y_train)


# ============================================================
# 7. RESULTS
# ============================================================

print("\n" + "=" * 70)
print("BEST HYPERPARAMETERS")
print("=" * 70)

print(grid_search.best_params_)

print("\nBest Cross-Validation F1 Score:")
print(round(grid_search.best_score_, 4))


# ============================================================
# 8. OOB SCORE
# ============================================================

best_pipeline = grid_search.best_estimator_

oob_score = best_pipeline.named_steps["model"].oob_score_

print("\nOOB Score:")
print(round(oob_score, 4))


# ============================================================
# 9. TEST SET PERFORMANCE
# ============================================================

test_accuracy = best_pipeline.score(X_test, y_test)

print("\nTest Accuracy:")
print(round(test_accuracy, 4))


print("\n" + "=" * 70)
print("RANDOM FOREST TUNING COMPLETED")
print("=" * 70)