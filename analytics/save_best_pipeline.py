import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. LOAD CLEANED DATA
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
# 4. BEST TUNED RANDOM FOREST
# ============================================================

random_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    max_features="sqrt",
    random_state=42,
    oob_score=True
)


# ============================================================
# 5. COMPLETE PIPELINE
# ============================================================

full_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", random_forest)
    ]
)


# ============================================================
# 6. TRAIN COMPLETE PIPELINE
# ============================================================

full_pipeline.fit(X_train, y_train)


# ============================================================
# 7. SAVE COMPLETE PIPELINE
# ============================================================

model_path = "best_titanic_model.joblib"

joblib.dump(
    full_pipeline,
    model_path
)

print("=" * 70)
print("MODEL SAVED")
print("=" * 70)

print("File:", model_path)


# ============================================================
# 8. LOAD THE SAVED PIPELINE
# ============================================================

loaded_pipeline = joblib.load(model_path)

print("\nModel reloaded successfully!")


# ============================================================
# 9. PREDICT USING RAW INPUT
# ============================================================

sample_input = X_test.iloc[[0]]

prediction = loaded_pipeline.predict(sample_input)

probability = loaded_pipeline.predict_proba(sample_input)


print("\n" + "=" * 70)
print("RAW INPUT PREDICTION")
print("=" * 70)

print("\nRaw input:")
print(sample_input)

print("\nPredicted survival:")
print(prediction[0])

print("\nPrediction probabilities:")
print(probability[0])


# ============================================================
# 10. CONFIRM PIPELINE
# ============================================================

print("\n" + "=" * 70)
print("PIPELINE VERIFICATION")
print("=" * 70)

print("Preprocessing + model are stored together.")
print("Prediction was successfully generated from raw input.")
print("Joblib save/reload verification completed successfully.")