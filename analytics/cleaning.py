import pandas as pd

# Load the locally saved Titanic dataset
df = pd.read_csv("titanic.csv")

print("Original shape:", df.shape)

# ============================================================
# 1. Drop rows with missing embarked
#    Missing percentage = 0.22% (< 5%)
# ============================================================

df = df.dropna(subset=["embarked"])

# ============================================================
# 2. Impute missing age with median
#    Missing percentage = 19.87% (5–30%)
# ============================================================

age_median = df["age"].median()
df["age"] = df["age"].fillna(age_median)

# ============================================================
# 3. Drop columns with very high/redundant information
# ============================================================

df = df.drop(
    columns=[
        "deck",
        "class",
        "who",
        "adult_male",
        "embark_town",
        "alive",
        "alone"
    ]
)

print("\nCleaned shape:", df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nCleaned columns:")
print(df.columns.tolist())

# Save cleaned dataset
df.to_csv("titanic_cleaned.csv", index=False)

print("\nCleaned dataset saved as titanic_cleaned.csv")