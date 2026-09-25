import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load cleaned Titanic dataset
df = pd.read_csv("titanic_cleaned.csv")

# Select variables for standardization
features = ["age", "fare"]

print("BEFORE STANDARDIZATION")
print(df[features].describe().loc[["mean", "std"]])

# Standardize using z-score
scaler = StandardScaler()

df_standardized = df.copy()

df_standardized[features] = scaler.fit_transform(
    df[features]
)

print("\n" + "=" * 70)
print("AFTER STANDARDIZATION")
print("=" * 70)

print(
    df_standardized[features]
    .describe()
    .loc[["mean", "std"]]
)

print("\nFirst 5 standardized rows:")
print(df_standardized[features].head())

print("\nStandardization completed successfully.")
print("NOTE: These standardized values are for EDA sanity checking only.")
print("They will NOT be used as the modeling input.")