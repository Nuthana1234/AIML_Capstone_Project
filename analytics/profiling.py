import pandas as pd

# Load the locally saved Titanic dataset
df = pd.read_csv("titanic.csv")

# ============================================================
# 1. Dataset shape
# ============================================================

print("DATASET SHAPE")
print(df.shape)

# ============================================================
# 2. Dataset information
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

df.info()

# ============================================================
# 3. Descriptive statistics
# ============================================================

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

print(df.describe())

# ============================================================
# 4. Missing values
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing_count = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100

missing_summary = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing Percentage": missing_percentage.round(2)
})

print(missing_summary)

# Show only columns containing missing values
print("\nColumns with missing values:")
print(missing_summary[missing_summary["Missing Count"] > 0])