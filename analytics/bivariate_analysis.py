import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned Titanic dataset
df = pd.read_csv("titanic_cleaned.csv")


# ============================================================
# 1. Survival Rate by Sex
# ============================================================

survival_by_sex = (
    df.groupby("sex")["survived"]
    .mean()
    .reset_index()
)

print("=" * 70)
print("SURVIVAL RATE BY SEX")
print("=" * 70)
print(survival_by_sex)

plt.figure(figsize=(7, 5))
sns.barplot(
    data=survival_by_sex,
    x="sex",
    y="survived"
)
plt.ylabel("Survival Rate")
plt.xlabel("Sex")
plt.title("Survival Rate by Sex")
plt.tight_layout()
plt.show()


# ============================================================
# 2. Survival Rate by Passenger Class
# ============================================================

survival_by_class = (
    df.groupby("pclass")["survived"]
    .mean()
    .reset_index()
)

print("\n" + "=" * 70)
print("SURVIVAL RATE BY PCLASS")
print("=" * 70)
print(survival_by_class)

plt.figure(figsize=(7, 5))
sns.barplot(
    data=survival_by_class,
    x="pclass",
    y="survived"
)
plt.ylabel("Survival Rate")
plt.xlabel("Passenger Class")
plt.title("Survival Rate by Passenger Class")
plt.tight_layout()
plt.show()


# ============================================================
# 3. Survival Rate by Sex + Passenger Class
# ============================================================

survival_by_sex_class = (
    df.groupby(["sex", "pclass"])["survived"]
    .mean()
    .reset_index()
)

print("\n" + "=" * 70)
print("SURVIVAL RATE BY SEX AND PCLASS")
print("=" * 70)
print(survival_by_sex_class)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=survival_by_sex_class,
    x="pclass",
    y="survived",
    hue="sex"
)
plt.ylabel("Survival Rate")
plt.xlabel("Passenger Class")
plt.title("Survival Rate by Sex and Passenger Class")
plt.tight_layout()
plt.show()


# ============================================================
# 4. Correlation Matrix
# ============================================================

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation_matrix = df[correlation_columns].corr()

print("\n" + "=" * 70)
print("CORRELATION MATRIX")
print("=" * 70)
print(correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


# ============================================================
# 5. Two Strongest Absolute Off-Diagonal Correlations
# ============================================================

corr_pairs = correlation_matrix.where(
    ~pd.DataFrame(
        __import__("numpy").eye(len(correlation_matrix)),
        index=correlation_matrix.index,
        columns=correlation_matrix.columns
    ).astype(bool)
).stack()

corr_pairs = corr_pairs.abs().sort_values(ascending=False)

print("\n" + "=" * 70)
print("STRONGEST ABSOLUTE CORRELATIONS")
print("=" * 70)

print(corr_pairs.head(2))