import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned Titanic dataset
df = pd.read_csv("titanic_cleaned.csv")


# ============================================================
# 1. Survival Rate by Sex — Boolean Masking
# ============================================================

female_mask = df["sex"] == "female"
male_mask = df["sex"] == "male"

survival_by_sex = pd.DataFrame({
    "sex": ["female", "male"],
    "survival_rate": [
        df.loc[female_mask, "survived"].mean(),
        df.loc[male_mask, "survived"].mean()
    ]
})

print("=" * 70)
print("SURVIVAL RATE BY SEX")
print("=" * 70)
print(survival_by_sex)

plt.figure(figsize=(7, 5))
sns.barplot(
    data=survival_by_sex,
    x="sex",
    y="survival_rate"
)
plt.ylabel("Survival Rate")
plt.xlabel("Sex")
plt.title("Survival Rate by Sex")
plt.tight_layout()
plt.show()


# ============================================================
# 2. Survival Rate by Passenger Class — Boolean Masking
# ============================================================

class_results = []

for passenger_class in sorted(df["pclass"].unique()):
    class_mask = df["pclass"] == passenger_class

    class_results.append({
        "pclass": passenger_class,
        "survival_rate": df.loc[class_mask, "survived"].mean()
    })

survival_by_class = pd.DataFrame(class_results)

print("\n" + "=" * 70)
print("SURVIVAL RATE BY PCLASS")
print("=" * 70)
print(survival_by_class)

plt.figure(figsize=(7, 5))
sns.barplot(
    data=survival_by_class,
    x="pclass",
    y="survival_rate"
)
plt.ylabel("Survival Rate")
plt.xlabel("Passenger Class")
plt.title("Survival Rate by Passenger Class")
plt.tight_layout()
plt.show()


# ============================================================
# 3. Survival Rate by Sex + Passenger Class — Boolean Masking
# ============================================================

sex_class_results = []

for sex in sorted(df["sex"].unique()):
    for passenger_class in sorted(df["pclass"].unique()):

        combined_mask = (
            (df["sex"] == sex) &
            (df["pclass"] == passenger_class)
        )

        sex_class_results.append({
            "sex": sex,
            "pclass": passenger_class,
            "survival_rate": df.loc[
                combined_mask,
                "survived"
            ].mean()
        })

survival_by_sex_class = pd.DataFrame(sex_class_results)

print("\n" + "=" * 70)
print("SURVIVAL RATE BY SEX AND PCLASS")
print("=" * 70)
print(survival_by_sex_class)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=survival_by_sex_class,
    x="pclass",
    y="survival_rate",
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