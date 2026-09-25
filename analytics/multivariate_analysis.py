import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned Titanic dataset
df = pd.read_csv("titanic_cleaned.csv")


# ============================================================
# Chart 1: Age + Fare + Survival
# ============================================================

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="age",
    y="fare",
    hue="survived",
    style="sex"
)

plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Age, Fare and Survival by Sex")
plt.tight_layout()
plt.show()


# ============================================================
# Chart 2: Sex + Pclass + Survival
# ============================================================

survival_sex_class = (
    df.groupby(["sex", "pclass"])["survived"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(9, 6))

sns.barplot(
    data=survival_sex_class,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.title("Survival Rate by Sex and Passenger Class")
plt.tight_layout()
plt.show()


# ============================================================
# Chart 3: Age + Pclass + Survival
# ============================================================

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="pclass",
    y="age",
    hue="survived"
)

plt.xlabel("Passenger Class")
plt.ylabel("Age")
plt.title("Age Distribution by Passenger Class and Survival")
plt.tight_layout()
plt.show()


# ============================================================
# Chart 4: Family Size + Fare + Survival
# ============================================================

df["family_size"] = df["sibsp"] + df["parch"] + 1

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="family_size",
    y="fare",
    hue="survived",
    size="pclass",
    sizes=(40, 200),
    alpha=0.7
)

plt.xlabel("Family Size")
plt.ylabel("Fare")
plt.title("Family Size, Fare and Survival")
plt.tight_layout()
plt.show()


# ============================================================
# Print summary information
# ============================================================

print("=" * 70)
print("MULTIVARIATE ANALYSIS COMPLETED")
print("=" * 70)

print("\nFamily size statistics:")
print(df["family_size"].describe())

print("\nSurvival rate by family size:")
print(
    df.groupby("family_size")["survived"]
    .mean()
    .round(3)
)

print("\nSurvival rate by sex and passenger class:")
print(
    df.groupby(["sex", "pclass"])["survived"]
    .mean()
    .round(3)
)