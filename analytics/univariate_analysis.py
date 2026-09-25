import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned Titanic dataset
df = pd.read_csv("titanic_cleaned.csv")

# ============================================================
# 1. Histograms
# ============================================================

plt.figure(figsize=(8, 5))
plt.hist(df["age"], bins=20, edgecolor="black")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Distribution of Passenger Age")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(df["fare"], bins=20, edgecolor="black")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.title("Distribution of Passenger Fare")
plt.tight_layout()
plt.show()


# ============================================================
# 2. Box Plots
# ============================================================

plt.figure(figsize=(8, 5))
plt.boxplot(df["age"])
plt.ylabel("Age")
plt.title("Box Plot of Passenger Age")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.boxplot(df["fare"])
plt.ylabel("Fare")
plt.title("Box Plot of Passenger Fare")
plt.tight_layout()
plt.show()


# ============================================================
# 3. IQR Outlier Counts
# ============================================================

def count_iqr_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = series[
        (series < lower_bound) |
        (series > upper_bound)
    ]

    return len(outliers), q1, q3, iqr, lower_bound, upper_bound


age_outliers = count_iqr_outliers(df["age"])
fare_outliers = count_iqr_outliers(df["fare"])

print("=" * 70)
print("IQR OUTLIER ANALYSIS")
print("=" * 70)

print("\nAge:")
print("Q1:", age_outliers[1])
print("Q3:", age_outliers[2])
print("IQR:", age_outliers[3])
print("Lower Bound:", age_outliers[4])
print("Upper Bound:", age_outliers[5])
print("Number of outliers:", age_outliers[0])

print("\nFare:")
print("Q1:", fare_outliers[1])
print("Q3:", fare_outliers[2])
print("IQR:", fare_outliers[3])
print("Lower Bound:", fare_outliers[4])
print("Upper Bound:", fare_outliers[5])
print("Number of outliers:", fare_outliers[0])


# ============================================================
# 4. Fare Mean, Median and Mode
# ============================================================

fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode()[0]

print("\n" + "=" * 70)
print("FARE STATISTICS")
print("=" * 70)

print("Mean:", fare_mean)
print("Median:", fare_median)
print("Mode:", fare_mode)


# ============================================================
# 5. Skewness Classification
# ============================================================

print("\n" + "=" * 70)
print("SKEWNESS CLASSIFICATION")
print("=" * 70)

if fare_mean > fare_median > fare_mode:
    print("Fare distribution: Positively skewed (right-skewed)")
elif fare_mean < fare_median < fare_mode:
    print("Fare distribution: Negatively skewed (left-skewed)")
else:
    print("Fare distribution: Not clearly classified using mean/median/mode ordering")