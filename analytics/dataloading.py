import pandas as pd
import seaborn as sns

# Load Titanic dataset once
df = sns.load_dataset("titanic")

print("Titanic dataset loaded successfully!")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

# Save immediately as offline fallback
df.to_csv("titanic.csv", index=False)

print("\nTitanic dataset saved as titanic.csv")