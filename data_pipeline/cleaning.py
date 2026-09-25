import pandas as pd
import numpy as np


# --------------------------------------------------
# 1. Load raw scraped data
# --------------------------------------------------

df = pd.read_csv("raw_books.csv")

print("Raw data shape:", df.shape)
print("\nRaw columns:")
print(df.columns.tolist())


# --------------------------------------------------
# 2. Clean GBP price
# --------------------------------------------------

def clean_price(value):
    try:
        return float(str(value).replace("£", "").strip())
    except (ValueError, TypeError):
        return np.nan


df["price_gbp"] = df["price_gbp_raw"].apply(clean_price)


# --------------------------------------------------
# 3. Clean star rating
# --------------------------------------------------

rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

df["rating"] = df["rating_raw"].map(rating_map)


# --------------------------------------------------
# 4. Clean availability
# --------------------------------------------------

df["in_stock"] = (
    df["availability_raw"]
    .str.contains("In stock", case=False, na=False)
)


# --------------------------------------------------
# 5. Handle parsing failures
# --------------------------------------------------

print("\nMissing values before handling:")
print(df[["price_gbp", "rating", "in_stock"]].isna().sum())

# Numeric parsing failures are handled using median imputation.
df["price_gbp"] = df["price_gbp"].fillna(
    df["price_gbp"].median()
)

df["rating"] = df["rating"].fillna(
    df["rating"].median()
).round().astype(int)


# --------------------------------------------------
# 6. Convert GBP to INR
# --------------------------------------------------

GBP_TO_INR = 105.50

df["price_inr"] = df["price_gbp"] * GBP_TO_INR


# --------------------------------------------------
# 7. Keep required columns
# --------------------------------------------------

df = df[
    [
        "title",
        "price_gbp",
        "price_inr",
        "rating",
        "in_stock",
        "category",
        "book_url"
    ]
]


# --------------------------------------------------
# 8. Verify data types
# --------------------------------------------------

print("\nCleaned data types:")
print(df.dtypes)


# --------------------------------------------------
# 9. Display cleaned data
# --------------------------------------------------

print("\nFirst 5 cleaned rows:")
print(df.head())


# --------------------------------------------------
# 10. Save cleaned dataset
# --------------------------------------------------

df.to_csv(
    "cleaned_books.csv",
    index=False
)

print("\nCleaned data saved successfully!")
print("Final shape:", df.shape)