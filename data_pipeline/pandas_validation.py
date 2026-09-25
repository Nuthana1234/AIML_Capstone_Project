import sqlite3
import pandas as pd

# Connect to SQLite database
connection = sqlite3.connect("books.db")

print("Connected to books.db successfully!\n")


# ============================================================
# 1. Read Query Result using pd.read_sql()
# ============================================================

query1 = """
SELECT title, rating, price_gbp
FROM books
WHERE rating = 5
"""

result1 = pd.read_sql(query1, connection)

print("RESULT 1: Five-star books")
print(result1)
print("\n" + "=" * 70 + "\n")


# ============================================================
# 2. Read JOIN Result using pd.read_sql()
# ============================================================

join_query = """
SELECT
    books.title,
    books.price_gbp,
    books.rating,
    categories.category_name
FROM books
JOIN categories
    ON books.category_id = categories.category_id
"""

sql_join_result = pd.read_sql(join_query, connection)

print("RESULT 2: SQL JOIN using pd.read_sql()")
print(sql_join_result.head())
print("\n" + "=" * 70 + "\n")


# ============================================================
# 3. Reproduce the same JOIN using pd.merge()
# ============================================================

books_df = pd.read_sql(
    """
    SELECT
        title,
        price_gbp,
        rating,
        category_id
    FROM books
    """,
    connection
)

categories_df = pd.read_sql(
    """
    SELECT
        category_id,
        category_name
    FROM categories
    """,
    connection
)

pandas_merge_result = pd.merge(
    books_df,
    categories_df,
    on="category_id",
    how="inner"
)

# Select the same columns and same order as SQL JOIN
pandas_merge_result = pandas_merge_result[
    [
        "title",
        "price_gbp",
        "rating",
        "category_name"
    ]
]

print("RESULT 3: JOIN reproduced using pd.merge()")
print(pandas_merge_result.head())
print("\n" + "=" * 70 + "\n")


# ============================================================
# 4. Compare SQL JOIN and Pandas JOIN
# ============================================================

sql_check = sql_join_result.sort_values(
    by=["title", "price_gbp", "rating", "category_name"]
).reset_index(drop=True)

pandas_check = pandas_merge_result.sort_values(
    by=["title", "price_gbp", "rating", "category_name"]
).reset_index(drop=True)

are_equal = sql_check.equals(pandas_check)

print("Are SQL JOIN and pandas.merge() results equivalent?")
print(are_equal)

if are_equal:
    print("\nSUCCESS: Both JOIN results are equivalent.")
else:
    print("\nWARNING: The JOIN results are different.")


# Close database connection
connection.close()

print("\nDatabase connection closed.")