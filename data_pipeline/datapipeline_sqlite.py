import sqlite3
import pandas as pd


# --------------------------------------------------
# 1. Load cleaned data
# --------------------------------------------------

df = pd.read_csv("cleaned_books.csv")

print("Loaded cleaned data:", df.shape)


# --------------------------------------------------
# 2. Connect to SQLite database
# --------------------------------------------------

connection = sqlite3.connect("books.db")

cursor = connection.cursor()


# --------------------------------------------------
# 3. Create categories table
# --------------------------------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT NOT NULL UNIQUE
)
""")


# --------------------------------------------------
# 4. Create books table
# --------------------------------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER,
    book_url TEXT,

    FOREIGN KEY (category_id)
        REFERENCES categories(category_id)
)
""")


# --------------------------------------------------
# 5. Insert categories
# --------------------------------------------------

categories = df["category"].dropna().unique()

for category in categories:

    cursor.execute(
        """
        INSERT OR IGNORE INTO categories (category_name)
        VALUES (?)
        """,
        (category,)
    )


# --------------------------------------------------
# 6. Insert books
# --------------------------------------------------

for _, row in df.iterrows():

    # Find category ID
    cursor.execute(
        """
        SELECT category_id
        FROM categories
        WHERE category_name = ?
        """,
        (row["category"],)
    )

    category_id = cursor.fetchone()[0]

    cursor.execute(
        """
        INSERT INTO books (
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id,
            book_url
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["title"],
            row["price_gbp"],
            row["price_inr"],
            row["rating"],
            int(row["in_stock"]),
            category_id,
            row["book_url"]
        )
    )


# --------------------------------------------------
# 7. Save changes
# --------------------------------------------------

connection.commit()


# --------------------------------------------------
# 8. Verify database
# --------------------------------------------------

category_count = cursor.execute(
    "SELECT COUNT(*) FROM categories"
).fetchone()[0]

book_count = cursor.execute(
    "SELECT COUNT(*) FROM books"
).fetchone()[0]

print("\nDatabase created successfully!")

print("Number of categories:", category_count)
print("Number of books:", book_count)


# --------------------------------------------------
# 9. Test JOIN
# --------------------------------------------------

test_query = """
SELECT
    books.title,
    books.price_gbp,
    categories.category_name
FROM books
JOIN categories
    ON books.category_id = categories.category_id
LIMIT 5
"""

result = pd.read_sql(test_query, connection)

print("\nSample JOIN result:")
print(result)


# --------------------------------------------------
# 10. Close connection
# --------------------------------------------------

connection.close()

print("\nDatabase connection closed.")