import sqlite3
import pandas as pd

# Load cleaned data
df = pd.read_csv("cleaned_books.csv")

print("Loaded cleaned data:", df.shape)


# Connect to SQLite database
# If books.db does not exist, SQLite will create it
connection = sqlite3.connect("books.db")
cursor = connection.cursor()


# --------------------------------------------------
# Create Categories Table
# --------------------------------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT NOT NULL UNIQUE
)
""")


# --------------------------------------------------
# Create Books Table
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
# Insert Categories
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
# Insert Books
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

    # Insert book
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


# Save changes
connection.commit()


# --------------------------------------------------
# Verify Database
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
# Test JOIN
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


# Close connection
connection.close()

print("\nDatabase connection closed.")