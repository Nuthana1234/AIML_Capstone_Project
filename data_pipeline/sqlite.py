import sqlite3
import pandas as pd

# Connect to SQLite database
connection = sqlite3.connect("books.db")

print("Connected to books.db successfully!\n")


# --------------------------------------------------
# Query 1: SELECT + WHERE
# Find books with a rating of 5
# --------------------------------------------------

query1 = """
SELECT title, rating, price_gbp
FROM books
WHERE rating = 5
"""

result1 = pd.read_sql(query1, connection)

print("QUERY 1: Books with rating 5")
print(result1)
print("\n" + "=" * 70 + "\n")


# --------------------------------------------------
# Query 2: ORDER BY
# Find the most expensive books
# --------------------------------------------------

query2 = """
SELECT title, price_gbp
FROM books
ORDER BY price_gbp DESC
"""

result2 = pd.read_sql(query2, connection)

print("QUERY 2: Books ordered by price (highest first)")
print(result2)
print("\n" + "=" * 70 + "\n")


# --------------------------------------------------
# Query 3: LIMIT
# Display the 10 cheapest books
# --------------------------------------------------

query3 = """
SELECT title, price_gbp
FROM books
ORDER BY price_gbp ASC
LIMIT 10
"""

result3 = pd.read_sql(query3, connection)

print("QUERY 3: 10 cheapest books")
print(result3)
print("\n" + "=" * 70 + "\n")


# --------------------------------------------------
# Query 4: DISTINCT
# Display all unique categories
# --------------------------------------------------

query4 = """
SELECT DISTINCT category_id
FROM books
"""

result4 = pd.read_sql(query4, connection)

print("QUERY 4: Distinct category IDs")
print(result4)
print("\n" + "=" * 70 + "\n")


# --------------------------------------------------
# Query 5: BETWEEN
# Find books priced between £20 and £30
# --------------------------------------------------

query5 = """
SELECT title, price_gbp, rating
FROM books
WHERE price_gbp BETWEEN 20 AND 30
"""

result5 = pd.read_sql(query5, connection)

print("QUERY 5: Books priced between £20 and £30")
print(result5)
print("\n" + "=" * 70 + "\n")


# --------------------------------------------------
# Query 6: JOIN
# Combine books with their category names
# --------------------------------------------------

query6 = """
SELECT
    books.title,
    books.price_gbp,
    books.rating,
    categories.category_name
FROM books
JOIN categories
    ON books.category_id = categories.category_id
ORDER BY books.price_gbp DESC
LIMIT 10
"""

result6 = pd.read_sql(query6, connection)

print("QUERY 6: Books with category names using JOIN")
print(result6)
print("\n" + "=" * 70 + "\n")


# Save query outputs
result1.to_csv("query1_output.csv", index=False)
result2.to_csv("query2_output.csv", index=False)
result3.to_csv("query3_output.csv", index=False)
result4.to_csv("query4_output.csv", index=False)
result5.to_csv("query5_output.csv", index=False)
result6.to_csv("query6_output.csv", index=False)

print("All query outputs saved successfully!")

connection.close()

print("Database connection closed.")