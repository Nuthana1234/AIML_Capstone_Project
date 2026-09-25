import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd

BASE_URL = "https://books.toscrape.com/"
headers = {
    "User-Agent": "Mozilla/5.0"
}


def get_book_category(book_url):
    response = requests.get(book_url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    breadcrumb = soup.select("ul.breadcrumb li")

    if len(breadcrumb) >= 3:
        return breadcrumb[2].get_text(strip=True)

    return None


def scrape_books(num_pages=5):
    books = []

    for page_number in range(1, num_pages + 1):

        page_url = urljoin(
            BASE_URL,
            f"catalogue/page-{page_number}.html"
        )

        print(f"Scraping page {page_number}...")

        response = requests.get(page_url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        book_containers = soup.select("article.product_pod")

        for book in book_containers:

            title = book.h3.a.get("title")

            price = book.select_one(
                ".price_color"
            ).get_text(strip=True)

            rating = book.select_one(
                "p.star-rating"
            ).get("class")[1]

            availability = book.select_one(
                ".availability"
            ).get_text(" ", strip=True)

            relative_url = book.h3.a.get("href")

            book_url = urljoin(
                page_url,
                relative_url
            )

            category = get_book_category(book_url)

            books.append({
                "title": title,
                "price_gbp_raw": price,
                "rating_raw": rating,
                "availability_raw": availability,
                "category": category,
                "book_url": book_url
            })

    return pd.DataFrame(books)


if __name__ == "__main__":

    df = scrape_books(num_pages=5)

    print("\nScraping completed!")
    print("Total books scraped:", len(df))

    print("\nFirst 5 books:")
    print(df.head())

    df.to_csv(
        "raw_books.csv",
        index=False
    )

    print("\nData saved successfully!")