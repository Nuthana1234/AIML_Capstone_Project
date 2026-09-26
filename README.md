# AI/ML Capstone Project

This repository contains the complete capstone project for the Certificate Program in Artificial Intelligence and Machine Learning.

## Modules

### Module 1 — Data Pipeline

* Scraped **100 books across 5 catalogue pages** from Books to Scrape using Requests and BeautifulSoup.
* Captured title, price, rating, availability, and category.
* Cleaned data and converted GBP to INR using **1 GBP = ₹105.50**.
* Stored the data in a normalized SQLite database with primary-key/foreign-key relationships.
* Implemented SQL queries covering filtering, sorting, limiting, distinct values, ranges, and joins.
* Validated SQL results using Pandas and `pd.merge()`.

### Module 2 — Analytics & Machine Learning

* Analyzed the **891 × 15 Titanic dataset**.
* Performed profiling, missing-value analysis, cleaning, univariate, bivariate, correlation, and multivariate analysis.
* Used boolean masking for survival analysis by sex, passenger class, and their combination.
* Applied standardization and stratified train-test splitting.
* Compared Logistic Regression, Decision Tree, and Random Forest classifiers.
* Evaluated class imbalance using baseline Logistic Regression, `class_weight="balanced"`, and SMOTE.
* Tuned Random Forest using 5-fold GridSearchCV.
* Built a Linear Regression model for fare prediction.
* Saved the final classification pipeline using Joblib.

### Key Results

| Model/Method          |   Accuracy |            F1 | ROC-AUC |
| --------------------- | ---------: | ------------: | ------: |
| Logistic Regression   |     0.7809 |        0.6880 |  0.8265 |
| Decision Tree         |     0.7697 |        0.6870 |  0.7412 |
| Random Forest         |     0.7978 |        0.7143 |  0.8211 |
| Class Weight Balanced |     0.7978 |        0.7273 |  0.8305 |
| SMOTE                 |     0.7921 |        0.7176 |  0.8396 |
| Tuned Random Forest   | **0.8034** | **0.7436 CV** |       — |

Fare regression:

* MAE: **21.0986**
* RMSE: **41.7021**
* R²: **0.3482**
* Adjusted R²: **0.3173**

The tuned Random Forest was retained as the final serialized classification pipeline.

### Module 3 — Support Assistant

* Built a policy-based RAG customer support assistant using **8 policy documents**.
* Used local `all-MiniLM-L6-v2` embeddings and ChromaDB with cosine similarity.
* Implemented retrieval and workflow orchestration using LangGraph.
* Used Pydantic for structured responses containing answer, sources, and confidence.
* Exposed the assistant through FastAPI with `POST /ask`.
* Containerized and tested the application using Docker.

## Technologies

Python • Pandas • NumPy • Scikit-learn • Matplotlib • Seaborn • Requests • BeautifulSoup • SQLite • Sentence Transformers • ChromaDB • LangGraph • FastAPI • Pydantic • Docker

## Docker

```bash
docker build -t aiml-capstone .
docker run --rm -p 8000:8000 aiml-capstone
```

Swagger UI:

```text
http://localhost:8000/docs
```

## Repository

GitHub: [https://github.com/Nuthana1234/AIML_Capstone_Project](https://github.com/Nuthana1234/AIML_Capstone_Project)

## Author

**Nuthana Sree**
B.Tech Computer Science and Engineering
Dhanekula Institute of Engineering & Technology
