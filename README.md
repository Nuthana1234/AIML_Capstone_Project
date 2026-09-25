# Capstone Project — Artificial Intelligence and Machine Learning

A three-module capstone project covering web data collection, data analytics and machine learning, and a RAG-based customer support assistant.

## Repository Structure

```text
CapstoneProject/
│
├── data_pipeline/
│   ├── scraper.py
│   ├── cleaning.py
│   ├── database.py
│   ├── queries.py
│   ├── pandas_validation.py
│   ├── raw_books.csv
│   ├── cleaned_books.csv
│   ├── books.db
│   └── query*_output.csv
│
├── analytics/
│   ├── data_loading.py
│   ├── profiling.py
│   ├── cleaning.py
│   ├── univariate.py
│   ├── bivariate.py
│   ├── multivariate.py
│   ├── standardization.py
│   ├── modeling.py
│   ├── class_imbalance.py
│   ├── hyperparameter_tuning.py
│   ├── fare_regression.py
│   ├── final_comparison.py
│   ├── save_best_pipeline.py
│   ├── titanic.csv
│   ├── titanic_cleaned.csv
│   └── best_titanic_model.joblib
│
├── support_assistant/
│   ├── ingestion.py
│   ├── retrieval.py
│   ├── prompt.py
│   ├── schemas.py
│   ├── graph.py
│   └── api.py
│
├── documents/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md