# Capstone Project — Artificial Intelligence and Machine Learning

This repository contains the complete capstone project for the Certificate Program in Artificial Intelligence and Machine Learning.

## Project Status

All three capstone modules have been implemented:

- Module 1 — Data Pipeline
- Module 2 — Titanic Analytics and Machine Learning
- Module 3 — Zepto RAG Support Assistant

---

## Repository Structure

```text
CapstoneProject/
│
├── data_pipeline/
│   ├── books.db
│   ├── raw_books.csv
│   ├── cleaned_books.csv
│   ├── scraper.py
│   ├── cleaning.py
│   ├── database.py
│   ├── datapipeline_sqlite.py
│   ├── sqlite.py
│   ├── pandas_validation.py
│   ├── query1_output.csv
│   ├── query2_output.csv
│   ├── query3_output.csv
│   ├── query4_output.csv
│   ├── query5_output.csv
│   └── query6_output.csv
│
├── analytics/
│   ├── data loading and preprocessing
│   ├── exploratory analysis
│   ├── visualizations
│   ├── classification models
│   ├── regression analysis
│   └── saved model pipeline
│
├── support_assistant/
│   ├── documents/
│   │   ├── doc_01.txt
│   │   ├── doc_02.txt
│   │   ├── doc_03.txt
│   │   ├── doc_04.txt
│   │   ├── doc_05.txt
│   │   ├── doc_06.txt
│   │   ├── doc_07.txt
│   │   └── doc_08.txt
│   ├── api.py
│   ├── graph.py
│   ├── ingestion.py
│   ├── prompt.py
│   ├── retrieval.py
│   └── schemas.py
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── uv.lock
└── README.md
Module 1 — Data Pipeline
Objective

Build an end-to-end data pipeline using books from Books to Scrape.

Workflow
Books to Scrape
      ↓
Web Scraping
      ↓
Raw CSV
      ↓
Data Cleaning
      ↓
GBP → INR Conversion
      ↓
SQLite Database
      ↓
SQL Queries
      ↓
Pandas Validation
Scraping

The scraper uses:

Python
Requests
BeautifulSoup
Pandas

The first 5 catalogue pages were scraped.

Results:

100 books
29 categories

Collected fields:

Title
Price in GBP
Star rating
Availability
Category
Book URL
Data Cleaning

The following transformations were performed:

price_gbp converted to float
Star ratings converted from text to integers from 1–5
Availability converted to Boolean in_stock
Parsing failures handled using median imputation where applicable

Required fixed currency conversion:

1 GBP = 105.50 INR

The converted price is stored as:

price_inr
SQLite Database

A normalized SQLite database was created:

data_pipeline/books.db

The database contains:

Categories table
Books table

The tables are connected using primary-key/foreign-key relationships.

SQL Queries

Six SQL queries were created and their outputs were saved.

The queries demonstrate:

SELECT
WHERE
ORDER BY
LIMIT
DISTINCT
BETWEEN
JOIN

The SQL JOIN result was reproduced using pd.merge() and checked for equivalence.

Module 2 — Titanic Analytics and Machine Learning
Objective

Perform exploratory data analysis, visualization, classification, regression, model tuning, and pipeline serialization using the Titanic dataset.

Data Loading

The Titanic dataset was initially loaded using:

sns.load_dataset("titanic")

It was immediately saved locally as:

analytics/titanic.csv

Dataset size:

891 rows × 15 columns
Data Profiling

The analysis includes:

df.info()
df.describe()
Dataset shape
Missing-value counts
Missing-value percentages

Important missing values:

Column	Missing
age	19.87%
embarked	0.22%
deck	77.22%
embark_town	0.22%
Data Cleaning

The cleaning strategy includes:

Median imputation for age
Removal of rows with missing embarked
Removal of the highly incomplete deck column
Removal of redundant/unsuitable columns
Univariate Analysis

The analysis includes:

Age histogram
Fare histogram
Age box plot
Fare box plot
IQR outlier detection
Fare mean, median, and mode
Skewness analysis

Results:

Age outliers: 65
Fare outliers: 114

Fare mean:   32.0967
Fare median: 14.4542
Fare mode:    8.05

Fare is right-skewed because the mean is greater than the median.

Bivariate Analysis

Survival rates were analyzed by:

Sex
Passenger class
Sex and passenger class

Selected survival rates:

Group	Survival Rate
Female	74.04%
Male	18.89%
1st Class	62.62%
2nd Class	47.28%
3rd Class	24.24%

The required correlation matrix uses:

survived
pclass
age
sibsp
parch
fare

Strongest absolute correlations:

pclass ↔ fare : -0.5482
pclass ↔ age  : -0.3365
Multivariate Analysis

Multiple multivariate charts were created using combinations of:

Age
Fare
Sex
Passenger class
Survival
Family size

Each visualization includes a written interpretation.

Standardization

Age and fare were standardized using z-scores for EDA validation.

After standardization:

Mean ≈ 0
Standard deviation ≈ 1

The standardized values were not used as modeling inputs.

Classification Models

A stratified train/test split was used.

The preprocessing pipeline includes:

Missing-value handling
One-hot encoding
StandardScaler
ColumnTransformer
Pipeline

Three models were trained using the same split.

Baseline Results
Model	Accuracy	Precision	Recall	F1	ROC-AUC
Logistic Regression	0.7809	0.7544	0.6324	0.6880	0.8265
Decision Tree	0.7697	0.7143	0.6618	0.6870	0.7412
Random Forest	0.7978	0.7759	0.6618	0.7143	0.8211

Confusion matrices and a model comparison table are included.

The Decision Tree was visualized using plot_tree() with feature and class names.

Class Imbalance

The analysis compares:

Baseline
class_weight="balanced"
SMOTE applied only to the training data

This prevents test-set information leakage.

Random Forest Hyperparameter Tuning

GridSearchCV was used to tune:

n_estimators
max_depth
max_features

Best configuration:

n_estimators = 100
max_depth    = 5
max_features = sqrt

Results:

Best CV F1:    0.7436
OOB Score:     0.8073
Test Accuracy: 0.8034

The Random Forest estimator used:

oob_score=True
Regression Side Task

A multivariate Linear Regression model was developed to predict passenger fare.

Results:

Metric	Result
MAE	21.0986
RMSE	41.7021
R²	0.3482
Adjusted R²	0.3173

A residual plot was created to examine model errors and variance behavior.

Saved Model Pipeline

The complete preprocessing and final estimator pipeline is saved using Joblib.

The saved pipeline was reloaded and tested with raw input data.

Example verification:

Prediction: 0

Class probabilities:
[0.81396309, 0.18603691]
Module 3 — Zepto RAG Support Assistant
Objective

Build a Retrieval-Augmented Generation support assistant using exactly eight Zepto policy documents.

The assistant handles questions about:

Delivery
Returns and refunds
Membership
Order tracking
Order cancellation
Damaged or missing items
Gift cards
Customer support
Policy Documents
doc_01 — Delivery Policy
doc_02 — Returns & Refunds
doc_03 — Membership Tiers
doc_04 — Order Tracking
doc_05 — Order Cancellation
doc_06 — Damaged/Missing Items
doc_07 — Gift Cards
doc_08 — Customer Support Hours
RAG Architecture
Policy Documents
      ↓
Document Ingestion
      ↓
Chunk Creation
      ↓
Sentence Transformer Embeddings
      ↓
ChromaDB
      ↓
User Query
      ↓
Query Embedding
      ↓
Top-3 Retrieval
      ↓
LangGraph
      ↓
Answer + Sources + Confidence
Embeddings

The local embedding model used is:

all-MiniLM-L6-v2

Embeddings are stored in ChromaDB.

No paid API or API key is required for the graded baseline.

LangGraph

The assistant uses a StateGraph with three main nodes:

classify_intent
       ↓
   ┌───┴────┐
   ↓        ↓
retrieve   direct
_and_      _answer
answer
classify_intent

The mock classifier identifies:

Delivery
Returns/refunds
Membership
Tracking
Cancellation
Gift cards
Support hours
General queries
retrieve_and_answer

For policy queries:

Embed the query.
Retrieve the top 3 documents.
Build the structured support prompt.
Generate the deterministic mock response.
Return answer, sources, and confidence.
direct_answer

General queries receive a deterministic support response without policy retrieval.

Structured Prompt

The support prompt contains:

Role
Context
Task
Format
Length
Negative constraint
Few-shot example

The negative constraint prevents unsupported policies, prices, delivery times, refund rules, or other facts from being invented.

Mock LLM Mode

The graded baseline uses:

MOCK_LLM=1

or leaves the environment variable unset.

The mock mode is deterministic and does not require an external LLM API.

If:

MOCK_LLM=0

a real LLM provider must be configured.

Pydantic Output Schema

The response follows:

{
  "answer": "string",
  "sources": ["doc_01"],
  "confidence": 0.9
}

The confidence value is validated between 0.0 and 1.0.

FastAPI

The support assistant exposes:

POST /ask

Example request:

{
  "query": "How long do I have to report a damaged grocery item?"
}

Example response:

{
  "answer": "Based on the retrieved context: Damaged or spoiled grocery items must be reported within 24 hours...",
  "sources": [
    "doc_06",
    "doc_02",
    "doc_04"
  ],
  "confidence": 0.9
}
Example API Calls
Request 1
{
  "query": "How long do I have to report a damaged grocery item?"
}
Request 2
{
  "query": "How much does Zepto Pass cost?"
}

Both requests were successfully tested through the FastAPI /ask endpoint in mock mode.

Swagger UI:

http://127.0.0.1:8000/docs
Running the Support Assistant

From the project root:

.\.venv\Scripts\python.exe -m uvicorn support_assistant.api:app --reload

Then open:

http://127.0.0.1:8000/docs

Use the POST /ask endpoint to test the assistant.

Docker

The repository includes:

Dockerfile
.dockerignore
requirements.txt

Build:

docker build -t zepto-support-assistant .

Run:

docker run -p 8000:8000 zepto-support-assistant

The Docker configuration is provided for local containerization and reproducibility.

Technologies Used
Data Pipeline
Python
Requests
BeautifulSoup
Pandas
SQLite
SQL
Analytics and Machine Learning
Python
Pandas
NumPy
Seaborn
Matplotlib
Scikit-learn
Imbalanced-learn
Joblib
Support Assistant
Python
Sentence Transformers
ChromaDB
LangGraph
Pydantic
FastAPI
Uvicorn
Docker
Reproducibility

The repository contains the generated datasets, SQLite database, SQL query outputs, policy documents, source code, and saved artifacts required for the project workflow.

The virtual environment, IDE configuration, environment files, and generated ChromaDB cache are excluded through .gitignore.

The Titanic dataset is saved locally as a CSV after the initial load so that subsequent analysis can use the local copy.

Academic Integrity

All implementation, analysis, interpretations, and project documentation in this repository were developed as part of the capstone project.

External documentation was consulted where required for understanding libraries and frameworks.

No paid services are required for the graded baseline implementation.

Author

Y.Nuthana Sree
