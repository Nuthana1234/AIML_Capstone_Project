import streamlit as st


# ============================================================
# PAGE TITLE
# ============================================================

st.title("AI/ML Capstone Dashboard")

st.caption(
    "An end-to-end project combining data engineering, "
    "analytics, machine learning, and an AI support assistant."
)


# ============================================================
# HERO SECTION
# ============================================================

st.html(
    """
    <div style="
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
        padding: 32px;
        border-radius: 18px;
        margin: 12px 0 28px 0;
        color: white;
    ">

        <div style="
            font-size: 14px;
            font-weight: 600;
            color: #bfdbfe;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        ">
            End-to-End AI/ML Capstone
        </div>

        <div style="
            font-size: 30px;
            font-weight: 750;
            margin-bottom: 12px;
            line-height: 1.2;
        ">
            Data Engineering → Analytics → Machine Learning → AI Assistant
        </div>

        <div style="
            font-size: 16px;
            color: #dbeafe;
            line-height: 1.6;
            max-width: 900px;
        ">
            A complete AI/ML workflow covering web scraping, data cleaning,
            SQLite database design, exploratory data analysis, machine learning,
            model evaluation, regression, semantic retrieval, RAG, FastAPI,
            Docker, and Streamlit.
        </div>

    </div>
    """
)


# ============================================================
# PROJECT OVERVIEW
# ============================================================

st.subheader("📌 Project Overview")

st.write(
    """
    This capstone project is divided into three major modules:

    **Module 1 — Data Pipeline:**  
    Collects and processes book data from a public website, cleans the
    dataset, stores it in SQLite, and validates SQL results using Pandas.

    **Module 2 — Analytics & Machine Learning:**  
    Performs exploratory analysis on the Titanic dataset, studies
    relationships between variables, trains classification models,
    handles class imbalance, tunes a Random Forest model, and performs
    fare regression.

    **Module 3 — AI Support Assistant:**  
    Uses policy documents, semantic embeddings, ChromaDB retrieval,
    LangGraph orchestration, and FastAPI to answer customer-support
    questions using retrieved policy information.
    """
)


# ============================================================
# KEY PROJECT METRICS
# ============================================================

st.subheader("📊 Key Project Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Books Collected",
        "100",
        help="Books collected during the data-pipeline scraping process.",
    )

with col2:
    st.metric(
        "Categories",
        "29",
        help="Unique book categories collected from the source.",
    )

with col3:
    st.metric(
        "Policy Documents",
        "8",
        help="Policy documents used by the support assistant.",
    )

with col4:
    st.metric(
        "Best RF Accuracy",
        "80.34%",
        help="Test accuracy of the tuned Random Forest classifier.",
    )


# ============================================================
# MODULE CARDS
# ============================================================

st.subheader("🧩 Project Modules")

module1, module2, module3 = st.columns(3)


with module1:

    st.html(
        """
        <div style="
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 22px;
            min-height: 250px;
        ">

            <div style="
                font-size: 28px;
                margin-bottom: 8px;
            ">
                🗄️
            </div>

            <div style="
                font-size: 20px;
                font-weight: 700;
                margin-bottom: 10px;
                color: #111827;
            ">
                Module 1
            </div>

            <div style="
                font-size: 16px;
                font-weight: 650;
                margin-bottom: 10px;
                color: #1d4ed8;
            ">
                Data Pipeline
            </div>

            <div style="
                color: #4b5563;
                line-height: 1.6;
            ">
                Web scraping, data cleaning, currency conversion,
                SQLite database design, SQL queries, and Pandas validation.
            </div>

        </div>
        """
    )


with module2:

    st.html(
        """
        <div style="
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 22px;
            min-height: 250px;
        ">

            <div style="
                font-size: 28px;
                margin-bottom: 8px;
            ">
                📊
            </div>

            <div style="
                font-size: 20px;
                font-weight: 700;
                margin-bottom: 10px;
                color: #111827;
            ">
                Module 2
            </div>

            <div style="
                font-size: 16px;
                font-weight: 650;
                margin-bottom: 10px;
                color: #1d4ed8;
            ">
                Analytics & ML
            </div>

            <div style="
                color: #4b5563;
                line-height: 1.6;
            ">
                EDA, correlation analysis, classification models,
                class-imbalance handling, Random Forest tuning,
                regression, and model serialization.
            </div>

        </div>
        """
    )


with module3:

    st.html(
        """
        <div style="
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 22px;
            min-height: 250px;
        ">

            <div style="
                font-size: 28px;
                margin-bottom: 8px;
            ">
                🤖
            </div>

            <div style="
                font-size: 20px;
                font-weight: 700;
                margin-bottom: 10px;
                color: #111827;
            ">
                Module 3
            </div>

            <div style="
                font-size: 16px;
                font-weight: 650;
                margin-bottom: 10px;
                color: #1d4ed8;
            ">
                AI Support Assistant
            </div>

            <div style="
                color: #4b5563;
                line-height: 1.6;
            ">
                Policy-document ingestion, embeddings, ChromaDB,
                semantic retrieval, LangGraph workflow,
                FastAPI API, and Docker deployment.
            </div>

        </div>
        """
    )


# ============================================================
# MODEL RESULTS
# ============================================================

st.subheader("🏆 Key Results")

result1, result2, result3, result4 = st.columns(4)

with result1:
    st.metric(
        "Tuned RF CV F1",
        "0.7436",
    )

with result2:
    st.metric(
        "RF OOB Score",
        "0.8073",
    )

with result3:
    st.metric(
        "Fare Regression R²",
        "0.3482",
    )

with result4:
    st.metric(
        "RAG Retrieval",
        "Top 3",
    )


# ============================================================
# ARCHITECTURE
# ============================================================

st.subheader("🏗️ System Architecture")

st.html(
    """
    <div style="
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 28px;
        margin-top: 10px;
    ">

        <div style="
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: center;
            gap: 12px;
        ">

            <div style="
                background: #eff6ff;
                border: 1px solid #bfdbfe;
                padding: 18px;
                border-radius: 12px;
                min-width: 150px;
                text-align: center;
            ">
                <div style="font-size: 25px;">🌐</div>
                <strong>Data Sources</strong>
                <div style="font-size: 13px; color: #4b5563;">
                    Web + Datasets
                </div>
            </div>

            <div style="
                font-size: 26px;
                color: #64748b;
            ">
                →
            </div>

            <div style="
                background: #f0fdf4;
                border: 1px solid #bbf7d0;
                padding: 18px;
                border-radius: 12px;
                min-width: 150px;
                text-align: center;
            ">
                <div style="font-size: 25px;">🧹</div>
                <strong>Data Pipeline</strong>
                <div style="font-size: 13px; color: #4b5563;">
                    Clean + Store
                </div>
            </div>

            <div style="
                font-size: 26px;
                color: #64748b;
            ">
                →
            </div>

            <div style="
                background: #fefce8;
                border: 1px solid #fde68a;
                padding: 18px;
                border-radius: 12px;
                min-width: 150px;
                text-align: center;
            ">
                <div style="font-size: 25px;">📊</div>
                <strong>Analytics & ML</strong>
                <div style="font-size: 13px; color: #4b5563;">
                    EDA + Models
                </div>
            </div>

            <div style="
                font-size: 26px;
                color: #64748b;
            ">
                →
            </div>

            <div style="
                background: #faf5ff;
                border: 1px solid #e9d5ff;
                padding: 18px;
                border-radius: 12px;
                min-width: 150px;
                text-align: center;
            ">
                <div style="font-size: 25px;">🤖</div>
                <strong>AI Assistant</strong>
                <div style="font-size: 13px; color: #4b5563;">
                    RAG + API
                </div>
            </div>

        </div>

    </div>
    """
)


# ============================================================
# TECHNOLOGY STACK
# ============================================================

st.subheader("🛠️ Technology Stack")

tech1, tech2, tech3, tech4 = st.columns(4)

with tech1:

    st.markdown("### 🐍 Python")

    st.markdown(
        """
        - Pandas
        - NumPy
        - Scikit-learn
        - BeautifulSoup
        """
    )


with tech2:

    st.markdown("### 🗄️ Data")

    st.markdown(
        """
        - SQLite
        - SQL
        - CSV
        - Pandas validation
        """
    )


with tech3:

    st.markdown("### 🤖 AI / ML")

    st.markdown(
        """
        - Random Forest
        - Logistic Regression
        - Regression
        - Sentence Transformers
        - ChromaDB
        """
    )


with tech4:

    st.markdown("### 🚀 Deployment")

    st.markdown(
        """
        - FastAPI
        - LangGraph
        - Docker
        - Streamlit
        - Git / GitHub
        """
    )


# ============================================================
# FINAL PROJECT STATUS
# ============================================================

st.divider()

st.success(
    "Capstone project modules completed — Data Pipeline, Analytics & ML, "
    "and AI Support Assistant."
)

st.caption(
    "Use the sidebar to explore Analytics & ML or interact with the AI Support Assistant."
)