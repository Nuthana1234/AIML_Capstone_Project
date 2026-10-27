
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI/ML Capstone",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>
    /* Main application background */
    .stApp {
        background-color: #f7f9fc;
    }

    .main .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #101827;
    }

    [data-testid="stSidebar"] > div {
        background-color: #101827;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {
        color: #f8fafc;
    }

    /* Sidebar navigation links */
    [data-testid="stSidebar"] [data-testid="stNavLink"] {
        border-radius: 8px;
    }

    [data-testid="stSidebar"] [data-testid="stNavLink"]:hover {
        background-color: #263247;
    }

    [data-testid="stSidebar"] [data-testid="stNavLink"][aria-current="page"] {
        background-color: #263247;
    }

    /* Main content headings */
    .main h1,
    .main h2,
    .main h3 {
        color: #0f172a;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1rem;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid #dbe3ef;
        font-weight: 600;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# NAVIGATION
# ============================================================

pages = {
    "Project": [
        st.Page(
            "pages/dashboard.py",
            title="Dashboard",
            icon="🏠",
            default=True,
        ),
    ],
    "AI / ML": [
        st.Page(
            "pages/analytics.py",
            title="Analytics & ML",
            icon="📊",
        ),
        st.Page(
            "pages/support_assistant.py",
            title="AI Support Assistant",
            icon="🤖",
        ),
    ],
}


# ============================================================
# CUSTOM SIDEBAR CONTENT
# ============================================================

with st.sidebar:

    # Project title and subtitle
    st.markdown(
        """
        <div style="
            padding: 12px 0 20px 0;
            border-bottom: 1px solid #263247;
            margin-bottom: 20px;
        ">
            <div style="
                font-size: 22px;
                font-weight: 800;
                color: #ffffff;
                line-height: 1.4;
            ">
                🤖 AI/ML Capstone
            </div>
            <div style="
                font-size: 13px;
                color: #cbd5e1;
                margin-top: 6px;
            ">
                End-to-End AI/ML System
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Technology stack
    st.markdown("### 📚 Project Stack")

    st.markdown(
        """
        <div style="
            line-height: 1.9;
            color: #cbd5e1;
            font-size: 14px;
        ">
            🕷️ Web Scraping<br>
            🗄️ SQLite<br>
            🐼 Pandas<br>
            📊 Data Analytics<br>
            🤖 Machine Learning<br>
            🔎 RAG<br>
            🧠 ChromaDB<br>
            🔗 LangGraph<br>
            ⚡ FastAPI<br>
            🎈 Streamlit<br>
            🐳 Docker
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Academic information
    st.markdown("### 🎓 Academic Project")

    st.markdown(
        """
        <div style="
            color: #cbd5e1;
            font-size: 13px;
            line-height: 1.8;
        ">
            <strong>Project:</strong><br>
            AI/ML Capstone Project
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RENDER SELECTED PAGE
# ============================================================

pg = st.navigation(
    pages,
    position="sidebar",
    expanded=True,
)

pg.run()
