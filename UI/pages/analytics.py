from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "analytics" / "titanic.csv"
MODEL_PATH = PROJECT_ROOT / "analytics" / "best_titanic_model.joblib"


# ============================================================
# PAGE HEADER
# ============================================================

st.title("📊 Analytics & Machine Learning")

st.caption(
    "Explore the Titanic dataset, evaluate models and make "
    "predictions using the serialized tuned Random Forest pipeline."
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data(path):
    return pd.read_csv(path)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


# ============================================================
# FILE STATUS
# ============================================================

data_exists = DATA_PATH.exists()
model_exists = MODEL_PATH.exists()

status1, status2 = st.columns(2)

with status1:
    if data_exists:
        st.success("✅ Titanic dataset loaded")
    else:
        st.error(f"❌ Dataset not found:\n{DATA_PATH}")

with status2:
    if model_exists:
        st.success("✅ Tuned model loaded")
    else:
        st.error(f"❌ Model not found:\n{MODEL_PATH}")


if not data_exists:
    st.stop()


df = load_data(DATA_PATH)


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.header("📌 Dataset Overview")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Rows", f"{df.shape[0]:,}")

with m2:
    st.metric("Columns", df.shape[1])

with m3:
    st.metric(
        "Survived",
        int(df["survived"].sum()) if "survived" in df.columns else "N/A",
    )

with m4:
    st.metric(
        "Survival Rate",
        f"{df['survived'].mean() * 100:.2f}%"
        if "survived" in df.columns
        else "N/A",
    )


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📄 Data Preview",
        "🔎 Exploratory Analysis",
        "🤖 Model Results",
        "🎯 Prediction",
    ]
)


# ============================================================
# TAB 1 — DATA PREVIEW
# ============================================================

with tab1:

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True,
    )

    st.subheader("Missing Values")

    missing = (
        df.isnull()
        .sum()
        .to_frame("Missing Values")
    )

    missing["Percentage"] = (
        missing["Missing Values"]
        / len(df)
        * 100
    ).round(2)

    st.dataframe(
        missing,
        use_container_width=True,
    )


# ============================================================
# TAB 2 — EXPLORATORY ANALYSIS
# ============================================================

with tab2:

    st.subheader("Survival Analysis")

    if {"sex", "survived"}.issubset(df.columns):

        survival_gender = (
            df.groupby("sex")["survived"]
            .mean()
            .mul(100)
            .round(2)
        )

        st.bar_chart(
            survival_gender,
            use_container_width=True,
        )

    if {"pclass", "survived"}.issubset(df.columns):

        st.subheader("Survival Rate by Passenger Class")

        survival_class = (
            df.groupby("pclass")["survived"]
            .mean()
            .mul(100)
            .round(2)
        )

        st.bar_chart(
            survival_class,
            use_container_width=True,
        )

    st.subheader("Numeric Correlations")

    numeric = df.select_dtypes(include="number")

    if not numeric.empty:

        correlation = numeric.corr().round(3)

        st.dataframe(
            correlation,
            use_container_width=True,
        )


# ============================================================
# TAB 3 — MODEL RESULTS
# ============================================================

with tab3:

    st.subheader("Classification Model Comparison")

    results = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
            ],
            "Accuracy": [
                0.7809,
                0.7697,
                0.7978,
            ],
            "Precision": [
                0.7544,
                0.7143,
                0.7759,
            ],
            "Recall": [
                0.6324,
                0.6618,
                0.6618,
            ],
            "F1": [
                0.6880,
                0.6870,
                0.7143,
            ],
            "ROC-AUC": [
                0.8265,
                0.7412,
                0.8211,
            ],
        }
    )

    st.dataframe(
        results.style.format(
            {
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
                "ROC-AUC": "{:.4f}",
            }
        ),
        use_container_width=True,
    )


    st.subheader("Tuned Random Forest")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Test Accuracy",
            "0.8034",
        )

    with c2:
        st.metric(
            "CV F1",
            "0.7436",
        )

    with c3:
        st.metric(
            "OOB Score",
            "0.8073",
        )


    st.subheader("Fare Regression")

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.metric(
            "MAE",
            "21.0986",
        )

    with r2:
        st.metric(
            "RMSE",
            "41.7021",
        )

    with r3:
        st.metric(
            "R²",
            "0.3482",
        )

    with r4:
        st.metric(
            "Adjusted R²",
            "0.3173",
        )


# ============================================================
# TAB 4 — PREDICTION
# ============================================================

with tab4:

    st.subheader("Passenger Survival Prediction")

    if not model_exists:

        st.warning(
            "The serialized model could not be found. "
            "Prediction is unavailable."
        )

    else:

        model = load_model(MODEL_PATH)

        col1, col2 = st.columns(2)

        with col1:

            pclass = st.selectbox(
                "Passenger Class",
                [1, 2, 3],
                index=2,
            )

            sex = st.selectbox(
                "Sex",
                ["male", "female"],
            )

            age = st.number_input(
                "Age",
                min_value=0.0,
                max_value=100.0,
                value=30.0,
                step=1.0,
            )

            sibsp = st.number_input(
                "Siblings / Spouses",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
            )

        with col2:

            parch = st.number_input(
                "Parents / Children",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
            )

            fare = st.number_input(
                "Fare",
                min_value=0.0,
                max_value=600.0,
                value=30.0,
                step=1.0,
            )

            embarked = st.selectbox(
                "Embarked",
                ["S", "C", "Q"],
            )


        if st.button(
            "🔮 Predict Survival",
            type="primary",
            use_container_width=True,
        ):

            input_data = pd.DataFrame(
                [
                    {
                        "pclass": pclass,
                        "sex": sex,
                        "age": age,
                        "sibsp": sibsp,
                        "parch": parch,
                        "fare": fare,
                        "embarked": embarked,
                    }
                ]
            )

            try:

                prediction = model.predict(input_data)[0]

                probability = model.predict_proba(
                    input_data
                )[0]

                if prediction == 1:

                    st.success(
                        "🟢 Prediction: Passenger likely survived."
                    )

                else:

                    st.error(
                        "🔴 Prediction: Passenger likely did not survive."
                    )


                st.write(
                    f"Survival probability: "
                    f"**{probability[1] * 100:.2f}%**"
                )

                st.progress(
                    float(probability[1])
                )

            except Exception as exc:

                st.error(
                    "Prediction failed."
                )

                st.exception(exc)