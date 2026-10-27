import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000/ask"


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🤖 AI Support Assistant")

st.caption(
    "RAG-based Zepto policy assistant powered by FastAPI, "
    "LangGraph, embeddings and ChromaDB."
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("### 💡 Try these questions")

    suggestions = [
        "How long does delivery take?",
        "How much does Zepto Pass cost?",
        "Can I return a product?",
        "How can I track my order?",
        "Can I cancel my order?",
        "What should I do if an item is damaged?",
    ]

    for suggestion in suggestions:

        if st.button(
            suggestion,
            use_container_width=True,
        ):

            st.session_state.pending_question = suggestion


    st.markdown("---")

    st.markdown("### 🔌 Backend")

    st.caption(
        "FastAPI endpoint:"
    )

    st.code(
        API_URL,
        language="text",
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander("📚 Sources"):

                for source in message["sources"]:

                    st.write(
                        f"• {source}"
                    )

            if message.get("confidence") is not None:

                st.caption(
                    f"Confidence: "
                    f"{message['confidence']:.2f}"
                )


# ============================================================
# HANDLE SUGGESTION
# ============================================================

pending_question = st.session_state.pop(
    "pending_question",
    None,
)


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about Zepto policies..."
)


if question is None:
    question = pending_question


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # -------------------------------
    # User message
    # -------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # -------------------------------
    # Assistant response
    # -------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching policy documents..."
        ):

            try:

                response = requests.post(
                    API_URL,
                    json={
                        "query": question
                    },
                    timeout=60,
                )


                response.raise_for_status()

                data = response.json()

                answer = data.get(
                    "answer",
                    "No answer returned.",
                )

                sources = data.get(
                    "sources",
                    [],
                )

                confidence = data.get(
                    "confidence",
                    None,
                )


                st.markdown(answer)


                if sources:

                    with st.expander(
                        "📚 Retrieved Sources"
                    ):

                        for source in sources:

                            st.write(
                                f"• {source}"
                            )


                if confidence is not None:

                    st.caption(
                        f"Confidence: "
                        f"{confidence:.2f}"
                    )


                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "confidence": confidence,
                    }
                )


            except requests.exceptions.ConnectionError:

                error_message = (
                    "❌ Cannot connect to the FastAPI backend.\n\n"
                    "Make sure Terminal 1 is running:\n\n"
                    "`python -m uvicorn support_assistant.api:app "
                    "--host 0.0.0.0 --port 8000`"
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                        "confidence": None,
                    }
                )


            except requests.exceptions.Timeout:

                error_message = (
                    "⏱️ The request timed out. "
                    "Please try again."
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                        "confidence": None,
                    }
                )


            except requests.exceptions.HTTPError as exc:

                error_message = (
                    f"❌ FastAPI returned an HTTP error: "
                    f"{exc}"
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                        "confidence": None,
                    }
                )


            except Exception as exc:

                error_message = (
                    f"❌ Unexpected error: {exc}"
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                        "confidence": None,
                    }
                )