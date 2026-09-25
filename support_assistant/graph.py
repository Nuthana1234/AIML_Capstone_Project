import os
from support_assistant.prompt import build_support_prompt
from typing import TypedDict
from langgraph.graph import StateGraph, END
from support_assistant.retrieval import retrieve_documents
from support_assistant.schemas import SupportResponse

MOCK_LLM = os.getenv("MOCK_LLM", "1") == "1"
# -----------------------------
# 1. Define graph state
# -----------------------------

class SupportState(TypedDict, total=False):
    query: str
    intent: str
    retrieved_docs: list
    answer: str
    sources: list
    confidence: float


# -----------------------------
# 2. Intent classification
# -----------------------------

def classify_intent(state: SupportState):
    query = state["query"].lower()

    if any(word in query for word in [
        "delivery", "deliver", "arrive", "minutes"
    ]):
        intent = "delivery"

    elif any(word in query for word in [
        "return", "refund", "damaged", "spoiled", "missing"
    ]):
        intent = "return"

    elif any(word in query for word in [
        "membership", "pass", "basic", "pass+"
    ]):
        intent = "membership"

    elif any(word in query for word in [
        "track", "tracking", "rider", "where is my order"
    ]):
        intent = "tracking"

    elif any(word in query for word in [
        "cancel", "cancellation"
    ]):
        intent = "cancel"

    elif any(word in query for word in [
        "gift card", "giftcard"
    ]):
        intent = "gift_card"

    elif any(word in query for word in [
        "support", "customer service", "chat", "phone", "email"
    ]):
        intent = "support_hours"

    else:
        intent = "general"

    return {"intent": intent}


# -----------------------------
# 3. Retrieve and answer
# -----------------------------

def retrieve_and_answer(state):
    query = state["query"]

    results = retrieve_documents(query, top_k=3)

    sources = [result["document_id"] for result in results]

    if MOCK_LLM:
        context = "\n\n".join(
            result["text"] for result in results
        )

        prompt = build_support_prompt(
            query=query,
            context=context
        )

        top_chunk = results[0]["text"]


        answer = f"Based on the retrieved context: {top_chunk}"
        confidence = 0.9
    else:
        raise RuntimeError(
            "MOCK_LLM=0 requires a real LLM provider configuration."
        )
    response = SupportResponse(
        answer=answer,
        sources=sources,
        confidence=confidence
    )

    return {
        "retrieved_docs": results,
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence
    }


# -----------------------------
# 4. Direct answer
# -----------------------------

def direct_answer(state):
    if MOCK_LLM:
        answer = (
            "I can help with Zepto policy questions about delivery, returns, "
            "refunds, membership, tracking, cancellation, gift cards, "
            "and customer support."
        )
        confidence = 0.8
    else:
        raise RuntimeError(
            "MOCK_LLM=0 requires a real LLM provider configuration."
        )

    response = SupportResponse(
        answer=answer,
        sources=[],
        confidence=confidence
    )

    return {
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence
    }

# -----------------------------
# 5. Conditional routing
# -----------------------------

def route_intent(state: SupportState):
    if state["intent"] == "general":
        return "direct"

    return "retrieve"


# -----------------------------
# 6. Build LangGraph
# -----------------------------

workflow = StateGraph(SupportState)

workflow.add_node("classify_intent", classify_intent)
workflow.add_node("retrieve_and_answer", retrieve_and_answer)
workflow.add_node("direct_answer", direct_answer)

workflow.set_entry_point("classify_intent")

workflow.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "retrieve": "retrieve_and_answer",
        "direct": "direct_answer"
    }
)

workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)

graph = workflow.compile()


# -----------------------------
# 7. Test the graph
# -----------------------------

if __name__ == "__main__":

    queries = [
        "How long do I have to report a damaged grocery item?",
        "How much does Zepto Pass cost?",
        "Hello, what can you help me with?"
    ]

    for query in queries:

        print("\n" + "=" * 60)
        print("QUERY:", query)

        result = graph.invoke({
            "query": query
        })

        print("INTENT:", result["intent"])
        print("ANSWER:", result["answer"])
        print("SOURCES:", result["sources"])
        print("CONFIDENCE:", result["confidence"])