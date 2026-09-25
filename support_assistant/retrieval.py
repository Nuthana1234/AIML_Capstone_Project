from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# Paths
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"


# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# Connect to existing ChromaDB
client = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = client.get_collection(
    name="zepto_policies"
)


def retrieve_documents(query, top_k=3):
    """Retrieve the most relevant policy chunks."""

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved = []

    for i in range(len(results["documents"][0])):
        retrieved.append(
            {
                "document_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "distance": results["distances"][0][i],
            }
        )

    return retrieved


if __name__ == "__main__":

    query = "How long do I have to report a damaged grocery item?"

    results = retrieve_documents(query)

    print("\nQuery:")
    print(query)

    print("\nTop retrieved documents:\n")

    for rank, result in enumerate(results, start=1):
        print(f"--- Result {rank} ---")
        print("Document:", result["document_id"])
        print("Distance:", result["distance"])
        print("Text:", result["text"])
        print()