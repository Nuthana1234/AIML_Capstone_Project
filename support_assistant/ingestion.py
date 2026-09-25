from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# Paths
BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"


# Load local embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# Create persistent ChromaDB client
client = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = client.get_or_create_collection(
    name="zepto_policies"
)


def load_documents():
    """Load all eight Zepto policy documents."""
    documents = []

    for file_path in sorted(DOCUMENTS_DIR.glob("doc_*.txt")):
        try:
            text = file_path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            text = file_path.read_text(encoding="cp1252").strip()
        documents.append(
            {
                "document_id": file_path.stem,
                "text": text,
            }
        )

    return documents


def create_chunks(documents):
    """
    Create one chunk per policy document.

    The capstone allows simple per-document chunking
    because each policy document is short.
    """
    chunks = []

    for document in documents:
        chunks.append(
            {
                "id": document["document_id"],
                "text": document["text"],
                "metadata": {
                    "source": document["document_id"]
                },
            }
        )

    return chunks


def build_vector_store():
    documents = load_documents()

    print(f"Documents loaded: {len(documents)}")

    chunks = create_chunks(documents)

    print(f"Chunks created: {len(chunks)}")

    texts = [chunk["text"] for chunk in chunks]

    print("Generating embeddings...")
    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True
    ).tolist()

    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[chunk["metadata"] for chunk in chunks],
    )

    print("Embeddings stored in ChromaDB.")
    print("Collection:", collection.name)
    print("Stored chunks:", collection.count())


if __name__ == "__main__":
    build_vector_store()