import numpy as np

from FlagEmbedding import BGEM3FlagModel

from rag.document_loader import load_all_documents
from rag.chunker import create_chunks


MODEL_NAME = "BAAI/bge-m3"

BATCH_SIZE = 4
MAX_LENGTH = 512
TOP_K = 5


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    norms = np.clip(norms, 1e-12, None)

    return vectors / norms


def build_retriever():
    print("Loading documents...")
    documents = load_all_documents()

    print("Creating chunks...")
    chunks = create_chunks(documents)

    print(f"Chunks: {len(chunks)}")

    print(f"\nLoading embedding model: {MODEL_NAME}")

    model = BGEM3FlagModel(
        MODEL_NAME,
        use_fp16=False,
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("\nEmbedding document chunks...")

    output = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )

    document_vectors = output["dense_vecs"]

    document_vectors = normalize_vectors(
        document_vectors
    )

    print(
        f"Document embedding shape: "
        f"{document_vectors.shape}"
    )

    return model, chunks, document_vectors


def search(
    query: str,
    model,
    chunks: list[dict],
    document_vectors: np.ndarray,
    top_k: int = TOP_K,
):
    query_output = model.encode(
        [query],
        batch_size=1,
        max_length=MAX_LENGTH,
    )

    query_vector = query_output["dense_vecs"]

    query_vector = normalize_vectors(
        query_vector
    )[0]

    # cosine similarity
    scores = document_vectors @ query_vector

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for rank, index in enumerate(
        top_indices,
        start=1,
    ):
        chunk = chunks[index]

        results.append(
            {
                "rank": rank,
                "score": float(scores[index]),
                "source": chunk["source"],
                "page": chunk["page"],
                "section": chunk["section"],
                "text": chunk["text"],
            }
        )

    return results


def print_results(
    query: str,
    results: list[dict],
):
    print("\n" + "=" * 100)
    print(f"QUERY: {query}")
    print("=" * 100)

    for result in results:
        print(
            f"\n[{result['rank']}] "
            f"score={result['score']:.4f}"
        )

        print(
            f"source={result['source']} | "
            f"page={result['page']} | "
            f"section={result['section']}"
        )

        print("-" * 100)

        print(
            result["text"][:1000]
        )


if __name__ == "__main__":
    model, chunks, document_vectors = (
        build_retriever()
    )

    test_queries = [
        "What is condition-based maintenance?",
        "How is vibration monitoring used for predictive maintenance?",
        "What causes heat dissipation failure in the AI4I dataset?",
    ]

    for query in test_queries:
        results = search(
            query=query,
            model=model,
            chunks=chunks,
            document_vectors=document_vectors,
            top_k=5,
        )

        print_results(
            query,
            results,
        )