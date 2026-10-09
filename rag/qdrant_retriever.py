from FlagEmbedding import BGEM3FlagModel

from rag.vector_store import (
    COLLECTION_NAME,
    get_qdrant_client,
)


MODEL_NAME = "BAAI/bge-m3"

TOP_K = 15
MAX_LENGTH = 512


def load_embedding_model():
    print(f"Loading embedding model: {MODEL_NAME}")

    return BGEM3FlagModel(
        MODEL_NAME,
        use_fp16=False,
    )


def search_qdrant(
    query: str,
    model,
    top_k: int = TOP_K,
):
    # 사용자 질문 하나만 embedding
    output = model.encode(
        [query],
        batch_size=1,
        max_length=MAX_LENGTH,
    )

    query_vector = output["dense_vecs"][0]

    client = get_qdrant_client()

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    results = []

    for rank, point in enumerate(
        response.points,
        start=1,
    ):
        payload = point.payload or {}

        results.append(
            {
                "rank": rank,
                "score": float(point.score),
                "chunk_id": payload.get("chunk_id"),
                "source": payload.get("source"),
                "page": payload.get("page"),
                "section": payload.get("section"),
                "text": payload.get("text"),
            }
        )

    client.close()

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

        text = result["text"] or ""

        print(text[:1000])


if __name__ == "__main__":
    model = load_embedding_model()

    test_queries = [
        "What is condition-based maintenance?",
        "How is vibration monitoring used for predictive maintenance?",
        "What causes heat dissipation failure in the AI4I dataset?",
    ]

    for query in test_queries:
        results = search_qdrant(
            query=query,
            model=model,
            top_k=5,
        )

        print_results(
            query,
            results,
        )