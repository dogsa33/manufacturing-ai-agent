from rag.qdrant_retriever import (
    load_embedding_model,
    search_qdrant,
)
from rag.reranker import (
    load_reranker,
    rerank,
)


CANDIDATE_K = 15
FINAL_K = 5


def print_results(
    query: str,
    results: list[dict],
):
    print("\n" + "=" * 100)
    print(f"QUERY: {query}")
    print("=" * 100)

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"\n[{rank}] "
            f"dense={result['dense_score']:.4f} | "
            f"rerank={result['rerank_score']:.4f}"
        )

        print(
            f"source={result['source']} | "
            f"page={result['page']} | "
            f"section={result['section']}"
        )

        print("-" * 100)

        print(
            (result["text"] or "")[:800]
        )


if __name__ == "__main__":
    embedding_model = load_embedding_model()
    reranker = load_reranker()

    test_queries = [
        "What is condition-based maintenance?",
        "How is vibration monitoring used for predictive maintenance?",
        "What causes heat dissipation failure in the AI4I dataset?",

        # 실제 서비스 환경을 고려한 한국어 질문도 테스트
        "상태 기반 정비란 무엇인가?",
        "진동 모니터링은 예지 정비에서 어떻게 활용되는가?",
        "AI4I 데이터셋에서 열 방산 고장은 어떤 조건에서 발생하는가?",
    ]

    for query in test_queries:

        candidates = search_qdrant(
            query=query,
            model=embedding_model,
            top_k=CANDIDATE_K,
        )

        results = rerank(
            query=query,
            candidates=candidates,
            reranker=reranker,
            top_k=FINAL_K,
        )

        print_results(
            query,
            results,
        )