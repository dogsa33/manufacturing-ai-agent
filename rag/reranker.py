from FlagEmbedding import FlagReranker


RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

FINAL_TOP_K = 5


def load_reranker():
    print(
        f"Loading reranker model: "
        f"{RERANKER_MODEL_NAME}"
    )

    reranker = FlagReranker(
        RERANKER_MODEL_NAME,
        use_fp16=False,
    )

    return reranker


def rerank(
    query: str,
    candidates: list[dict],
    reranker,
    top_k: int = FINAL_TOP_K,
) -> list[dict]:

    if not candidates:
        return []

    pairs = [
        [
            query,
            candidate["text"],
        ]
        for candidate in candidates
    ]

    scores = reranker.compute_score(
        pairs,
        normalize=True,
    )

    # 후보가 하나일 경우 scalar로 반환될 수 있음
    if not isinstance(scores, list):
        scores = [scores]

    reranked = []

    for candidate, score in zip(
        candidates,
        scores,
    ):
        result = candidate.copy()

        # 기존 Qdrant dense score 보존
        result["dense_score"] = result.pop(
            "score"
        )

        result["rerank_score"] = float(
            score
        )

        reranked.append(result)

    reranked.sort(
        key=lambda x: x["rerank_score"],
        reverse=True,
    )

    return reranked[:top_k]