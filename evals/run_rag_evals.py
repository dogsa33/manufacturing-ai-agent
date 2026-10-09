from evals.rag_eval_cases import RAG_EVAL_CASES

from rag.qdrant_retriever import (
    load_embedding_model,
    search_qdrant,
)

from rag.reranker import (
    load_reranker,
    rerank,
)


DENSE_TOP_K = 15
RERANK_TOP_K = 5


def is_relevant(
    result: dict,
    relevant_specs: list[dict],
) -> bool:
    """
    검색 결과가 정답 문서 조건 중 하나와 일치하는지 확인한다.

    하나의 relevant spec 안에서는
    source_contains AND section_contains 조건을 모두 만족해야 한다.

    여러 relevant spec 사이는 OR 관계이다.
    """

    source = (result.get("source") or "").lower()
    section = (result.get("section") or "").lower()

    for spec in relevant_specs:
        source_contains = (
            spec.get("source_contains") or ""
        ).lower()

        section_contains = (
            spec.get("section_contains") or ""
        ).lower()

        source_match = (
            not source_contains
            or source_contains in source
        )

        section_match = (
            not section_contains
            or section_contains in section
        )

        if source_match and section_match:
            return True

    return False


def first_relevant_rank(
    results: list[dict],
    relevant_specs: list[dict],
):
    """
    첫 번째 relevant document의 rank를 반환한다.
    없으면 None.
    """

    for rank, result in enumerate(
        results,
        start=1,
    ):
        if is_relevant(
            result,
            relevant_specs,
        ):
            return rank

    return None


def reciprocal_rank(rank):
    if rank is None:
        return 0.0

    return 1.0 / rank


def hit_at_k(rank, k):
    if rank is None:
        return 0

    return int(rank <= k)


def print_case_result(
    case_id,
    query,
    dense_rank,
    rerank_rank,
    dense_results,
    reranked_results,
):
    print("\n" + "=" * 100)
    print(f"CASE : {case_id}")
    print(f"QUERY: {query}")
    print("-" * 100)

    print(
        f"Dense first relevant rank   : "
        f"{dense_rank}"
    )

    print(
        f"Reranker first relevant rank: "
        f"{rerank_rank}"
    )

    print("\n[DENSE TOP 5]")

    for rank, result in enumerate(
        dense_results[:5],
        start=1,
    ):
        print(
            f"{rank}. "
            f"{result['source']} | "
            f"{result['section']} | "
            f"score={result['score']:.4f}"
        )

    print("\n[RERANK TOP 5]")

    for rank, result in enumerate(
        reranked_results[:5],
        start=1,
    ):
        print(
            f"{rank}. "
            f"{result['source']} | "
            f"{result['section']} | "
            f"dense={result['dense_score']:.4f} | "
            f"rerank={result['rerank_score']:.4f}"
        )


def calculate_summary(records):
    n = len(records)

    if n == 0:
        return {}

    return {
        "cases": n,

        "dense_hit@5": sum(
            hit_at_k(
                record["dense_rank"],
                5,
            )
            for record in records
        ) / n,

        "dense_hit@10": sum(
            hit_at_k(
                record["dense_rank"],
                10,
            )
            for record in records
        ) / n,

        "dense_hit@15": sum(
            hit_at_k(
                record["dense_rank"],
                15,
            )
            for record in records
        ) / n,

        "dense_mrr": sum(
            reciprocal_rank(
                record["dense_rank"]
            )
            for record in records
        ) / n,

        "rerank_hit@1": sum(
            hit_at_k(
                record["rerank_rank"],
                1,
            )
            for record in records
        ) / n,

        "rerank_hit@3": sum(
            hit_at_k(
                record["rerank_rank"],
                3,
            )
            for record in records
        ) / n,

        "rerank_hit@5": sum(
            hit_at_k(
                record["rerank_rank"],
                5,
            )
            for record in records
        ) / n,

        "rerank_mrr": sum(
            reciprocal_rank(
                record["rerank_rank"]
            )
            for record in records
        ) / n,
    }


def print_summary(
    title: str,
    summary: dict,
):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    print(
        f"Cases          : "
        f"{summary['cases']}"
    )

    print("\nDense Retrieval")

    print(
        f"Hit@5          : "
        f"{summary['dense_hit@5']:.3f}"
    )

    print(
        f"Hit@10         : "
        f"{summary['dense_hit@10']:.3f}"
    )

    print(
        f"Hit@15         : "
        f"{summary['dense_hit@15']:.3f}"
    )

    print(
        f"MRR            : "
        f"{summary['dense_mrr']:.3f}"
    )

    print("\nReranker")

    print(
        f"Hit@1          : "
        f"{summary['rerank_hit@1']:.3f}"
    )

    print(
        f"Hit@3          : "
        f"{summary['rerank_hit@3']:.3f}"
    )

    print(
        f"Hit@5          : "
        f"{summary['rerank_hit@5']:.3f}"
    )

    print(
        f"MRR            : "
        f"{summary['rerank_mrr']:.3f}"
    )


def main():
    print("Loading embedding model...")

    embedding_model = load_embedding_model()

    print("\nLoading reranker...")

    reranker_model = load_reranker()

    records = []

    for case in RAG_EVAL_CASES:
        case_id = case["id"]
        query = case["query"]
        relevant_specs = case["relevant"]

        print(
            f"\nEvaluating: "
            f"{case_id}"
        )

        # --------------------------------
        # 1. Dense Retrieval
        # --------------------------------

        dense_results = search_qdrant(
            query=query,
            model=embedding_model,
            top_k=DENSE_TOP_K,
        )

        dense_rank = first_relevant_rank(
            dense_results,
            relevant_specs,
        )

        # --------------------------------
        # 2. Reranking
        # --------------------------------

        reranked_results = rerank(
            query=query,
            candidates=dense_results,
            reranker=reranker_model,
            top_k=RERANK_TOP_K,
        )

        rerank_rank = first_relevant_rank(
            reranked_results,
            relevant_specs,
        )

        records.append(
            {
                "case_id": case_id,
                "query": query,
                "dense_rank": dense_rank,
                "rerank_rank": rerank_rank,
            }
        )

        print_case_result(
            case_id=case_id,
            query=query,
            dense_rank=dense_rank,
            rerank_rank=rerank_rank,
            dense_results=dense_results,
            reranked_results=reranked_results,
        )

    # --------------------------------
    # Overall
    # --------------------------------

    overall_summary = calculate_summary(
        records
    )

    print_summary(
        "OVERALL RETRIEVAL EVALUATION",
        overall_summary,
    )

    # --------------------------------
    # English
    # --------------------------------

    english_records = [
        record
        for record in records
        if record["case_id"].endswith("_en")
    ]

    if english_records:
        print_summary(
            "ENGLISH QUERY EVALUATION",
            calculate_summary(
                english_records
            ),
        )

    # --------------------------------
    # Korean
    # --------------------------------

    korean_records = [
        record
        for record in records
        if record["case_id"].endswith("_ko")
    ]

    if korean_records:
        print_summary(
            "KOREAN CROSS-LINGUAL EVALUATION",
            calculate_summary(
                korean_records
            ),
        )


if __name__ == "__main__":
    main()