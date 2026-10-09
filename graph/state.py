from typing import TypedDict


class RAGState(TypedDict, total=False):
    # 사용자 원본 질문
    query: str

    # 현재 retrieval에 사용하는 질문
    retrieval_query: str

    # Qdrant dense retrieval 결과
    dense_results: list[dict]

    # reranker 최종 결과
    retrieved_docs: list[dict]

    # LLM document grading 결과
    documents_relevant: bool
    grade_reason: str

    # query rewrite 횟수
    rewrite_count: int

    # 최종 응답
    answer: str