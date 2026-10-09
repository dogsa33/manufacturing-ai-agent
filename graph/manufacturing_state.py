from typing import Any, TypedDict


class ManufacturingState(TypedDict, total=False):

    # ========================================================
    # Current User Turn
    # ========================================================

    query: str

    # ========================================================
    # Router
    # ========================================================

    route: str
    tool_name: str
    tool_arguments: dict[str, Any]

    # ========================================================
    # Tool Argument Validation
    # ========================================================

    missing_arguments: list[str]
    needs_clarification: bool

    # ========================================================
    # Pending Tool Context
    # 다음 사용자 턴까지 유지해야 하는 정보
    # ========================================================

    pending_tool_name: str
    pending_tool_arguments: dict[str, Any]
    pending_missing_arguments: list[str]

    # ========================================================
    # MCP Tool
    # ========================================================

    tool_result: Any

    # ========================================================
    # RAG
    # ========================================================

    rag_answer: str
    rag_documents: list[dict]
    rag_relevant: bool
    rag_rewrite_count: int
    rag_retrieval_query: str

    # ========================================================
    # Final Answer
    # ========================================================

    answer: str

    # ========================================================
    # Validator
    # ========================================================

    validation_passed: bool
    validation_feedback: str

    # ========================================================
    # Replan
    # ========================================================

    retry_count: int