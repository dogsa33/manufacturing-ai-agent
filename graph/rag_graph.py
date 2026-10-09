import os

from openai import OpenAI
from dotenv import load_dotenv

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from graph.state import RAGState

from rag.qdrant_retriever import (
    load_embedding_model,
    search_qdrant,
)

from rag.reranker import (
    load_reranker,
    rerank,
)


# ============================================================
# Configuration
# ============================================================

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL")

if not OPENAI_MODEL:
    raise RuntimeError(
        "OPENAI_MODEL is not defined in .env"
    )


DENSE_TOP_K = 15
FINAL_TOP_K = 5

MAX_REWRITE_COUNT = 2


# ============================================================
# Clients / Models
# ============================================================

client = OpenAI()


print("Loading RAG models...")

embedding_model = load_embedding_model()
reranker_model = load_reranker()

print("RAG models loaded.")


# ============================================================
# Helper
# ============================================================

def format_documents(
    docs: list[dict],
) -> str:
    """
    Retrieved documents를 LLM prompt용 text로 변환한다.
    """

    parts = []

    for index, doc in enumerate(
        docs,
        start=1,
    ):
        parts.append(
            f"""
DOCUMENT {index}

Source:
{doc.get("source")}

Page:
{doc.get("page")}

Section:
{doc.get("section")}

Content:
{doc.get("text")}
""".strip()
        )

    return "\n\n" + ("\n\n" + "=" * 80 + "\n\n").join(parts)


# ============================================================
# Nodes
# ============================================================

def prepare_query(
    state: RAGState,
) -> dict:

    query = state["query"]

    print("\n[NODE] prepare_query")
    print(f"query: {query}")

    return {
        "retrieval_query": query,
        "rewrite_count": 0,
    }


def retrieve(
    state: RAGState,
) -> dict:

    query = state["retrieval_query"]

    print("\n[NODE] retrieve")
    print(f"retrieval_query: {query}")

    results = search_qdrant(
        query=query,
        model=embedding_model,
        top_k=DENSE_TOP_K,
    )

    print(
        f"Dense candidates: "
        f"{len(results)}"
    )

    return {
        "dense_results": results,
    }


def rerank_documents(
    state: RAGState,
) -> dict:

    query = state["retrieval_query"]
    candidates = state["dense_results"]

    print("\n[NODE] rerank")

    results = rerank(
        query=query,
        candidates=candidates,
        reranker=reranker_model,
        top_k=FINAL_TOP_K,
    )

    print(
        f"Final retrieved docs: "
        f"{len(results)}"
    )

    if results:
        top = results[0]

        print(
            "Top document:",
            top["source"],
            "|",
            top["section"],
        )

    return {
        "retrieved_docs": results,
    }


def grade_documents(
    state: RAGState,
) -> dict:
    """
    검색 문서가 원래 사용자 질문에 답할 충분한
    근거를 포함하는지 LLM이 판단한다.

    rerank_score의 절대값은 사용하지 않는다.
    """

    print("\n[NODE] grade_documents")

    original_query = state["query"]
    retrieval_query = state["retrieval_query"]
    docs = state["retrieved_docs"]

    context = format_documents(docs)

    prompt = f"""
You are a strict retrieval evaluator for a manufacturing RAG system.

Determine whether the retrieved documents contain sufficient factual evidence
to answer the ORIGINAL USER QUESTION.

Important rules:

1. Judge the actual document content, not similarity scores.
2. A document that is merely related to the topic is not sufficient.
3. The documents should contain information that directly helps answer
   the user's question.
4. Do not use your own outside knowledge.
5. If the evidence is incomplete or only indirectly related, answer NO.
6. Output exactly one token:
   YES
   or
   NO

ORIGINAL USER QUESTION:
{original_query}

CURRENT RETRIEVAL QUERY:
{retrieval_query}

RETRIEVED DOCUMENTS:
{context}
"""

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
    )

    result = response.output_text.strip().upper()

    relevant = result.startswith("YES")

    print(
        "Document grade:",
        "RELEVANT" if relevant else "NOT RELEVANT",
    )

    return {
        "documents_relevant": relevant,
        "grade_reason": result,
    }


def rewrite_query(
    state: RAGState,
) -> dict:
    """
    검색 결과가 부족하면 query를 retrieval 친화적으로 다시 작성한다.
    """

    print("\n[NODE] rewrite_query")

    original_query = state["query"]

    current_query = state["retrieval_query"]

    rewrite_count = state.get(
        "rewrite_count",
        0,
    )

    docs = state.get(
        "retrieved_docs",
        [],
    )

    context = format_documents(
        docs[:3]
    )

    prompt = f"""
Rewrite the search query for a technical manufacturing document retrieval system.

The document corpus is primarily written in English and includes:
- predictive maintenance
- reliability-centered maintenance
- condition monitoring
- manufacturing monitoring
- the AI4I predictive-maintenance dataset
- maintenance safety documentation

Your goal is to create ONE improved retrieval query.

Rules:

1. Preserve important technical identifiers such as:
   AI4I, HDF, TWF, PWF, OSF, RNF, vibration, torque, temperature, etc.
2. Prefer precise engineering terminology.
3. Because most source documents are English, you may rewrite a Korean
   question into an English technical retrieval query.
4. Do not answer the question.
5. Return only the rewritten search query.
6. Do not add quotation marks or explanation.
7. Do not introduce standards, threshold values, failure modes,
   equipment names, or technical concepts that are not present in either
   the original user question or the retrieved context.
8. Your task is query reformulation, not query expansion using outside knowledge.

ORIGINAL USER QUESTION:
{original_query}

CURRENT SEARCH QUERY:
{current_query}

PREVIOUSLY RETRIEVED CONTEXT:
{context}
"""

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
    )

    rewritten = response.output_text.strip()

    new_count = rewrite_count + 1

    print(
        f"rewrite_count: "
        f"{new_count}/{MAX_REWRITE_COUNT}"
    )

    print(
        f"rewritten query: "
        f"{rewritten}"
    )

    return {
        "retrieval_query": rewritten,
        "rewrite_count": new_count,
    }


def generate_answer(
    state: RAGState,
) -> dict:
    """
    최종 Top-K 문서만 사용하여 답변을 생성한다.
    """

    print("\n[NODE] generate_answer")

    query = state["query"]

    docs = state["retrieved_docs"]

    relevant = state.get(
        "documents_relevant",
        False,
    )

    context = format_documents(docs)

    evidence_instruction = (
        """
The retrieval evaluator determined that the documents contain sufficient
evidence. Answer the question using only the supplied documents.
"""
        if relevant
        else
        """
The retrieval evaluator could not confirm sufficient evidence even after
the allowed retrieval attempts.

Do not invent an answer.

Explain which parts can be supported by the supplied documents and clearly
state when the retrieved evidence is insufficient.
"""
    )

    prompt = f"""
You are a manufacturing AI assistant.

Answer the user's question using ONLY the retrieved source documents below.

Rules:

1. Do not use factual information that is absent from the supplied documents.
2. Answer in the same language as the user's question.
3. Be concise but technically precise.
4. Distinguish dataset-specific rules from general manufacturing knowledge.
5. In particular, AI4I failure-generation conditions are rules of a synthetic
   dataset and must not be presented as universal physical failure thresholds.
6. Cite evidence inline using this format:

   [source | p.page | section]

7. If the documents do not support a claim, explicitly say that the available
   documents do not establish it.

{evidence_instruction}

USER QUESTION:
{query}

RETRIEVED DOCUMENTS:
{context}
"""

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
    )

    answer = response.output_text.strip()

    print("Answer generated.")

    return {
        "answer": answer,
    }


# ============================================================
# Conditional Routing
# ============================================================

def route_after_grade(
    state: RAGState,
) -> str:

    relevant = state.get(
        "documents_relevant",
        False,
    )

    rewrite_count = state.get(
        "rewrite_count",
        0,
    )

    if relevant:
        print(
            "\n[ROUTE] relevant "
            "→ generate"
        )

        return "generate"

    if rewrite_count < MAX_REWRITE_COUNT:
        print(
            "\n[ROUTE] insufficient evidence "
            "→ rewrite"
        )

        return "rewrite"

    print(
        "\n[ROUTE] max rewrites reached "
        "→ generate cautious answer"
    )

    return "generate"


# ============================================================
# Graph
# ============================================================

def build_graph():

    builder = StateGraph(
        RAGState
    )

    builder.add_node(
        "prepare_query",
        prepare_query,
    )

    builder.add_node(
        "retrieve",
        retrieve,
    )

    builder.add_node(
        "rerank",
        rerank_documents,
    )

    builder.add_node(
        "grade",
        grade_documents,
    )

    builder.add_node(
        "rewrite",
        rewrite_query,
    )

    builder.add_node(
        "generate",
        generate_answer,
    )

    # START
    builder.add_edge(
        START,
        "prepare_query",
    )

    # Normal retrieval pipeline
    builder.add_edge(
        "prepare_query",
        "retrieve",
    )

    builder.add_edge(
        "retrieve",
        "rerank",
    )

    builder.add_edge(
        "rerank",
        "grade",
    )

    # Conditional edge
    builder.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "generate": "generate",
            "rewrite": "rewrite",
        },
    )

    # Corrective retrieval loop
    builder.add_edge(
        "rewrite",
        "retrieve",
    )

    # Finish
    builder.add_edge(
        "generate",
        END,
    )

    return builder.compile()


graph = build_graph()


# ============================================================
# Manual Test
# ============================================================

if __name__ == "__main__":

    result = graph.invoke(
        {
            "query": (
                "AI4I 데이터셋에서 "
                "열 방산 고장은 어떤 조건에서 발생하는가?"
            )
        }
    )

    print("\n" + "=" * 100)
    print("GRAPH FINISHED")
    print("=" * 100)

    print(
        "\nOriginal query:",
        result["query"],
    )

    print(
        "\nFinal retrieval query:",
        result["retrieval_query"],
    )

    print(
        "\nRewrite count:",
        result["rewrite_count"],
    )

    print(
        "\nDocuments relevant:",
        result["documents_relevant"],
    )

    print(
        "\nFinal answer:\n"
    )

    print(
        result["answer"]
    )