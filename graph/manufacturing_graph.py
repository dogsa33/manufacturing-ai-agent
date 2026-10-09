import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from graph.manufacturing_state import (
    ManufacturingState,
)

from graph.mcp_client import (
    list_mcp_tools,
    call_mcp_tool_data,
)

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.checkpoint.memory import MemorySaver

# ============================================================
# Configuration
# ============================================================

load_dotenv()

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5.6-luna",
)

client = AsyncOpenAI()


# Main Agent가 Validator에서 실패했을 때
# 최대 2번까지 다시 시도
MAX_AGENT_RETRIES = 2


# ============================================================
# MCP Tool Catalog Cache
# ============================================================

_tool_catalog_cache = None


async def get_tool_catalog():
    """
    MCP Server의 Tool 목록과 schema를 읽는다.

    최초 1회만 MCP Server에 연결하고,
    이후에는 현재 Python 프로세스에서 cache를 사용한다.
    """

    global _tool_catalog_cache

    if _tool_catalog_cache is not None:
        return _tool_catalog_cache

    tools = await list_mcp_tools()

    catalog = []

    for tool in tools:

        input_schema = getattr(
            tool,
            "inputSchema",
            None,
        )

        if input_schema is None:
            input_schema = getattr(
                tool,
                "input_schema",
                None,
            )

        catalog.append(
            {
                "name": tool.name,
                "description": (
                    tool.description or ""
                ),
                "input_schema": input_schema,
            }
        )

    _tool_catalog_cache = catalog

    return catalog


# ============================================================
# JSON Helper
# ============================================================

def parse_json_output(
    text: str,
) -> dict:
    """
    LLM이 JSON을 ```json ... ``` 형태로 반환해도
    Python dict로 변환할 수 있도록 처리한다.
    """

    text = text.strip()

    if text.startswith("```"):

        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if (
            lines
            and lines[-1].strip().startswith("```")
        ):
            lines = lines[:-1]

        text = "\n".join(lines)

    return json.loads(text)


# ============================================================
# Begin User Turn
# ============================================================

async def begin_turn(
    state: ManufacturingState,
) -> dict:
    """
    새로운 사용자 메시지가 들어올 때
    이전 실행의 일시적인 상태는 초기화한다.

    단,
    pending_tool_* 정보는 초기화하지 않는다.
    사용자가 clarification에 답할 때 필요하기 때문이다.
    """

    print("\n[NODE] begin_turn")

    return {
        "route": "",
        "tool_name": "",
        "tool_arguments": {},

        "missing_arguments": [],
        "needs_clarification": False,

        "tool_result": {},

        "rag_answer": "",
        "rag_documents": [],
        "rag_relevant": False,
        "rag_rewrite_count": 0,
        "rag_retrieval_query": "",

        "answer": "",

        "validation_passed": False,
        "validation_feedback": "",

        "retry_count": 0,
    }


# ============================================================
# Router Node
# ============================================================

async def route_query(
    state: ManufacturingState,
) -> dict:

    print("\n[NODE] router")

    query = state["query"]

    retry_count = state.get(
        "retry_count",
        0,
    )

    validation_feedback = state.get(
        "validation_feedback",
        "",
    )

    previous_route = state.get(
        "route",
        "",
    )

    previous_tool = state.get(
        "tool_name",
        "",
    )

    # ========================================================
    # Pending Tool Context
    # ========================================================

    pending_tool_name = state.get(
        "pending_tool_name",
        "",
    )

    pending_tool_arguments = state.get(
        "pending_tool_arguments",
        {},
    )

    pending_missing_arguments = state.get(
        "pending_missing_arguments",
        [],
    )

    tool_catalog = await get_tool_catalog()

    catalog_text = json.dumps(
        tool_catalog,
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""
You are the router for a manufacturing AI system.

Your job is ONLY to select the correct execution route.

Select exactly ONE route:

1. "tool"
2. "rag"
3. "direct"


============================================================
TOOL ROUTE
============================================================

Use "tool" whenever the user's intent matches an available MCP Tool.

IMPORTANT:

Missing Tool arguments are NOT a reason to choose "direct".

If the user clearly asks for a Tool-supported operation but does not provide
all required arguments:

- still choose route="tool"
- select the correct Tool
- extract only the arguments actually provided by the user

The application has a separate argument-validation node that will ask
the user for missing values.

IMPORTANT TOOL SELECTION RULE:

1. Use `predict_machine_failure` when the user asks only for:
   - failure prediction
   - failure probability
   - risk level
   - whether the condition is risky
   - general failure-risk analysis without explicitly asking for reasons,
     influential variables, risk drivers, or sensitivity

   Examples:
   - "고장 위험을 예측해줘."
   - "고장 위험을 분석해줘."
   - "이 조건이 위험한지 알려줘."
   - "고장 확률이 얼마나 되는지 알려줘."

2. Use `explain_machine_failure` only when the user explicitly asks
   WHY the prediction is risky or asks for variable-level explanation.

   Explanation intent includes requests such as:
   - "왜 위험한지"
   - "왜 이렇게 예측됐는지"
   - "주요 변수를 분석해줘"
   - "어떤 변수가 영향을 줬는지"
   - "위험요인을 설명해줘"
   - "risk driver"
   - "local sensitivity"
   - "변수별 영향"
   - "주요 원인을 모델 기준으로 설명"

3. `explain_machine_failure` already provides the failure prediction,
   failure probability, risk level, and local sensitivity analysis.
   Therefore, if BOTH prediction and explicit explanation intent are present,
   use `explain_machine_failure` directly.

4. IMPORTANT:
   The Korean words "분석", "분석해줘", or "위험 분석" by themselves
   do NOT mean explanation intent.
   Do not select `explain_machine_failure` unless the user explicitly asks
   for reasons, influential variables, risk drivers, or local sensitivity.

5. Do NOT choose a tool based only on the first keyword.
   Determine whether explicit explanation intent exists in the full request.

Examples:

User:
"Product Type L의 고장 위험을 예측해줘."

Correct:

{{
  "route": "tool",
  "tool_name": "predict_machine_failure",
  "tool_arguments": {{
    "product_type": "L"
  }}
}}

User:
"다음 제조조건의 고장 위험을 분석해줘. Product Type은 L, Air temperature는 301.0, Process temperature는 310.5, Rotational speed는 1300, Torque는 65.0, Tool wear는 200이야."

Correct:

{{
  "route": "tool",
  "tool_name": "predict_machine_failure",
  "tool_arguments": {{
    "product_type": "L",
    "air_temperature": 301.0,
    "process_temperature": 310.5,
    "rotational_speed": 1300,
    "torque": 65.0,
    "tool_wear": 200
  }}
}}

User:
"다음 제조조건의 고장 위험을 먼저 예측하고, 왜 위험하게 판단됐는지도 주요 변수 기준으로 분석해줘. Product Type은 L, Air temperature는 301.0, Process temperature는 310.5, Rotational speed는 1300, Torque는 65.0, Tool wear는 200이야."

Correct:

{{
  "route": "tool",
  "tool_name": "explain_machine_failure",
  "tool_arguments": {{
    "product_type": "L",
    "air_temperature": 301.0,
    "process_temperature": 310.5,
    "rotational_speed": 1300,
    "torque": 65.0,
    "tool_wear": 200
  }}
}}

User:
"Torque의 정상과 고장 조건을 비교해줘."

Correct:

{{
  "route": "tool",
  "tool_name": "compare_process_condition",
  "tool_arguments": {{
    "feature": "Torque"
  }}
}}

User:
"고장 유형별 발생 건수를 알려줘."

Correct:

{{
  "route": "tool",
  "tool_name": "failure_type_summary",
  "tool_arguments": {{}}
}}


============================================================
RAG ROUTE
============================================================

Use "rag" when the user asks what documents, manuals, or technical
references say.

Examples:

- AI4I에서 HDF는 어떤 조건에서 발생하는가?
- Condition-based maintenance란 무엇인가?
- NASA RCM 문서에서 예방정비는 어떻게 설명하는가?
- OSHA LOTO 관련 안전 지침은 무엇인가?


============================================================
DIRECT ROUTE
============================================================

Use "direct" ONLY for:

- greetings
- conversational questions
- meta questions about this AI system
- questions that genuinely require neither MCP Tools nor RAG

IMPORTANT:

Do NOT use "direct" just because Tool inputs are incomplete.

For example:

"Product Type L의 고장 위험을 예측해줘."

is NOT direct.

It is:

route="tool"
tool_name="predict_machine_failure"


============================================================
AVAILABLE MCP TOOLS
============================================================

{catalog_text}


============================================================
PENDING TOOL CONTEXT
============================================================

Pending tool:
{pending_tool_name}

Already collected arguments:
{json.dumps(
    pending_tool_arguments,
    ensure_ascii=False,
    indent=2,
)}

Still missing arguments:
{json.dumps(
    pending_missing_arguments,
    ensure_ascii=False,
    indent=2,
)}

If a pending Tool exists and the current message appears to provide
the requested missing values:

- continue with route="tool"
- use the SAME pending Tool
- extract newly supplied values
- do not ask again for values already collected

If the current user clearly starts a completely different task,
you may choose another route.


============================================================
RETRY INFORMATION
============================================================

Retry count:
{retry_count}

Previous route:
{previous_route}

Previous tool:
{previous_tool}

Validator feedback:
{validation_feedback}


============================================================
FINAL RULES
============================================================

- Choose Tool based on user intent, not based on whether all arguments exist.
- Never invent Tool arguments.
- Extract only values actually supplied by the user.
- Tool argument names must follow the input schema exactly.
- If the Tool needs more values, the downstream argument checker handles it.
- AI4I documentation definitions normally use RAG.
- AI4I dataset statistics and ML predictions use MCP Tools.
- If pending Tool context exists and this is a continuation, preserve it.

Return ONLY valid JSON:

{{
  "route": "tool" | "rag" | "direct",
  "tool_name": "tool name or empty string",
  "tool_arguments": {{}}
}}

CURRENT USER MESSAGE:

{query}
"""

    response = await client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
    )

    decision = parse_json_output(
        response.output_text
    )

    route = decision.get(
        "route",
        "direct",
    )

    tool_name = decision.get(
        "tool_name",
        "",
    )

    tool_arguments = decision.get(
        "tool_arguments",
        {},
    )

    # ========================================================
    # Pending argument merge
    # ========================================================

    if (
        route == "tool"
        and pending_tool_name
        and tool_name == pending_tool_name
    ):

        tool_arguments = {
            **pending_tool_arguments,
            **tool_arguments,
        }

        print(
            "Merged pending arguments."
        )

    print(
        f"route: {route}"
    )

    if route == "tool":

        print(
            f"tool: {tool_name}"
        )

        print(
            "arguments:",
            tool_arguments,
        )

    return {
        "route": route,
        "tool_name": tool_name,
        "tool_arguments": tool_arguments,
        "retry_count": retry_count,
    }


# ============================================================
# Tool Argument Validation Node
# ============================================================

async def check_tool_arguments(
    state: ManufacturingState,
) -> dict:

    print(
        "\n[NODE] check_tool_arguments"
    )

    tool_name = state["tool_name"]

    arguments = state.get(
        "tool_arguments",
        {},
    )

    tool_catalog = await get_tool_catalog()

    selected_tool = None

    for tool in tool_catalog:

        if tool["name"] == tool_name:

            selected_tool = tool
            break

    if selected_tool is None:

        raise RuntimeError(
            f"Unknown MCP tool: {tool_name}"
        )

    schema = (
        selected_tool.get(
            "input_schema"
        )
        or {}
    )

    required = schema.get(
        "required",
        [],
    )

    missing = [
        argument_name
        for argument_name in required
        if argument_name not in arguments
        or arguments[argument_name] is None
    ]

    needs_clarification = (
        len(missing) > 0
    )

    print(
        "missing arguments:",
        missing,
    )

    # --------------------------------------------------------
    # 아직 값이 부족함
    # → 다음 사용자 턴을 위해 저장
    # --------------------------------------------------------

    if needs_clarification:

        return {
            "missing_arguments": missing,

            "needs_clarification": True,

            "pending_tool_name": (
                tool_name
            ),

            "pending_tool_arguments": (
                arguments
            ),

            "pending_missing_arguments": (
                missing
            ),
        }

    # --------------------------------------------------------
    # 필요한 값이 모두 모임
    # → pending context 제거
    # --------------------------------------------------------

    return {
        "missing_arguments": [],

        "needs_clarification": False,

        "pending_tool_name": "",

        "pending_tool_arguments": {},

        "pending_missing_arguments": [],
    }


# ============================================================
# Missing Argument Clarification Node
# ============================================================

async def clarify_missing_arguments(
    state: ManufacturingState,
) -> dict:
    """
    필수 입력이 부족하면 MCP Tool을 호출하지 않고
    사용자에게 필요한 값을 요청한다.
    """

    print(
        "\n[NODE] clarify_missing_arguments"
    )

    tool_name = state["tool_name"]

    missing = state.get(
        "missing_arguments",
        [],
    )

    query = state["query"]

    prompt = f"""
You are a manufacturing AI assistant.

The user requested an analysis that requires the MCP Tool:

{tool_name}

However, some required input values are missing.

Missing arguments:

{json.dumps(
    missing,
    ensure_ascii=False,
    indent=2,
)}

Original user question:

{query}

Ask the user only for the missing information.

Rules:

1. Do not invent values.
2. Do not execute or simulate the prediction.
3. Explain the required input names in natural Korean.
4. Keep the response concise.
5. Do not ask again for values already provided by the user.
"""

    response = await client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
    )

    return {
        "answer": (
            response.output_text.strip()
        ),
    }


# ============================================================
# MCP Tool Execution Node
# ============================================================

async def execute_tool(
    state: ManufacturingState,
) -> dict:
    """
    Router가 선택한 MCP Tool을 실행한다.
    """

    print("\n[NODE] execute_tool")

    tool_name = state["tool_name"]

    arguments = state.get(
        "tool_arguments",
        {},
    )

    print(
        f"tool: {tool_name}"
    )

    print(
        f"arguments: {arguments}"
    )

    result = await call_mcp_tool_data(
        tool_name=tool_name,
        arguments=arguments,
    )

    print(
        "MCP tool completed."
    )

    return {
        "tool_result": result,
    }


# ============================================================
# MCP Tool Answer Generation
# ============================================================

async def generate_tool_answer(
    state: ManufacturingState,
) -> dict:
    """
    MCP Tool 결과를 바탕으로 답변을 생성한다.

    Validator 재시도인 경우 이전 답변과
    validation_feedback을 참고해서 수정한다.
    """

    print(
        "\n[NODE] generate_tool_answer"
    )

    query = state["query"]

    tool_name = state["tool_name"]

    result = state["tool_result"]

    previous_answer = state.get(
        "answer",
        "",
    )

    validation_feedback = state.get(
        "validation_feedback",
        "",
    )

    if validation_feedback:

        print(
            "Using validator feedback "
            "for tool answer regeneration."
        )

    prompt = f"""
You are a manufacturing data-analysis assistant.

Answer the user's question using ONLY the MCP Tool result.

============================================================
USER QUESTION
============================================================

{query}


============================================================
MCP TOOL
============================================================

{tool_name}


============================================================
MCP TOOL RESULT
============================================================

{json.dumps(
    result,
    ensure_ascii=False,
    indent=2,
)}


============================================================
PREVIOUS ANSWER
============================================================

{previous_answer}


============================================================
VALIDATOR FEEDBACK
============================================================

{validation_feedback}


============================================================
RULES
============================================================

1. Do not invent values.

2. Preserve units and numerical values from the Tool result.

3. Prediction output must be described as a model prediction,
   not as a measured physical fact.

4. Results from explain_machine_failure or risk_driver_chart
   are local sensitivity results.

5. Do not describe local sensitivity as:
   - causal effect
   - SHAP value
   - proof of physical causality

6. Answer in the same language as the user.

7. Be concise and technically precise.

8. If VALIDATOR FEEDBACK is not empty,
   explicitly correct the problem identified by the Validator.

9. Do not repeat the same mistake from PREVIOUS ANSWER.

10. Validator feedback may request clarification, qualification,
    omission of unsupported claims, or inclusion of information
    already present in the Tool result.

11. Even when correcting the answer, use ONLY the MCP Tool result
    as factual evidence.
"""

    response = await client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
    )

    answer = response.output_text.strip()

    return {
        "answer": answer,
    }


# ============================================================
# RAG Node
# ============================================================

async def run_rag(
    state: ManufacturingState,
) -> dict:
    """
    기존 Corrective RAG Graph를 실행한다.

    Validator 재시도인 경우:
    RAG 검색 결과 + 기존 답변 + Validator feedback을 사용해
    최종 답변을 다시 교정한다.
    """

    print("\n[NODE] run_rag")

    from graph.rag_graph import (
        graph as rag_graph,
    )

    query = state["query"]

    previous_answer = state.get(
        "answer",
        "",
    )

    validation_feedback = state.get(
        "validation_feedback",
        "",
    )

    # ========================================================
    # Existing Corrective RAG
    # ========================================================

    result = await rag_graph.ainvoke(
        {
            "query": query,
        }
    )

    answer = result["answer"]

    retrieved_docs = result.get(
        "retrieved_docs",
        [],
    )

    rag_relevant = result.get(
        "documents_relevant",
        False,
    )

    rewrite_count = result.get(
        "rewrite_count",
        0,
    )

    retrieval_query = result.get(
        "retrieval_query",
        query,
    )

    # ========================================================
    # Outer Validator Feedback Correction
    # ========================================================

    if validation_feedback:

        print(
            "Using validator feedback "
            "for RAG answer regeneration."
        )

        evidence_docs = []

        for doc in retrieved_docs:

            evidence_docs.append(
                {
                    "source": doc.get(
                        "source"
                    ),

                    "page": doc.get(
                        "page"
                    ),

                    "section": doc.get(
                        "section"
                    ),

                    "text": (
                        doc.get(
                            "text",
                            "",
                        )
                        or ""
                    )[:2000],
                }
            )

        correction_prompt = f"""
You are correcting a document-grounded manufacturing RAG answer.

Use ONLY the retrieved document evidence below.

============================================================
USER QUESTION
============================================================

{query}


============================================================
PREVIOUS ANSWER
============================================================

{previous_answer}


============================================================
NEW RAG ANSWER
============================================================

{answer}


============================================================
VALIDATOR FEEDBACK
============================================================

{validation_feedback}


============================================================
DOCUMENT RELEVANCE
============================================================

{rag_relevant}


============================================================
RETRIEVED DOCUMENTS
============================================================

{json.dumps(
    evidence_docs,
    ensure_ascii=False,
    indent=2,
)}


============================================================
RULES
============================================================

1. Correct the exact problem identified by the Validator.

2. Use ONLY the retrieved documents as factual evidence.

3. Do not introduce external technical knowledge.

4. Do not invent threshold values, standards, failure modes,
   equipment names, or procedures.

5. If documents are insufficient, explicitly say that the
   supplied documents do not establish the requested fact.

6. AI4I failure-generation conditions are synthetic dataset rules,
   not universal physical manufacturing thresholds.

7. Preserve useful source citations in the format:

   [source | p.page | section]

8. Answer in the same language as the user.

9. Do not repeat the same unsupported claim from PREVIOUS ANSWER.
"""

        correction_response = (
            await client.responses.create(
                model=OPENAI_MODEL,
                input=correction_prompt,
            )
        )

        answer = (
            correction_response
            .output_text
            .strip()
        )

    return {
        "rag_answer": answer,

        "answer": answer,

        "rag_documents":
            retrieved_docs,

        "rag_relevant":
            rag_relevant,

        "rag_rewrite_count":
            rewrite_count,

        "rag_retrieval_query":
            retrieval_query,
    }


# ============================================================
# Direct Answer Node
# ============================================================

async def direct_answer(
    state: ManufacturingState,
) -> dict:
    """
    Tool 또는 RAG가 필요 없는 질문을 처리한다.

    Validator 재시도 시 이전 답변과 feedback을 참고한다.
    """

    print("\n[NODE] direct_answer")

    query = state["query"]

    previous_answer = state.get(
        "answer",
        "",
    )

    validation_feedback = state.get(
        "validation_feedback",
        "",
    )

    if validation_feedback:

        print(
            "Using validator feedback "
            "for direct answer regeneration."
        )

    tool_catalog = await get_tool_catalog()

    capabilities = [
        {
            "name": tool["name"],
            "description": (
                tool["description"]
            ),
        }
        for tool in tool_catalog
    ]

    prompt = f"""
You are the conversational layer of a manufacturing AI system.

You must describe ONLY capabilities that are actually implemented.


============================================================
CURRENT IMPLEMENTED MCP TOOLS
============================================================

{json.dumps(
    capabilities,
    ensure_ascii=False,
    indent=2,
)}


============================================================
CURRENT RAG KNOWLEDGE SCOPE
============================================================

- AI4I 2020 predictive-maintenance dataset documentation
- NASA Reliability-Centered Maintenance guide
- DOE Operations & Maintenance Best Practices
- NIST manufacturing/process-monitoring documentation
- OSHA lockout/tagout safety documentation


============================================================
USER QUESTION
============================================================

{query}


============================================================
PREVIOUS ANSWER
============================================================

{previous_answer}


============================================================
VALIDATOR FEEDBACK
============================================================

{validation_feedback}


============================================================
RULES
============================================================

1. Do not claim capabilities that are not implemented above.

2. Do not claim that this system currently supports:
   - OEE analysis
   - takt time analysis
   - inventory optimization
   - demand forecasting
   - MTBF/MTTR analysis
   - energy analysis

   unless an available Tool explicitly supports it.

3. Do not invent manufacturing data.

4. If the user asks what the system can do,
   summarize only actual MCP Tool and RAG capabilities.

5. Answer in the same language as the user.

6. If VALIDATOR FEEDBACK is not empty,
   correct the exact issue identified by the Validator.

7. Do not repeat unsupported claims from PREVIOUS ANSWER.
"""

    response = await client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
    )

    answer = response.output_text.strip()

    return {
        "answer": answer,
    }


# ============================================================
# Main Validator Node
# ============================================================

async def validate_answer(
    state: ManufacturingState,
) -> dict:
    """
    최종 답변이 실제 실행 결과와 근거에 부합하는지
    검증한다.
    """

    print("\n[NODE] validator")

    query = state["query"]

    route = state["route"]

    answer = state.get(
        "answer",
        "",
    )

    evidence = {}

    # --------------------------------------------------------
    # Tool evidence
    # --------------------------------------------------------

    if route == "tool":

        evidence = {
            "tool_name": state.get(
                "tool_name"
            ),

            "tool_arguments": state.get(
                "tool_arguments",
                {},
            ),

            "tool_result": state.get(
                "tool_result",
                {},
            ),
        }

    # --------------------------------------------------------
    # RAG evidence
    # --------------------------------------------------------

    elif route == "rag":

        docs = state.get(
            "rag_documents",
            [],
        )

        evidence = {
            "rag_relevant": state.get(
                "rag_relevant"
            ),

            "rewrite_count": state.get(
                "rag_rewrite_count"
            ),

            "retrieval_query": state.get(
                "rag_retrieval_query"
            ),

            "documents": [
                {
                    "source": doc.get(
                        "source"
                    ),

                    "page": doc.get(
                        "page"
                    ),

                    "section": doc.get(
                        "section"
                    ),

                    "text": (
                        doc.get("text")
                        or ""
                    )[:1500],
                }

                for doc in docs
            ],
        }

    # --------------------------------------------------------
    # Direct evidence
    # --------------------------------------------------------

    elif route == "direct":

        tool_catalog = (
            await get_tool_catalog()
        )

        evidence = {
            "available_tools": [
                {
                    "name": tool["name"],

                    "description": (
                        tool["description"]
                    ),
                }

                for tool in tool_catalog
            ],

            "rag_scope": [
                "AI4I documentation",
                "NASA Reliability-Centered Maintenance",
                "DOE Operations & Maintenance",
                "NIST manufacturing monitoring",
                "OSHA lockout/tagout",
            ],
        }

    prompt = f"""
You are a strict Validator for a manufacturing AI Agent.

Determine whether the FINAL ANSWER is correct and supported
by the execution evidence.


USER QUESTION:
{query}


ROUTE:
{route}


FINAL ANSWER:
{answer}


EXECUTION EVIDENCE:
{json.dumps(
    evidence,
    ensure_ascii=False,
    indent=2,
)}


VALIDATION RULES:

1. TOOL ANSWERS

- Numerical claims must agree with tool_result.
- Do not allow invented values.
- Prediction output must not be presented as measured fact.
- Local sensitivity must not be described as causal analysis or SHAP.


2. RAG ANSWERS

- Factual claims must be supported by retrieved documents.
- AI4I failure-generation rules are synthetic dataset rules.
- They must not be described as universal physical manufacturing thresholds.
- If rag_relevant=false, the answer must explicitly acknowledge
  insufficient evidence.


3. DIRECT ANSWERS

- Do not allow claims about capabilities that are not actually represented
  by the MCP Tools or RAG scope.
- Do not allow fabricated manufacturing data.


4. GENERAL

- The answer must address the user's actual question.


Return ONLY valid JSON:

{{
  "passed": true,
  "feedback": "short explanation"
}}
"""

    response = await client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
    )

    result = parse_json_output(
        response.output_text
    )

    passed = bool(
        result.get(
            "passed",
            False,
        )
    )

    feedback = result.get(
        "feedback",
        "",
    )

    print(
        "validation:",
        "PASS"
        if passed
        else "FAIL",
    )

    print(
        "feedback:",
        feedback,
    )

    return {
        "validation_passed": passed,
        "validation_feedback": feedback,
    }


# ============================================================
# Replan Node
# ============================================================

async def replan(
    state: ManufacturingState,
) -> dict:
    """
    Validator FAIL 시 retry_count를 증가시키고
    Router로 다시 보낸다.

    Router는 validation_feedback을 읽고
    새로운 route/tool 선택을 시도한다.
    """

    print("\n[NODE] replan")

    current_retry = state.get(
        "retry_count",
        0,
    )

    new_retry = current_retry + 1

    print(
        f"retry_count: "
        f"{new_retry}/{MAX_AGENT_RETRIES}"
    )

    print(
        "validator feedback:",
        state.get(
            "validation_feedback",
            "",
        ),
    )

    return {
        "retry_count": new_retry,
    }


# ============================================================
# Safe Fallback Node
# ============================================================

async def safe_fallback(
    state: ManufacturingState,
) -> dict:
    """
    최대 재시도 횟수를 넘겨도 Validator를 통과하지 못하면
    잘못된 답변을 반환하지 않고 안전하게 종료한다.
    """

    print("\n[NODE] safe_fallback")

    feedback = state.get(
        "validation_feedback",
        "",
    )

    answer = (
        "요청에 대해 검증 가능한 답변을 생성하지 못했습니다. "
        "잘못된 정보를 제공하지 않기 위해 답변 생성을 중단합니다."
    )

    if feedback:

        answer += (
            f"\n\n검증 사유: "
            f"{feedback}"
        )

    return {
        "answer": answer,
    }


# ============================================================
# Router Conditional Edge
# ============================================================

def choose_route(
    state: ManufacturingState,
) -> str:

    route = state.get(
        "route",
        "direct",
    )

    if route == "tool":
        return "tool"

    if route == "rag":
        return "rag"

    return "direct"


# ============================================================
# Tool Argument Conditional Edge
# ============================================================

def route_after_argument_check(
    state: ManufacturingState,
) -> str:

    needs_clarification = state.get(
        "needs_clarification",
        False,
    )

    if needs_clarification:

        print(
            "\n[ROUTE] "
            "missing arguments → clarify"
        )

        return "clarify"

    print(
        "\n[ROUTE] "
        "arguments complete → execute tool"
    )

    return "execute"


# ============================================================
# Validator Conditional Edge
# ============================================================

def route_after_validation(
    state: ManufacturingState,
) -> str:

    passed = state.get(
        "validation_passed",
        False,
    )

    retry_count = state.get(
        "retry_count",
        0,
    )

    # Validator 성공
    if passed:

        print(
            "\n[ROUTE] "
            "validation PASS → END"
        )

        return "end"

    # 아직 retry 가능
    if (
        retry_count
        < MAX_AGENT_RETRIES
    ):

        print(
            "\n[ROUTE] "
            "validation FAIL → replan"
        )

        return "replan"

    # 최대 retry 도달
    print(
        "\n[ROUTE] "
        "max retries reached → fallback"
    )

    return "fallback"


# ============================================================
# Build Manufacturing Graph
# ============================================================

def build_manufacturing_graph(
        checkpointer=None,
):

    builder = StateGraph(
        ManufacturingState
    )

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    builder.add_node(
        "begin_turn",
        begin_turn,
    )

    builder.add_node(
        "router",
        route_query,
    )

    builder.add_node(
        "check_tool_arguments",
        check_tool_arguments,
    )

    builder.add_node(
        "clarify",
        clarify_missing_arguments,
    )

    builder.add_node(
        "execute_tool",
        execute_tool,
    )

    builder.add_node(
        "generate_tool_answer",
        generate_tool_answer,
    )

    builder.add_node(
        "rag",
        run_rag,
    )

    builder.add_node(
        "direct",
        direct_answer,
    )

    builder.add_node(
        "validator",
        validate_answer,
    )

    builder.add_node(
        "replan",
        replan,
    )

    builder.add_node(
        "fallback",
        safe_fallback,
    )

    # --------------------------------------------------------
    # START → Router
    # --------------------------------------------------------

    builder.add_edge(
        START,
        "begin_turn",
    )

    builder.add_edge(
        "begin_turn",
        "router",
    )

    # --------------------------------------------------------
    # Router → Tool / RAG / Direct
    # --------------------------------------------------------

    builder.add_conditional_edges(
        "router",
        choose_route,
        {
            "tool": "check_tool_arguments",
            "rag": "rag",
            "direct": "direct",
        },
    )

    builder.add_conditional_edges(
        "check_tool_arguments",
        route_after_argument_check,
        {
            "execute": "execute_tool",
            "clarify": "clarify",
        },
    )

    builder.add_edge(
        "clarify",
        END,
    )

    # --------------------------------------------------------
    # Tool Route
    # --------------------------------------------------------

    builder.add_edge(
        "execute_tool",
        "generate_tool_answer",
    )

    builder.add_edge(
        "generate_tool_answer",
        "validator",
    )

    # --------------------------------------------------------
    # RAG Route
    # --------------------------------------------------------

    builder.add_edge(
        "rag",
        "validator",
    )

    # --------------------------------------------------------
    # Direct Route
    # --------------------------------------------------------

    builder.add_edge(
        "direct",
        "validator",
    )

    # --------------------------------------------------------
    # Validator
    # --------------------------------------------------------

    builder.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "end": END,
            "replan": "replan",
            "fallback": "fallback",
        },
    )

    # --------------------------------------------------------
    # FAIL → Replan → Router
    # --------------------------------------------------------

    builder.add_edge(
        "replan",
        "router",
    )

    # --------------------------------------------------------
    # Max retry → Fallback → END
    # --------------------------------------------------------

    builder.add_edge(
        "fallback",
        END,
    )

    if checkpointer is None:
        checkpointer = MemorySaver()

    return builder.compile(
        checkpointer=checkpointer
    )


# ============================================================
# Compiled Graph
# ============================================================

manufacturing_graph = (
    build_manufacturing_graph()
)


# ============================================================
# Manual Test
# ============================================================

async def main():

    result = await manufacturing_graph.ainvoke(
        {
            "query": (
                "Product Type L의 전체 샘플 수와 "
                "고장 건수, 고장률을 알려줘."
            ),

            # 최초 실행이므로 retry 0
            "retry_count": 0,
        }
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "MANUFACTURING GRAPH FINISHED"
    )

    print(
        "=" * 100
    )

    print(
        "\nRoute:",
        result.get(
            "route"
        ),
    )

    print(
        "Tool:",
        result.get(
            "tool_name"
        ),
    )

    print(
        "Validation:",
        result.get(
            "validation_passed"
        ),
    )

    print(
        "Retry count:",
        result.get(
            "retry_count"
        ),
    )

    print(
        "\nAnswer:\n"
    )

    print(
        result.get(
            "answer"
        )
    )


if __name__ == "__main__":

    asyncio.run(
        main()
    )