import asyncio
import json
import sys
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# Project Path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# Persistent Streamlit Thread
# ============================================================

DATABASE_DIR = (
    PROJECT_ROOT
    / "database"
)

DATABASE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

STREAMLIT_STATE_PATH = (
    DATABASE_DIR
    / "streamlit_session.json"
)


def save_thread_id(
    thread_id: str,
) -> None:

    with open(
        STREAMLIT_STATE_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "thread_id": thread_id,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


def load_or_create_thread_id() -> str:

    if STREAMLIT_STATE_PATH.exists():

        try:

            with open(
                STREAMLIT_STATE_PATH,
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(
                    file
                )

            thread_id = data.get(
                "thread_id"
            )

            if (
                isinstance(
                    thread_id,
                    str,
                )
                and thread_id.strip()
            ):

                return thread_id

        except (
            OSError,
            json.JSONDecodeError,
        ):

            pass

    thread_id = (
        "streamlit_"
        + str(
            uuid.uuid4()
        )
    )

    save_thread_id(
        thread_id
    )

    return thread_id


# ============================================================
# LangGraph Manufacturing Agent
# ============================================================

from graph.persistent_graph import (
    invoke_persistent_graph,
)


# ============================================================
# Streamlit Page Configuration
# ============================================================

st.set_page_config(
    page_title="Manufacturing AI Agent",
    page_icon="🏭",
    layout="wide",
)


# ============================================================
# Streamlit Session State
# ============================================================

# LangGraph MemorySaver에서 사용할 thread_id
if "thread_id" not in st.session_state:

    st.session_state.thread_id = (
        load_or_create_thread_id()
    )


# Streamlit 화면에 표시할 Chat History
if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# Persistent Async Event Loop
# ============================================================

if (
    "event_loop" not in st.session_state
    or st.session_state.event_loop.is_closed()
):

    st.session_state.event_loop = (
        asyncio.new_event_loop()
    )


def run_async(coro):
    """
    하나의 Streamlit Session 동안
    동일한 asyncio Event Loop를 재사용한다.
    """

    loop = st.session_state.event_loop

    asyncio.set_event_loop(
        loop
    )

    return loop.run_until_complete(
        coro
    )


# ============================================================
# Chart Path Extraction
# ============================================================

def extract_chart_paths(
    tool_result,
):
    """
    MCP Visualization Tool의 structured_content에서
    생성된 Plotly HTML 경로를 추출한다.
    """

    chart_paths = []

    if not isinstance(
        tool_result,
        dict,
    ):
        return chart_paths

    file_path = tool_result.get(
        "file_path"
    )

    if not file_path:
        return chart_paths

    path = Path(
        file_path
    )

    if (
        path.exists()
        and path.suffix.lower() == ".html"
    ):

        chart_paths.append(
            str(path)
        )

    return chart_paths


# ============================================================
# LangGraph State → Streamlit Result
# ============================================================

def convert_graph_result(
    state: dict,
) -> dict:
    """
    LangGraph의 최종 State를
    Streamlit UI에서 사용하기 쉬운 형태로 변환한다.
    """

    route = state.get(
        "route",
        "",
    )

    answer = state.get(
        "answer",
        "",
    )

    tool_name = state.get(
        "tool_name",
        "",
    )

    tool_arguments = state.get(
        "tool_arguments",
        {},
    )

    tool_result = state.get(
        "tool_result"
    )

    needs_clarification = state.get(
        "needs_clarification",
        False,
    )

    # ========================================================
    # 실제 MCP Tool 실행 Trace
    # ========================================================

    tool_calls = []

    # clarification 단계에서는 Tool을 선택했지만
    # 실제 MCP Tool 실행은 하지 않았으므로 Trace에 넣지 않는다.
    if (
        route == "tool"
        and tool_name
        and not needs_clarification
        and tool_result is not None
        and tool_result != {}
    ):

        tool_calls.append(
            {
                "tool_name": tool_name,
                "arguments": tool_arguments,
            }
        )

    # ========================================================
    # Chart
    # ========================================================

    chart_paths = extract_chart_paths(
        tool_result
    )

    # ========================================================
    # Validator
    # ========================================================

    validation = None
    attempts = 0

    # clarification 응답은 아직 최종 분석 결과가 아니므로
    # Validator가 실행되지 않는다.
    if not needs_clarification:

        retry_count = state.get(
            "retry_count",
            0,
        )

        attempts = (
            retry_count + 1
        )

        validation = {
            "passed": state.get(
                "validation_passed",
                False,
            ),

            "feedback": state.get(
                "validation_feedback",
                "",
            ),

            "route": route,

            "retry_count": retry_count,

            "rag_rewrite_count": state.get(
                "rag_rewrite_count",
                0,
            ),
        }

    return {
        "answer": answer,

        "route": route,

        "tool_calls": tool_calls,

        "tool_result": tool_result,

        "chart_paths": chart_paths,

        "validation": validation,

        "attempts": attempts,

        "needs_clarification":
            needs_clarification,

        "missing_arguments": state.get(
            "missing_arguments",
            [],
        ),

        "pending_tool_name": state.get(
            "pending_tool_name",
            "",
        ),
    }


# ============================================================
# LangGraph Runner
# ============================================================

async def run_agent(
    question: str,
):

    result = await invoke_persistent_graph(
        query=question,
        thread_id=(
            st.session_state.thread_id
        ),
    )

    return convert_graph_result(
        result
    )


# ============================================================
# Reset Conversation
# ============================================================

def reset_conversation():
    """
    UI History를 제거하고 새로운 LangGraph thread_id를 생성한다.

    새로운 thread_id를 사용하므로
    이전 대화의 pending state를 더 이상 참조하지 않는다.
    """

    st.session_state.messages = []

    st.session_state.thread_id = (
        "streamlit_"
        + str(uuid.uuid4())
    )

    save_thread_id(
        st.session_state.thread_id
    )


# ============================================================
# Plotly HTML Renderer
# ============================================================

def render_chart(
    chart_path: str,
):
    """
    MCP Visualization Tool이 생성한 Plotly HTML을
    Streamlit 채팅창 안에 표시한다.
    """

    path = Path(
        chart_path
    )

    if not path.exists():

        st.warning(
            "그래프 파일을 찾을 수 없습니다.\n\n"
            f"`{chart_path}`"
        )

        return

    try:

        html_content = path.read_text(
            encoding="utf-8"
        )

        components.html(
            html_content,
            height=600,
            scrolling=True,
        )

    except Exception as error:

        st.warning(
            "그래프 표시 중 오류가 발생했습니다.\n\n"
            f"`{type(error).__name__}: {error}`"
        )


# ============================================================
# MCP Tool Trace Renderer
# ============================================================

def render_tool_trace(
    tool_calls,
):
    """
    LangGraph 실행 과정에서
    실제 호출된 MCP Tool을 표시한다.
    """

    if not tool_calls:
        return

    with st.expander(
        f"사용한 MCP Tool "
        f"({len(tool_calls)}개)"
    ):

        for index, tool in enumerate(
            tool_calls,
            start=1,
        ):

            st.markdown(
                f"### {index}. "
                f"`{tool['tool_name']}`"
            )

            arguments = tool.get(
                "arguments",
                {},
            )

            if not arguments:
                continue

            if isinstance(
                arguments,
                dict,
            ):

                st.json(
                    arguments
                )

            elif isinstance(
                arguments,
                str,
            ):

                try:

                    parsed_arguments = (
                        json.loads(
                            arguments
                        )
                    )

                    st.json(
                        parsed_arguments
                    )

                except json.JSONDecodeError:

                    st.code(
                        arguments,
                        language="text",
                    )

            else:

                st.code(
                    str(arguments),
                    language="text",
                )


# ============================================================
# Validator Renderer
# ============================================================

def render_validation(
    validation,
    attempts,
):
    """
    LangGraph Main Validator 결과를 표시한다.
    """

    if not validation:
        return

    passed = validation.get(
        "passed",
        False,
    )

    if passed:

        status_text = "PASS ✅"

    else:

        status_text = "FAIL ❌"

    with st.expander(
        f"Validation: {status_text}"
    ):

        route = validation.get(
            "route",
            ""
        )

        retry_count = validation.get(
            "retry_count",
            0,
        )

        rag_rewrite_count = (
            validation.get(
                "rag_rewrite_count",
                0,
            )
        )

        col1, col2, col3 = (
            st.columns(3)
        )

        with col1:

            st.metric(
                "Route",
                route.upper()
                if route
                else "-",
            )

        with col2:

            st.metric(
                "Attempts",
                attempts,
            )

        with col3:

            st.metric(
                "Agent Retries",
                retry_count,
            )

        if route == "rag":

            st.metric(
                "RAG Query Rewrites",
                rag_rewrite_count,
            )

        feedback = validation.get(
            "feedback",
            "",
        )

        if feedback:

            st.divider()

            st.markdown(
                "**Validator Feedback**"
            )

            st.write(
                feedback
            )


# ============================================================
# Clarification Renderer
# ============================================================

def render_clarification_status(
    message,
):
    """
    Tool 입력값이 부족해서
    다음 사용자 응답을 기다리고 있는 상태를 표시한다.
    """

    if not message.get(
        "needs_clarification",
        False,
    ):
        return

    pending_tool = message.get(
        "pending_tool_name",
        "",
    )

    missing = message.get(
        "missing_arguments",
        [],
    )

    with st.expander(
        "추가 입력 대기"
    ):

        if pending_tool:

            st.markdown(
                "**대기 중인 Tool**"
            )

            st.code(
                pending_tool,
                language="text",
            )

        if missing:

            st.markdown(
                "**필요한 추가 입력값**"
            )

            for item in missing:

                st.markdown(
                    f"- `{item}`"
                )


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.title(
        "Manufacturing AI Agent"
    )

    st.caption(
        "LangGraph · MCP · RAG 기반 "
        "제조설비 분석 Agent"
    )

    st.divider()

    # --------------------------------------------------------
    # Capabilities
    # --------------------------------------------------------

    st.subheader(
        "지원 기능"
    )

    st.markdown(
        """
- 제조 데이터 요약
- 정상 / 고장 조건 비교
- 고장 유형 분석
- Machine Failure 예측
- Local Sensitivity Explanation
- Plotly 시각화
- MCP Tool Routing
- 문서 기반 RAG
- BGE-M3 + Qdrant + Reranker
- Validator / Re-planning
- Missing Argument Clarification
- LangGraph Thread Memory
"""
    )

    st.divider()

    # --------------------------------------------------------
    # Architecture
    # --------------------------------------------------------

    st.subheader(
        "V2 Architecture"
    )

    st.code(
        """
User
 ↓
Streamlit
 ↓
LangGraph StateGraph
 ↓
Router
 ├─ MCP Tool
 │   ↓
 │ DuckDB / RF / Plotly
 │
 ├─ Corrective RAG
 │   ↓
 │ BGE-M3
 │   ↓
 │ Qdrant
 │   ↓
 │ Reranker
 │
 └─ Direct
 ↓
Validator
 ├─ PASS → END
 └─ FAIL → Re-plan
              ↓
            Router

Thread State
 ↓
MemorySaver
        """,
        language="text",
    )

    st.divider()

    # --------------------------------------------------------
    # Thread
    # --------------------------------------------------------

    st.caption(
        "Current LangGraph Thread ID"
    )

    st.code(
        st.session_state.thread_id,
        language="text",
    )

    st.caption(
        "현재 MemorySaver는 서버 프로세스가 "
        "실행되는 동안 유지되는 short-term memory입니다."
    )

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    if st.button(
        "대화 초기화",
        use_container_width=True,
    ):

        reset_conversation()

        st.rerun()


# ============================================================
# Main Header
# ============================================================

st.title(
    "Manufacturing AI Agent"
)

st.caption(
    "제조 데이터 분석 · "
    "설비 고장 예측 · "
    "정비 문서 RAG"
)


# ============================================================
# Example Questions
# ============================================================

with st.expander(
    "예시 질문"
):

    st.markdown(
        """
### 제조 통계

`Product Type L의 전체 샘플 수와 고장 건수, 고장률을 알려줘.`

### 정상 / 고장 조건 비교

`Torque가 정상 제품과 고장 제품에서 어떻게 다른지 비교해줘.`

### 고장 예측

`Product Type은 L, Air temperature는 301.0, Process temperature는 310.5, Rotational speed는 1300, Torque는 65.0, Tool wear는 200이야. 고장 위험을 예측해줘.`

### 입력값을 나눠서 제공

첫 번째 질문:

`Product Type L의 고장 위험을 예측해줘.`

Agent가 부족한 입력값을 요청하면 다음 메시지:

`Air temperature 301.0 K, Process temperature 310.5 K, Rotational speed 1300 rpm, Torque 65.0 Nm, Tool wear 200 min이야.`

### 문서 기반 RAG

`AI4I 데이터셋에서 열 방산 고장은 어떤 조건에서 발생하는가?`

`Condition-based maintenance란 무엇인가?`

### 시각화

`Torque의 정상 제품과 고장 제품 분포를 그래프로 보여줘.`

### Product Type별 시각화

`Product Type별 고장률을 그래프로 보여줘.`
"""
    )


# ============================================================
# Existing Chat History
# ============================================================

for message in (
    st.session_state.messages
):

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"]
            == "assistant"
        ):

            # ----------------------------------------------
            # Clarification
            # ----------------------------------------------

            render_clarification_status(
                message
            )

            # ----------------------------------------------
            # MCP Tool Trace
            # ----------------------------------------------

            render_tool_trace(
                message.get(
                    "tool_calls",
                    [],
                )
            )

            # ----------------------------------------------
            # Plotly Charts
            # ----------------------------------------------

            for chart_path in (
                message.get(
                    "chart_paths",
                    [],
                )
            ):

                render_chart(
                    chart_path
                )

            # ----------------------------------------------
            # Validator
            # ----------------------------------------------

            render_validation(
                message.get(
                    "validation"
                ),

                message.get(
                    "attempts",
                    0,
                ),
            )


# ============================================================
# Chat Input
# ============================================================

user_input = st.chat_input(
    "제조 데이터 또는 정비 문서에 대해 질문하세요."
)


# ============================================================
# New Conversation Turn
# ============================================================

if user_input:

    # ========================================================
    # User Message
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_input
        )


    # ========================================================
    # Assistant Message
    # ========================================================

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "LangGraph가 요청을 분석하고 있습니다..."
        ):

            try:

                # ------------------------------------------
                # LangGraph
                # ------------------------------------------

                agent_result = run_async(
                    run_agent(
                        user_input
                    )
                )

                response = (
                    agent_result[
                        "answer"
                    ]
                )

                # ------------------------------------------
                # Final Answer
                # ------------------------------------------

                st.markdown(
                    response
                )

                # ------------------------------------------
                # Clarification State
                # ------------------------------------------

                render_clarification_status(
                    agent_result
                )

                # ------------------------------------------
                # MCP Tool Trace
                # ------------------------------------------

                render_tool_trace(
                    agent_result[
                        "tool_calls"
                    ]
                )

                # ------------------------------------------
                # Plotly Charts
                # ------------------------------------------

                for chart_path in (
                    agent_result[
                        "chart_paths"
                    ]
                ):

                    render_chart(
                        chart_path
                    )

                # ------------------------------------------
                # Validation
                # ------------------------------------------

                render_validation(
                    agent_result[
                        "validation"
                    ],

                    agent_result[
                        "attempts"
                    ],
                )


            # =================================================
            # Error Handling
            # =================================================

            except Exception as error:

                response = (
                    "Agent 실행 중 오류가 "
                    "발생했습니다.\n\n"
                    f"`{type(error).__name__}: "
                    f"{error}`"
                )

                agent_result = {
                    "tool_calls": [],
                    "chart_paths": [],
                    "validation": None,
                    "attempts": 0,
                    "needs_clarification": False,
                    "missing_arguments": [],
                    "pending_tool_name": "",
                }

                st.error(
                    response
                )


    # ========================================================
    # Save Assistant UI History
    # ========================================================

    st.session_state.messages.append(
        {
            "role":
                "assistant",

            "content":
                response,

            "tool_calls":
                agent_result[
                    "tool_calls"
                ],

            "chart_paths":
                agent_result[
                    "chart_paths"
                ],

            "validation":
                agent_result[
                    "validation"
                ],

            "attempts":
                agent_result[
                    "attempts"
                ],

            "needs_clarification":
                agent_result.get(
                    "needs_clarification",
                    False,
                ),

            "missing_arguments":
                agent_result.get(
                    "missing_arguments",
                    [],
                ),

            "pending_tool_name":
                agent_result.get(
                    "pending_tool_name",
                    "",
                ),
        }
    )