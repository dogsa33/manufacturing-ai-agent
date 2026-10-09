import asyncio
import json
import sys
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from agents import (
    Agent,
    SQLiteSession,
)

from agents.items import (
    ToolCallItem,
    ToolCallOutputItem,
)

from agents.mcp import (
    MCPServerStdio,
)

from agent.manufacturing_agent import (
    AGENT_INSTRUCTIONS,
)

from agent.validated_agent import (
    VALIDATOR_INSTRUCTIONS,
    ValidationResult,
)

from agent.ui_harness import (
    run_validated_turn,
)


# =========================================================
# Project Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SESSION_DB_PATH = (
    PROJECT_ROOT
    / "database"
    / "agent_sessions.db"
)


# =========================================================
# Streamlit Page Configuration
# =========================================================

st.set_page_config(
    page_title="Manufacturing AI Agent",
    page_icon="🏭",
    layout="wide",
)


# =========================================================
# Streamlit Session State
# =========================================================

if "session_id" not in st.session_state:

    st.session_state.session_id = (
        "streamlit_"
        + str(uuid.uuid4())
    )


if "messages" not in st.session_state:

    st.session_state.messages = []


# =========================================================
# Persistent Async Event Loop
# =========================================================

if (
    "event_loop" not in st.session_state
    or st.session_state.event_loop.is_closed()
):

    st.session_state.event_loop = (
        asyncio.new_event_loop()
    )


def run_async(coro):
    """
    Streamlit Session 동안 하나의 asyncio Event Loop를
    계속 재사용한다.

    asyncio.run()을 반복 호출하면 Event Loop가 닫히면서
    'Event loop is closed' 오류가 발생할 수 있으므로
    persistent loop를 사용한다.
    """

    loop = st.session_state.event_loop

    asyncio.set_event_loop(
        loop
    )

    return loop.run_until_complete(
        coro
    )


# =========================================================
# Tool Trace Extraction
# =========================================================

def extract_tool_trace(result):
    """
    Agent 실행 결과에서 다음을 추출한다.

    - MCP Tool Call
    - Tool Output
    - 생성된 HTML Chart Path
    """

    tool_calls = []
    tool_outputs = []
    chart_paths = []

    for item in result.new_items:

        # -------------------------------------------------
        # MCP Tool Call
        # -------------------------------------------------

        if isinstance(
            item,
            ToolCallItem,
        ):

            raw_item = item.raw_item

            arguments = getattr(
                raw_item,
                "arguments",
                None,
            )

            # SDK 객체가 아니라 dict인 경우 대응
            if (
                arguments is None
                and isinstance(
                    raw_item,
                    dict,
                )
            ):

                arguments = raw_item.get(
                    "arguments"
                )

            tool_calls.append(
                {
                    "tool_name":
                        item.tool_name,

                    "arguments":
                        arguments,
                }
            )

        # -------------------------------------------------
        # MCP Tool Output
        # -------------------------------------------------

        elif isinstance(
            item,
            ToolCallOutputItem,
        ):

            output = item.output

            tool_outputs.append(
                str(output)
            )

            parsed_output = None

            # dict 형태라면 그대로 사용
            if isinstance(
                output,
                dict,
            ):

                parsed_output = output

            # 문자열 JSON이면 parsing
            elif isinstance(
                output,
                str,
            ):

                try:

                    parsed_output = (
                        json.loads(output)
                    )

                except json.JSONDecodeError:

                    parsed_output = None

            # -------------------------------------------------
            # Visualization Tool 결과 확인
            # -------------------------------------------------

            if (
                isinstance(
                    parsed_output,
                    dict,
                )
                and "file_path"
                in parsed_output
            ):

                file_path = Path(
                    parsed_output[
                        "file_path"
                    ]
                )

                if (
                    file_path.exists()
                    and file_path.suffix.lower()
                    == ".html"
                ):

                    chart_paths.append(
                        str(file_path)
                    )

    return {
        "tool_calls":
            tool_calls,

        "tool_outputs":
            tool_outputs,

        "chart_paths":
            chart_paths,
    }


# =========================================================
# Agent Runner
# =========================================================

async def run_agent(
    question: str,
) -> dict:
    """
    Main Agent + MCP + SQLite Session + Validator Harness를
    실행하고 Streamlit에서 사용할 결과를 반환한다.
    """

    python_executable = (
        sys.executable
    )

    # -----------------------------------------------------
    # Persistent Conversation Memory
    # -----------------------------------------------------

    session = SQLiteSession(
        session_id=(
            st.session_state.session_id
        ),
        db_path=str(
            SESSION_DB_PATH
        ),
    )

    # -----------------------------------------------------
    # MCP Server
    # -----------------------------------------------------

    async with MCPServerStdio(
        name=(
            "Manufacturing Analysis "
            "MCP Server"
        ),

        params={
            "command":
                python_executable,

            "args": [
                "-m",
                "mcp_server.server",
            ],

            "cwd":
                str(PROJECT_ROOT),
        },

        cache_tools_list=True,
        use_structured_content=True,

    ) as mcp_server:

        # =================================================
        # Main Manufacturing Agent
        # =================================================

        main_agent = Agent(
            name=(
                "Manufacturing "
                "Analysis Agent"
            ),

            instructions=(
                AGENT_INSTRUCTIONS
            ),

            mcp_servers=[
                mcp_server,
            ],
        )

        # =================================================
        # Validator Agent
        # =================================================

        validator_agent = Agent(
            name=(
                "Manufacturing "
                "Agent Validator"
            ),

            instructions=(
                VALIDATOR_INSTRUCTIONS
            ),

            output_type=(
                ValidationResult
            ),
        )

        # =================================================
        # Validator / Re-planning Harness
        # =================================================

        harness_result = (
            await run_validated_turn(
                main_agent=main_agent,
                validator_agent=validator_agent,
                session=session,
                user_request=question,
                max_attempts=2,
            )
        )

        result = (
            harness_result[
                "result"
            ]
        )

        validation = (
            harness_result[
                "validation"
            ]
        )

        attempts = (
            harness_result[
                "attempts"
            ]
        )

        # =================================================
        # Tool Trace
        # =================================================

        trace = extract_tool_trace(
            result
        )

        # =================================================
        # Return to Streamlit
        # =================================================

        return {
            "answer":
                str(
                    result.final_output
                ),

            "tool_calls":
                trace[
                    "tool_calls"
                ],

            "tool_outputs":
                trace[
                    "tool_outputs"
                ],

            "chart_paths":
                trace[
                    "chart_paths"
                ],

            "validation": {
                "passed":
                    validation.passed,

                "grounded":
                    validation.grounded,

                "complete":
                    validation.complete,

                "safe_interpretation":
                    validation.safe_interpretation,

                "missing_requirements":
                    validation.missing_requirements,

                "feedback":
                    validation.feedback,
            },

            "attempts":
                attempts,
        }


# =========================================================
# Agent Session Clear
# =========================================================

async def clear_agent_session():
    """
    현재 SQLiteSession의 Agent 대화 Context를 삭제한다.
    """

    session = SQLiteSession(
        session_id=(
            st.session_state.session_id
        ),

        db_path=str(
            SESSION_DB_PATH
        ),
    )

    await session.clear_session()


# =========================================================
# Reset Conversation
# =========================================================

def reset_conversation():
    """
    UI History와 Agent SQLite Session을 모두 초기화한다.
    """

    # 기존 Agent Conversation 삭제
    run_async(
        clear_agent_session()
    )

    # Streamlit UI History 삭제
    st.session_state.messages = []

    # 새로운 Session ID 생성
    st.session_state.session_id = (
        "streamlit_"
        + str(uuid.uuid4())
    )


# =========================================================
# Plotly HTML Renderer
# =========================================================

def render_chart(
    chart_path: str,
):
    """
    MCP Visualization Tool이 생성한 Plotly HTML 파일을
    Streamlit 채팅창 안에 직접 표시한다.
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


# =========================================================
# MCP Tool Trace Renderer
# =========================================================

def render_tool_trace(
    tool_calls,
):
    """
    Agent가 실제 호출한 MCP Tool을 UI에 표시한다.
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

            arguments = (
                tool["arguments"]
            )

            if arguments:

                try:

                    parsed_arguments = (
                        json.loads(
                            arguments
                        )
                    )

                    st.json(
                        parsed_arguments
                    )

                except (
                    json.JSONDecodeError,
                    TypeError,
                ):

                    st.code(
                        str(arguments),
                        language="text",
                    )


# =========================================================
# Validator Renderer
# =========================================================

def render_validation(
    validation,
    attempts,
):
    """
    Validator 결과와 Re-planning 횟수를 표시한다.
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

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Attempts",
                attempts,
            )

            st.metric(
                "Grounded",
                str(
                    validation.get(
                        "grounded",
                        False,
                    )
                ),
            )

        with col2:

            st.metric(
                "Complete",
                str(
                    validation.get(
                        "complete",
                        False,
                    )
                ),
            )

            st.metric(
                "Safe Interpretation",
                str(
                    validation.get(
                        "safe_interpretation",
                        False,
                    )
                ),
            )

        # -------------------------------------------------
        # Missing Requirements
        # -------------------------------------------------

        missing = validation.get(
            "missing_requirements",
            [],
        )

        if missing:

            st.divider()

            st.markdown(
                "**Missing Requirements**"
            )

            for item in missing:

                st.markdown(
                    f"- {item}"
                )

        # -------------------------------------------------
        # Validator Feedback
        # -------------------------------------------------

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


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.title(
        "Manufacturing AI Agent"
    )

    st.caption(
        "MCP 기반 제조설비 "
        "예지보전 · 데이터 분석 Agent"
    )

    st.divider()

    # -----------------------------------------------------
    # Capabilities
    # -----------------------------------------------------

    st.subheader(
        "지원 기능"
    )

    st.markdown(
        """
- 제조 데이터 요약
- 정상 / 고장 조건 비교
- 고장 유형 분석
- Machine Failure 예측
- Local Risk Explanation
- Plotly 시각화
- MCP Tool Trace
- Validator / Re-planning
- Persistent Session Memory
"""
    )

    st.divider()

    # -----------------------------------------------------
    # Architecture
    # -----------------------------------------------------

    st.subheader(
        "Agent Architecture"
    )

    st.code(
        """
User
 ↓
Streamlit
 ↓
Session Memory
 ↓
Main Agent
 ↓
MCP Server
 ↓
DuckDB / ML / Plotly
 ↓
Tool Observation
 ↓
Agent Answer
 ↓
Validator
 ├─ PASS
 └─ FAIL → Re-plan
        """,
        language="text",
    )

    st.divider()

    # -----------------------------------------------------
    # Session
    # -----------------------------------------------------

    st.caption(
        "Current Session ID"
    )

    st.code(
        st.session_state.session_id,
        language="text",
    )

    # -----------------------------------------------------
    # Reset
    # -----------------------------------------------------

    if st.button(
        "대화 초기화",
        use_container_width=True,
    ):

        reset_conversation()

        st.rerun()


# =========================================================
# Main Header
# =========================================================

st.title(
    "Manufacturing AI Agent"
)

st.caption(
    "제조 데이터 분석 · "
    "설비 고장 예측 · "
    "위험요인 분석"
)


# =========================================================
# Example Questions
# =========================================================

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

### 고장 위험 원인 분석

`그럼 왜 위험하게 판단된 거야?`

### 시각화

`Torque의 정상 제품과 고장 제품 분포를 그래프로 보여줘.`

### Product Type별 시각화

`Product Type별 고장률을 그래프로 보여줘.`
"""
    )


# =========================================================
# Existing Chat History
# =========================================================

for message in (
    st.session_state.messages
):

    with st.chat_message(
        message["role"]
    ):

        # -------------------------------------------------
        # Message Text
        # -------------------------------------------------

        st.markdown(
            message["content"]
        )

        # -------------------------------------------------
        # Assistant Metadata
        # -------------------------------------------------

        if (
            message["role"]
            == "assistant"
        ):

            # Tool Trace
            render_tool_trace(
                message.get(
                    "tool_calls",
                    [],
                )
            )

            # Charts
            for chart_path in (
                message.get(
                    "chart_paths",
                    [],
                )
            ):

                render_chart(
                    chart_path
                )

            # Validation
            render_validation(
                message.get(
                    "validation"
                ),

                message.get(
                    "attempts",
                    1,
                ),
            )


# =========================================================
# Chat Input
# =========================================================

user_input = st.chat_input(
    "제조 데이터에 대해 질문하세요."
)


# =========================================================
# New Conversation Turn
# =========================================================

if user_input:

    # =====================================================
    # User Message
    # =====================================================

    st.session_state.messages.append(
        {
            "role":
                "user",

            "content":
                user_input,
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_input
        )


    # =====================================================
    # Assistant Message
    # =====================================================

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "제조 데이터를 분석하고 "
            "결과를 검증하고 있습니다..."
        ):

            try:

                # -----------------------------------------
                # Main Agent + Validator Harness
                # -----------------------------------------

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

                # -----------------------------------------
                # Final Answer
                # -----------------------------------------

                st.markdown(
                    response
                )

                # -----------------------------------------
                # MCP Tool Trace
                # -----------------------------------------

                render_tool_trace(
                    agent_result[
                        "tool_calls"
                    ]
                )

                # -----------------------------------------
                # Plotly Charts
                # -----------------------------------------

                for chart_path in (
                    agent_result[
                        "chart_paths"
                    ]
                ):

                    render_chart(
                        chart_path
                    )

                # -----------------------------------------
                # Validation Result
                # -----------------------------------------

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
                    "tool_outputs": [],
                    "chart_paths": [],
                    "validation": None,
                    "attempts": 0,
                }

                st.error(
                    response
                )


    # =====================================================
    # Save Assistant UI History
    # =====================================================

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
        }
    )