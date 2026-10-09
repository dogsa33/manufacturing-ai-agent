import asyncio
import sys
from pathlib import Path

from agents import (
    Agent,
    Runner,
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


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SESSION_DB_PATH = (
    PROJECT_ROOT
    / "database"
    / "agent_sessions.db"
)


# =========================================================
# Trace Helper
# =========================================================

def print_trace(result):

    print("\n--- TOOL TRACE ---")

    tool_count = 0

    for item in result.new_items:

        if isinstance(
            item,
            ToolCallItem,
        ):
            tool_count += 1

            raw_item = item.raw_item

            arguments = getattr(
                raw_item,
                "arguments",
                None,
            )

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

            print(
                f"[CALL {tool_count}] "
                f"{item.tool_name}"
            )

            print(
                "Arguments:",
                arguments,
            )

        elif isinstance(
            item,
            ToolCallOutputItem,
        ):

            print(
                "[OUTPUT]",
                item.output,
            )

    print(
        "Total tool calls:",
        tool_count,
    )


# =========================================================
# Run Turn
# =========================================================

async def run_turn(
    agent,
    session,
    question: str,
):

    print("\n" + "=" * 70)
    print("USER")
    print("=" * 70)

    print(question)

    result = await Runner.run(
        agent,
        question,
        session=session,
    )

    print_trace(
        result
    )

    print("\n" + "=" * 70)
    print("AGENT")
    print("=" * 70)

    print(
        result.final_output
    )

    return result


# =========================================================
# Main
# =========================================================

async def main():

    python_executable = (
        sys.executable
    )

    # -----------------------------------------------------
    # Persistent Session
    # -----------------------------------------------------

    session = SQLiteSession(
        session_id="manufacturing_demo_001",
        db_path=str(
            SESSION_DB_PATH
        ),
    )

    # 테스트를 매번 동일한 상태에서 시작하기 위해
    # 기존 대화 삭제
    await session.clear_session()

    print(
        "Session DB:",
        SESSION_DB_PATH,
    )

    print(
        "Session ID:",
        "manufacturing_demo_001",
    )

    # -----------------------------------------------------
    # MCP
    # -----------------------------------------------------

    async with MCPServerStdio(
        name=(
            "Manufacturing Analysis "
            "MCP Server"
        ),
        params={
            "command": (
                python_executable
            ),
            "args": [
                "-m",
                "mcp_server.server",
            ],
            "cwd": str(
                PROJECT_ROOT
            ),
        },
        cache_tools_list=True,
        use_structured_content=True,
    ) as mcp_server:

        # -------------------------------------------------
        # Agent
        # -------------------------------------------------

        agent = Agent(
            name=(
                "Manufacturing Analysis "
                "Agent"
            ),
            instructions=(
                AGENT_INSTRUCTIONS
            ),
            mcp_servers=[
                mcp_server,
            ],
        )

        # =================================================
        # TURN 1
        # =================================================

        await run_turn(
            agent=agent,
            session=session,
            question=(
                "Product Type은 L, "
                "Air temperature는 301.0, "
                "Process temperature는 310.5, "
                "Rotational speed는 1300, "
                "Torque는 65.0, "
                "Tool wear는 200이야. "
                "고장 위험을 예측해줘."
            ),
        )

        # =================================================
        # TURN 2
        # 이전 제조조건을 다시 적지 않음
        # =================================================

        await run_turn(
            agent=agent,
            session=session,
            question=(
                "그럼 왜 위험하게 "
                "판단된 거야?"
            ),
        )

        # =================================================
        # TURN 3
        # 대화 Context를 활용한 후속 질문
        # =================================================

        await run_turn(
            agent=agent,
            session=session,
            question=(
                "가장 영향이 큰 변수는 "
                "정상 기준과 얼마나 차이나?"
            ),
        )

        # =================================================
        # Stored Session 확인
        # =================================================

        items = await session.get_items()

        print("\n" + "=" * 70)
        print("SESSION STATUS")
        print("=" * 70)

        print(
            "Stored items:",
            len(items),
        )

        print(
            "Database:",
            SESSION_DB_PATH,
        )


if __name__ == "__main__":
    asyncio.run(main())