from pathlib import Path

from langgraph.checkpoint.sqlite.aio import (
    AsyncSqliteSaver,
)

from graph.manufacturing_graph import (
    build_manufacturing_graph,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DATABASE_DIR = (
    PROJECT_ROOT
    / "database"
)

DATABASE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_DB_PATH = (
    DATABASE_DIR
    / "langgraph_checkpoints.sqlite"
)


# ============================================================
# Persistent Graph Invocation
# ============================================================

async def invoke_persistent_graph(
    query: str,
    thread_id: str,
) -> dict:
    """
    SQLite-backed LangGraph 실행.

    같은 thread_id를 사용하면
    Python 프로세스가 종료되었다가 다시 실행되어도
    이전 checkpoint를 불러올 수 있다.
    """

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    async with (
        AsyncSqliteSaver
        .from_conn_string(
            str(
                CHECKPOINT_DB_PATH
            )
        )
    ) as checkpointer:

        graph = (
            build_manufacturing_graph(
                checkpointer=checkpointer,
            )
        )

        result = await graph.ainvoke(
            {
                "query": query,
            },
            config=config,
        )

    return result