"""Unit tests for Node 4 (Executor & Logger) following TDD and .ai_rules standards."""

from unittest.mock import MagicMock, patch
import duckdb
import pandas as pd
import pytest

from executor_node import execution_node
from state import AgentState


REQUIRED_STATE_KEYS = {
    "asset_symbol",
    "current_price",
    "historical_data",
    "sentiment_score",
    "risk_cleared",
    "order_details",
    "error_message",
}


def _assert_valid_agent_state(state: AgentState) -> None:
    """Validate that ``state`` conforms to the ``AgentState`` schema and types."""
    assert set(state.keys()) == REQUIRED_STATE_KEYS
    assert isinstance(state["asset_symbol"], str)
    assert isinstance(state["current_price"], float)
    assert isinstance(state["historical_data"], pd.DataFrame)
    assert isinstance(state["sentiment_score"], float)
    assert isinstance(state["risk_cleared"], bool)
    assert isinstance(state["order_details"], dict)
    assert state["error_message"] is None or isinstance(state["error_message"], str)


class _PersistentInMemoryConn:
    """Proxy around an in-memory DuckDB connection that ignores ``close()`` so tests can inspect rows."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn

    def execute(self, *args, **kwargs):
        return self._conn.execute(*args, **kwargs)

    def close(self) -> None:
        pass


def _build_ready_state(risk_cleared: bool) -> AgentState:
    """Build a sample ``AgentState`` ready for Node 4 testing."""
    return {
        "asset_symbol": "SOL/USD",
        "current_price": 100.0,
        "historical_data": pd.DataFrame({"close": [99.0, 100.0]}),
        "sentiment_score": 0.85,
        "risk_cleared": risk_cleared,
        "order_details": {
            "symbol": "SOL/USD",
            "side": "buy",
            "order_type": "bracket",
            "entry_price": 100.0,
            "stop_loss_price": 98.5,
            "target_price": 103.0,
            "position_size": 0.1,
            "dry_run": True,
        },
        "error_message": None,
    }


def test_execution_node_executes_and_logs_to_duckdb() -> None:
    """Test A (Execution): ``risk_cleared=True`` calls mock MCP and inserts into in-memory DuckDB."""
    state = _build_ready_state(risk_cleared=True)
    _assert_valid_agent_state(state)

    in_mem_conn = duckdb.connect(":memory:")
    conn_proxy = _PersistentInMemoryConn(in_mem_conn)

    try:
        with (
            patch(
                "executor_node.execute_mcp_bracket_order",
                return_value="mock-trade-001",
            ) as mock_mcp,
            patch("executor_node.duckdb.connect", return_value=conn_proxy) as mock_db_connect,
        ):
            updated_state = execution_node(state)

        mock_mcp.assert_called_once_with(state["order_details"])
        mock_db_connect.assert_called_once()
        _assert_valid_agent_state(updated_state)
        assert updated_state["order_details"]["trade_id"] == "mock-trade-001"
        assert updated_state["error_message"] is None

        rows = in_mem_conn.execute(
            "SELECT trade_id, symbol, size, stop_loss, target FROM trades"
        ).fetchall()
        assert len(rows) == 1
        trade_id, symbol, size, stop_loss, target = rows[0]
        assert trade_id == "mock-trade-001"
        assert symbol == "SOL/USD"
        assert size == pytest.approx(0.1)
        assert stop_loss == pytest.approx(98.5)
        assert target == pytest.approx(103.0)
    finally:
        in_mem_conn.close()


def test_execution_node_safety_guard_blocks_when_risk_not_cleared() -> None:
    """Test B (Safety Guard): ``risk_cleared=False`` never calls mock MCP or DuckDB."""
    state = _build_ready_state(risk_cleared=False)
    _assert_valid_agent_state(state)

    with (
        patch("executor_node.execute_mcp_bracket_order") as mock_mcp,
        patch("executor_node.duckdb.connect") as mock_db_connect,
    ):
        updated_state = execution_node(state)

    mock_mcp.assert_not_called()
    mock_db_connect.assert_not_called()
    _assert_valid_agent_state(updated_state)
    assert updated_state == state
    assert "trade_id" not in updated_state["order_details"]
