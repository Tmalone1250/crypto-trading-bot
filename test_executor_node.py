"""Unit tests for Node 4 (Executor & Logger) with crypto-compliant MCP order routing."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import duckdb
import pandas as pd
import pytest

from executor_node import execution_node, place_crypto_order
from state import AgentState


REQUIRED_STATE_KEYS = {
    "asset_symbol",
    "current_price",
    "historical_data",
    "sentiment_score",
    "risk_cleared",
    "order_details",
    "error_message",
    "account_data",
    "open_positions",
    "active_orders",
    "decision",
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
    assert isinstance(state["account_data"], dict)
    assert isinstance(state["open_positions"], list)
    assert isinstance(state["active_orders"], list)
    assert isinstance(state["decision"], str)


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
        "asset_symbol": "BTC/USD",
        "current_price": 95000.0,
        "historical_data": pd.DataFrame({"close": [94800.0, 95000.0]}),
        "sentiment_score": 0.85,
        "risk_cleared": risk_cleared,
        "order_details": {
            "symbol": "BTC/USD",
            "side": "buy",
            "order_type": "limit",
            "time_in_force": "gtc",
            "limit_price": 95000.0,
            "entry_price": 95000.0,
            "stop_loss_price": 93500.0,
            "target_price": 98000.0,
            "position_size": 0.000263,
            "dry_run": True,
        },
        "error_message": None,
        "account_data": {},
        "open_positions": [],
        "active_orders": [],
        "decision": "BUY",
    }


def test_execution_node_executes_and_logs_to_duckdb() -> None:
    """Test A (Execution): ``risk_cleared=True`` calls ``place_crypto_order`` and logs to DuckDB."""
    state = _build_ready_state(risk_cleared=True)
    _assert_valid_agent_state(state)

    in_mem_conn = duckdb.connect(":memory:")
    conn_proxy = _PersistentInMemoryConn(in_mem_conn)

    try:
        with (
            patch(
                "executor_node.place_crypto_order",
                return_value="mock-crypto-trade-001",
            ) as mock_mcp,
            patch("executor_node.duckdb.connect", return_value=conn_proxy) as mock_db_connect,
        ):
            updated_state = execution_node(state)

        mock_mcp.assert_called_once_with(state["order_details"])
        mock_db_connect.assert_called_once()
        _assert_valid_agent_state(updated_state)
        assert updated_state["order_details"]["trade_id"] == "mock-crypto-trade-001"
        assert updated_state["error_message"] is None

        rows = in_mem_conn.execute(
            "SELECT trade_id, symbol, size, stop_loss, target FROM trades"
        ).fetchall()
        assert len(rows) == 1
        trade_id, symbol, size, stop_loss, target = rows[0]
        assert trade_id == "mock-crypto-trade-001"
        assert symbol == "BTC/USD"
        assert size == pytest.approx(0.000263)
        assert stop_loss == pytest.approx(93500.0)
        assert target == pytest.approx(98000.0)
    finally:
        in_mem_conn.close()


def test_execution_node_safety_guard_blocks_when_risk_not_cleared() -> None:
    """Test B (Safety Guard): ``risk_cleared=False`` never calls ``place_crypto_order`` or DuckDB."""
    state = _build_ready_state(risk_cleared=False)
    _assert_valid_agent_state(state)

    with (
        patch("executor_node.place_crypto_order") as mock_mcp,
        patch("executor_node.duckdb.connect") as mock_db_connect,
    ):
        updated_state = execution_node(state)

    mock_mcp.assert_not_called()
    mock_db_connect.assert_not_called()
    _assert_valid_agent_state(updated_state)
    assert updated_state == state
    assert "trade_id" not in updated_state["order_details"]


def test_place_crypto_order_calls_mcp_and_validates_tif() -> None:
    """Verify ``place_crypto_order`` invokes MCP ``place_crypto_order`` with ``gtc``/``ioc``."""
    order_details = _build_ready_state(risk_cleared=True)["order_details"]
    expected_id = "276e2673-764b-4ab6-a611-caf665ca6340"
    mock_tool_result = SimpleNamespace(
        isError=False,
        structuredContent=None,
        content=[SimpleNamespace(text=json.dumps({"id": expected_id, "status": "accepted"}))],
    )

    mock_session = MagicMock()
    mock_session.initialize = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value=mock_tool_result)

    mock_session_cm = MagicMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_cm.__aexit__ = AsyncMock(return_value=None)

    mock_stdio_cm = MagicMock()
    mock_stdio_cm.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
    mock_stdio_cm.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("executor_node.stdio_client", return_value=mock_stdio_cm),
        patch("executor_node.ClientSession", return_value=mock_session_cm),
    ):
        trade_id = place_crypto_order(order_details)

    assert trade_id == expected_id
    mock_session.call_tool.assert_awaited_once_with(
        "place_crypto_order",
        arguments={
            "symbol": "BTC/USD",
            "side": "buy",
            "type": "limit",
            "time_in_force": "gtc",
            "qty": "0.000263",
            "limit_price": "95000.0",
        },
    )

    with pytest.raises(ValueError, match="Unsupported crypto order_type"):
        place_crypto_order({**order_details, "order_type": "bracket"})
