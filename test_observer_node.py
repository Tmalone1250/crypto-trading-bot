"""Unit tests for Node 1 (Market Observer) with Alpaca MCP & strategy filter."""

from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from observer_node import BAR_LIMIT, BAR_TIMEFRAME, market_observer_node
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


def _sample_mcp_bars(count: int = 3) -> list[dict]:
    """Return synthetic Alpaca MCP crypto bar records."""
    return [
        {
            "timestamp": f"2026-11-14T0{i}:00:00Z",
            "open": 90000.0 + (i * 100.0),
            "high": 90600.0 + (i * 100.0),
            "low": 89900.0 + (i * 100.0),
            "close": 90500.0 + (i * 250.25),
            "volume": 12.5 + i,
        }
        for i in range(count)
    ]


@pytest.fixture
def base_state() -> AgentState:
    """Return a valid initial ``AgentState`` for observer node tests."""
    return {
        "asset_symbol": "BTC/USD",
        "current_price": 0.0,
        "historical_data": pd.DataFrame(),
        "sentiment_score": 0.0,
        "risk_cleared": False,
        "order_details": {},
        "error_message": None,
    }


def test_market_observer_node_success_when_signal_triggers(base_state: AgentState) -> None:
    """Verify live MCP bars are formatted to float64 and state updates when filter returns True."""
    _assert_valid_agent_state(base_state)
    bars = _sample_mcp_bars(3)

    with (
        patch("observer_node.get_crypto_bars", return_value=bars) as mock_mcp,
        patch("observer_node.check_entry_signal", return_value=True) as mock_filter,
    ):
        updated_state = market_observer_node(base_state)

    mock_mcp.assert_called_once_with(symbol="BTC/USD", timeframe=BAR_TIMEFRAME, limit=BAR_LIMIT)
    mock_filter.assert_called_once()
    _assert_valid_agent_state(updated_state)

    df = updated_state["historical_data"]
    for col in ("open", "high", "low", "close", "volume"):
        assert df[col].dtype == np.float64

    assert updated_state["current_price"] == pytest.approx(bars[-1]["close"])
    assert updated_state["error_message"] is None


def test_market_observer_node_raises_value_error_when_no_setup(base_state: AgentState) -> None:
    """Verify ``ValueError`` is raised to short-circuit the graph when filter returns False."""
    _assert_valid_agent_state(base_state)
    bars = _sample_mcp_bars(3)

    with (
        patch("observer_node.get_crypto_bars", return_value=bars) as mock_mcp,
        patch("observer_node.check_entry_signal", return_value=False) as mock_filter,
    ):
        with pytest.raises(ValueError, match="No structural setup detected"):
            market_observer_node(base_state)

    mock_mcp.assert_called_once_with(symbol="BTC/USD", timeframe=BAR_TIMEFRAME, limit=BAR_LIMIT)
    mock_filter.assert_called_once()


def test_market_observer_node_api_timeout_sets_error_message(base_state: AgentState) -> None:
    """Verify an MCP timeout/exception sets ``error_message`` gracefully without crashing."""
    _assert_valid_agent_state(base_state)

    with patch(
        "observer_node.get_crypto_bars",
        side_effect=TimeoutError("Alpaca MCP request timed out"),
    ) as mock_mcp:
        updated_state = market_observer_node(base_state)

    mock_mcp.assert_called_once_with(symbol="BTC/USD", timeframe=BAR_TIMEFRAME, limit=BAR_LIMIT)
    _assert_valid_agent_state(updated_state)
    assert updated_state["error_message"] is not None
    assert "timed out" in updated_state["error_message"]
    assert updated_state["current_price"] == 0.0
    assert updated_state["historical_data"].empty
