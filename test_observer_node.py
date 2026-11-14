"""Unit tests for Node 1 (Market Observer) following TDD and .ai_rules standards."""

from unittest.mock import patch
import pandas as pd
import pytest

from observer_node import market_observer_node
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


def test_market_observer_node_success(base_state: AgentState) -> None:
    """Verify successful state update with a mocked OHLCV DataFrame."""
    _assert_valid_agent_state(base_state)

    mock_df = pd.DataFrame(
        {
            "open": [90000.0, 90500.0],
            "high": [90600.0, 91200.0],
            "low": [89900.0, 90400.0],
            "close": [90500.0, 91050.25],
            "volume": [12.5, 18.2],
        }
    )

    with patch("observer_node.get_mock_alpaca_data", return_value=mock_df) as mock_fetch:
        updated_state = market_observer_node(base_state)

    mock_fetch.assert_called_once_with("BTC/USD")
    _assert_valid_agent_state(updated_state)
    assert updated_state["current_price"] == pytest.approx(91050.25)
    pd.testing.assert_frame_equal(updated_state["historical_data"], mock_df)
    assert updated_state["error_message"] is None


def test_market_observer_node_api_timeout_sets_error_message(base_state: AgentState) -> None:
    """Verify an API timeout/exception sets ``error_message`` without crashing."""
    _assert_valid_agent_state(base_state)

    with patch(
        "observer_node.get_mock_alpaca_data",
        side_effect=TimeoutError("Alpaca MCP request timed out"),
    ) as mock_fetch:
        updated_state = market_observer_node(base_state)

    mock_fetch.assert_called_once_with("BTC/USD")
    _assert_valid_agent_state(updated_state)
    assert updated_state["error_message"] is not None
    assert "timed out" in updated_state["error_message"]
    assert updated_state["current_price"] == 0.0
    assert updated_state["historical_data"].empty
