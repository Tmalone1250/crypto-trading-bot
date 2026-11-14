"""Unit tests for Node 3 (Risk Manager) following TDD and .ai_rules standards."""

import pandas as pd
import pytest

from risk_node import MAX_RISK_USD, risk_manager_node
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


def _build_synthetic_ohlcv(
    num_candles: int,
    base_price: float,
    wick_spread: float,
) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame with controlled True Range."""
    rows = []
    for i in range(num_candles):
        close = base_price + (i * 0.05)
        rows.append(
            {
                "open": close - 0.02,
                "high": close + wick_spread,
                "low": close - wick_spread,
                "close": close,
                "volume": 100.0 + i,
            }
        )
    return pd.DataFrame(rows)


def test_risk_manager_node_cleared_low_volatility() -> None:
    """Test A (Cleared): Low-volatility data keeps 1.5x ATR risk <= $0.80."""
    # With price ~100.0 and wick_spread=0.5 (TR ~1.0), 1.5x ATR ~1.5 -> $0.15 risk on $10.
    df = _build_synthetic_ohlcv(num_candles=20, base_price=100.0, wick_spread=0.5)
    current_price = float(df["close"].iloc[-1])

    state: AgentState = {
        "asset_symbol": "SOL/USD",
        "current_price": current_price,
        "historical_data": df,
        "sentiment_score": 0.82,
        "risk_cleared": False,
        "order_details": {},
        "error_message": None,
    }
    _assert_valid_agent_state(state)

    updated_state = risk_manager_node(state)

    _assert_valid_agent_state(updated_state)
    assert updated_state["risk_cleared"] is True
    assert updated_state["error_message"] is None

    order = updated_state["order_details"]
    assert order["order_type"] == "bracket"
    assert order["dry_run"] is True
    assert order["stop_loss_price"] < current_price
    assert order["target_price"] > current_price
    assert order["position_size"] > 0.0
    assert order["projected_dollar_risk"] <= MAX_RISK_USD


def test_risk_manager_node_vetoed_high_volatility() -> None:
    """Test B (Vetoed): High-volatility candle wicks push 1.5x ATR risk > $0.80."""
    # With price ~100.0 and wick_spread=8.0 (TR ~16.0), 1.5x ATR ~24.0 -> $2.40 risk on $10.
    df = _build_synthetic_ohlcv(num_candles=20, base_price=100.0, wick_spread=8.0)
    current_price = float(df["close"].iloc[-1])

    state: AgentState = {
        "asset_symbol": "SOL/USD",
        "current_price": current_price,
        "historical_data": df,
        "sentiment_score": 0.82,
        "risk_cleared": False,
        "order_details": {},
        "error_message": None,
    }
    _assert_valid_agent_state(state)

    updated_state = risk_manager_node(state)

    _assert_valid_agent_state(updated_state)
    assert updated_state["risk_cleared"] is False
    assert updated_state["order_details"] == {}
    assert updated_state["error_message"] is not None
    assert "Risk veto" in updated_state["error_message"]
