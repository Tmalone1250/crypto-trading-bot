"""Unit tests for Node 3 (Risk Manager) following TDD and .ai_rules standards."""

import pandas as pd
import pytest

from risk_node import (
    ACCOUNT_SIZE_USD,
    MAX_RISK_USD,
    MIN_CRYPTO_ORDER_SIZE,
    QTY_PRECISION,
    risk_manager_node,
)
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
    "action",
    "target_price",
    "rationale",
    "market_regime",
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
    assert isinstance(state["action"], str)
    assert state["target_price"] is None or isinstance(state["target_price"], float)
    assert isinstance(state["rationale"], str)
    assert isinstance(state["market_regime"], str)


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


def test_risk_manager_node_cleared_low_volatility_and_btc_sizing() -> None:
    """Test A (Cleared): $25 base size clears 0.0001 BTC min_order_size and 1.5x ATR risk <= $2.00."""
    df = _build_synthetic_ohlcv(num_candles=20, base_price=95000.0, wick_spread=150.0)
    current_price = float(df["close"].iloc[-1])

    state: AgentState = {
        "asset_symbol": "BTC/USD",
        "current_price": current_price,
        "historical_data": df,
        "sentiment_score": 0.82,
        "risk_cleared": False,
        "order_details": {},
        "error_message": None,
        "account_data": {},
        "open_positions": [],
        "active_orders": [],
        "action": "LIMIT_BUY",
        "target_price": 94800.0,
        "rationale": "Test limit buy",
        "market_regime": "ACTIVE",
    }
    _assert_valid_agent_state(state)

    updated_state = risk_manager_node(state)

    _assert_valid_agent_state(updated_state)
    assert updated_state["risk_cleared"] is True
    assert updated_state["error_message"] is None

    order = updated_state["order_details"]
    assert order["order_type"] in {"limit", "market"}
    assert order["time_in_force"] in {"gtc", "ioc"}
    assert order["dry_run"] is True
    assert order["stop_loss_price"] < current_price
    assert order["target_price"] > current_price
    assert order["position_size"] == pytest.approx(round(ACCOUNT_SIZE_USD / current_price, QTY_PRECISION))
    assert order["position_size"] >= MIN_CRYPTO_ORDER_SIZE
    assert order["projected_dollar_risk"] <= MAX_RISK_USD


def test_risk_manager_node_vetoed_high_volatility() -> None:
    """Test B (Vetoed): High-volatility candle wicks push 1.5x ATR risk above $2.00."""
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
        "account_data": {},
        "open_positions": [],
        "active_orders": [],
        "action": "MARKET_BUY",
        "target_price": None,
        "rationale": "Test high volatility veto",
        "market_regime": "ACTIVE",
    }
    _assert_valid_agent_state(state)

    updated_state = risk_manager_node(state)

    _assert_valid_agent_state(updated_state)
    assert updated_state["risk_cleared"] is False
    assert updated_state["order_details"] == {}
    assert updated_state["error_message"] is not None
    assert "Risk veto" in updated_state["error_message"]
