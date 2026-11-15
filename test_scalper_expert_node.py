"""Unit tests for the Scalper Expert Node (Python-only FLAT-regime micro-scalper)."""

import pandas as pd
import pytest

from scalper_expert_node import scalper_expert_node
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
    "realized_pnl",
}


def _assert_valid_agent_state(state: AgentState) -> None:
    """Validate that ``state`` conforms to the ``AgentState`` schema and types."""
    assert set(state.keys()) == REQUIRED_STATE_KEYS
    assert isinstance(state["realized_pnl"], float)
    assert isinstance(state["action"], str)
    assert state["target_price"] is None or isinstance(state["target_price"], float)
    assert isinstance(state["rationale"], str)


def _flat_5bar_df() -> pd.DataFrame:
    """Return a 5-bar DataFrame with range 90000.0 to 90200.0 (lower 20% <= 90040.0)."""
    return pd.DataFrame(
        {
            "high": [90180.0, 90200.0, 90150.0, 90120.0, 90100.0],
            "low": [90050.0, 90080.0, 90020.0, 90000.0, 90010.0],
            "close": [90100.0, 90140.0, 90090.0, 90030.0, 90020.0],
        }
    )


def _base_state(
    current_price: float,
    open_positions: list[dict] | None = None,
) -> AgentState:
    """Build a valid FLAT-regime ``AgentState`` for scalper unit tests."""
    return {
        "asset_symbol": "BTC/USD",
        "current_price": current_price,
        "historical_data": _flat_5bar_df(),
        "sentiment_score": 0.0,
        "risk_cleared": False,
        "order_details": {},
        "error_message": None,
        "account_data": {"buying_power": "2000.00", "cash": "1000.00"},
        "open_positions": open_positions or [],
        "active_orders": [],
        "action": "HOLD",
        "target_price": None,
        "rationale": "",
        "market_regime": "FLAT",
        "realized_pnl": 0.0,
    }


def test_scalper_expert_node_limit_buy_in_lower_20_percent() -> None:
    """Verify ``current_price`` in lower 20% of 5-bar range triggers ``LIMIT_BUY``."""
    # Range: 90000..90200 -> lower 20% <= 90040.0
    state = _base_state(current_price=90020.0)
    updated = scalper_expert_node(state)

    _assert_valid_agent_state(updated)
    assert updated["action"] == "LIMIT_BUY"
    assert updated["target_price"] == pytest.approx(90020.0)
    assert "lower-20%" in updated["rationale"]


def test_scalper_expert_node_limit_sell_when_up_half_percent() -> None:
    """Verify open position up >= 0.5% from ``avg_entry_price`` triggers ``LIMIT_SELL``."""
    # avg_entry = 89500.0; +0.5% = 89947.5; current = 90050.0 (+0.61%)
    state = _base_state(
        current_price=90050.0,
        open_positions=[{"symbol": "BTCUSD", "avg_entry_price": "89500.0", "qty": "0.001"}],
    )
    updated = scalper_expert_node(state)

    _assert_valid_agent_state(updated)
    assert updated["action"] == "LIMIT_SELL"
    assert updated["target_price"] == pytest.approx(90050.0)
    assert "take-profit" in updated["rationale"]


def test_scalper_expert_node_holds_when_neither_condition_met() -> None:
    """Verify ``HOLD`` when price is above lower 20% and open position is < +0.5%."""
    state = _base_state(
        current_price=90100.0,
        open_positions=[{"symbol": "BTCUSD", "avg_entry_price": "90000.0", "qty": "0.001"}],
    )
    updated = scalper_expert_node(state)

    _assert_valid_agent_state(updated)
    assert updated["action"] == "HOLD"
    assert updated["target_price"] is None
