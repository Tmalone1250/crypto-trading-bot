"""Unit tests for the Regime Supervisor Node (deterministic volatility gate)."""

import pandas as pd
import pytest

from state import AgentState
from supervisor_node import FLAT_RATIONALE, supervisor_node


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


def _make_state(df: pd.DataFrame) -> AgentState:
    """Build a valid ``AgentState`` with the given OHLCV DataFrame."""
    return {
        "asset_symbol": "BTC/USD",
        "current_price": 90000.0,
        "historical_data": df,
        "sentiment_score": 0.0,
        "risk_cleared": False,
        "order_details": {},
        "error_message": None,
        "account_data": {},
        "open_positions": [],
        "active_orders": [],
        "action": "MARKET_BUY",
        "target_price": None,
        "rationale": "",
        "market_regime": "",
    }


def test_supervisor_node_flat_regime_when_range_under_threshold(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify 5-bar high-low range < 0.25% sets ``FLAT``, ``HOLD``, and skips LLM."""
    # Only the last 5 candles should be inspected; earlier volatile bars are ignored.
    df = pd.DataFrame(
        {
            "high": [95000.0, 95000.0, 90100.0, 90110.0, 90120.0, 90115.0, 90105.0],
            "low": [85000.0, 85000.0, 90000.0, 90010.0, 90020.0, 90015.0, 90005.0],
            "close": [90000.0] * 7,
        }
    )
    state = _make_state(df)
    updated_state = supervisor_node(state)

    _assert_valid_agent_state(updated_state)
    assert updated_state["market_regime"] == "FLAT"
    assert updated_state["action"] == "HOLD"
    assert updated_state["rationale"] == FLAT_RATIONALE

    captured = capsys.readouterr().out
    assert "[Supervisor]" in captured
    assert "Regime: FLAT" in captured


def test_supervisor_node_active_regime_when_range_meets_or_exceeds_threshold(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify 5-bar high-low range >= 0.25% sets ``ACTIVE`` and wakes Alpha LLM."""
    # (90300 - 90000) / 90000 * 100 = 0.3333% >= 0.25%
    df = pd.DataFrame(
        {
            "high": [90100.0, 90150.0, 90300.0, 90200.0, 90250.0],
            "low": [90000.0, 90050.0, 90100.0, 90080.0, 90120.0],
            "close": [90050.0, 90100.0, 90250.0, 90150.0, 90200.0],
        }
    )
    state = _make_state(df)
    updated_state = supervisor_node(state)

    _assert_valid_agent_state(updated_state)
    assert updated_state["market_regime"] == "ACTIVE"

    captured = capsys.readouterr().out
    assert "[Supervisor]" in captured
    assert "Regime: ACTIVE" in captured


def test_supervisor_node_empty_dataframe_defaults_to_flat() -> None:
    """Verify empty DataFrame safely classifies as ``FLAT`` and ``HOLD``."""
    state = _make_state(pd.DataFrame())
    updated_state = supervisor_node(state)

    _assert_valid_agent_state(updated_state)
    assert updated_state["market_regime"] == "FLAT"
    assert updated_state["action"] == "HOLD"
    assert updated_state["rationale"] == FLAT_RATIONALE
