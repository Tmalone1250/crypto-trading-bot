"""Unit tests for the Regime Supervisor Node (deterministic volatility gate)."""

from datetime import datetime
from zoneinfo import ZoneInfo
import duckdb
import pandas as pd
import pytest

from state import AgentState
import supervisor_node as supervisor_mod
from supervisor_node import (
    FLAT_RATIONALE,
    THREE_LOSS_RATIONALE,
    detect_session_open,
    supervisor_node,
)


REQUIRED_STATE_KEYS = {
    "asset_symbol",
    "asset_symbols",
    "market_data_by_symbol",
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
    "trade_intents",
    "last_alpha_eval_price",
    "mcp_client",
}


def _assert_valid_agent_state(state: AgentState) -> None:
    """Validate that ``state`` conforms to the ``AgentState`` schema and types."""
    assert set(state.keys()) == REQUIRED_STATE_KEYS
    assert isinstance(state["asset_symbol"], str)
    assert isinstance(state["asset_symbols"], list)
    assert isinstance(state["market_data_by_symbol"], dict)
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
    assert isinstance(state["realized_pnl"], float)
    assert isinstance(state["trade_intents"], list)
    assert isinstance(state["last_alpha_eval_price"], float)


def _make_state(df: pd.DataFrame) -> AgentState:
    """Build a valid ``AgentState`` with the given OHLCV DataFrame."""
    return {
        "asset_symbol": "BTC/USD",
        "asset_symbols": ["BTC/USD", "ETH/USD", "SOL/USD"],
        "market_data_by_symbol": {},
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
        "realized_pnl": 0.0,
        "trade_intents": [],
        "last_alpha_eval_price": 0.0,
        "mcp_client": None,
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


def test_supervisor_node_gates_active_on_0_20_percent_move_from_last_alpha_eval() -> None:
    """Verify >= 0.25% range still defers to ``FLAT`` if price moved < 0.20% from ``last_alpha_eval_price``."""
    active_df = pd.DataFrame(
        {
            "high": [90100.0, 90150.0, 90300.0, 90200.0, 90250.0],
            "low": [90000.0, 90050.0, 90100.0, 90080.0, 90120.0],
            "close": [90050.0, 90100.0, 90250.0, 90150.0, 90200.0],
        }
    )

    # 90100 vs 90000 = 0.111% (< 0.20%) -> FLAT (save tokens, let Scalper run)
    state_small_move = _make_state(active_df)
    state_small_move["last_alpha_eval_price"] = 90000.0
    state_small_move["current_price"] = 90100.0
    res_small = supervisor_node(state_small_move)
    assert res_small["market_regime"] == "FLAT"
    assert res_small["action"] == "HOLD"

    # 90200 vs 90000 = 0.222% (>= 0.20%) -> ACTIVE (wake Alpha LLM)
    state_large_move = _make_state(active_df)
    state_large_move["last_alpha_eval_price"] = 90000.0
    state_large_move["current_price"] = 90200.0
    res_large = supervisor_node(state_large_move)
    assert res_large["market_regime"] == "ACTIVE"


def test_detect_session_open_ny_london_asia() -> None:
    """Verify session open detector catches NY (08:30 CST), London, and Asia opens."""
    ny_open = datetime(2026, 11, 16, 8, 30, tzinfo=ZoneInfo("America/Chicago"))
    london_open = datetime(2026, 11, 16, 8, 0, tzinfo=ZoneInfo("Europe/London"))
    asia_open = datetime(2026, 11, 16, 9, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
    non_open = datetime(2026, 11, 16, 11, 15, tzinfo=ZoneInfo("America/Chicago"))

    assert detect_session_open(ny_open) == "NEW_YORK"
    assert detect_session_open(london_open) == "LONDON"
    assert detect_session_open(asia_open) == "ASIA"
    assert detect_session_open(non_open) is None


def test_supervisor_anchors_fair_price_and_enforces_three_loss_rule(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify NY open anchors Fair Price in DuckDB and Three-Loss Rule aborts to FLAT."""
    db_file = str(tmp_path / "test_supervisor.duckdb")
    monkeypatch.setattr(supervisor_mod, "DB_PATH", db_file)

    ny_open = datetime(2026, 11, 16, 8, 30, tzinfo=ZoneInfo("America/Chicago"))
    df = pd.DataFrame(
        {"high": [91200.0], "low": [91000.0], "close": [91150.0]}
    )
    state = _make_state(df)
    state["current_price"] = 91150.0

    updated = supervisor_node(state, now=ny_open)
    assert updated["market_regime"] == "SESSION_OPEN_WATCH"

    conn = duckdb.connect(db_file)
    try:
        row = conn.execute(
            "SELECT strategy, phase, target_price FROM setup_state_machine LIMIT 1"
        ).fetchone()
        assert row == ("FAIR_PRICE_REVERSION", "PHASE_0_ANCHORED", 91150.0)

        # Now insert consecutive_losses = 3 for today and verify Three-Loss Rule aborts.
        conn.execute(
            "INSERT INTO session_metrics (date, realized_pnl, consecutive_losses) VALUES (?, ?, ?)",
            [ny_open.date(), -450.0, 3],
        )
    finally:
        conn.close()

    aborted = supervisor_node(state, now=ny_open)
    assert aborted["market_regime"] == "FLAT"
    assert aborted["action"] == "HOLD"
    assert aborted["rationale"] == THREE_LOSS_RATIONALE


def test_supervisor_transitions_circuit_breaker_to_suspended() -> None:
    """Verify CIRCUIT_BREAKER action transitions market_regime to SUSPENDED."""
    df = pd.DataFrame({"high": [91200.0], "low": [91000.0], "close": [91150.0]})
    state = _make_state(df)
    state["action"] = "CIRCUIT_BREAKER"

    updated = supervisor_node(state)
    assert updated["market_regime"] == "SUSPENDED"
    assert updated["action"] == "CIRCUIT_BREAKER"
