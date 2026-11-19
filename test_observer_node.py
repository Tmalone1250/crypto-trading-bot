"""Unit tests for Node 1 (Market Observer) with Alpaca MCP data ingestion."""

from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from observer_node import (
    BAR_LIMIT,
    BAR_TIMEFRAME,
    LOOKBACK_DAYS,
    _format_bars_dataframe,
    get_crypto_bars,
    get_live_crypto_price,
    market_observer_node,
)
from state import AgentState


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
    "macro_context",
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
        "asset_symbols": ["BTC/USD", "ETH/USD", "SOL/USD"],
        "market_data_by_symbol": {},
        "current_price": 0.0,
        "historical_data": pd.DataFrame(),
        "sentiment_score": 0.0,
        "risk_cleared": False,
        "order_details": {},
        "error_message": None,
        "account_data": {},
        "open_positions": [],
        "active_orders": [],
        "action": "HOLD",
        "target_price": None,
        "rationale": "",
        "market_regime": "FLAT",
        "realized_pnl": 0.0,
        "trade_intents": [],
        "last_alpha_eval_price": 0.0,
        "macro_context": "",
        "mcp_client": None,
    }


def test_market_observer_node_success(base_state: AgentState) -> None:
    """Verify live MCP bars are formatted to float64 and state updates without static filtering."""
    _assert_valid_agent_state(base_state)
    bars = _sample_mcp_bars(3)

    with (
        patch("observer_node.get_crypto_bars", return_value=bars) as mock_mcp,
        patch("observer_node.get_live_crypto_price", return_value=None),
        patch("observer_node._fetch_macro_context", return_value="No high-impact events."),
        patch("observer_node.time.sleep"),
    ):
        updated_state = market_observer_node(base_state)

    mock_mcp.assert_any_call(symbol="BTC/USD", timeframe=BAR_TIMEFRAME, limit=BAR_LIMIT)
    _assert_valid_agent_state(updated_state)

    df = updated_state["historical_data"]
    for col in ("open", "high", "low", "close", "volume"):
        assert df[col].dtype == np.float64

    market_map = updated_state["market_data_by_symbol"]
    for sym in ("BTC/USD", "ETH/USD", "SOL/USD"):
        for key in (sym, f"{sym}_1H", f"{sym}_15M", f"{sym}_1M"):
            assert key in market_map
            assert len(market_map[key]) == 3

    assert updated_state["current_price"] == pytest.approx(bars[-1]["close"])
    assert updated_state["error_message"] is None


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


def test_format_bars_dataframe_parses_json_and_fastmcp_text() -> None:
    """Verify nested FastMCP ``data.bars`` JSON payloads, ``BTCUSD`` keys, and text parse cleanly."""
    enveloped_btc_slash = json.dumps(
        {
            "data": {
                "bars": {"BTC/USD": _sample_mcp_bars(2)},
                "next_page_token": None,
            }
        }
    )
    df_slash = _format_bars_dataframe(enveloped_btc_slash, "BTC/USD")
    assert len(df_slash) == 2
    assert df_slash["close"].dtype == np.float64
    assert float(df_slash["close"].iloc[-1]) == pytest.approx(90750.25)

    enveloped_btcusd = json.dumps(
        {
            "data": {
                "bars": {"BTCUSD": _sample_mcp_bars(2)},
                "next_page_token": None,
            }
        }
    )
    df_noslash = _format_bars_dataframe(enveloped_btcusd, "BTC/USD")
    assert len(df_noslash) == 2
    assert df_noslash["close"].dtype == np.float64
    assert float(df_noslash["close"].iloc[-1]) == pytest.approx(90750.25)

    fastmcp_text = (
        "Historical Crypto Bars for BTC/USD:\n"
        "Timestamp: 2026-11-14T00:00:00Z | Open: $90,000.00 | High: $90,600.00 | "
        "Low: $89,900.00 | Close: $90,500.00 | Volume: 12.5\n"
        "Timestamp: 2026-11-14T01:00:00Z | Open: $90,500.00 | High: $91,100.00 | "
        "Low: $90,400.00 | Close: $90,950.50 | Volume: 15.2\n"
    )
    df_text = _format_bars_dataframe(fastmcp_text, "BTC/USD")
    assert len(df_text) == 2
    assert df_text["close"].dtype == np.float64
    assert float(df_text["close"].iloc[-1]) == pytest.approx(90950.50)


def test_get_crypto_bars_calls_mcp_stdio_session_with_12_day_start() -> None:
    """Verify ``get_crypto_bars`` passes ``symbols``, ``timeframe``, ``limit``, and 12-day ``start``."""
    bars = _sample_mcp_bars(2)
    enveloped_payload = {"data": {"bars": {"BTC/USD": bars}, "next_page_token": None}}
    mock_tool_result = SimpleNamespace(
        isError=False,
        structuredContent=None,
        content=[SimpleNamespace(text=json.dumps(enveloped_payload))],
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

    fixed_now = datetime(2026, 11, 14, 18, 0, 0, tzinfo=timezone.utc)
    expected_start = (fixed_now - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    with (
        patch("observer_node.stdio_client", return_value=mock_stdio_cm),
        patch("observer_node.ClientSession", return_value=mock_session_cm),
        patch("observer_node.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = fixed_now
        result = get_crypto_bars("BTC/USD", timeframe="1Hour", limit=250)

    mock_session.initialize.assert_awaited_once()
    mock_session.call_tool.assert_awaited_once_with(
        "get_crypto_bars",
        arguments={
            "symbols": "BTC/USD",
            "timeframe": "1Hour",
            "limit": 250,
            "start": expected_start,
        },
    )
    assert json.loads(result)["data"]["bars"]["BTC/USD"] == bars


def test_get_live_crypto_price_uses_native_alpaca_sdk(base_state: AgentState) -> None:
    """Verify ``get_live_crypto_price`` and ``market_observer_node`` use the native Alpaca SDK."""
    mock_sdk_client = MagicMock()
    mock_sdk_client.get_crypto_latest_quote.return_value = {
        "BTC/USD": SimpleNamespace(ask_price=91510.0, bid_price=91490.0)
    }

    with patch(
        "observer_node.get_alpaca_crypto_data_client",
        return_value=mock_sdk_client,
    ):
        price = get_live_crypto_price("BTC/USD")

    assert mock_sdk_client.get_crypto_latest_quote.call_count == 1
    req_arg = mock_sdk_client.get_crypto_latest_quote.call_args[0][0]
    assert req_arg.symbol_or_symbols == ["BTC/USD"] or req_arg.symbol_or_symbols == "BTC/USD"
    assert price == pytest.approx(91500.0)

    bars = _sample_mcp_bars(3)
    state_with_live: dict = {**base_state, "live_price": 0.0}
    with (
        patch("observer_node.get_crypto_bars", return_value=bars),
        patch("observer_node.get_alpaca_crypto_data_client", return_value=mock_sdk_client),
        patch("observer_node._fetch_macro_context", return_value="FOMC Rate Decision in 30m"),
        patch("observer_node.time.sleep"),
    ):
        updated = market_observer_node(state_with_live)  # type: ignore[arg-type]

    assert updated["current_price"] == pytest.approx(91500.0)
    assert updated["live_price"] == pytest.approx(91500.0)  # type: ignore[typeddict-item]
    assert len(updated["market_data_by_symbol"]["BTC/USD"]) == 3
    assert float(updated["market_data_by_symbol"]["BTC/USD"]["close"].iloc[-1]) == pytest.approx(91500.0)
    assert "FOMC Rate Decision" in updated["macro_context"]
