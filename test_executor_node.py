"""Unit tests for Node 4 (Executor & Logger) with full lifecycle action routing."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import duckdb
import pandas as pd
import pytest

from executor_node import (
    CANCEL_ALL_TOOL_NAME,
    cancel_all_orders,
    execution_node,
    monitor_trade_and_breakeven,
    place_crypto_order,
)
from state import AgentState, TradeIntent


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


class _PersistentInMemoryConn:
    """Proxy around an in-memory DuckDB connection that ignores ``close()`` so tests can inspect rows."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn

    def execute(self, *args, **kwargs):
        return self._conn.execute(*args, **kwargs)

    def close(self) -> None:
        pass


def _build_ready_state(
    risk_cleared: bool,
    action: str = "LIMIT_BUY",
    target_price: float | None = 94250.5,
) -> AgentState:
    """Build a sample ``AgentState`` ready for Node 4 testing."""
    return {
        "asset_symbol": "BTC/USD",
        "asset_symbols": ["BTC/USD", "ETH/USD", "SOL/USD"],
        "market_data_by_symbol": {},
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
        "action": action,
        "target_price": target_price,
        "rationale": "Test execution rationale",
        "market_regime": "ACTIVE",
        "realized_pnl": 0.0,
        "trade_intents": [],
        "last_alpha_eval_price": 0.0,
        "mcp_client": None,
    }


def test_execution_node_limit_buy_uses_target_price_and_logs_to_duckdb() -> None:
    """Verify ``LIMIT_BUY`` passes ``type='limit'`` and ``limit_price=str(state['target_price'])``."""
    state = _build_ready_state(risk_cleared=True, action="LIMIT_BUY", target_price=94250.5)
    _assert_valid_agent_state(state)

    in_mem_conn = duckdb.connect(":memory:")
    conn_proxy = _PersistentInMemoryConn(in_mem_conn)

    try:
        with (
            patch(
                "executor_node.place_crypto_order",
                return_value="mock-limit-trade-001",
            ) as mock_mcp,
            patch("executor_node.duckdb.connect", return_value=conn_proxy) as mock_db_connect,
        ):
            updated_state = execution_node(state)

        mock_mcp.assert_called_once()
        passed_details = mock_mcp.call_args.args[0]
        assert passed_details["side"] == "buy"
        assert passed_details["type"] == "limit"
        assert passed_details["qty"] == "0.000263"
        assert passed_details["limit_price"] == "94250.5"

        mock_db_connect.assert_called_once()
        _assert_valid_agent_state(updated_state)
        assert updated_state["order_details"]["trade_id"] == "mock-limit-trade-001"
        assert updated_state["order_details"]["action"] == "LIMIT_BUY"
        assert updated_state["error_message"] is None

        rows = in_mem_conn.execute(
            "SELECT trade_id, symbol, size, stop_loss, target FROM trades"
        ).fetchall()
        assert len(rows) == 1
        trade_id, symbol, size, stop_loss, target = rows[0]
        assert trade_id == "mock-limit-trade-001"
        assert symbol == "BTC/USD"
        assert size == pytest.approx(0.000263)
        assert stop_loss == pytest.approx(93500.0)
        assert target == pytest.approx(98000.0)
    finally:
        in_mem_conn.close()


def test_execution_node_market_buy_submits_market_order() -> None:
    """Verify ``MARKET_BUY`` maps to ``type='market'`` with string ``qty`` and no ``limit_price``."""
    state = _build_ready_state(risk_cleared=True, action="MARKET_BUY", target_price=None)
    _assert_valid_agent_state(state)

    in_mem_conn = duckdb.connect(":memory:")
    conn_proxy = _PersistentInMemoryConn(in_mem_conn)

    try:
        with (
            patch(
                "executor_node.place_crypto_order",
                return_value="mock-market-trade-002",
            ) as mock_mcp,
            patch("executor_node.duckdb.connect", return_value=conn_proxy),
        ):
            updated_state = execution_node(state)

        mock_mcp.assert_called_once()
        passed_details = mock_mcp.call_args.args[0]
        assert passed_details["side"] == "buy"
        assert passed_details["type"] == "market"
        assert passed_details["qty"] == "0.000263"
        assert "limit_price" not in passed_details

        _assert_valid_agent_state(updated_state)
        assert updated_state["order_details"]["trade_id"] == "mock-market-trade-002"
        assert updated_state["order_details"]["action"] == "MARKET_BUY"
    finally:
        in_mem_conn.close()


def test_execution_node_limit_sell_and_market_sell_pass_side_sell() -> None:
    """Verify ``LIMIT_SELL`` and ``MARKET_SELL`` pass ``side='sell'`` with string ``qty``/``limit_price``."""
    in_mem_conn = duckdb.connect(":memory:")
    conn_proxy = _PersistentInMemoryConn(in_mem_conn)

    try:
        limit_sell_state = _build_ready_state(
            risk_cleared=True,
            action="LIMIT_SELL",
            target_price=97500.0,
        )
        with (
            patch(
                "executor_node.place_crypto_order",
                return_value="mock-limit-sell-001",
            ) as mock_mcp,
            patch("executor_node.duckdb.connect", return_value=conn_proxy),
        ):
            updated_limit_sell = execution_node(limit_sell_state)

        passed_limit_sell = mock_mcp.call_args.args[0]
        assert passed_limit_sell["side"] == "sell"
        assert passed_limit_sell["type"] == "limit"
        assert passed_limit_sell["qty"] == "0.000263"
        assert passed_limit_sell["limit_price"] == "97500.0"
        _assert_valid_agent_state(updated_limit_sell)
        assert updated_limit_sell["order_details"]["action"] == "LIMIT_SELL"
        assert updated_limit_sell["order_details"]["trade_id"] == "mock-limit-sell-001"

        market_sell_state = _build_ready_state(
            risk_cleared=True,
            action="MARKET_SELL",
            target_price=None,
        )
        with (
            patch(
                "executor_node.place_crypto_order",
                return_value="mock-market-sell-002",
            ) as mock_mcp,
            patch("executor_node.duckdb.connect", return_value=conn_proxy),
        ):
            updated_market_sell = execution_node(market_sell_state)

        passed_market_sell = mock_mcp.call_args.args[0]
        assert passed_market_sell["side"] == "sell"
        assert passed_market_sell["type"] == "market"
        assert passed_market_sell["qty"] == "0.000263"
        assert "limit_price" not in passed_market_sell
        _assert_valid_agent_state(updated_market_sell)
        assert updated_market_sell["order_details"]["action"] == "MARKET_SELL"
        assert updated_market_sell["order_details"]["trade_id"] == "mock-market-sell-002"

        # Verify War Chest (realized_pnl) compounds when selling a profitable position.
        war_chest_state = _build_ready_state(
            risk_cleared=True,
            action="LIMIT_SELL",
            target_price=97000.0,
        )
        war_chest_state["realized_pnl"] = 1.50
        war_chest_state["open_positions"] = [
            {"symbol": "BTCUSD", "avg_entry_price": "95000.0", "qty": "0.000263"}
        ]
        with (
            patch("executor_node.place_crypto_order", return_value="mock-war-chest-003"),
            patch("executor_node.duckdb.connect", return_value=conn_proxy),
        ):
            updated_war_chest = execution_node(war_chest_state)

        # (97000 - 95000) * 0.000263 = 0.526; 1.50 + 0.526 = 2.026
        assert updated_war_chest["realized_pnl"] == pytest.approx(2.026)
    finally:
        in_mem_conn.close()


def test_execution_node_executes_list_of_trade_intents_and_updates_war_chest() -> None:
    """Verify ``execution_node`` loops through ``trade_intents`` and compounds ``realized_pnl``."""
    in_mem_conn = duckdb.connect(":memory:")
    conn_proxy = _PersistentInMemoryConn(in_mem_conn)

    intents = [
        TradeIntent(
            symbol="BTC/USD",
            action="LIMIT_BUY",
            target_price=90000.0,
            qty=0.002222,
            notional_usd=200.0,
            rationale="Lower-20% buy",
        ),
        TradeIntent(
            symbol="SOL/USD",
            action="LIMIT_SELL",
            target_price=102.0,
            qty=2.0,
            notional_usd=204.0,
            rationale="Fee-adjusted +2% sell",
        ),
    ]
    state = _build_ready_state(risk_cleared=True)
    state["trade_intents"] = intents
    state["open_positions"] = [{"symbol": "SOLUSD", "avg_entry_price": "100.0", "qty": "2.0"}]
    state["realized_pnl"] = 1.0

    try:
        with (
            patch(
                "executor_node.place_crypto_order",
                side_effect=["intent-order-1", "intent-order-2"],
            ) as mock_mcp,
            patch("executor_node.duckdb.connect", return_value=conn_proxy),
        ):
            updated = execution_node(state)

        assert mock_mcp.call_count == 2
        first_args = mock_mcp.call_args_list[0].args[0]
        second_args = mock_mcp.call_args_list[1].args[0]

        assert first_args["symbol"] == "BTC/USD"
        assert first_args["side"] == "buy"
        assert first_args["type"] == "limit"
        assert first_args["qty"] == "0.002222"
        assert first_args["limit_price"] == "90000.0"

        assert second_args["symbol"] == "SOL/USD"
        assert second_args["side"] == "sell"
        assert second_args["type"] == "limit"
        assert second_args["qty"] == "2.0"
        assert second_args["limit_price"] == "102.0"

        _assert_valid_agent_state(updated)
        assert len(updated["order_details"]["executed_intents"]) == 2
        # (102.0 - 100.0) * 2.0 = 4.0; 1.0 + 4.0 = 5.0
        assert updated["realized_pnl"] == pytest.approx(5.0)
    finally:
        in_mem_conn.close()


def test_execution_node_cancel_all_calls_cancel_all_orders_mcp() -> None:
    """Verify ``CANCEL_ALL`` maps to ``cancel_all_orders`` without placing an order."""
    state = _build_ready_state(risk_cleared=True, action="CANCEL_ALL", target_price=None)
    _assert_valid_agent_state(state)

    mock_tool_result = SimpleNamespace(
        is_error=False,
        structured_content={"data": {"id": "cancel-batch-001"}},
        content=[SimpleNamespace(text=json.dumps({"data": {"id": "cancel-batch-001"}}))],
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
        patch("executor_node.place_crypto_order") as mock_place_order,
    ):
        updated_state = execution_node(state)

    mock_place_order.assert_not_called()
    mock_session.call_tool.assert_awaited_once_with(CANCEL_ALL_TOOL_NAME, arguments={})
    _assert_valid_agent_state(updated_state)
    assert updated_state["order_details"]["action"] == "CANCEL_ALL"
    assert updated_state["order_details"]["trade_id"] == "cancel-batch-001"
    assert updated_state["error_message"] is None


def test_execution_node_safety_guard_blocks_when_risk_not_cleared() -> None:
    """Test B (Safety Guard): ``risk_cleared=False`` never calls MCP or DuckDB."""
    state = _build_ready_state(risk_cleared=False)
    _assert_valid_agent_state(state)

    with (
        patch("executor_node.place_crypto_order") as mock_mcp,
        patch("executor_node.cancel_all_orders") as mock_cancel,
        patch("executor_node.duckdb.connect") as mock_db_connect,
    ):
        updated_state = execution_node(state)

    mock_mcp.assert_not_called()
    mock_cancel.assert_not_called()
    mock_db_connect.assert_not_called()
    _assert_valid_agent_state(updated_state)
    assert updated_state == state
    assert "trade_id" not in updated_state["order_details"]


def test_place_crypto_order_calls_mcp_with_string_casts_and_validates_tif() -> None:
    """Verify ``place_crypto_order`` casts ``qty`` and ``limit_price`` to ``str`` and validates ``type``."""
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


def test_monitor_trade_and_breakeven_triggers_replace_order_at_momentum_threshold() -> None:
    """Verify 1s polling moves stop-loss to entry_price via replace_order at +0.15% PnL."""
    mock_client = MagicMock()
    # Sequential prices: exact entry, slight drawdown, then > +0.0015 momentum threshold
    mock_client.get_latest_trade.side_effect = [
        SimpleNamespace(price=100000.0),
        SimpleNamespace(price=99900.0),
        SimpleNamespace(price=100200.0),
    ]
    mock_client.replace_order.return_value = {"id": "ord-sl-123", "stop_price": "100000.0"}

    with patch("executor_node.time.sleep") as mock_sleep:
        result = monitor_trade_and_breakeven(
            client=mock_client,
            order_id="ord-sl-123",
            entry_price=100000.0,
            is_long=True,
        )

    assert result["breakeven_triggered"] is True
    assert result["order_id"] == "ord-sl-123"
    assert result["entry_price"] == pytest.approx(100000.0)
    assert result["trigger_price"] == pytest.approx(100200.0)
    assert mock_client.get_latest_trade.call_count == 3
    assert mock_sleep.call_count == 2
    mock_client.replace_order.assert_called_once_with(
        order_id="ord-sl-123",
        stop_price="100000.0",
    )


def test_monitor_trade_and_breakeven_exits_after_max_attempts() -> None:
    """Verify polling stops cleanly after max_attempts when threshold is never reached."""
    mock_client = MagicMock()
    mock_client.get_latest_trade.return_value = SimpleNamespace(price=100000.0)

    with patch("executor_node.time.sleep") as mock_sleep:
        result = monitor_trade_and_breakeven(
            client=mock_client,
            order_id="ord-sl-timeout",
            entry_price=100000.0,
            is_long=True,
            max_attempts=3,
        )

    assert result["breakeven_triggered"] is False
    assert result["attempts"] == 3
    assert mock_client.get_latest_trade.call_count == 3
    assert mock_sleep.call_count == 2
    mock_client.replace_order.assert_not_called()


def test_alpaca_paper_config_and_market_intent_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify config.py pins Paper Trading URL/keys and executor omits extended_hours."""
    from config import ALPACA_PAPER_URL, get_alpaca_paper_config
    from executor_node import _intent_to_order_details

    monkeypatch.setenv("APCA_API_KEY_ID", "paper-key-id")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "paper-secret-key")
    cfg = get_alpaca_paper_config()

    assert cfg["base_url"] == ALPACA_PAPER_URL
    assert cfg["api_key"] == "paper-key-id"
    assert cfg["secret_key"] == "paper-secret-key"
    assert cfg["paper"] is True

    intent = TradeIntent(
        symbol="BTC/USD",
        action="MARKET_BUY",
        target_price=100000.0,
        qty=0.002,
        notional_usd=200.0,
        raw_stop_loss=99500.0,
        rationale="LVN squeeze market buy",
    )
    _, act, _, details = _intent_to_order_details(intent)
    assert act == "MARKET_BUY"
    assert details["type"] == "market"
    assert details["time_in_force"] == "gtc"
    assert "extended_hours" not in details
