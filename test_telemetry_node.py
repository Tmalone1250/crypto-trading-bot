"""Unit tests for the Telemetry Logger Node (DuckDB ``agent_telemetry``)."""

import json
from unittest.mock import patch
import duckdb
import pandas as pd
import pytest

from state import AgentState, TradeIntent
from telemetry_node import telemetry_node


class _PersistentInMemoryConn:
    """Proxy around an in-memory DuckDB connection that ignores ``close()`` so tests can inspect rows."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn

    def execute(self, *args, **kwargs):
        return self._conn.execute(*args, **kwargs)

    def close(self) -> None:
        pass


def _sample_state(**overrides) -> AgentState:
    """Build a valid ``AgentState`` for telemetry logging tests."""
    base: AgentState = {
        "asset_symbol": "BTC/USD",
        "asset_symbols": ["BTC/USD", "ETH/USD", "SOL/USD"],
        "market_data_by_symbol": {},
        "current_price": 91000.0,
        "historical_data": pd.DataFrame(),
        "sentiment_score": 0.42,
        "risk_cleared": False,
        "order_details": {},
        "error_message": None,
        "account_data": {"portfolio_value": "99999.89", "buying_power": "399916.80"},
        "open_positions": [],
        "active_orders": [],
        "action": "LIMIT_BUY",
        "target_price": 90400.0,
        "rationale": "Place resting bid at structural support.",
        "market_regime": "ACTIVE",
        "realized_pnl": 0.0,
        "trade_intents": [],
        "last_alpha_eval_price": 0.0,
        "mcp_client": None,
    }
    base.update(overrides)
    return base


def test_telemetry_node_creates_table_and_inserts_snapshot() -> None:
    """Verify ``telemetry_node`` creates ``agent_telemetry`` and inserts the cycle row."""
    in_mem_conn = duckdb.connect(":memory:")
    conn_proxy = _PersistentInMemoryConn(in_mem_conn)
    state = _sample_state()

    try:
        with patch("telemetry_node.duckdb.connect", return_value=conn_proxy) as mock_connect:
            returned_state = telemetry_node(state)

        mock_connect.assert_called_once()
        assert returned_state == state

        rows = in_mem_conn.execute(
            """
            SELECT timestamp, portfolio_value, market_regime, action,
                   target_price, rationale, scalper_intents
            FROM agent_telemetry
            """
        ).fetchall()
        assert len(rows) == 1
        ts, port_val, regime, action, target_price, rationale, scalper_intents = rows[0]
        assert ts is not None
        assert port_val == pytest.approx(99999.89)
        assert regime == "ACTIVE"
        assert action == "LIMIT_BUY"
        assert target_price == pytest.approx(90400.0)
        assert rationale == "Place resting bid at structural support."
        assert json.loads(scalper_intents) == []
    finally:
        in_mem_conn.close()


def test_telemetry_node_handles_hold_with_none_target_price() -> None:
    """Verify ``telemetry_node`` logs ``HOLD`` cycles with ``target_price=None`` cleanly."""
    in_mem_conn = duckdb.connect(":memory:")
    conn_proxy = _PersistentInMemoryConn(in_mem_conn)
    state = _sample_state(
        account_data={},
        market_regime="FLAT",
        action="HOLD",
        target_price=None,
        rationale="No setup triggered.",
    )

    try:
        with patch("telemetry_node.duckdb.connect", return_value=conn_proxy):
            telemetry_node(state)

        rows = in_mem_conn.execute(
            """
            SELECT portfolio_value, market_regime, action, target_price, rationale, scalper_intents
            FROM agent_telemetry
            """
        ).fetchall()
        assert len(rows) == 1
        port_val, regime, action, target_price, rationale, scalper_intents = rows[0]
        assert port_val is None
        assert regime == "FLAT"
        assert action == "HOLD"
        assert target_price is None
        assert rationale == "No setup triggered."
        assert json.loads(scalper_intents) == []
    finally:
        in_mem_conn.close()


def test_telemetry_node_serializes_trade_intents_and_overrides_flat_scalp_headline() -> None:
    """Verify ``trade_intents`` serialize to JSON, ``SCALP_ACTIVE`` overrides headline, and ALTER TABLE upgrades legacy schema."""
    in_mem_conn = duckdb.connect(":memory:")
    # Simulate an older agent_telemetry table missing the scalper_intents column.
    in_mem_conn.execute(
        """
        CREATE TABLE agent_telemetry (
            timestamp TIMESTAMP NOT NULL,
            portfolio_value DOUBLE,
            market_regime VARCHAR,
            action VARCHAR,
            target_price DOUBLE,
            rationale VARCHAR
        )
        """
    )
    conn_proxy = _PersistentInMemoryConn(in_mem_conn)

    intents = [
        TradeIntent(
            symbol="BTC/USD",
            action="LIMIT_BUY",
            target_price=90020.0,
            qty=0.002222,
            notional_usd=200.0,
            rationale="Lower-20% range buy",
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
    state = _sample_state(
        market_regime="FLAT",
        action="LIMIT_BUY",
        target_price=90020.0,
        rationale="Lower-20% range buy",
        trade_intents=intents,
    )

    try:
        with patch("telemetry_node.duckdb.connect", return_value=conn_proxy):
            telemetry_node(state)

        rows = in_mem_conn.execute(
            "SELECT market_regime, action, rationale, scalper_intents FROM agent_telemetry"
        ).fetchall()
        assert len(rows) == 1
        regime, action, rationale, scalper_intents = rows[0]
        assert regime == "FLAT"
        assert action == "SCALP_ACTIVE"
        assert rationale == "Scalper executing 2 grid actions."

        parsed_intents = json.loads(scalper_intents)
        assert len(parsed_intents) == 2
        assert parsed_intents[0]["symbol"] == "BTC/USD"
        assert parsed_intents[0]["action"] == "LIMIT_BUY"
        assert parsed_intents[1]["symbol"] == "SOL/USD"
        assert parsed_intents[1]["action"] == "LIMIT_SELL"
    finally:
        in_mem_conn.close()
