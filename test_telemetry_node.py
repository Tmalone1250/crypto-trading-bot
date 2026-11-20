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
                   target_price, rationale, scalper_intents, watched_assets
            FROM agent_telemetry
            """
        ).fetchall()
        assert len(rows) == 1
        ts, port_val, regime, action, target_price, rationale, scalper_intents, watched_assets = rows[0]
        assert ts is not None
        assert port_val == pytest.approx(99999.89)
        assert regime == "ACTIVE"
        assert action == "LIMIT_BUY"
        assert target_price == pytest.approx(90400.0)
        assert rationale == "Place resting bid at structural support."
        assert json.loads(scalper_intents) == []
        parsed_watched = json.loads(watched_assets)
        assert parsed_watched == {
            "BTC/USD": 91000.0,
            "ETH/USD": None,
            "SOL/USD": None,
        }
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


def test_telemetry_node_serializes_multi_asset_watched_prices() -> None:
    """Verify ``watched_assets``, ``open_positions``, and ``asset_phases`` serialize cleanly."""
    in_mem_conn = duckdb.connect(":memory:")
    conn_proxy = _PersistentInMemoryConn(in_mem_conn)
    state = _sample_state(
        current_price=91250.5,
        market_data_by_symbol={
            "ETH/USD": pd.DataFrame([{"close": 3125.75}]),
            "SOL/USD": pd.DataFrame([{"close": 185.40}]),
        },
        open_positions=[
            {
                "symbol": "ETHUSD",
                "qty": "0.15",
                "avg_entry_price": "3100.0",
                "unrealized_pl": "3.86",
            }
        ],
        rationale="BTC: HOLD | ETH: PHASE_1_SWEPT | SOL: PHASE_2_CHOCH",
    )

    try:
        with patch("telemetry_node.duckdb.connect", return_value=conn_proxy):
            telemetry_node(state)

        row = in_mem_conn.execute(
            "SELECT watched_assets, open_positions, asset_phases FROM agent_telemetry LIMIT 1"
        ).fetchone()
        assert row is not None
        watched_raw, open_pos_raw, phases_raw = row
        assert json.loads(watched_raw) == {
            "BTC/USD": 91250.5,
            "ETH/USD": 3125.75,
            "SOL/USD": 185.4,
        }
        assert json.loads(open_pos_raw) == [
            {
                "symbol": "ETHUSD",
                "qty": 0.15,
                "avg_entry_price": 3100.0,
                "unrealized_pl": 3.86,
            }
        ]
        assert json.loads(phases_raw) == {
            "BTC/USD": "HOLD",
            "ETH/USD": "PHASE_1_SWEPT",
            "SOL/USD": "PHASE_2_CHOCH",
        }
    finally:
        in_mem_conn.close()


def test_action_ledger_csv_export_endpoint_and_filters(tmp_path) -> None:
    """Verify ``action_log`` writes and ``GET /api/export/actions.csv`` schema + query filters."""
    import csv
    import io
    import urllib.request

    from action_ledger import ACTION_LOG_COLUMNS, log_action
    from telemetry_api import start_telemetry_server

    db_file = str(tmp_path / "test_actions.duckdb")
    log_action(
        action="LIMIT_BUY",
        symbol="BTC/USD",
        asset_id="ord-buy-1",
        qty=0.002,
        notional=200.0,
        price_or_value=100000.0,
        strategy="FLAT",
        reason="Lower-20% range buy",
        mode="paper",
        db_path=db_file,
    )
    log_action(
        action="MARKET_SELL",
        symbol="SOL/USD",
        asset_id="ord-sell-2",
        qty=2.0,
        notional=191.0,
        price_or_value=95.5,
        pnl_amount=-9.0,
        pnl_pct=-4.5,
        strategy="GRID_STOP",
        reason="Grid safety stop-loss triggered (-4.0%)",
        mode="paper",
        db_path=db_file,
    )
    log_action(
        action="RISK_BLOCK",
        symbol="ETH/USD",
        price_or_value=3000.0,
        strategy="RISK_MANAGER",
        reason="Risk veto: high volatility",
        mode="paper",
        db_path=db_file,
    )

    server, _ = start_telemetry_server(host="127.0.0.1", port=0, db_path=db_file)
    port = server.server_address[1]
    try:
        # 1. Unfiltered export returns all 3 real actions with exact 14-column header
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/export/actions.csv?limit=1000", timeout=5
        ) as resp:
            assert resp.status == 200
            assert resp.headers.get("Content-Type", "").startswith("text/csv")
            assert 'filename="alpaca_actions.csv"' in resp.headers.get("Content-Disposition", "")
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"
            body = resp.read().decode("utf-8")

        rows = list(csv.reader(io.StringIO(body)))
        assert tuple(rows[0]) == ACTION_LOG_COLUMNS
        assert len(rows) == 4  # header + 3 actions
        assert rows[1][1] == "alpaca_crypto"
        assert rows[1][8] == "USD"

        # 2. Filtered export (?actions=BUY,SELL) excludes RISK_BLOCK and keeps LIMIT_BUY & MARKET_SELL
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/export/actions.csv?actions=BUY,SELL&limit=10",
            timeout=5,
        ) as resp_filtered:
            assert resp_filtered.status == 200
            filtered_rows = list(csv.reader(io.StringIO(resp_filtered.read().decode("utf-8"))))

        assert tuple(filtered_rows[0]) == ACTION_LOG_COLUMNS
        assert len(filtered_rows) == 3  # header + LIMIT_BUY + MARKET_SELL
        exported_actions = {r[3] for r in filtered_rows[1:]}
        assert exported_actions == {"LIMIT_BUY", "MARKET_SELL"}
    finally:
        server.shutdown()
        server.server_close()
