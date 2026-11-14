"""Node 4: Executor & Logger for the LangGraph execution engine.

Guards order placement behind ``risk_cleared == True``, delegates bracket order
execution to the Alpaca MCP toolset (simulated here via a placeholder helper),
and persists trade records to local ``DuckDB`` storage (``agent_trades.duckdb``).
"""

from datetime import datetime, timezone
import os
from typing import Any
import uuid
import duckdb

from state import AgentState

DB_PATH: str = "agent_trades.duckdb"


def _is_dry_run() -> bool:
    """Default to paper-trading/dry-run unless explicitly toggled for production."""
    return os.environ.get("TRADING_MODE", "dry_run").lower() != "production"


def execute_mcp_bracket_order(order_details: dict[str, Any]) -> str:
    """Placeholder helper simulating bracket order submission via ``alpaca-mcp-server``.

    Never uses raw ``requests`` or ``aiohttp`` calls; all exchange execution
    must flow through the Alpaca MCP toolset.

    Args:
        order_details: Bracket order payload containing ``symbol``, ``side``,
            ``entry_price``, ``stop_loss_price``, ``target_price``, and ``position_size``.

    Returns:
        Simulated order/trade identifier string.
    """
    # TODO: Wire to alpaca-mcp-server place_order / place_crypto_order tool.
    prefix = "paper-dryrun" if _is_dry_run() else "live-mcp"
    _ = order_details
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _log_trade_to_duckdb(
    trade_id: str,
    symbol: str,
    size: float,
    stop_loss: float,
    target: float,
    timestamp: datetime,
) -> None:
    """Ensure the ``trades`` table exists in DuckDB and insert a trade record."""
    conn = duckdb.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                trade_id VARCHAR PRIMARY KEY,
                symbol VARCHAR NOT NULL,
                size DOUBLE NOT NULL,
                stop_loss DOUBLE NOT NULL,
                target DOUBLE NOT NULL,
                timestamp TIMESTAMP NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO trades (trade_id, symbol, size, stop_loss, target, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [trade_id, symbol, size, stop_loss, target, timestamp],
        )
    finally:
        conn.close()


def execution_node(state: AgentState) -> AgentState:
    """Execute a risk-cleared bracket order via Alpaca MCP and log to DuckDB.

    Args:
        state: Current ``AgentState`` dictionary.

    Returns:
        Updated ``AgentState`` dictionary with ``order_details["trade_id"]``
        populated on success, or the original state if ``risk_cleared`` is False.
    """
    if state["risk_cleared"] is not True:
        return state

    symbol = state["asset_symbol"]
    order_details = dict(state["order_details"])

    try:
        trade_id = execute_mcp_bracket_order(order_details)
        timestamp = datetime.now(timezone.utc).replace(tzinfo=None)

        _log_trade_to_duckdb(
            trade_id=trade_id,
            symbol=symbol,
            size=float(order_details["position_size"]),
            stop_loss=float(order_details["stop_loss_price"]),
            target=float(order_details["target_price"]),
            timestamp=timestamp,
        )

        updated_order_details = {
            **order_details,
            "trade_id": trade_id,
        }

        return {
            **state,
            "order_details": updated_order_details,
            "error_message": None,
        }
    except Exception as exc:
        return {
            **state,
            "error_message": f"Execution node error for {symbol}: {exc}",
        }
