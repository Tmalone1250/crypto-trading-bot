"""Node 4: Executor & Logger for the LangGraph execution engine.

Guards order placement behind ``risk_cleared == True``, submits crypto-compliant
``market`` or ``limit`` orders (with ``gtc`` or ``ioc`` ``time_in_force``) via
``alpaca-mcp-server``'s ``place_crypto_order`` tool, and logs trade records to
local ``DuckDB`` storage (``agent_trades.duckdb``).
"""

import asyncio
from datetime import datetime, timezone
import json
import os
import re
import shlex
from typing import Any
import uuid

from dotenv import load_dotenv
import duckdb
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from state import AgentState

load_dotenv()

DB_PATH: str = "agent_trades.duckdb"
SUPPORTED_CRYPTO_ORDER_TYPES: frozenset[str] = frozenset({"market", "limit", "stop_limit"})
SUPPORTED_CRYPTO_TIF: frozenset[str] = frozenset({"gtc", "ioc"})

_UUID_REGEX = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)


def _is_dry_run() -> bool:
    """Default to paper-trading/dry-run unless explicitly toggled for production."""
    return os.environ.get("TRADING_MODE", "dry_run").lower() != "production"


def _extract_order_id(result: Any) -> str:
    """Extract an order/trade ID from an MCP ``place_crypto_order`` ``CallToolResult``."""
    is_error = bool(
        getattr(result, "is_error", False) or getattr(result, "isError", False)
    )
    content = getattr(result, "content", None) or []

    if is_error:
        error_parts = [
            str(getattr(item, "text", item))
            for item in content
        ]
        raise RuntimeError(" ".join(error_parts) or "Alpaca MCP order placement failed.")

    structured = getattr(result, "structured_content", None)
    if structured is None:
        structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict) and structured.get("id"):
        return str(structured["id"])

    text = "\n".join(
        text_val
        for item in content
        if isinstance((text_val := getattr(item, "text", None)), str)
    ).strip()

    if text:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and parsed.get("id"):
                return str(parsed["id"])
        except json.JSONDecodeError:
            pass

        uuid_match = _UUID_REGEX.search(text)
        if uuid_match:
            return uuid_match.group(0)

    return f"mcp-crypto-{uuid.uuid4().hex[:12]}"


async def _submit_crypto_order_mcp(arguments: dict[str, Any]) -> str:
    """Open an MCP stdio session to ``alpaca-mcp-server`` and invoke ``place_crypto_order``."""
    command = os.environ.get("ALPACA_MCP_COMMAND", "uvx")
    args = shlex.split(os.environ.get("ALPACA_MCP_ARGS", "alpaca-mcp-server"))

    env = dict(os.environ)
    for key_name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
        val = os.environ.get(key_name)
        if val is not None:
            env[key_name] = val
    env.setdefault("ALPACA_PAPER_TRADE", "true" if _is_dry_run() else "false")

    server_params = StdioServerParameters(
        command=command,
        args=args,
        env=env,
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool("place_crypto_order", arguments=arguments)
            return _extract_order_id(result)


def place_crypto_order(order_details: dict[str, Any]) -> str:
    """Submit a crypto order via ``alpaca-mcp-server``'s ``place_crypto_order`` tool.

    Enforces Alpaca crypto constraints:
      - ``order_type`` must be ``"market"``, ``"limit"``, or ``"stop_limit"``
        (equities-only ``"bracket"`` orders are not used).
      - ``time_in_force`` must be ``"gtc"`` or ``"ioc"``.

    Args:
        order_details: Order dictionary populated by Node 3.

    Returns:
        Order/trade ID returned by ``alpaca-mcp-server``.
    """
    symbol = str(order_details["symbol"])
    side = str(order_details.get("side", "buy")).lower()
    order_type = str(order_details.get("order_type", "limit")).lower()
    time_in_force = str(order_details.get("time_in_force", "gtc")).lower()
    qty = float(order_details["position_size"])

    if order_type not in SUPPORTED_CRYPTO_ORDER_TYPES:
        raise ValueError(
            f"Unsupported crypto order_type '{order_type}'. "
            f"Supported: {sorted(SUPPORTED_CRYPTO_ORDER_TYPES)}"
        )

    if time_in_force not in SUPPORTED_CRYPTO_TIF:
        raise ValueError(
            f"Unsupported crypto time_in_force '{time_in_force}'. "
            f"Supported: {sorted(SUPPORTED_CRYPTO_TIF)}"
        )

    mcp_args: dict[str, Any] = {
        "symbol": symbol,
        "side": side,
        "order_type": order_type,
        "time_in_force": time_in_force,
        "qty": qty,
    }
    if order_type in {"limit", "stop_limit"}:
        limit_price = order_details.get("limit_price", order_details.get("entry_price"))
        if limit_price is not None:
            mcp_args["limit_price"] = float(limit_price)

    return asyncio.run(_submit_crypto_order_mcp(mcp_args))


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
    """Execute a risk-cleared crypto order via Alpaca MCP and log to DuckDB.

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
        trade_id = place_crypto_order(order_details)
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
