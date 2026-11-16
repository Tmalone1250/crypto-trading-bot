"""Node 4: Executor & Logger for the LangGraph execution engine.

Guards execution behind ``risk_cleared == True``, routes the LLM's lifecycle
``action`` (``MARKET_BUY``, ``LIMIT_BUY``, or ``CANCEL_ALL``) to the appropriate
``alpaca-mcp-server`` MCP tool (``place_crypto_order`` or ``cancel_all_orders``),
ensures all quantities and limit prices are explicitly cast to ``str``, and logs
executed trades to local ``DuckDB`` storage (``agent_trades.duckdb``).
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
CANCEL_ALL_TOOL_NAME: str = "cancel_all_orders"
SUPPORTED_CRYPTO_ORDER_TYPES: frozenset[str] = frozenset({"market", "limit", "stop_limit"})
SUPPORTED_CRYPTO_TIF: frozenset[str] = frozenset({"gtc", "ioc"})

_UUID_REGEX = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)


def _is_dry_run() -> bool:
    """Default to paper-trading/dry-run unless explicitly toggled for production."""
    return os.environ.get("TRADING_MODE", "dry_run").lower() != "production"


def _build_server_params() -> StdioServerParameters:
    """Construct ``StdioServerParameters`` for ``alpaca-mcp-server``."""
    command = os.environ.get("ALPACA_MCP_COMMAND", "uvx")
    args = shlex.split(os.environ.get("ALPACA_MCP_ARGS", "alpaca-mcp-server"))

    env = dict(os.environ)
    for key_name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
        val = os.environ.get(key_name)
        if val is not None:
            env[key_name] = val
    env.setdefault("ALPACA_PAPER_TRADE", "true" if _is_dry_run() else "false")

    return StdioServerParameters(
        command=command,
        args=args,
        env=env,
    )


def _extract_order_id(result: Any) -> str:
    """Extract an order/trade ID from an MCP ``CallToolResult`` (including FastMCP envelopes)."""
    is_error = bool(
        getattr(result, "is_error", False) or getattr(result, "isError", False)
    )
    content = getattr(result, "content", None) or []

    if is_error:
        error_parts = [
            str(getattr(item, "text", item))
            for item in content
        ]
        raise RuntimeError(" ".join(error_parts) or "Alpaca MCP order execution failed.")

    structured = getattr(result, "structured_content", None)
    if structured is None:
        structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        if structured.get("id"):
            return str(structured["id"])
        data_sec = structured.get("data")
        if isinstance(data_sec, dict) and data_sec.get("id"):
            return str(data_sec["id"])

    text = "\n".join(
        text_val
        for item in content
        if isinstance((text_val := getattr(item, "text", None)), str)
    ).strip()

    if text:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                if parsed.get("id"):
                    return str(parsed["id"])
                data_sec = parsed.get("data")
                if isinstance(data_sec, dict) and data_sec.get("id"):
                    return str(data_sec["id"])
        except json.JSONDecodeError:
            pass

        uuid_match = _UUID_REGEX.search(text)
        if uuid_match:
            return uuid_match.group(0)

    return f"mcp-crypto-{uuid.uuid4().hex[:12]}"


async def _submit_crypto_order_mcp(arguments: dict[str, Any]) -> str:
    """Open an MCP stdio session to ``alpaca-mcp-server`` and invoke ``place_crypto_order``."""
    server_params = _build_server_params()
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool("place_crypto_order", arguments=arguments)
            return _extract_order_id(result)


async def _cancel_all_orders_mcp() -> Any:
    """Open an MCP stdio session to ``alpaca-mcp-server`` and invoke ``cancel_all_orders``."""
    server_params = _build_server_params()
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(CANCEL_ALL_TOOL_NAME, arguments={})
            is_error = bool(
                getattr(result, "is_error", False) or getattr(result, "isError", False)
            )
            content = getattr(result, "content", None) or []
            if is_error:
                error_parts = [str(getattr(item, "text", item)) for item in content]
                raise RuntimeError(
                    " ".join(error_parts) or "Alpaca MCP cancel_all_orders failed."
                )
            return _extract_order_id(result)


def cancel_all_orders() -> str:
    """Cancel all open orders via ``alpaca-mcp-server``'s ``cancel_all_orders`` tool."""
    return str(asyncio.run(_cancel_all_orders_mcp()))


def cancel_orders() -> str:
    """Alias for ``cancel_all_orders``."""
    return cancel_all_orders()


def place_crypto_order(order_details: dict[str, Any]) -> str:
    """Submit a crypto order via ``alpaca-mcp-server``'s ``place_crypto_order`` tool.

    Enforces Alpaca crypto constraints:
      - ``type`` / ``order_type`` must be ``"market"``, ``"limit"``, or ``"stop_limit"``.
      - ``time_in_force`` must be ``"gtc"`` or ``"ioc"``.
      - ``qty`` and ``limit_price`` are explicitly cast to ``str``.

    Args:
        order_details: Order dictionary populated by Node 3 / Node 4.

    Returns:
        Order/trade ID returned by ``alpaca-mcp-server``.
    """
    symbol = str(order_details["symbol"])
    side = str(order_details.get("side", "buy")).lower()
    order_type = str(
        order_details.get("type", order_details.get("order_type", "limit"))
    ).lower()
    time_in_force = str(order_details.get("time_in_force", "gtc")).lower()
    raw_qty = order_details.get(
        "trade_qty", order_details.get("qty", order_details["position_size"])
    )
    qty_str = str(raw_qty)

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
        "type": order_type,
        "time_in_force": time_in_force,
        "qty": qty_str,
    }
    if order_type in {"limit", "stop_limit"}:
        limit_price = order_details.get("limit_price", order_details.get("entry_price"))
        if limit_price is not None:
            mcp_args["limit_price"] = str(limit_price)

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


def _calculate_sell_profit(
    state: AgentState,
    order_details: dict[str, Any],
    qty: float,
    symbol: str | None = None,
) -> float:
    """Estimate rough realized profit in USD for an executed sell order."""
    resolved_symbol = symbol or state["asset_symbol"]
    target_symbol = resolved_symbol.replace("/", "").upper()
    avg_entry = 0.0
    for pos in state.get("open_positions") or []:
        pos_sym = str(pos.get("symbol", "")).replace("/", "").upper()
        if pos_sym == target_symbol:
            try:
                avg_entry = float(pos.get("avg_entry_price", 0.0))
            except (TypeError, ValueError):
                avg_entry = 0.0
            break

    if avg_entry <= 0.0:
        try:
            avg_entry = float(order_details.get("entry_price", 0.0))
        except (TypeError, ValueError):
            avg_entry = 0.0

    raw_sell_price = order_details.get("limit_price", state.get("current_price", 0.0))
    try:
        sell_price = float(raw_sell_price)
    except (TypeError, ValueError):
        sell_price = float(state.get("current_price", 0.0))

    if avg_entry <= 0.0 or sell_price <= 0.0 or qty <= 0.0:
        return 0.0
    return round((sell_price - avg_entry) * qty, 6)


def _intent_to_order_details(intent: Any) -> tuple[str, str, float, dict[str, Any]]:
    """Normalize a ``TradeIntent`` (or dict) into ``(symbol, action, qty, order_details)``."""
    if isinstance(intent, dict):
        sym = str(intent["symbol"])
        act = str(intent["action"]).upper()
        target_price = float(intent["target_price"])
        qty = float(intent["qty"])
    else:
        sym = str(intent.symbol)
        act = str(intent.action).upper()
        target_price = float(intent.target_price)
        qty = float(intent.qty)

    side = "sell" if act == "LIMIT_SELL" else "buy"
    details: dict[str, Any] = {
        "symbol": sym,
        "side": side,
        "type": "limit",
        "order_type": "limit",
        "time_in_force": "gtc",
        "qty": str(qty),
        "limit_price": str(target_price),
        "position_size": qty,
        "stop_loss_price": 0.0,
        "target_price": target_price,
        "dry_run": _is_dry_run(),
    }
    return sym, act, qty, details


def execution_node(state: AgentState) -> AgentState:
    """Execute risk-cleared trade intents or single lifecycle actions via Alpaca MCP.

    Args:
        state: Current ``AgentState`` dictionary.

    Returns:
        Updated ``AgentState`` dictionary with ``order_details`` and ``realized_pnl``
        updated on success, or the original state if ``risk_cleared`` is False.
    """
    if state["risk_cleared"] is not True:
        return state

    symbol = state["asset_symbol"]
    action = str(state.get("action", "LIMIT_BUY")).upper()
    order_details = dict(state["order_details"])
    trade_intents = state.get("trade_intents") or []

    try:
        if trade_intents:
            realized_pnl = float(state.get("realized_pnl", 0.0))
            executed_intents: list[dict[str, Any]] = []
            last_trade_id = ""

            for intent in trade_intents:
                intent_sym, intent_act, intent_qty, intent_details = _intent_to_order_details(intent)
                trade_id = place_crypto_order(intent_details)
                last_trade_id = trade_id
                timestamp = datetime.now(timezone.utc).replace(tzinfo=None)

                _log_trade_to_duckdb(
                    trade_id=trade_id,
                    symbol=intent_sym,
                    size=intent_qty,
                    stop_loss=float(intent_details.get("stop_loss_price", 0.0)),
                    target=float(intent_details.get("target_price", 0.0)),
                    timestamp=timestamp,
                )

                if intent_act == "LIMIT_SELL":
                    rough_profit = _calculate_sell_profit(
                        state=state,
                        order_details=intent_details,
                        qty=intent_qty,
                        symbol=intent_sym,
                    )
                    realized_pnl = round(realized_pnl + rough_profit, 6)

                executed_intents.append(
                    {
                        **intent_details,
                        "action": intent_act,
                        "trade_id": trade_id,
                    }
                )

            return {
                **state,
                "order_details": {
                    **order_details,
                    "trade_id": last_trade_id,
                    "executed_intents": executed_intents,
                },
                "realized_pnl": realized_pnl,
                "error_message": None,
            }
        if action == "CANCEL_ALL":
            cancel_id = cancel_all_orders()
            updated_order_details = {
                **order_details,
                "action": "CANCEL_ALL",
                "trade_id": cancel_id,
            }
            return {
                **state,
                "order_details": updated_order_details,
                "error_message": None,
            }

        raw_qty = order_details.get(
            "trade_qty", order_details.get("qty", order_details.get("position_size", 0.0))
        )
        order_details["qty"] = str(raw_qty)

        if action == "MARKET_BUY":
            order_details["side"] = "buy"
            order_details["type"] = "market"
            order_details["order_type"] = "market"
            order_details.pop("limit_price", None)
        elif action == "LIMIT_BUY":
            order_details["side"] = "buy"
            order_details["type"] = "limit"
            order_details["order_type"] = "limit"
            target_limit = (
                state.get("target_price")
                if state.get("target_price") is not None
                else order_details.get("limit_price", state["current_price"])
            )
            order_details["limit_price"] = str(target_limit)
        elif action == "MARKET_SELL":
            order_details["side"] = "sell"
            order_details["type"] = "market"
            order_details["order_type"] = "market"
            order_details.pop("limit_price", None)
        elif action == "LIMIT_SELL":
            order_details["side"] = "sell"
            order_details["type"] = "limit"
            order_details["order_type"] = "limit"
            target_limit = (
                state.get("target_price")
                if state.get("target_price") is not None
                else order_details.get("limit_price", state["current_price"])
            )
            order_details["limit_price"] = str(target_limit)
        else:
            order_details.setdefault("side", "buy")
            if order_details.get("limit_price") is not None:
                order_details["limit_price"] = str(order_details["limit_price"])

        trade_id = place_crypto_order(order_details)
        timestamp = datetime.now(timezone.utc).replace(tzinfo=None)

        _log_trade_to_duckdb(
            trade_id=trade_id,
            symbol=symbol,
            size=float(order_details.get("position_size", raw_qty)),
            stop_loss=float(order_details.get("stop_loss_price", 0.0)),
            target=float(order_details.get("target_price", 0.0)),
            timestamp=timestamp,
        )

        updated_order_details = {
            **order_details,
            "action": action,
            "trade_id": trade_id,
        }

        realized_pnl = float(state.get("realized_pnl", 0.0))
        if action in {"LIMIT_SELL", "MARKET_SELL"}:
            rough_profit = _calculate_sell_profit(
                state=state,
                order_details=order_details,
                qty=float(order_details.get("position_size", raw_qty)),
            )
            realized_pnl = round(realized_pnl + rough_profit, 6)

        return {
            **state,
            "order_details": updated_order_details,
            "realized_pnl": realized_pnl,
            "error_message": None,
        }
    except Exception as exc:
        return {
            **state,
            "error_message": f"Execution node error for {symbol}: {exc}",
        }
