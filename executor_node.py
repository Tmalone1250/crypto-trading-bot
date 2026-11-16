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
import time
from typing import Any
import uuid

from dotenv import load_dotenv
import duckdb
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import get_alpaca_paper_config
from state import AgentState

load_dotenv()

DB_PATH: str = "agent_trades.duckdb"
CANCEL_ALL_TOOL_NAME: str = "cancel_all_orders"
BREAKEVEN_MOMENTUM_THRESHOLD: float = 0.0015  # +0.15% unrealized PnL trigger
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
    paper_cfg = get_alpaca_paper_config()
    command = os.environ.get("ALPACA_MCP_COMMAND", "uvx")
    args = shlex.split(os.environ.get("ALPACA_MCP_ARGS", "alpaca-mcp-server"))

    env = dict(os.environ)
    if paper_cfg["api_key"]:
        env["ALPACA_API_KEY"] = paper_cfg["api_key"]
        env["APCA_API_KEY_ID"] = paper_cfg["api_key"]
    if paper_cfg["secret_key"]:
        env["ALPACA_SECRET_KEY"] = paper_cfg["secret_key"]
        env["APCA_API_SECRET_KEY"] = paper_cfg["secret_key"]
    env.setdefault("APCA_API_BASE_URL", paper_cfg["base_url"])
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


def cancel_all_orders(mcp_client: Any | None = None) -> str:
    """Cancel all open orders via ``alpaca-mcp-server``'s ``cancel_all_orders`` tool."""
    if mcp_client is not None:
        result = mcp_client.call_tool(CANCEL_ALL_TOOL_NAME, arguments={})
        is_error = bool(
            getattr(result, "is_error", False) or getattr(result, "isError", False)
        )
        content = getattr(result, "content", None) or []
        if is_error:
            error_parts = [str(getattr(item, "text", item)) for item in content]
            raise RuntimeError(
                " ".join(error_parts) or "Alpaca MCP cancel_all_orders failed."
            )
        return str(_extract_order_id(result))
    return str(asyncio.run(_cancel_all_orders_mcp()))


def cancel_orders(mcp_client: Any | None = None) -> str:
    """Alias for ``cancel_all_orders``."""
    return cancel_all_orders(mcp_client=mcp_client)


def place_crypto_order(
    order_details: dict[str, Any],
    mcp_client: Any | None = None,
) -> str:
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

    if mcp_client is not None:
        result = mcp_client.call_tool("place_crypto_order", arguments=mcp_args)
        return _extract_order_id(result)

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

    side = "sell" if act in {"LIMIT_SELL", "MARKET_SELL"} else "buy"
    order_type = "market" if act.startswith("MARKET_") else "limit"
    details: dict[str, Any] = {
        "symbol": sym,
        "side": side,
        "type": order_type,
        "order_type": order_type,
        "time_in_force": "gtc",
        "qty": str(qty),
        "position_size": qty,
        "stop_loss_price": 0.0,
        "target_price": target_price,
        "dry_run": _is_dry_run(),
    }
    if order_type == "limit":
        details["limit_price"] = str(target_price)
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
    mcp_client = state.get("mcp_client")

    try:
        if trade_intents:
            realized_pnl = float(state.get("realized_pnl", 0.0))
            executed_intents: list[dict[str, Any]] = []
            last_trade_id = ""

            for intent in trade_intents:
                intent_sym, intent_act, intent_qty, intent_details = _intent_to_order_details(intent)
                trade_id = (
                    place_crypto_order(intent_details, mcp_client=mcp_client)
                    if mcp_client is not None
                    else place_crypto_order(intent_details)
                )
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

                if intent_act in {"LIMIT_SELL", "MARKET_SELL"}:
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
            cancel_id = (
                cancel_all_orders(mcp_client=mcp_client)
                if mcp_client is not None
                else cancel_all_orders()
            )
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

        trade_id = (
            place_crypto_order(order_details, mcp_client=mcp_client)
            if mcp_client is not None
            else place_crypto_order(order_details)
        )
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


def _extract_price_from_quote(payload: Any) -> float | None:
    """Extract a numeric BTC/USD price from an Alpaca SDK or MCP quote payload."""
    if isinstance(payload, (int, float)):
        return float(payload)

    if hasattr(payload, "content") or hasattr(payload, "structured_content") or hasattr(payload, "structuredContent"):
        structured = getattr(payload, "structured_content", None) or getattr(payload, "structuredContent", None)
        if isinstance(structured, dict):
            extracted = _extract_price_from_quote(structured)
            if extracted is not None:
                return extracted
        content = getattr(payload, "content", None) or []
        for item in content:
            text_val = getattr(item, "text", None)
            if isinstance(text_val, str) and text_val.strip():
                try:
                    parsed = json.loads(text_val)
                    extracted = _extract_price_from_quote(parsed)
                    if extracted is not None:
                        return extracted
                except json.JSONDecodeError:
                    pass

    if isinstance(payload, dict):
        for key in ("BTC/USD", "BTCUSD", "data", "quote", "trade"):
            if key in payload:
                nested = _extract_price_from_quote(payload[key])
                if nested is not None:
                    return nested
        for price_key in ("price", "p", "last_price", "close"):
            if payload.get(price_key) is not None:
                return float(payload[price_key])
        ask = payload.get("ask_price", payload.get("ap"))
        bid = payload.get("bid_price", payload.get("bp"))
        if ask is not None and bid is not None:
            return (float(ask) + float(bid)) / 2.0
        if ask is not None:
            return float(ask)
        if bid is not None:
            return float(bid)
        return None

    for attr in ("price", "last_price"):
        val = getattr(payload, attr, None)
        if val is not None:
            return float(val)
    ask_attr = getattr(payload, "ask_price", None)
    bid_attr = getattr(payload, "bid_price", None)
    if ask_attr is not None and bid_attr is not None:
        return (float(ask_attr) + float(bid_attr)) / 2.0
    if ask_attr is not None:
        return float(ask_attr)
    if bid_attr is not None:
        return float(bid_attr)
    return None


def _is_unconfigured_mock(attr: Any) -> bool:
    """Return True when ``attr`` is an unconfigured ``unittest.mock`` callable."""
    return (
        hasattr(attr, "assert_called")
        and getattr(attr, "side_effect", None) is None
        and hasattr(attr, "return_value")
        and hasattr(attr.return_value, "assert_called")
    )


def _fetch_live_btc_price(client: Any) -> float:
    """Fetch the current live BTC/USD price from an Alpaca MCP or SDK client."""
    price: float | None = None
    if hasattr(client, "get_latest_trade") and not _is_unconfigured_mock(client.get_latest_trade):
        price = _extract_price_from_quote(client.get_latest_trade("BTC/USD"))
    elif hasattr(client, "get_crypto_latest_quote") and not _is_unconfigured_mock(
        client.get_crypto_latest_quote
    ):
        price = _extract_price_from_quote(client.get_crypto_latest_quote("BTC/USD"))
    elif hasattr(client, "get_latest_price") and not _is_unconfigured_mock(client.get_latest_price):
        price = _extract_price_from_quote(client.get_latest_price("BTC/USD"))
    elif hasattr(client, "call_tool") and not _is_unconfigured_mock(client.call_tool):
        result = client.call_tool("get_crypto_latest_quote", {"symbols": "BTC/USD"})
        price = _extract_price_from_quote(result)
    else:
        raise AttributeError(
            "Unsupported Alpaca client: expected get_latest_trade, "
            "get_crypto_latest_quote, get_latest_price, or call_tool."
        )

    if price is None or price <= 0.0:
        raise ValueError("Unable to parse a positive live BTC/USD price from Alpaca client.")
    return float(price)


def _replace_stop_loss_order(client: Any, order_id: str, entry_price: float) -> Any:
    """Invoke Alpaca replace_order (SDK or MCP) to move stop_loss to ``entry_price``."""
    stop_price_str = str(entry_price)
    if hasattr(client, "replace_order"):
        return client.replace_order(order_id=str(order_id), stop_price=stop_price_str)
    if hasattr(client, "replace_order_by_id"):
        return client.replace_order_by_id(
            order_id=str(order_id),
            order_data={"stop_price": stop_price_str},
        )
    if hasattr(client, "call_tool"):
        return client.call_tool(
            "replace_order_by_id",
            {"order_id": str(order_id), "stop_price": stop_price_str},
        )
    raise AttributeError(
        "Unsupported Alpaca client: expected replace_order, replace_order_by_id, or call_tool."
    )


def monitor_trade_and_breakeven(
    client: Any,
    order_id: str,
    entry_price: float,
    is_long: bool,
    poll_interval: float = 1.0,
    max_attempts: int = 60,
) -> dict[str, Any]:
    """Poll live BTC/USD every 1s and move stop-loss to breakeven at +0.15% PnL.

    Args:
        client: Active Alpaca MCP or SDK client instance.
        order_id: Alpaca stop-loss order ID to update via ``replace_order``.
        entry_price: Executed trade entry price used for PnL and breakeven SL.
        is_long: ``True`` for long positions, ``False`` for short positions.
        poll_interval: Seconds between price polls (defaults to ``1.0``).
        max_attempts: Maximum polling iterations before exiting (defaults to ``60``).

    Returns:
        Summary dictionary with ``breakeven_triggered``, ``order_id``,
        ``entry_price``, and trigger metrics when triggered.
    """
    entry = float(entry_price)
    if entry <= 0.0:
        raise ValueError("entry_price must be positive.")

    attempts = 0
    while attempts < max_attempts:
        live_price = _fetch_live_btc_price(client)
        unrealized_pnl_pct = (
            (live_price - entry) / entry
            if is_long
            else (entry - live_price) / entry
        )

        if unrealized_pnl_pct >= BREAKEVEN_MOMENTUM_THRESHOLD:
            replace_result = _replace_stop_loss_order(client, str(order_id), entry)
            print(
                f"[Executor] Breakeven triggered for {order_id}: "
                f"PnL={unrealized_pnl_pct:.4%} >= {BREAKEVEN_MOMENTUM_THRESHOLD:.2%}, "
                f"moved SL to {entry:.2f}"
            )
            return {
                "breakeven_triggered": True,
                "order_id": str(order_id),
                "entry_price": entry,
                "trigger_price": live_price,
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "replace_result": replace_result,
            }

        attempts += 1
        if attempts < max_attempts:
            time.sleep(poll_interval)

    return {
        "breakeven_triggered": False,
        "order_id": str(order_id),
        "entry_price": entry,
        "attempts": attempts,
    }
