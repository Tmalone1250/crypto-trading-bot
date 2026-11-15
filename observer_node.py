"""Node 1: Market Observer node for the LangGraph execution engine.

Connects to ``alpaca-mcp-server`` over MCP stdio transport to invoke
``get_crypto_bars`` for the latest 250 hourly OHLCV candles, normalizes numeric
types to ``float64``, and gates execution through ``strategy_filter.check_entry_signal``
to preserve Node 2's 15 RPM Gemini free-tier rate limit.
"""

import asyncio
import json
import os
import shlex
from typing import Any

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import numpy as np
import pandas as pd

from state import AgentState
from strategy_filter import check_entry_signal

load_dotenv()

BAR_TIMEFRAME: str = "1Hour"
BAR_LIMIT: int = 250
OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


def _extract_mcp_result(result: Any) -> Any:
    """Extract structured or JSON/text payload from an MCP ``CallToolResult``."""
    if getattr(result, "isError", False):
        error_parts = [
            getattr(item, "text", str(item))
            for item in (getattr(result, "content", None) or [])
        ]
        raise RuntimeError(" ".join(error_parts) or "Alpaca MCP tool returned an error.")

    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured

    content = getattr(result, "content", None) or []
    text_chunks = [
        item.text for item in content if getattr(item, "text", None) is not None
    ]
    if len(text_chunks) == 1:
        return text_chunks[0]
    if text_chunks:
        return "\n".join(text_chunks)
    return []


async def _fetch_crypto_bars_mcp(
    symbol: str,
    timeframe: str,
    limit: int,
) -> Any:
    """Open an MCP stdio session to ``alpaca-mcp-server`` and call ``get_crypto_bars``."""
    command = os.environ.get("ALPACA_MCP_COMMAND", "uvx")
    args = shlex.split(os.environ.get("ALPACA_MCP_ARGS", "alpaca-mcp-server"))

    env = dict(os.environ)
    for key_name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
        val = os.environ.get(key_name)
        if val is not None:
            env[key_name] = val

    server_params = StdioServerParameters(
        command=command,
        args=args,
        env=env,
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(
                "get_crypto_bars",
                arguments={
                    "symbols": symbol,
                    "timeframe": timeframe,
                    "limit": limit,
                },
            )
            return _extract_mcp_result(result)


def get_crypto_bars(
    symbol: str,
    timeframe: str = BAR_TIMEFRAME,
    limit: int = BAR_LIMIT,
) -> Any:
    """Invoke the ``alpaca-mcp-server`` ``get_crypto_bars`` tool via MCP stdio transport.

    All exchange market data requests flow through ``alpaca-mcp-server`` rather
    than raw ``requests`` or ``aiohttp`` HTTP calls.

    Args:
        symbol: Target crypto pair (e.g., ``"BTC/USD"``).
        timeframe: Bar interval string (default ``"1Hour"``).
        limit: Number of historical bars to retrieve (default ``250``).

    Returns:
        Raw bar payload from the Alpaca MCP tool.
    """
    return asyncio.run(_fetch_crypto_bars_mcp(symbol, timeframe, limit))


def _symbol_key_candidates(symbol: str) -> list[str]:
    """Generate ordered symbol key variants (e.g., ``BTC/USD`` and ``BTCUSD``)."""
    candidates = [symbol, symbol.replace("/", ""), symbol.upper(), symbol.replace("/", "").upper()]
    if "/" not in symbol and len(symbol) > 3:
        candidates.append(f"{symbol[:-3]}/{symbol[-3:]}")
        candidates.append(f"{symbol[:-3]}/{symbol[-3:]}".upper())
    # Preserve order while deduplicating.
    return list(dict.fromkeys(candidates))


def _extract_records_from_dict(payload: dict[str, Any], symbol: str) -> Any:
    """Extract bar records from nested Alpaca MCP response dicts across key variants."""
    container: Any = payload
    for wrapper_key in ("bars", "data", "result"):
        if wrapper_key in payload:
            container = payload[wrapper_key]
            break

    if isinstance(container, list):
        return container

    if isinstance(container, dict):
        for candidate in _symbol_key_candidates(symbol):
            if candidate in container:
                return container[candidate]
        # Fallback: if dict has a single symbol key, return its value.
        if len(container) == 1:
            only_val = next(iter(container.values()))
            if isinstance(only_val, list):
                return only_val
        return []

    return []


def _format_bars_dataframe(raw_bars: Any, symbol: str) -> pd.DataFrame:
    """Normalize an Alpaca MCP ``get_crypto_bars`` response into a ``float64`` OHLCV DataFrame.

    Safely parses JSON string payloads and handles ``BTCUSD`` vs ``BTC/USD`` key variations.
    """
    if isinstance(raw_bars, str):
        stripped = raw_bars.strip()
        if not stripped:
            return pd.DataFrame()
        raw_bars = json.loads(stripped)

    if isinstance(raw_bars, pd.DataFrame):
        df = raw_bars.copy()
    elif isinstance(raw_bars, dict):
        records = _extract_records_from_dict(raw_bars, symbol)
        df = pd.DataFrame(records)
    elif isinstance(raw_bars, list):
        df = pd.DataFrame(raw_bars)
    else:
        raise TypeError(f"Unsupported MCP bars payload type: {type(raw_bars).__name__}")

    if df.empty:
        return df

    # Support both full names and Alpaca abbreviated keys (o, h, l, c, v, t).
    rename_map = {
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "t": "timestamp",
    }
    df = df.rename(
        columns={
            k: v for k, v in rename_map.items() if k in df.columns and v not in df.columns
        }
    )

    missing = [col for col in OHLCV_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns from MCP response: {missing}")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

    for col in OHLCV_COLUMNS:
        df[col] = df[col].astype(np.float64)

    return df


def market_observer_node(state: AgentState) -> AgentState:
    """Fetch live hourly bars via Alpaca MCP and gate on the Iteration 2 strategy filter.

    Raises:
        ValueError: When ``check_entry_signal(df)`` returns ``False``, short-circuiting
            the LangGraph execution back to the caller's ``try/except`` loop without
            invoking Node 2 (Gemini Alpha).
    """
    symbol = state["asset_symbol"]

    try:
        raw_bars = get_crypto_bars(symbol=symbol, timeframe=BAR_TIMEFRAME, limit=BAR_LIMIT)
        df = _format_bars_dataframe(raw_bars, symbol)
    except Exception as exc:
        return {
            **state,
            "error_message": f"Market observer error for {symbol}: {exc}",
        }

    if df.empty:
        return {
            **state,
            "error_message": f"No market data returned for {symbol}.",
        }

    if not check_entry_signal(df):
        raise ValueError(
            "No structural setup detected. Halting graph to preserve LLM rate limits."
        )

    latest_price = float(df["close"].iloc[-1])

    return {
        **state,
        "historical_data": df,
        "current_price": latest_price,
        "error_message": None,
    }
