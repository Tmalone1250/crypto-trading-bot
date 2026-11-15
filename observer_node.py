"""Node 1: Market Observer node for the LangGraph execution engine.

Connects to ``alpaca-mcp-server`` over MCP stdio transport to invoke
``get_crypto_bars`` for the latest 250 hourly OHLCV candles (with a 12-day UTC
``start`` lookback), normalizes numeric types to ``float64``, and gates execution
through ``strategy_filter.check_entry_signal`` to preserve Node 2's 15 RPM
Gemini free-tier rate limit.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
import re
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
LOOKBACK_DAYS: int = 12
OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

_NUM_PATTERN = r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?"
_TEXT_BAR_REGEX = re.compile(
    rf"(?:(?:t|time|timestamp|date)\s*[:=]\s*(?P<timestamp>\S+).*?)?"
    rf"(?:o|open)\s*[:=]\s*\$?(?P<open>{_NUM_PATTERN})\D+"
    rf"(?:h|high)\s*[:=]\s*\$?(?P<high>{_NUM_PATTERN})\D+"
    rf"(?:l|low)\s*[:=]\s*\$?(?P<low>{_NUM_PATTERN})\D+"
    rf"(?:c|close)\s*[:=]\s*\$?(?P<close>{_NUM_PATTERN})\D+"
    rf"(?:v|vol|volume)\s*[:=]\s*\$?(?P<volume>{_NUM_PATTERN})",
    re.IGNORECASE | re.DOTALL,
)


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

    start_iso = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).isoformat()

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(
                "get_crypto_bars",
                arguments={
                    "symbols": symbol,
                    "timeframe": timeframe,
                    "limit": limit,
                    "start": start_iso,
                },
            )
            return _extract_mcp_result(result)


def get_crypto_bars(
    symbol: str,
    timeframe: str = BAR_TIMEFRAME,
    limit: int = BAR_LIMIT,
) -> Any:
    """Invoke the ``alpaca-mcp-server`` ``get_crypto_bars`` tool via MCP stdio transport.

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
    candidates = [
        symbol,
        symbol.replace("/", ""),
        symbol.upper(),
        symbol.replace("/", "").upper(),
    ]
    if "/" not in symbol and len(symbol) > 3:
        candidates.append(f"{symbol[:-3]}/{symbol[-3:]}")
        candidates.append(f"{symbol[:-3]}/{symbol[-3:]}".upper())
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
        if len(container) == 1:
            only_val = next(iter(container.values()))
            if isinstance(only_val, list):
                return only_val
        return []

    return []


def _parse_fastmcp_text_bars(text: str) -> list[dict[str, Any]]:
    """Parse FastMCP formatted text output for ``get_crypto_bars`` into bar dicts."""
    records: list[dict[str, Any]] = []
    for match in _TEXT_BAR_REGEX.finditer(text):
        group = match.groupdict()
        row: dict[str, Any] = {
            "open": float(group["open"].replace(",", "")),
            "high": float(group["high"].replace(",", "")),
            "low": float(group["low"].replace(",", "")),
            "close": float(group["close"].replace(",", "")),
            "volume": float(group["volume"].replace(",", "")),
        }
        if group.get("timestamp"):
            row["timestamp"] = group["timestamp"].rstrip(",;|")
        records.append(row)
    return records


def _parse_string_payload(raw_text: str) -> Any:
    """Parse a string MCP payload as JSON first, falling back to FastMCP text parsing."""
    stripped = raw_text.strip()
    if not stripped:
        return []

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Check for embedded JSON object/array inside markdown/text output.
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start_idx = stripped.find(start_char)
        end_idx = stripped.rfind(end_char)
        if start_idx != -1 and end_idx > start_idx:
            try:
                return json.loads(stripped[start_idx : end_idx + 1])
            except json.JSONDecodeError:
                pass

    return _parse_fastmcp_text_bars(stripped)


def _format_bars_dataframe(raw_bars: Any, symbol: str) -> pd.DataFrame:
    """Normalize an Alpaca MCP ``get_crypto_bars`` response into a ``float64`` OHLCV DataFrame.

    Safely parses JSON strings, FastMCP text outputs, and ``BTCUSD`` vs ``BTC/USD`` keys.
    """
    if isinstance(raw_bars, str):
        raw_bars = _parse_string_payload(raw_bars)

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
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
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
