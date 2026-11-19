"""Node 1: Market Observer node for the LangGraph execution engine.

Connects to ``alpaca-mcp-server`` over MCP stdio transport to invoke
``get_crypto_bars`` for the latest 250 hourly OHLCV candles (with a 12-day UTC
``start`` lookback) and normalizes numeric types to ``float64`` for downstream
dynamic LLM reasoning.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
import re
import shlex
from typing import Any

from alpaca.data.requests import CryptoLatestQuoteRequest
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import numpy as np
import pandas as pd
import requests

from config import get_alpaca_crypto_data_client
from state import AgentState

load_dotenv()

BAR_TIMEFRAME: str = "1Hour"
BAR_LIMIT: int = 250
LOOKBACK_DAYS: int = 12
OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
MACRO_CALENDAR_URL: str = (
    "https://www.showmytrades.com/api/public/calendar/v1?currency=USD&impact=high&days=1"
)

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
    is_error = bool(
        getattr(result, "is_error", False) or getattr(result, "isError", False)
    )
    content = getattr(result, "content", None) or []

    if is_error:
        error_parts = [
            str(getattr(item, "text", item))
            for item in content
        ]
        raise RuntimeError(" ".join(error_parts) or "Alpaca MCP tool returned an error.")

    structured = getattr(result, "structured_content", None)
    if structured is None:
        structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured

    text_chunks = [
        text_val
        for item in content
        if isinstance((text_val := getattr(item, "text", None)), str)
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

    start_date = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).strftime(
        "%Y-%m-%d"
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
                    "start": start_date,
                },
            )
            return _extract_mcp_result(result)


def get_crypto_bars(
    symbol: str,
    timeframe: str = BAR_TIMEFRAME,
    limit: int = BAR_LIMIT,
    mcp_client: Any | None = None,
) -> Any:
    """Invoke the ``alpaca-mcp-server`` ``get_crypto_bars`` tool via MCP stdio transport.

    Args:
        symbol: Target crypto pair (e.g., ``"BTC/USD"``).
        timeframe: Bar interval string (default ``"1Hour"``).
        limit: Number of historical bars to retrieve (default ``250``).
        mcp_client: Optional persistent ``PersistentAlpacaMCPClient`` instance.

    Returns:
        Raw bar payload from the Alpaca MCP tool.
    """
    if mcp_client is not None:
        start_date = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).strftime(
            "%Y-%m-%d"
        )
        result = mcp_client.call_tool(
            "get_crypto_bars",
            arguments={
                "symbols": symbol,
                "timeframe": timeframe,
                "limit": limit,
                "start": start_date,
            },
        )
        return _extract_mcp_result(result)
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
    """Extract bar records from FastMCP security-enveloped dicts across symbol variants.

    Specifically handles the nested ``payload.get('data', {}).get('bars', {}).get(symbol, [])``
    structure returned by ``alpaca-mcp-server`` as well as direct ``bars`` dicts and
    slash-less symbol keys (e.g., ``BTCUSD`` vs ``BTC/USD``).
    """
    data_section: Any = payload.get("data", payload)
    if isinstance(data_section, str):
        try:
            data_section = json.loads(data_section)
        except json.JSONDecodeError:
            pass

    if isinstance(data_section, dict):
        bars_container: Any = data_section.get("bars", data_section)
    else:
        bars_container = data_section

    if isinstance(bars_container, list):
        return bars_container

    if isinstance(bars_container, dict):
        for candidate in _symbol_key_candidates(symbol):
            if candidate in bars_container and isinstance(bars_container[candidate], list):
                return bars_container[candidate]
        if len(bars_container) == 1:
            only_val = next(iter(bars_container.values()))
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


def _extract_sdk_quote_price(quote_response: Any, symbol: str) -> float | None:
    """Extract ``.ask_price`` (or midpoint) float directly from a native Alpaca SDK quote response."""
    quote_obj: Any = quote_response
    if isinstance(quote_response, dict):
        for candidate in _symbol_key_candidates(symbol):
            if candidate in quote_response:
                quote_obj = quote_response[candidate]
                break
    elif hasattr(quote_response, "data") and isinstance(quote_response.data, dict):
        for candidate in _symbol_key_candidates(symbol):
            if candidate in quote_response.data:
                quote_obj = quote_response.data[candidate]
                break

    if isinstance(quote_obj, dict):
        ask = quote_obj.get("ask_price", quote_obj.get("ap"))
        bid = quote_obj.get("bid_price", quote_obj.get("bp"))
    else:
        ask = getattr(quote_obj, "ask_price", None)
        bid = getattr(quote_obj, "bid_price", None)

    if ask is not None and float(ask) > 0.0 and bid is not None and float(bid) > 0.0:
        return (float(ask) + float(bid)) / 2.0
    if ask is not None and float(ask) > 0.0:
        return float(ask)
    if bid is not None and float(bid) > 0.0:
        return float(bid)
    return None


def get_live_crypto_price(symbol: str = "BTC/USD", data_client: Any = None) -> float | None:
    """Fetch the live real-time crypto quote directly via the native Alpaca Python SDK."""
    try:
        client = data_client if data_client is not None else get_alpaca_crypto_data_client()
        request = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
        quote_response = client.get_crypto_latest_quote(request)
        return _extract_sdk_quote_price(quote_response, symbol)
    except Exception:
        return None


def _fetch_macro_context(timeout: float = 10.0) -> str:
    """Fetch upcoming high-impact USD macroeconomic events from the public calendar API."""
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        response = requests.get(MACRO_CALENDAR_URL, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return (
            f"Current UTC Time: {now_iso}\n"
            f"Macroeconomic calendar unavailable: {exc}"
        )

    events = payload if isinstance(payload, list) else payload.get("events", payload.get("data", []))
    if not isinstance(events, list) or not events:
        return (
            f"Current UTC Time: {now_iso}\n"
            "High-Impact USD Macro Calendar (Next 24h): No high-impact events scheduled."
        )

    lines = [
        f"Current UTC Time: {now_iso}",
        "High-Impact USD Macro Calendar (Next 24h):",
    ]
    for ev in events[:15]:
        if not isinstance(ev, dict):
            continue
        title = ev.get("title") or ev.get("event") or ev.get("name") or "Unknown Event"
        ev_time = ev.get("date") or ev.get("time") or ev.get("timestamp") or "TBD"
        impact = ev.get("impact") or "high"
        currency = ev.get("currency") or "USD"
        forecast = ev.get("forecast")
        previous = ev.get("previous")
        extras = []
        if forecast is not None:
            extras.append(f"forecast={forecast}")
        if previous is not None:
            extras.append(f"previous={previous}")
        extra_str = f" ({', '.join(extras)})" if extras else ""
        lines.append(f"- [{ev_time}] {currency} ({impact}): {title}{extra_str}")

    return "\n".join(lines)


def market_observer_node(state: AgentState) -> AgentState:
    """Fetch live bars, native SDK real-time quote, and macro calendar context."""
    symbol = state["asset_symbol"]
    mcp_client = state.get("mcp_client")

    try:
        if mcp_client is not None:
            raw_bars = get_crypto_bars(
                symbol=symbol,
                timeframe=BAR_TIMEFRAME,
                limit=BAR_LIMIT,
                mcp_client=mcp_client,
            )
        else:
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

    live_price = get_live_crypto_price(symbol=symbol)
    latest_price = live_price if live_price is not None else float(df["close"].iloc[-1])
    macro_context = _fetch_macro_context()

    market_data_by_symbol = dict(state.get("market_data_by_symbol") or {})
    market_data_by_symbol[symbol] = df
    for extra_sym in state.get("asset_symbols") or []:
        if extra_sym == symbol or extra_sym in market_data_by_symbol:
            continue
        extra_price = get_live_crypto_price(symbol=extra_sym)
        if extra_price is not None and extra_price > 0.0:
            market_data_by_symbol[extra_sym] = pd.DataFrame(
                [
                    {
                        "open": extra_price,
                        "high": extra_price,
                        "low": extra_price,
                        "close": extra_price,
                        "volume": 0.0,
                    }
                ]
            )

    updated_state: dict[str, Any] = {
        **state,
        "historical_data": df,
        "market_data_by_symbol": market_data_by_symbol,
        "current_price": latest_price,
        "macro_context": macro_context,
        "error_message": None,
    }
    if "live_price" in state:
        updated_state["live_price"] = latest_price
    return updated_state  # type: ignore[return-value]
