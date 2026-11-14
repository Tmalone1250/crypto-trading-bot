"""Node 1: Market Observer node for the LangGraph execution engine.

Fetches the latest 250 hourly OHLCV candles via the native Alpaca MCP tool
(``get_crypto_bars``), normalizes numeric types to ``float64``, and gates
execution through ``strategy_filter.check_entry_signal`` to preserve Node 2's
15 RPM Gemini free-tier rate limit.
"""

from typing import Any
import numpy as np
import pandas as pd

from state import AgentState
from strategy_filter import check_entry_signal

BAR_TIMEFRAME: str = "1Hour"
BAR_LIMIT: int = 250
OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


def get_crypto_bars(
    symbol: str,
    timeframe: str = BAR_TIMEFRAME,
    limit: int = BAR_LIMIT,
) -> Any:
    """Invoke the native ``alpaca-mcp-server`` ``get_crypto_bars`` tool.

    All exchange market data requests must flow through the Alpaca MCP toolset
    rather than raw ``requests`` or ``aiohttp`` HTTP calls.

    Args:
        symbol: Target crypto pair (e.g., ``"BTC/USD"``).
        timeframe: Bar interval string (default ``"1Hour"``).
        limit: Number of historical bars to retrieve (default ``250``).

    Returns:
        Raw bar payload from the Alpaca MCP tool.
    """
    raise NotImplementedError(
        "Wire native alpaca-mcp-server `get_crypto_bars` tool binding in runtime."
    )


def _format_bars_dataframe(raw_bars: Any, symbol: str) -> pd.DataFrame:
    """Normalize an Alpaca MCP ``get_crypto_bars`` response into a ``float64`` OHLCV DataFrame."""
    if isinstance(raw_bars, pd.DataFrame):
        df = raw_bars.copy()
    elif isinstance(raw_bars, dict):
        if "bars" in raw_bars:
            bars_payload = raw_bars["bars"]
            if isinstance(bars_payload, dict):
                records = bars_payload.get(symbol, [])
            else:
                records = bars_payload
        else:
            records = raw_bars.get(symbol, [])
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
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns and v not in df.columns})

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
