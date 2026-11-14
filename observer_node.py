"""Node 1: Market Observer node for the LangGraph execution engine.

Captures inbound market data (to be wired to the Alpaca MCP / WebSocket stream)
and packages OHLCV history and the latest price into ``AgentState``.
"""

from typing import Optional
import pandas as pd

from state import AgentState


def get_mock_alpaca_data(symbol: str) -> Optional[pd.DataFrame]:
    """Placeholder helper for fetching market data via the Alpaca MCP server.

    Args:
        symbol: Trading pair symbol (e.g., ``"BTC/USD"``).

    Returns:
        A Pandas DataFrame containing OHLCV columns (including ``"close"``),
        or ``None`` until wired to the Alpaca MCP toolset.
    """
    # TODO: Wire to alpaca-mcp-server / Alpaca WebSocket stream in integration step.
    pass


def market_observer_node(state: AgentState) -> AgentState:
    """Observe market data for the configured asset and update ``AgentState``.

    Calls ``get_mock_alpaca_data`` to retrieve recent candlestick data, updates
    ``historical_data`` and ``current_price`` on success, and captures any
    runtime or data errors in ``error_message`` without crashing the graph.

    Args:
        state: Current ``AgentState`` dictionary.

    Returns:
        Updated ``AgentState`` dictionary.
    """
    symbol = state["asset_symbol"]

    try:
        df = get_mock_alpaca_data(symbol)

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return {
                **state,
                "error_message": f"No market data returned for {symbol}.",
            }

        if "close" not in df.columns:
            return {
                **state,
                "error_message": f"Missing 'close' column in market data for {symbol}.",
            }

        # Cast explicitly to Python float to avoid numpy.float64 type-check issues.
        latest_price = float(df["close"].iloc[-1])

        return {
            **state,
            "historical_data": df,
            "current_price": latest_price,
            "error_message": None,
        }
    except Exception as exc:
        return {
            **state,
            "error_message": f"Market observer error for {symbol}: {exc}",
        }
