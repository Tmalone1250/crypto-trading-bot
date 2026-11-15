"""Global LangGraph state schema for the autonomous crypto swing-trading agent."""

from typing import Any, Optional, TypedDict
import pandas as pd


class AgentState(TypedDict):
    """Shared state dictionary passed sequentially across all LangGraph nodes.

    Attributes:
        asset_symbol: Target trading pair symbol (e.g., ``"BTC/USD"``).
        current_price: Latest observed asset price in quote currency.
        historical_data: OHLCV candlestick DataFrame used for pattern and ATR math.
        sentiment_score: Polarity score from Node 2 (Gemini Alpha) in ``[-1.0, 1.0]``.
        risk_cleared: Gate flag set by Node 3 (TA-Lib Risk Manager).
        order_details: Execution payload and metadata populated by Node 3 and Node 4.
        error_message: Optional error string captured when a node fails gracefully.
        account_data: Alpaca account balances, buying power, and status dictionary.
        open_positions: List of currently held open positions from Alpaca.
        active_orders: List of currently active/open orders from Alpaca.
        action: Autonomous quant manager action (``"MARKET_BUY"``, ``"LIMIT_BUY"``, ``"MARKET_SELL"``, ``"LIMIT_SELL"``, ``"CANCEL_ALL"``, or ``"HOLD"``).
        target_price: Optional limit price specified by the LLM for ``"LIMIT_BUY"`` or ``"LIMIT_SELL"``.
        rationale: Concise quantitative reasoning produced by the LLM for the selected action.
    """

    asset_symbol: str
    current_price: float
    historical_data: pd.DataFrame
    sentiment_score: float
    risk_cleared: bool
    order_details: dict[str, Any]
    error_message: Optional[str]
    account_data: dict[str, Any]
    open_positions: list[dict[str, Any]]
    active_orders: list[dict[str, Any]]
    action: str
    target_price: Optional[float]
    rationale: str
