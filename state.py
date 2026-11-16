"""Global LangGraph state schema for the autonomous crypto swing-trading agent."""

from typing import Any, Literal, Optional, TypedDict
import pandas as pd
from pydantic import BaseModel, Field


class TradeIntent(BaseModel):
    """Structured maker-only trade intent emitted by the multi-asset Scalper Expert."""

    symbol: str = Field(..., description="Target trading pair (e.g., 'BTC/USD').")
    action: Literal["LIMIT_BUY", "LIMIT_SELL"] = Field(
        ...,
        description="Maker-only limit action ('LIMIT_BUY' or 'LIMIT_SELL').",
    )
    target_price: float = Field(..., gt=0.0, description="Limit order price.")
    qty: float = Field(..., gt=0.0, description="Order quantity in base asset.")
    notional_usd: float = Field(
        default=200.0,
        gt=0.0,
        description="Fixed USD notional per tranche (default $200.00).",
    )
    rationale: str = Field(..., description="Quantitative rationale for this intent.")


class AgentState(TypedDict):
    """Shared state dictionary passed sequentially across all LangGraph nodes.

    Attributes:
        asset_symbol: Primary target trading pair symbol (e.g., ``"BTC/USD"``).
        asset_symbols: Ordered list of crypto pairs scanned by the multi-asset scalper.
        market_data_by_symbol: Optional per-symbol OHLCV DataFrames for multi-asset scanning.
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
        market_regime: Deterministic volatility regime from the Supervisor node (``"FLAT"`` or ``"ACTIVE"``).
        realized_pnl: Compounding realized profit/loss War Chest in USD (default ``0.0``).
        trade_intents: List of ``TradeIntent`` objects produced by the multi-asset scalper.
    """

    asset_symbol: str
    asset_symbols: list[str]
    market_data_by_symbol: dict[str, pd.DataFrame]
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
    market_regime: str
    realized_pnl: float
    trade_intents: list[TradeIntent]
