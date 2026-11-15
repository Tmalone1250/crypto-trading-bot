"""Node 2: Alpha Logic Gate (Autonomous Quant Manager) for the LangGraph execution engine.

Uses ``ChatGoogleGenerativeAI`` (``gemini-3.5-flash-lite``) with Pydantic
structured output to evaluate the agent's live portfolio exposure (buying power,
open positions, and active open orders) against a condensed summary of recent
OHLCV price action, returning a structured lifecycle action
(``MARKET_BUY``, ``LIMIT_BUY``, ``CANCEL_ALL``, or ``HOLD``), an optional
``target_price``, a ``rationale``, and a bounded ``sentiment_score`` in ``[-1.0, 1.0]``.
"""

import os
from typing import Literal, Optional
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from state import AgentState

load_dotenv()

GEMINI_MODEL = "gemini-3.5-flash-lite"


class SentimentResponse(BaseModel):
    """Strict structured output schema enforced on the Gemini Alpha quant manager."""

    action: Literal[
        "MARKET_BUY",
        "LIMIT_BUY",
        "MARKET_SELL",
        "LIMIT_SELL",
        "CANCEL_ALL",
        "HOLD",
    ] = Field(
        ...,
        description=(
            "Lifecycle action to execute: 'MARKET_BUY' for immediate market entry, "
            "'LIMIT_BUY' to place a resting limit buy order at target_price, "
            "'MARKET_SELL' for immediate market exit/profit-taking, "
            "'LIMIT_SELL' to place a resting limit sell order at target_price, "
            "'CANCEL_ALL' if the trade thesis is invalidated and open orders must be "
            "pulled, or 'HOLD' to take no action."
        ),
    )
    target_price: Optional[float] = Field(
        default=None,
        description=(
            "Target limit price when action is 'LIMIT_BUY' or 'LIMIT_SELL' "
            "(required for limit orders, otherwise None)."
        ),
    )
    rationale: str = Field(
        ...,
        description="Concise quantitative rationale explaining the selected action.",
    )
    sentiment_score: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Market sentiment polarity score from -1.0 (strongly bearish) "
            "to 1.0 (strongly bullish), with 0.0 representing neutral."
        ),
    )


def _format_portfolio_context(state: AgentState) -> str:
    """Format available buying power, open positions, and active orders from ``AgentState``."""
    account = state.get("account_data") or {}
    buying_power = account.get("buying_power", account.get("cash", "0.00"))
    cash = account.get("cash", "N/A")
    portfolio_value = account.get("portfolio_value", "N/A")

    positions = state.get("open_positions") or []
    if not positions:
        holdings_text = "None (flat / no open positions)"
    else:
        lines = []
        for pos in positions:
            sym = pos.get("symbol", "UNKNOWN")
            side = pos.get("side", "long")
            qty = pos.get("qty", "0")
            entry = pos.get("avg_entry_price", "N/A")
            mkt_val = pos.get("market_value", "N/A")
            upl = pos.get("unrealized_pl", "N/A")
            lines.append(
                f"- {sym} ({side}): qty={qty}, avg_entry={entry}, "
                f"market_value=${mkt_val}, unrealized_pl=${upl}"
            )
        holdings_text = "\n".join(lines)

    orders = state.get("active_orders") or []
    if not orders:
        orders_text = "None (no resting open orders)"
    else:
        order_lines = []
        for ord_item in orders:
            oid = ord_item.get("id", "UNKNOWN")
            sym = ord_item.get("symbol", "UNKNOWN")
            side = ord_item.get("side", "buy")
            otype = ord_item.get("type", "limit")
            lprice = ord_item.get("limit_price", "N/A")
            qty = ord_item.get("qty", "N/A")
            order_lines.append(
                f"- id={oid}: {sym} {side} {otype} qty={qty} limit_price={lprice}"
            )
        orders_text = "\n".join(order_lines)

    market_regime = str(state.get("market_regime", "ACTIVE")).upper()

    return (
        f"Market Regime: {market_regime}\n"
        f"Available Buying Power: ${buying_power}\n"
        f"Cash Balance: ${cash}\n"
        f"Portfolio Value: ${portfolio_value}\n"
        f"Current Holdings:\n{holdings_text}\n"
        f"Active Open Orders:\n{orders_text}"
    )


def _format_condensed_market_summary(state: AgentState, lookback: int = 10) -> str:
    """Build a condensed statistical and recent-candle summary of ``historical_data``."""
    df = state["historical_data"]
    recent_df = df.tail(lookback)
    first_close = float(df["close"].iloc[0])
    latest_close = float(df["close"].iloc[-1])
    pct_change = ((latest_close - first_close) / first_close * 100.0) if first_close else 0.0
    period_high = float(df["high"].max()) if "high" in df.columns else latest_close
    period_low = float(df["low"].min()) if "low" in df.columns else latest_close
    avg_vol = float(df["volume"].mean()) if "volume" in df.columns else 0.0

    cols = [c for c in ("timestamp", "open", "high", "low", "close", "volume") if c in recent_df.columns]
    condensed_table = recent_df[cols].to_string(index=False)

    return (
        f"Asset: {state['asset_symbol']}\n"
        f"Current Price: ${state['current_price']:.2f}\n"
        f"Window Summary ({len(df)} bars): "
        f"High=${period_high:.2f}, Low=${period_low:.2f}, "
        f"Net Change={pct_change:+.2f}%, Avg Volume={avg_vol:.2f}\n"
        f"Most Recent {len(recent_df)} Candles:\n{condensed_table}"
    )


def alpha_logic_gate_node(state: AgentState) -> AgentState:
    """Evaluate portfolio exposure against market data with Gemini and output a lifecycle action.

    Uses ``gemini-3.5-flash-lite`` with Pydantic structured output to return
    ``action`` (``MARKET_BUY``, ``LIMIT_BUY``, ``CANCEL_ALL``, or ``HOLD``),
    ``target_price``, ``rationale``, and ``sentiment_score``. If ``historical_data``
    is empty or the Gemini API fails, safely defaults to ``action="HOLD"``,
    ``target_price=None``, ``rationale=""``, and ``sentiment_score=0.0``.

    Args:
        state: Current ``AgentState`` dictionary.

    Returns:
        Updated ``AgentState`` dictionary.
    """
    symbol = state["asset_symbol"]
    df = state["historical_data"]

    if df is None or df.empty:
        return {
            **state,
            "action": "HOLD",
            "target_price": None,
            "rationale": "Insufficient market data to evaluate setup.",
            "sentiment_score": 0.0,
            "error_message": f"Cannot compute alpha sentiment for {symbol}: historical_data is empty.",
        }

    portfolio_context = _format_portfolio_context(state)
    market_summary = _format_condensed_market_summary(state)

    system_prompt = (
        "You are an autonomous quantitative crypto portfolio manager with full order "
        "lifecycle authority. Evaluate your current portfolio exposure (available buying "
        "power, open positions, and active open orders) against the recent market data "
        "summary and choose exactly one action:\n"
        "- MARKET_BUY: Execute an immediate market buy order when momentum and structure "
        "warrant immediate entry (set target_price to null).\n"
        "- LIMIT_BUY: Place a resting limit buy order at a specific structural support or "
        "pullback level (you MUST provide target_price as a float).\n"
        "- MARKET_SELL: Execute an immediate market sell order to take profit or exit an "
        "open position immediately (set target_price to null).\n"
        "- LIMIT_SELL: Place a resting limit sell order at structural resistance or a "
        "target profit level (you MUST provide target_price as a float).\n"
        "- CANCEL_ALL: Cancel all active open orders when the market thesis is invalidated, "
        "price has moved away from resting limit levels, or risk must be pulled (set "
        "target_price to null).\n"
        "- HOLD: Take no action when current exposure and resting orders are appropriate or "
        "no high-conviction setup is present (set target_price to null).\n\n"
        "If you hold an open position, constantly evaluate the current price against your "
        "position's avg_entry_price. If the market reaches structural resistance or a "
        "profitable target (e.g., +2%), you MUST issue a LIMIT_SELL or MARKET_SELL to take "
        "profit and lock in gains.\n\n"
        "Always provide a concise quantitative rationale and a sentiment_score strictly "
        "between -1.0 (strongly bearish) and 1.0 (strongly bullish).\n\n"
        f"=== CURRENT PORTFOLIO STATE ===\n{portfolio_context}"
    )

    human_prompt = (
        f"=== MARKET DATA SUMMARY ===\n{market_summary}\n\n"
        "Output your structured SentimentResponse (action, target_price, rationale, sentiment_score)."
    )

    try:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
            temperature=0.0,
        )
        structured_llm = llm.with_structured_output(SentimentResponse)
        result: SentimentResponse = structured_llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
        )
        action = str(result.action).upper()
        target_price = float(result.target_price) if result.target_price is not None else None
        rationale = str(result.rationale)
        sentiment_score = float(result.sentiment_score)

        print(
            f"[Alpha] Action: {action} | Target Price: {target_price} "
            f"| Rationale: {rationale}"
        )

        return {
            **state,
            "action": action,
            "target_price": target_price,
            "rationale": rationale,
            "sentiment_score": sentiment_score,
            "error_message": None,
        }
    except Exception as exc:
        return {
            **state,
            "action": "HOLD",
            "target_price": None,
            "rationale": "",
            "sentiment_score": 0.0,
            "error_message": f"Alpha node error for {symbol}: {exc}",
        }
