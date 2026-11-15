"""Node 2: Alpha Logic Gate (Autonomous Quant Manager) for the LangGraph execution engine.

Uses ``ChatGoogleGenerativeAI`` (``gemini-3.5-flash-lite``) with Pydantic
structured output to evaluate the agent's live portfolio exposure (buying power
and open positions) against a condensed summary of recent OHLCV price action,
returning a structured trading decision (``BUY``, ``SELL``, or ``HOLD``) and a
bounded sentiment polarity score in ``[-1.0, 1.0]``.
"""

import os
from typing import Literal
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from state import AgentState

load_dotenv()

GEMINI_MODEL = "gemini-3.5-flash-lite"


class SentimentResponse(BaseModel):
    """Strict structured output schema enforced on the Gemini Alpha quant manager."""

    decision: Literal["BUY", "SELL", "HOLD"] = Field(
        ...,
        description=(
            "Autonomous portfolio action: 'BUY' to enter or add exposure, "
            "'SELL' to exit or reduce exposure, or 'HOLD' to take no action."
        ),
    )
    sentiment_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description=(
            "Market sentiment polarity score from -1.0 (strongly bearish) "
            "to 1.0 (strongly bullish), with 0.0 representing neutral."
        ),
    )


def _format_portfolio_context(state: AgentState) -> str:
    """Format available buying power and current open positions from ``AgentState``."""
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

    return (
        f"Available Buying Power: ${buying_power}\n"
        f"Cash Balance: ${cash}\n"
        f"Portfolio Value: ${portfolio_value}\n"
        f"Current Holdings:\n{holdings_text}"
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
    """Evaluate portfolio exposure against market data with Gemini and output a decision.

    Uses ``gemini-3.5-flash-lite`` with Pydantic structured output to return
    ``decision`` (``BUY``, ``SELL``, or ``HOLD``) and ``sentiment_score`` in
    ``[-1.0, 1.0]``. If ``historical_data`` is empty or the Gemini API fails,
    safely defaults ``decision`` to ``"HOLD"`` and ``sentiment_score`` to ``0.0``.

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
            "decision": "HOLD",
            "sentiment_score": 0.0,
            "error_message": f"Cannot compute alpha sentiment for {symbol}: historical_data is empty.",
        }

    portfolio_context = _format_portfolio_context(state)
    market_summary = _format_condensed_market_summary(state)

    system_prompt = (
        "You are an autonomous quantitative crypto portfolio manager. "
        "Evaluate your current portfolio exposure (available buying power and open "
        "holdings) against the recent market data summary. "
        "Decide whether to BUY (open/add a long position), SELL (close/reduce an "
        "existing position), or HOLD (stay flat or maintain current exposure without "
        "placing a new order), and output a sentiment_score strictly between -1.0 "
        "(strongly bearish) and 1.0 (strongly bullish).\n\n"
        f"=== CURRENT PORTFOLIO STATE ===\n{portfolio_context}"
    )

    human_prompt = (
        f"=== MARKET DATA SUMMARY ===\n{market_summary}\n\n"
        "Provide your structured decision (BUY, SELL, or HOLD) and sentiment_score."
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

        return {
            **state,
            "decision": str(result.decision).upper(),
            "sentiment_score": float(result.sentiment_score),
            "error_message": None,
        }
    except Exception as exc:
        return {
            **state,
            "decision": "HOLD",
            "sentiment_score": 0.0,
            "error_message": f"Alpha node error for {symbol}: {exc}",
        }
