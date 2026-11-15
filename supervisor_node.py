"""Regime Supervisor Node: deterministic volatility gate for the Multi-Agent Swarm.

Inspects the most recent 5 candles in ``state["historical_data"]``, computes the
high-to-low percentage range, and classifies the market regime as ``"FLAT"``
(< 0.25%) or ``"ACTIVE"`` (>= 0.25%) to preserve Gemini LLM rate limits during
low-volatility chop.
"""

from state import AgentState

LOOKBACK_CANDLES: int = 5
VOLATILITY_THRESHOLD_PCT: float = 0.25
FLAT_RATIONALE: str = "Market is flat. Skipping LLM execution to preserve rate limits."


def supervisor_node(state: AgentState) -> AgentState:
    """Classify the 5-candle volatility regime as ``FLAT`` or ``ACTIVE``."""
    df = state["historical_data"]

    if df is None or df.empty or not {"high", "low"}.issubset(df.columns):
        print(
            "[Supervisor] 5-Bar Range: 0.0000% (< 0.25%) | "
            "Regime: FLAT | Action: HOLD (insufficient OHLCV data)"
        )
        return {
            **state,
            "market_regime": "FLAT",
            "action": "HOLD",
            "rationale": FLAT_RATIONALE,
        }

    recent_df = df.tail(LOOKBACK_CANDLES)
    highest_high = float(recent_df["high"].max())
    lowest_low = float(recent_df["low"].min())

    if lowest_low <= 0.0:
        range_pct = 0.0
    else:
        range_pct = ((highest_high - lowest_low) / lowest_low) * 100.0

    if range_pct < VOLATILITY_THRESHOLD_PCT:
        print(
            f"[Supervisor] 5-Bar Range: {range_pct:.4f}% (< {VOLATILITY_THRESHOLD_PCT:.2f}%) | "
            "Regime: FLAT | Action: HOLD"
        )
        return {
            **state,
            "market_regime": "FLAT",
            "action": "HOLD",
            "rationale": FLAT_RATIONALE,
        }

    print(
        f"[Supervisor] 5-Bar Range: {range_pct:.4f}% (>= {VOLATILITY_THRESHOLD_PCT:.2f}%) | "
        "Regime: ACTIVE | Waking Alpha LLM"
    )
    return {
        **state,
        "market_regime": "ACTIVE",
    }
