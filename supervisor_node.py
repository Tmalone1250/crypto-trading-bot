"""Regime Supervisor Node: deterministic volatility & price-delta gate for the Swarm.

Inspects the most recent 5 candles in ``state["historical_data"]``, computes the
high-to-low percentage range, and gates LLM activation on BOTH:
  1. 5-bar range >= 0.25%
  2. Price movement >= 0.20% from ``state["last_alpha_eval_price"]`` (when > 0.0)

Otherwise classifies the regime as ``"FLAT"`` so the Python-only Scalper runs
and preserves Gemini API rate limits.
"""

from state import AgentState

LOOKBACK_CANDLES: int = 5
VOLATILITY_THRESHOLD_PCT: float = 0.25
ALPHA_REENTRY_MOVE_PCT: float = 0.20
FLAT_RATIONALE: str = "Market is flat. Skipping LLM execution to preserve rate limits."


def supervisor_node(state: AgentState) -> AgentState:
    """Classify the market regime as ``FLAT`` or ``ACTIVE``."""
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

    current_price = float(state.get("current_price", 0.0))
    if current_price <= 0.0 and "close" in recent_df.columns:
        current_price = float(recent_df["close"].iloc[-1])

    last_alpha_eval_price = float(state.get("last_alpha_eval_price", 0.0))
    if last_alpha_eval_price > 0.0 and current_price > 0.0:
        move_from_last_eval_pct = (
            abs(current_price - last_alpha_eval_price) / last_alpha_eval_price
        ) * 100.0
        if move_from_last_eval_pct < ALPHA_REENTRY_MOVE_PCT:
            print(
                f"[Supervisor] 5-Bar Range: {range_pct:.4f}% | "
                f"Move vs Last Alpha (${last_alpha_eval_price:.2f}): "
                f"{move_from_last_eval_pct:.4f}% (< {ALPHA_REENTRY_MOVE_PCT:.2f}%) | "
                "Regime: FLAT (deferring to Scalper)"
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
