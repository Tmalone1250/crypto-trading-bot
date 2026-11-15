"""Scalper Expert Node: Python-only high-frequency micro-scalper for FLAT regimes.

Operates without LLM calls when ``Supervisor`` classifies the market as ``FLAT``:
  1. If an open position is held and ``current_price`` is up >= 0.5% from
     ``avg_entry_price``, issues ``LIMIT_SELL`` to lock in micro-profits.
  2. Otherwise, inspects the last 3 candles and checks if ``current_price`` is
     in the lower 20% of the 5-bar high-low range, issuing ``LIMIT_BUY`` at
     ``current_price`` risking a small fixed fraction of ``buying_power``.
  3. Otherwise, issues ``HOLD``.
"""

from state import AgentState

SCALP_TAKE_PROFIT_PCT: float = 0.5
LOWER_RANGE_PCT: float = 0.20
SCALP_BUYING_POWER_RISK_FRACTION: float = 0.01
SCALP_MAX_NOTIONAL_USD: float = 25.0


def _find_open_position(state: AgentState) -> dict | None:
    """Return the matching open position for ``state['asset_symbol']`` if held."""
    target_symbol = state["asset_symbol"].replace("/", "").upper()
    for pos in state.get("open_positions") or []:
        pos_sym = str(pos.get("symbol", "")).replace("/", "").upper()
        if pos_sym == target_symbol:
            return pos
    return None


def scalper_expert_node(state: AgentState) -> AgentState:
    """Evaluate a FLAT market regime using deterministic micro-scalping rules."""
    df = state["historical_data"]
    current_price = float(state["current_price"])

    if df is None or df.empty or not {"high", "low", "close"}.issubset(df.columns):
        print("[Scalper] Action: HOLD (insufficient OHLCV data)")
        return {
            **state,
            "action": "HOLD",
            "target_price": None,
            "rationale": "Scalper hold: insufficient historical data.",
        }

    # Check open position for +0.5% micro-profit take-profit first.
    open_pos = _find_open_position(state)
    if open_pos is not None:
        try:
            avg_entry = float(open_pos.get("avg_entry_price", 0.0))
        except (TypeError, ValueError):
            avg_entry = 0.0

        if avg_entry > 0.0:
            gain_pct = ((current_price - avg_entry) / avg_entry) * 100.0
            if gain_pct >= SCALP_TAKE_PROFIT_PCT:
                rationale = (
                    f"Scalper take-profit: price ${current_price:.2f} is up "
                    f"{gain_pct:.2f}% from avg_entry ${avg_entry:.2f}."
                )
                print(f"[Scalper] Action: LIMIT_SELL @ ${current_price:.2f} | {rationale}")
                return {
                    **state,
                    "action": "LIMIT_SELL",
                    "target_price": current_price,
                    "rationale": rationale,
                }

    # Inspect the last 3 candles and evaluate current price against the 5-bar range.
    recent_3 = df.tail(3)
    recent_5 = df.tail(5)
    five_bar_high = float(recent_5["high"].max())
    five_bar_low = float(recent_5["low"].min())
    range_span = five_bar_high - five_bar_low
    lower_20_threshold = five_bar_low + (LOWER_RANGE_PCT * range_span)

    account = state.get("account_data") or {}
    try:
        buying_power = float(account.get("buying_power", account.get("cash", 0.0)))
    except (TypeError, ValueError):
        buying_power = 0.0

    scalp_risk_usd = round(
        min(buying_power * SCALP_BUYING_POWER_RISK_FRACTION, SCALP_MAX_NOTIONAL_USD),
        2,
    )

    if range_span > 0.0 and current_price <= lower_20_threshold:
        last_3_close = float(recent_3["close"].iloc[-1])
        rationale = (
            f"Scalper lower-20% range buy: price ${current_price:.2f} "
            f"(3-bar close ${last_3_close:.2f}) <= ${lower_20_threshold:.2f} "
            f"(5-bar ${five_bar_low:.2f}-${five_bar_high:.2f}); "
            f"risking ${scalp_risk_usd:.2f} of ${buying_power:.2f} buying power."
        )
        print(f"[Scalper] Action: LIMIT_BUY @ ${current_price:.2f} | {rationale}")
        return {
            **state,
            "action": "LIMIT_BUY",
            "target_price": current_price,
            "rationale": rationale,
        }

    rationale = (
        f"Scalper hold: price ${current_price:.2f} outside lower 20% "
        f"(<= ${lower_20_threshold:.2f}) with no +0.5% profit target hit."
    )
    print(f"[Scalper] Action: HOLD | {rationale}")
    return {
        **state,
        "action": "HOLD",
        "target_price": None,
        "rationale": rationale,
    }
