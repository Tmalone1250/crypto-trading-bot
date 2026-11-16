"""Scalper Expert Node: Multi-Asset Maker-Only Yield Harvester for FLAT regimes.

Operates without LLM calls when ``Supervisor`` classifies the market as ``FLAT``:
  - Iterates over ``state['asset_symbols']`` (defaulting to ``["BTC/USD", "ETH/USD", "SOL/USD"]``).
  - **Fee-Adjusted Take-Profit**: Evaluates for ``LIMIT_SELL`` only when
    ``current_price >= avg_entry_price * 1.015`` (1.5% gross profit to clear 0.30%
    round-trip maker fees).
  - **Multi-Tranche Grid**: If a position is already open, allows a new ``LIMIT_BUY``
    only when ``current_price <= avg_entry_price * 0.98`` (-2.0% drawdown tranche).
    If no position is open, buys when ``current_price`` is in the lower 20% of the
    5-bar high-low range. All buys are sized at a fixed ``$200.00`` per tranche.
  - **Maker-Only Enforcement**: Emits strictly ``LIMIT_BUY`` or ``LIMIT_SELL``
    ``TradeIntent`` objects (never market orders).
"""

from typing import Any
import pandas as pd

from state import AgentState, TradeIntent

DEFAULT_SCALP_SYMBOLS: tuple[str, ...] = ("BTC/USD", "ETH/USD", "SOL/USD")
FEE_ADJUSTED_PROFIT_MULTIPLIER: float = 1.015  # +1.5% gross target to clear 0.30% maker fees
TRANCHE_GRID_DROP_MULTIPLIER: float = 0.98     # -2.0% below avg_entry_price for next tranche
TRANCHE_NOTIONAL_USD: float = 200.0
LOWER_RANGE_PCT: float = 0.20
QTY_PRECISION: int = 6


def _find_open_position(state: AgentState, symbol: str) -> dict[str, Any] | None:
    """Return the matching open position for ``symbol`` if held."""
    target_symbol = symbol.replace("/", "").upper()
    for pos in state.get("open_positions") or []:
        pos_sym = str(pos.get("symbol", "")).replace("/", "").upper()
        if pos_sym == target_symbol:
            return pos
    return None


def _get_symbol_dataframe(state: AgentState, symbol: str) -> pd.DataFrame | None:
    """Resolve the OHLCV DataFrame for ``symbol`` from multi-asset or primary state."""
    market_map = state.get("market_data_by_symbol") or {}
    if symbol in market_map:
        return market_map[symbol]
    slashless = symbol.replace("/", "")
    if slashless in market_map:
        return market_map[slashless]
    if symbol == state.get("asset_symbol"):
        return state.get("historical_data")
    return None


def _evaluate_symbol_intent(
    state: AgentState,
    symbol: str,
    df: pd.DataFrame,
) -> TradeIntent | None:
    """Evaluate a single symbol for fee-adjusted ``LIMIT_SELL`` or grid ``LIMIT_BUY``."""
    if df is None or df.empty or not {"high", "low", "close"}.issubset(df.columns):
        return None

    if symbol == state.get("asset_symbol") and float(state.get("current_price", 0.0)) > 0.0:
        current_price = float(state["current_price"])
    else:
        current_price = float(df["close"].iloc[-1])

    if current_price <= 0.0:
        return None

    open_pos = _find_open_position(state, symbol)
    if open_pos is not None:
        try:
            avg_entry = float(open_pos.get("avg_entry_price", 0.0))
        except (TypeError, ValueError):
            avg_entry = 0.0

        try:
            held_qty = float(open_pos.get("qty", 0.0))
        except (TypeError, ValueError):
            held_qty = 0.0

        # 1. Fee-Adjusted Take-Profit: >= +1.5% above avg_entry_price.
        if avg_entry > 0.0 and current_price >= avg_entry * FEE_ADJUSTED_PROFIT_MULTIPLIER:
            sell_qty = held_qty if held_qty > 0.0 else round(TRANCHE_NOTIONAL_USD / current_price, QTY_PRECISION)
            gain_pct = ((current_price - avg_entry) / avg_entry) * 100.0
            rationale = (
                f"Fee-adjusted LIMIT_SELL on {symbol}: ${current_price:.2f} is up "
                f"{gain_pct:.2f}% (>= 1.50% target) from avg_entry ${avg_entry:.2f}."
            )
            return TradeIntent(
                symbol=symbol,
                action="LIMIT_SELL",
                target_price=current_price,
                qty=round(sell_qty, QTY_PRECISION),
                notional_usd=round(sell_qty * current_price, 2),
                rationale=rationale,
            )

        # 2. Multi-Tranche Grid: allow additional $200 LIMIT_BUY only if down >= 2% from avg_entry.
        if avg_entry > 0.0 and current_price <= avg_entry * TRANCHE_GRID_DROP_MULTIPLIER:
            buy_qty = round(TRANCHE_NOTIONAL_USD / current_price, QTY_PRECISION)
            drop_pct = ((current_price - avg_entry) / avg_entry) * 100.0
            rationale = (
                f"Multi-tranche grid LIMIT_BUY on {symbol}: ${current_price:.2f} is "
                f"{drop_pct:.2f}% (<= -2.00% tranche step) from avg_entry ${avg_entry:.2f}; "
                f"sizing ${TRANCHE_NOTIONAL_USD:.2f} tranche."
            )
            return TradeIntent(
                symbol=symbol,
                action="LIMIT_BUY",
                target_price=current_price,
                qty=buy_qty,
                notional_usd=TRANCHE_NOTIONAL_USD,
                rationale=rationale,
            )

        return None

    # 3. Flat / No Open Position: enter $200 tranche if in lower 20% of 5-bar range.
    recent_5 = df.tail(5)
    five_bar_high = float(recent_5["high"].max())
    five_bar_low = float(recent_5["low"].min())
    range_span = five_bar_high - five_bar_low
    lower_20_threshold = five_bar_low + (LOWER_RANGE_PCT * range_span)

    if range_span > 0.0 and current_price <= lower_20_threshold:
        buy_qty = round(TRANCHE_NOTIONAL_USD / current_price, QTY_PRECISION)
        rationale = (
            f"Lower-20% range LIMIT_BUY on {symbol}: ${current_price:.2f} <= "
            f"${lower_20_threshold:.2f} (5-bar ${five_bar_low:.2f}-${five_bar_high:.2f}); "
            f"sizing ${TRANCHE_NOTIONAL_USD:.2f} tranche."
        )
        return TradeIntent(
            symbol=symbol,
            action="LIMIT_BUY",
            target_price=current_price,
            qty=buy_qty,
            notional_usd=TRANCHE_NOTIONAL_USD,
            rationale=rationale,
        )

    return None


def scalper_expert_node(state: AgentState) -> AgentState:
    """Scan ``state['asset_symbols']`` and emit maker-only ``TradeIntent`` objects."""
    symbols = state.get("asset_symbols") or [state.get("asset_symbol", "BTC/USD")]
    intents: list[TradeIntent] = []

    for symbol in symbols:
        df = _get_symbol_dataframe(state, symbol)
        if df is None:
            continue
        intent = _evaluate_symbol_intent(state, symbol, df)
        if intent is not None:
            intents.append(intent)

    if intents:
        primary = intents[0]
        summary = "; ".join(f"{i.symbol}:{i.action}@{i.target_price}" for i in intents)
        print(f"[Scalper] Emitted {len(intents)} maker intent(s): {summary}")
        return {
            **state,
            "trade_intents": intents,
            "action": primary.action,
            "target_price": primary.target_price,
            "rationale": primary.rationale,
        }

    print("[Scalper] Emitted 0 intents | Action: HOLD")
    return {
        **state,
        "trade_intents": [],
        "action": "HOLD",
        "target_price": None,
        "rationale": "Scalper hold: no tickers met fee-adjusted or grid criteria.",
    }
