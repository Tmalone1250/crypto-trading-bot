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

from datetime import datetime, timezone
from typing import Any
import uuid
import duckdb
import pandas as pd

from config import (
    GRID_STOP_LOSS_PCT,
    GRID_TRANCHE_NOTIONAL,
    MAX_TRANCHES_PER_SYMBOL,
)
from state import AgentState, TradeIntent

DB_PATH: str = "agent_trades.duckdb"
FAIR_PRICE_STRATEGY: str = "FAIR_PRICE_REVERSION"
CRT_REBREAK_STRATEGY: str = "CRT_REBREAK"
SMC_GOLDEN_FVG_STRATEGY: str = "SMC_GOLDEN_FVG"
LVN_ABSORPTION_STRATEGY: str = "LVN_ABSORPTION_SQUEEZE"
DISPLACEMENT_THRESHOLD_PCT: float = 0.0020  # 0.20% minimum deviation from Fair Price anchor
DEFAULT_SCALP_SYMBOLS: tuple[str, ...] = ("BTC/USD", "ETH/USD", "SOL/USD")
FEE_ADJUSTED_PROFIT_MULTIPLIER: float = 1.015  # +1.5% gross target to clear 0.30% maker fees
TRANCHE_GRID_DROP_MULTIPLIER: float = 0.98     # -2.0% below avg_entry_price for next tranche
TRANCHE_NOTIONAL_USD: float = GRID_TRANCHE_NOTIONAL
TRANCHE_CAP_TOLERANCE: float = 0.95
TP_DRIFT_TOLERANCE_PCT: float = 0.0010         # 0.10% price shift threshold for resting TP sync
LOWER_RANGE_PCT: float = 0.20
QTY_PRECISION: int = 6


def _get_open_orders_for_symbol(
    state: AgentState,
    symbol: str,
    side: str | None = None,
) -> list[dict[str, Any]]:
    """Return open/active orders matching ``symbol`` (and optional ``side``)."""
    target_sym = symbol.replace("/", "").upper()
    raw_orders = list(state.get("open_orders") or []) + list(state.get("active_orders") or [])  # type: ignore[arg-type]
    matched: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for order in raw_orders:
        if not isinstance(order, dict):
            continue
        oid = str(order.get("id", ""))
        if oid and oid in seen_ids:
            continue
        ord_sym = str(order.get("symbol", "")).replace("/", "").upper()
        if ord_sym != target_sym:
            continue
        if side is not None and str(order.get("side", "")).lower() != side.lower():
            continue
        if oid:
            seen_ids.add(oid)
        matched.append(order)
    return matched


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

        # 0. Software Safety Stop-Loss: <= -4.0% below avg_entry_price.
        if avg_entry > 0.0 and current_price <= avg_entry * (1.0 - GRID_STOP_LOSS_PCT):
            sell_qty = (
                held_qty
                if held_qty > 0.0
                else round(TRANCHE_NOTIONAL_USD / current_price, QTY_PRECISION)
            )
            print(
                f"[Scalper] {symbol} Grid safety stop-loss triggered (-4.0%): "
                f"price=${current_price:.2f} <= ${avg_entry * (1.0 - GRID_STOP_LOSS_PCT):.2f}"
            )
            return TradeIntent(
                symbol=symbol,
                action="MARKET_SELL",
                target_price=current_price,
                qty=round(sell_qty, QTY_PRECISION),
                notional_usd=round(sell_qty * current_price, 2),
                rationale="Grid safety stop-loss triggered (-4.0%)",
            )

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

        # 1b. Resting +1.5% GTC Take-Profit Sync when an existing open sell order is stale.
        open_sells = _get_open_orders_for_symbol(state, symbol, side="sell")
        if avg_entry > 0.0 and held_qty > 0.0 and open_sells:
            tp_target = round(avg_entry * FEE_ADJUSTED_PROFIT_MULTIPLIER, 6)
            existing_sell = open_sells[0]
            try:
                existing_qty = float(existing_sell.get("qty", 0.0))
            except (TypeError, ValueError):
                existing_qty = 0.0
            try:
                existing_limit = float(existing_sell.get("limit_price", 0.0))
            except (TypeError, ValueError):
                existing_limit = 0.0

            qty_drifted = abs(existing_qty - held_qty) > 1e-6
            price_drifted = (
                existing_limit <= 0.0
                or (abs(existing_limit - tp_target) / tp_target) > TP_DRIFT_TOLERANCE_PCT
            )
            if qty_drifted or price_drifted:
                rationale = (
                    f"Resting +1.5% GTC TP sync on {symbol}: replacing stale sell "
                    f"(qty={existing_qty}->{held_qty}, tp=${existing_limit:.2f}->${tp_target:.2f})."
                )
                print(f"[Scalper] {rationale}")
                return TradeIntent(
                    symbol=symbol,
                    action="LIMIT_SELL",
                    target_price=tp_target,
                    qty=round(held_qty, QTY_PRECISION),
                    notional_usd=round(held_qty * tp_target, 2),
                    rationale=rationale,
                )

        # 2. Multi-Tranche Grid: allow additional $200 LIMIT_BUY only if down >= 2% from avg_entry.
        if avg_entry > 0.0 and current_price <= avg_entry * TRANCHE_GRID_DROP_MULTIPLIER:
            cost_basis = held_qty * avg_entry
            max_cost_basis = MAX_TRANCHES_PER_SYMBOL * GRID_TRANCHE_NOTIONAL * TRANCHE_CAP_TOLERANCE
            if cost_basis >= max_cost_basis:
                print(
                    f"[Scalper] {symbol} DCA buy blocked: cost_basis=${cost_basis:.2f} "
                    f"reached max {MAX_TRANCHES_PER_SYMBOL} tranches (${MAX_TRANCHES_PER_SYMBOL * GRID_TRANCHE_NOTIONAL:.2f})."
                )
                return None

            open_buys = _get_open_orders_for_symbol(state, symbol, side="buy")
            if open_buys:
                print(
                    f"[Scalper] {symbol} DCA buy blocked: {len(open_buys)} open buy order(s) already resting."
                )
                return None

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
    if _get_open_orders_for_symbol(state, symbol, side="buy"):
        return None
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


def _get_exec_1m_dataframe(state: AgentState, symbol: str) -> pd.DataFrame | None:
    """Resolve the 1-minute execution DataFrame (``{symbol}_1M``) or fall back to ``{symbol}``."""
    market_map = state.get("market_data_by_symbol") or {}
    for key in (f"{symbol}_1M", f"{symbol}:1M"):
        df_1m = market_map.get(key)
        if df_1m is not None and not df_1m.empty:
            return df_1m
    return _get_symbol_dataframe(state, symbol)


def _resolve_symbol_price(
    state: AgentState,
    symbol: str,
    df: pd.DataFrame | None,
) -> float:
    """Resolve the latest execution price for ``symbol``."""
    primary_sym = str(state.get("asset_symbol", "BTC/USD"))
    if symbol == primary_sym and float(state.get("current_price", 0.0)) > 0.0:
        return float(state["current_price"])
    if df is not None and not df.empty and "close" in df.columns:
        return float(df["close"].iloc[-1])
    return 0.0


def _fetch_symbol_setup(
    conn: duckdb.DuckDBPyConnection,
    strategy: str,
    phases: tuple[str, ...],
    symbol: str,
    primary_symbol: str,
) -> tuple[Any, ...] | None:
    """Fetch the latest active setup row for ``(strategy, symbol)`` from ``setup_state_machine``."""
    placeholders = ", ".join("?" for _ in phases)
    query = f"""
        SELECT setup_id, phase, sweep_low, target_price
        FROM setup_state_machine
        WHERE strategy = ?
          AND phase IN ({placeholders})
          AND (
              setup_id LIKE ?
              OR setup_id LIKE ?
              OR (? = ? AND setup_id NOT LIKE '%:%')
          )
        ORDER BY timestamp DESC
        LIMIT 1
    """
    params: list[Any] = [
        strategy,
        *phases,
        f"%:{symbol}:%",
        f"%:{symbol}",
        symbol,
        primary_symbol,
    ]
    return conn.execute(query, params).fetchone()


def _summarize_active_symbol_phases(symbols: list[str], primary_symbol: str) -> str | None:
    """Build a compact multi-symbol phase summary (e.g., ``BTC: PHASE_1 | ETH: HOLD``)."""
    conn = duckdb.connect(DB_PATH)
    try:
        _ensure_setup_table(conn)
        parts: list[str] = []
        any_active = False
        active_phases = (
            "PHASE_0_ANCHORED",
            "PHASE_1_DISPLACED",
            "PHASE_1_SWEPT",
            "PHASE_2_FIRST_BREAK",
            "PHASE_3_PULLBACK",
            "PHASE_1_POI_TAPPED",
            "PHASE_2_CHOCH",
            "PHASE_1_LVN_IDENTIFIED",
            "PHASE_2_ABSORPTION",
        )
        placeholders = ", ".join("?" for _ in active_phases)
        for sym in symbols:
            short = sym.split("/")[0]
            row = conn.execute(
                f"""
                SELECT phase
                FROM setup_state_machine
                WHERE phase IN ({placeholders})
                  AND (
                      setup_id LIKE ?
                      OR setup_id LIKE ?
                      OR (? = ? AND setup_id NOT LIKE '%:%')
                  )
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                [*active_phases, f"%:{sym}:%", f"%:{sym}", sym, primary_symbol],
            ).fetchone()
            if row is not None and row[0]:
                any_active = True
                parts.append(f"{short}: {row[0]}")
            else:
                parts.append(f"{short}: HOLD")
        return " | ".join(parts) if any_active else None
    except Exception:
        return None
    finally:
        conn.close()


def _evaluate_session_open_reversion_for_symbol(
    state: AgentState,
    symbol: str,
) -> tuple[TradeIntent | None, str]:
    """Evaluate J.J. Simon Fair Price Reversion Phase 1/2 for a single ``symbol``."""
    primary_sym = str(state.get("asset_symbol", "BTC/USD"))
    df = _get_exec_1m_dataframe(state, symbol)
    if df is None or df.empty or not {"open", "high", "low", "close"}.issubset(df.columns):
        return None, f"{symbol.split('/')[0]}: HOLD"

    current_price = _resolve_symbol_price(state, symbol, df)
    if current_price <= 0.0:
        return None, f"{symbol.split('/')[0]}: HOLD"

    conn = duckdb.connect(DB_PATH)
    try:
        _ensure_setup_table(conn)
        row = _fetch_symbol_setup(
            conn=conn,
            strategy=FAIR_PRICE_STRATEGY,
            phases=("PHASE_0_ANCHORED", "PHASE_1_DISPLACED"),
            symbol=symbol,
            primary_symbol=primary_sym,
        )

        if row is None or row[3] is None:
            return None, f"{symbol.split('/')[0]}: HOLD"

        setup_id, phase, fair_price = str(row[0]), str(row[1]), float(row[3])
        if fair_price <= 0.0:
            return None, f"{symbol.split('/')[0]}: HOLD"

        if phase == "PHASE_0_ANCHORED":
            displacement_pct = abs(current_price - fair_price) / fair_price
            if displacement_pct >= DISPLACEMENT_THRESHOLD_PCT:
                sweep_extreme = (
                    float(df.tail(5)["low"].min())
                    if current_price < fair_price
                    else float(df.tail(5)["high"].max())
                )
                conn.execute(
                    """
                    UPDATE setup_state_machine
                    SET phase = 'PHASE_1_DISPLACED', sweep_low = ?
                    WHERE setup_id = ?
                    """,
                    [sweep_extreme, setup_id],
                )
                print(
                    f"[Scalper] {setup_id} Phase 1 Displaced: price ${current_price:.2f} "
                    f"deviated {displacement_pct:.3%} from Fair Price ${fair_price:.2f}"
                )
                return None, f"{symbol.split('/')[0]}: PHASE_1_DISPLACED"

            return None, f"{symbol.split('/')[0]}: PHASE_0_ANCHORED"

        if phase == "PHASE_1_DISPLACED" and len(df) >= 2:
            prev = df.iloc[-2]
            curr = df.iloc[-1]
            prev_body = abs(float(prev["close"]) - float(prev["open"]))
            curr_body = abs(float(curr["close"]) - float(curr["open"]))
            curr_open = float(curr["open"])
            curr_close = float(curr["close"])
            prev_high = float(prev["high"])
            prev_low = float(prev["low"])

            triggered_action: str | None = None
            raw_sl: float | None = None

            if curr_body > prev_body:
                if curr_close < fair_price and curr_close > curr_open and curr_close > prev_high:
                    triggered_action = "LIMIT_BUY"
                    raw_sl = float(df.tail(5)["low"].min())
                elif curr_close > fair_price and curr_close < curr_open and curr_close < prev_low:
                    triggered_action = "LIMIT_SELL"
                    raw_sl = float(df.tail(5)["high"].max())

            if triggered_action is not None and raw_sl is not None:
                conn.execute(
                    """
                    UPDATE setup_state_machine
                    SET phase = 'PHASE_2_TRIGGERED'
                    WHERE setup_id = ?
                    """,
                    [setup_id],
                )
                entry_price = curr_close
                rationale = (
                    f"Fair Price Reversion {triggered_action} on {symbol} ({setup_id}): "
                    f"1m displacement candle broke structure toward ${fair_price:.2f} "
                    f"(entry=${entry_price:.2f}, raw_sl=${raw_sl:.2f})."
                )
                intent = TradeIntent(
                    symbol=symbol,
                    action=triggered_action,  # type: ignore[arg-type]
                    target_price=entry_price,
                    qty=round(TRANCHE_NOTIONAL_USD / entry_price, QTY_PRECISION),
                    notional_usd=TRANCHE_NOTIONAL_USD,
                    raw_stop_loss=raw_sl,
                    rationale=rationale,
                )
                print(f"[Scalper] {rationale}")
                return intent, f"{symbol.split('/')[0]}: PHASE_2_TRIGGERED"
    finally:
        conn.close()

    return None, f"{symbol.split('/')[0]}: PHASE_1_DISPLACED"


def _evaluate_session_open_reversion(
    state: AgentState,
    symbol: str | None = None,
) -> AgentState:
    """Execute J.J. Simon Fair Price Reversion across all watched symbols in ``SESSION_OPEN_WATCH``."""
    symbols = [symbol] if symbol is not None else list(
        state.get("asset_symbols") or [state.get("asset_symbol", "BTC/USD")]
    )
    intents: list[TradeIntent] = []
    status_parts: list[str] = []

    for sym in symbols:
        intent, status = _evaluate_session_open_reversion_for_symbol(state, sym)
        status_parts.append(status)
        if intent is not None:
            intents.append(intent)

    if intents:
        primary = intents[0]
        return {
            **state,
            "trade_intents": intents,
            "action": primary.action,
            "target_price": primary.target_price,
            "rationale": "; ".join(i.rationale for i in intents),
        }

    return {
        **state,
        "trade_intents": [],
        "action": "HOLD",
        "target_price": None,
        "rationale": " | ".join(status_parts)
        if status_parts
        else "Session Open Watch: awaiting Phase 2 reversion candle.",
    }


def _get_htf_1h_dataframe(
    state: AgentState,
    symbol: str,
    df_exec: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Resolve 1-Hour OHLCV candles from ``market_data_by_symbol`` or resample ``df_exec``."""
    market_map = state.get("market_data_by_symbol") or {}
    for key in (f"{symbol}_1H", f"{symbol}:1H", "1H"):
        htf = market_map.get(key)
        if htf is not None and not htf.empty:
            return htf

    if df_exec is None or df_exec.empty:
        return None

    work_df = df_exec
    if not isinstance(work_df.index, pd.DatetimeIndex):
        if "timestamp" in work_df.columns:
            work_df = work_df.set_index(pd.to_datetime(work_df["timestamp"]))
        else:
            return None

    if not {"high", "low", "close"}.issubset(work_df.columns):
        return None

    agg_spec: dict[str, str] = {"high": "max", "low": "min", "close": "last"}
    if "open" in work_df.columns:
        agg_spec["open"] = "first"

    resampled = work_df.resample("1h").agg(agg_spec).dropna(subset=["high", "low", "close"])
    return resampled if len(resampled) >= 2 else None


def _evaluate_crt_rebreak(
    state: AgentState,
    symbol: str | None = None,
) -> TradeIntent | None:
    """Execute the 24/7 CRT Re-break 4-phase state machine for ``symbol``."""
    primary_sym = str(state.get("asset_symbol", "BTC/USD"))
    target_sym = symbol or primary_sym
    df_exec = _get_exec_1m_dataframe(state, target_sym)
    current_price = _resolve_symbol_price(state, target_sym, df_exec)

    if current_price <= 0.0:
        return None

    conn = duckdb.connect(DB_PATH)
    try:
        _ensure_setup_table(conn)

        row = _fetch_symbol_setup(
            conn=conn,
            strategy=CRT_REBREAK_STRATEGY,
            phases=("PHASE_1_SWEPT", "PHASE_2_FIRST_BREAK", "PHASE_3_PULLBACK"),
            symbol=target_sym,
            primary_symbol=primary_sym,
        )

        # Phase 0 & Phase 1: Monitor 1H candles when no CRT setup is active for target_sym.
        if row is None:
            df_1h = _get_htf_1h_dataframe(state, target_sym, df_exec)
            if df_1h is None or len(df_1h) < 2 or not {"high", "low", "close"}.issubset(df_1h.columns):
                return None

            candle_1 = df_1h.iloc[-2]
            candle_2 = df_1h.iloc[-1]
            h1, l1 = float(candle_1["high"]), float(candle_1["low"])
            h2, l2, c2_close = float(candle_2["high"]), float(candle_2["low"]), float(candle_2["close"])

            direction: str | None = None
            target_extreme: float | None = None

            # Swept L1 and reversed back inside Candle 1 range -> LONG targeting H1
            if l2 < l1 and l1 < c2_close < h1:
                direction = "LONG"
                target_extreme = h1
            # Swept H1 and reversed back inside Candle 1 range -> SHORT targeting L1
            elif h2 > h1 and l1 < c2_close < h1:
                direction = "SHORT"
                target_extreme = l1

            if direction is None or target_extreme is None:
                return None

            midpoint = (h2 + l2) / 2.0
            now_naive = datetime.now(timezone.utc).replace(tzinfo=None)
            setup_id = (
                f"crt:{target_sym}:{direction}-"
                f"{now_naive.strftime('%Y%m%d%H%M')}-{uuid.uuid4().hex[:6]}"
            )
            conn.execute(
                """
                INSERT INTO setup_state_machine (
                    setup_id,
                    strategy,
                    phase,
                    sweep_low,
                    target_price,
                    timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [setup_id, CRT_REBREAK_STRATEGY, "PHASE_1_SWEPT", midpoint, target_extreme, now_naive],
            )
            print(
                f"[Scalper] CRT Phase 1 Swept ({setup_id}): M={midpoint:.2f}, "
                f"Target={target_extreme:.2f}"
            )
            return None

        setup_id, phase = str(row[0]), str(row[1])
        if row[2] is None or row[3] is None:
            return None
        midpoint, target_extreme = float(row[2]), float(row[3])
        is_long = ":LONG-" in setup_id or setup_id.startswith("CRT-LONG") or target_extreme > midpoint

        # Phase 2 (First Break): Price crosses M in trade direction (no entry yet).
        if phase == "PHASE_1_SWEPT":
            crossed_first = current_price > midpoint if is_long else current_price < midpoint
            if crossed_first:
                conn.execute(
                    "UPDATE setup_state_machine SET phase = 'PHASE_2_FIRST_BREAK' WHERE setup_id = ?",
                    [setup_id],
                )
                print(f"[Scalper] CRT {setup_id} -> PHASE_2_FIRST_BREAK @ ${current_price:.2f}")
            return None

        # Phase 3 (Pullback): Price pulls back across M against trade direction.
        if phase == "PHASE_2_FIRST_BREAK":
            pulled_back = current_price < midpoint if is_long else current_price > midpoint
            if pulled_back:
                conn.execute(
                    "UPDATE setup_state_machine SET phase = 'PHASE_3_PULLBACK' WHERE setup_id = ?",
                    [setup_id],
                )
                print(f"[Scalper] CRT {setup_id} -> PHASE_3_PULLBACK @ ${current_price:.2f}")
            return None

        # Phase 4 (Re-break Execution): Price crosses M again in trade direction -> emit TradeIntent.
        if phase == "PHASE_3_PULLBACK":
            rebreak = current_price >= midpoint if is_long else current_price <= midpoint
            if not rebreak:
                return None

            target_distance = abs(target_extreme - midpoint)
            if target_distance <= 0.0:
                return None

            sl_distance = target_distance / 1.5
            raw_stop_loss = (midpoint - sl_distance) if is_long else (midpoint + sl_distance)
            if raw_stop_loss <= 0.0:
                return None

            conn.execute(
                "UPDATE setup_state_machine SET phase = 'PHASE_4_EXECUTED' WHERE setup_id = ?",
                [setup_id],
            )
            action = "LIMIT_BUY" if is_long else "LIMIT_SELL"
            rationale = (
                f"CRT Re-break Phase 4 {action} on {target_sym} ({setup_id}): "
                f"entry at M=${midpoint:.2f}, target=${target_extreme:.2f}, "
                f"raw_sl=${raw_stop_loss:.2f} (1.5R)."
            )
            print(f"[Scalper] {rationale}")
            return TradeIntent(
                symbol=target_sym,
                action=action,  # type: ignore[arg-type]
                target_price=round(midpoint, 6),
                qty=round(TRANCHE_NOTIONAL_USD / midpoint, QTY_PRECISION),
                notional_usd=TRANCHE_NOTIONAL_USD,
                raw_stop_loss=round(raw_stop_loss, 6),
                rationale=rationale,
            )
    finally:
        conn.close()

    return None


def _ensure_setup_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Ensure ``setup_state_machine`` exists in DuckDB."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS setup_state_machine (
            setup_id VARCHAR PRIMARY KEY,
            strategy VARCHAR NOT NULL,
            phase VARCHAR NOT NULL,
            sweep_low DOUBLE,
            target_price DOUBLE,
            timestamp TIMESTAMP NOT NULL
        )
        """
    )


def _get_mtf_15m_dataframe(
    state: AgentState,
    symbol: str,
    df_exec: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Resolve 15-minute OHLCV candles from ``market_data_by_symbol`` or resample ``df_exec``."""
    market_map = state.get("market_data_by_symbol") or {}
    for key in (f"{symbol}_15M", f"{symbol}:15M", "15M"):
        mtf = market_map.get(key)
        if mtf is not None and not mtf.empty:
            return mtf

    if df_exec is None or df_exec.empty:
        return None

    work_df = df_exec
    if not isinstance(work_df.index, pd.DatetimeIndex):
        if "timestamp" in work_df.columns:
            work_df = work_df.set_index(pd.to_datetime(work_df["timestamp"]))
        else:
            return None

    if not {"high", "low", "close"}.issubset(work_df.columns):
        return None

    agg_spec: dict[str, str] = {"high": "max", "low": "min", "close": "last"}
    if "open" in work_df.columns:
        agg_spec["open"] = "first"

    resampled = work_df.resample("15min").agg(agg_spec).dropna(subset=["high", "low", "close"])
    return resampled if len(resampled) >= 3 else None


def _find_tapped_15m_fvg(
    df_15m: pd.DataFrame,
    current_price: float,
) -> tuple[str, float, float] | None:
    """Locate an unmitigated 15m FVG currently tapped by ``current_price``."""
    if len(df_15m) < 3 or not {"high", "low", "close"}.issubset(df_15m.columns):
        return None

    for idx in range(len(df_15m) - 1, 1, -1):
        c0 = df_15m.iloc[idx - 2]
        c2 = df_15m.iloc[idx]
        c0_high, c0_low = float(c0["high"]), float(c0["low"])
        c2_high, c2_low = float(c2["high"]), float(c2["low"])

        # Bullish Demand FVG: c0.high < c2.low
        if c0_high < c2_low:
            fvg_low, fvg_high = c0_high, c2_low
            subsequent = df_15m.iloc[idx + 1 :]
            mitigated = not subsequent.empty and (subsequent["close"] < fvg_low).any()
            if not mitigated and fvg_low <= current_price <= fvg_high:
                return ("LONG", fvg_low, fvg_high)

        # Bearish Supply FVG: c0.low > c2.high
        if c0_low > c2_high:
            fvg_low, fvg_high = c2_high, c0_low
            subsequent = df_15m.iloc[idx + 1 :]
            mitigated = not subsequent.empty and (subsequent["close"] > fvg_high).any()
            if not mitigated and fvg_low <= current_price <= fvg_high:
                return ("SHORT", fvg_low, fvg_high)

    return None


def _find_latest_1m_fvg(df_1m: pd.DataFrame, is_long: bool) -> tuple[float, float] | None:
    """Return ``(fvg_low, fvg_high)`` of the most recent 1m FVG matching direction."""
    if len(df_1m) < 3:
        return None
    for idx in range(len(df_1m) - 1, 1, -1):
        c0_high = float(df_1m.iloc[idx - 2]["high"])
        c0_low = float(df_1m.iloc[idx - 2]["low"])
        c2_high = float(df_1m.iloc[idx]["high"])
        c2_low = float(df_1m.iloc[idx]["low"])
        if is_long and c0_high < c2_low:
            return (c0_high, c2_low)
        if not is_long and c0_low > c2_high:
            return (c2_high, c0_low)
    return None


def _evaluate_smc_golden_fvg(
    state: AgentState,
    symbol: str | None = None,
) -> TradeIntent | None:
    """Strategy 3: MTF Market Shift + Golden Zone (50%-61.8%) FVG Limit Entry for ``symbol``."""
    primary_sym = str(state.get("asset_symbol", "BTC/USD"))
    target_sym = symbol or primary_sym
    df_exec = _get_exec_1m_dataframe(state, target_sym)
    current_price = _resolve_symbol_price(state, target_sym, df_exec)

    if current_price <= 0.0:
        return None

    conn = duckdb.connect(DB_PATH)
    try:
        _ensure_setup_table(conn)
        row = _fetch_symbol_setup(
            conn=conn,
            strategy=SMC_GOLDEN_FVG_STRATEGY,
            phases=("PHASE_1_POI_TAPPED", "PHASE_2_CHOCH"),
            symbol=target_sym,
            primary_symbol=primary_sym,
        )

        # Phase 1 (PHASE_1_POI_TAPPED): 15m FVG tap detection
        if row is None:
            df_15m = _get_mtf_15m_dataframe(state, target_sym, df_exec)
            if df_15m is None:
                return None
            tapped = _find_tapped_15m_fvg(df_15m, current_price)
            if tapped is None:
                return None

            direction, fvg_low, fvg_high = tapped
            now_naive = datetime.now(timezone.utc).replace(tzinfo=None)
            setup_id = (
                f"smc_fvg:{target_sym}:{direction}-"
                f"{now_naive.strftime('%Y%m%d%H%M')}-{uuid.uuid4().hex[:6]}"
            )
            conn.execute(
                """
                INSERT INTO setup_state_machine (
                    setup_id, strategy, phase, sweep_low, target_price, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [setup_id, SMC_GOLDEN_FVG_STRATEGY, "PHASE_1_POI_TAPPED", fvg_low, fvg_high, now_naive],
            )
            print(
                f"[Scalper] SMC Phase 1 POI Tapped ({setup_id}): "
                f"15m FVG [{fvg_low:.2f}, {fvg_high:.2f}]"
            )
            return None

        if df_exec is None or len(df_exec) < 5 or not {"open", "high", "low", "close"}.issubset(df_exec.columns):
            return None

        setup_id, phase = str(row[0]), str(row[1])
        is_long = ":LONG-" in setup_id or setup_id.startswith("SMC-LONG")

        # Phase 2 (PHASE_2_CHOCH): 1m liquidity sweep + full-body CHoCH + new 1m FVG
        if phase == "PHASE_1_POI_TAPPED":
            prior_bars = df_exec.iloc[:-1]
            curr = df_exec.iloc[-1]
            curr_open, curr_close = float(curr["open"]), float(curr["close"])
            fvg_1m = _find_latest_1m_fvg(df_exec, is_long=is_long)
            if fvg_1m is None:
                return None

            if is_long:
                prior_swing_high = float(prior_bars.iloc[:-1]["high"].max()) if len(prior_bars) > 1 else float(prior_bars["high"].max())
                prior_swing_low = float(prior_bars.iloc[:-1]["low"].min()) if len(prior_bars) > 1 else float(prior_bars["low"].min())
                sweep_extreme = float(df_exec["low"].min())
                swept = sweep_extreme <= prior_swing_low
                choch = curr_close > curr_open and curr_close > prior_swing_high
                disp_extreme = float(df_exec["high"].max())
            else:
                prior_swing_low = float(prior_bars.iloc[:-1]["low"].min()) if len(prior_bars) > 1 else float(prior_bars["low"].min())
                prior_swing_high = float(prior_bars.iloc[:-1]["high"].max()) if len(prior_bars) > 1 else float(prior_bars["high"].max())
                sweep_extreme = float(df_exec["high"].max())
                swept = sweep_extreme >= prior_swing_high
                choch = curr_close < curr_open and curr_close < prior_swing_low
                disp_extreme = float(df_exec["low"].min())

            if swept and choch:
                conn.execute(
                    """
                    UPDATE setup_state_machine
                    SET phase = 'PHASE_2_CHOCH', sweep_low = ?, target_price = ?
                    WHERE setup_id = ?
                    """,
                    [sweep_extreme, disp_extreme, setup_id],
                )
                print(
                    f"[Scalper] SMC Phase 2 CHoCH ({setup_id}): "
                    f"sweep={sweep_extreme:.2f}, disp={disp_extreme:.2f}"
                )
            return None

        # Phase 3 (PHASE_3_LIMIT_ARMED): 50%-61.8% Golden Zone aligned with 1m FVG
        if phase == "PHASE_2_CHOCH":
            if row[2] is None or row[3] is None:
                return None
            sweep_extreme, disp_extreme = float(row[2]), float(row[3])
            fvg_1m = _find_latest_1m_fvg(df_exec, is_long=is_long)
            if fvg_1m is None:
                return None
            fvg_low, fvg_high = fvg_1m

            leg_span = abs(disp_extreme - sweep_extreme)
            if leg_span <= 0.0:
                return None

            if is_long:
                fib_50 = disp_extreme - (0.50 * leg_span)
                fib_618 = disp_extreme - (0.618 * leg_span)
                gz_low, gz_high = min(fib_50, fib_618), max(fib_50, fib_618)
                raw_sl = sweep_extreme * 0.9999
                action = "LIMIT_BUY"
            else:
                fib_50 = disp_extreme + (0.50 * leg_span)
                fib_618 = disp_extreme + (0.618 * leg_span)
                gz_low, gz_high = min(fib_50, fib_618), max(fib_50, fib_618)
                raw_sl = sweep_extreme * 1.0001
                action = "LIMIT_SELL"

            overlap_low = max(gz_low, fvg_low)
            overlap_high = min(gz_high, fvg_high)
            if overlap_low > overlap_high:
                return None

            entry_price = (overlap_low + overlap_high) / 2.0
            conn.execute(
                "UPDATE setup_state_machine SET phase = 'PHASE_3_LIMIT_ARMED' WHERE setup_id = ?",
                [setup_id],
            )
            rationale = (
                f"SMC Golden FVG {action} on {target_sym} ({setup_id}): "
                f"Golden Zone [{gz_low:.2f}, {gz_high:.2f}] aligned with 1m FVG "
                f"[{fvg_low:.2f}, {fvg_high:.2f}] -> entry=${entry_price:.2f}, raw_sl=${raw_sl:.2f}."
            )
            print(f"[Scalper] {rationale}")
            return TradeIntent(
                symbol=target_sym,
                action=action,  # type: ignore[arg-type]
                target_price=round(entry_price, 6),
                qty=round(TRANCHE_NOTIONAL_USD / entry_price, QTY_PRECISION),
                notional_usd=TRANCHE_NOTIONAL_USD,
                raw_stop_loss=round(raw_sl, 6),
                rationale=rationale,
            )
    finally:
        conn.close()

    return None


def _compute_rolling_val_and_lvn(
    df_1m: pd.DataFrame,
    bins: int = 12,
) -> tuple[float, float] | None:
    """Compute 60-bar rolling Volume Profile and return ``(lvn_low, lvn_high)`` coinciding with VAL."""
    if df_1m is None or len(df_1m) < 60 or not {"high", "low", "close", "volume"}.issubset(df_1m.columns):
        return None

    window = df_1m.tail(60)
    price_min = float(window["low"].min())
    price_max = float(window["high"].max())
    if price_max <= price_min:
        return None

    bin_width = (price_max - price_min) / bins
    vol_by_bin = [0.0] * bins
    for _, bar in window.iterrows():
        typical = (float(bar["high"]) + float(bar["low"]) + float(bar["close"])) / 3.0
        vol = float(bar["volume"])
        idx = min(bins - 1, max(0, int((typical - price_min) / bin_width)))
        vol_by_bin[idx] += vol

    total_vol = sum(vol_by_bin)
    if total_vol <= 0.0:
        return None

    poc_idx = max(range(bins), key=lambda i: vol_by_bin[i])
    captured = vol_by_bin[poc_idx]
    low_idx = poc_idx
    high_idx = poc_idx

    while captured < 0.70 * total_vol and (low_idx > 0 or high_idx < bins - 1):
        left_vol = vol_by_bin[low_idx - 1] if low_idx > 0 else -1.0
        right_vol = vol_by_bin[high_idx + 1] if high_idx < bins - 1 else -1.0
        if left_vol >= right_vol and low_idx > 0:
            low_idx -= 1
            captured += vol_by_bin[low_idx]
        elif high_idx < bins - 1:
            high_idx += 1
            captured += vol_by_bin[high_idx]
        else:
            break

    mean_vol = total_vol / bins
    # Check if VAL bin (or adjacent bin) is a Low Volume Node (< 60% of mean bin volume)
    candidate_indices = [i for i in range(max(0, low_idx - 1), min(bins, low_idx + 2))]
    lvn_idx = min(candidate_indices, key=lambda i: vol_by_bin[i])
    if vol_by_bin[lvn_idx] >= 0.60 * mean_vol:
        return None

    lvn_low = price_min + (lvn_idx * bin_width)
    lvn_high = lvn_low + bin_width
    return (lvn_low, lvn_high)


def _evaluate_lvn_absorption_squeeze(
    state: AgentState,
    symbol: str | None = None,
) -> TradeIntent | None:
    """Strategy 4: 60-bar Rolling Volume Profile VAL/LVN Absorption Squeeze for ``symbol``."""
    primary_sym = str(state.get("asset_symbol", "BTC/USD"))
    target_sym = symbol or primary_sym
    df_exec = _get_exec_1m_dataframe(state, target_sym)
    if df_exec is None or df_exec.empty or not {"open", "high", "low", "close"}.issubset(df_exec.columns):
        return None

    current_price = _resolve_symbol_price(state, target_sym, df_exec)
    if current_price <= 0.0:
        return None

    conn = duckdb.connect(DB_PATH)
    try:
        _ensure_setup_table(conn)
        row = _fetch_symbol_setup(
            conn=conn,
            strategy=LVN_ABSORPTION_STRATEGY,
            phases=("PHASE_1_LVN_IDENTIFIED", "PHASE_2_ABSORPTION"),
            symbol=target_sym,
            primary_symbol=primary_sym,
        )

        # Phase 1 (PHASE_1_LVN_IDENTIFIED): 60-bar rolling Volume Profile VAL + LVN
        if row is None:
            lvn_zone = _compute_rolling_val_and_lvn(df_exec)
            if lvn_zone is None:
                return None
            lvn_low, lvn_high = lvn_zone
            now_naive = datetime.now(timezone.utc).replace(tzinfo=None)
            setup_id = (
                f"lvn:{target_sym}:LONG-"
                f"{now_naive.strftime('%Y%m%d%H%M')}-{uuid.uuid4().hex[:6]}"
            )
            conn.execute(
                """
                INSERT INTO setup_state_machine (
                    setup_id, strategy, phase, sweep_low, target_price, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [setup_id, LVN_ABSORPTION_STRATEGY, "PHASE_1_LVN_IDENTIFIED", lvn_low, lvn_high, now_naive],
            )
            print(
                f"[Scalper] LVN Phase 1 Identified ({setup_id}): "
                f"VAL/LVN zone [{lvn_low:.2f}, {lvn_high:.2f}]"
            )
            return None

        setup_id, phase = str(row[0]), str(row[1])
        if row[2] is None or row[3] is None:
            return None

        # Phase 2 (PHASE_2_ABSORPTION): Price enters LVN with high sell volume + narrow body / lower wick
        if phase == "PHASE_1_LVN_IDENTIFIED":
            if "volume" not in df_exec.columns or len(df_exec) < 20:
                return None
            lvn_low, lvn_high = float(row[2]), float(row[3])
            curr = df_exec.iloc[-1]
            curr_open, curr_high, curr_low, curr_close = (
                float(curr["open"]),
                float(curr["high"]),
                float(curr["low"]),
                float(curr["close"]),
            )
            curr_vol = float(curr["volume"])
            avg_vol = float(df_exec["volume"].tail(20).mean())
            candle_range = curr_high - curr_low
            if candle_range <= 0.0 or avg_vol <= 0.0:
                return None

            body = abs(curr_close - curr_open)
            lower_wick = min(curr_open, curr_close) - curr_low
            in_lvn = curr_low <= lvn_high and curr_close >= (lvn_low * 0.999)
            high_rel_vol = curr_vol >= 1.5 * avg_vol
            no_follow_through = (body <= 0.35 * candle_range) or (lower_wick >= 0.50 * candle_range)

            if in_lvn and high_rel_vol and no_follow_through:
                conn.execute(
                    """
                    UPDATE setup_state_machine
                    SET phase = 'PHASE_2_ABSORPTION', sweep_low = ?, target_price = ?
                    WHERE setup_id = ?
                    """,
                    [curr_low, curr_high, setup_id],
                )
                print(
                    f"[Scalper] LVN Phase 2 Absorption ({setup_id}): "
                    f"vol={curr_vol:.1f} (>1.5x avg), low=${curr_low:.2f}, high=${curr_high:.2f}"
                )
            return None

        # Phase 3 (PHASE_3_SQUEEZE_EXECUTION): Break above absorption candle high -> MARKET_BUY
        if phase == "PHASE_2_ABSORPTION":
            absorption_low, absorption_high = float(row[2]), float(row[3])
            if current_price > absorption_high:
                conn.execute(
                    "UPDATE setup_state_machine SET phase = 'PHASE_3_SQUEEZE_EXECUTION' WHERE setup_id = ?",
                    [setup_id],
                )
                rationale = (
                    f"LVN Absorption Squeeze MARKET_BUY on {target_sym} ({setup_id}): "
                    f"price ${current_price:.2f} broke above absorption high ${absorption_high:.2f}, "
                    f"raw_sl=${absorption_low:.2f}."
                )
                print(f"[Scalper] {rationale}")
                return TradeIntent(
                    symbol=target_sym,
                    action="MARKET_BUY",
                    target_price=round(current_price, 6),
                    qty=round(TRANCHE_NOTIONAL_USD / current_price, QTY_PRECISION),
                    notional_usd=TRANCHE_NOTIONAL_USD,
                    raw_stop_loss=round(absorption_low, 6),
                    rationale=rationale,
                )
    finally:
        conn.close()

    return None


def scalper_expert_node(state: AgentState) -> AgentState:
    """Scan ``state['asset_symbols']`` across all strategies and emit ``TradeIntent`` objects."""
    if (
        str(state.get("market_regime", "")).upper() == "SUSPENDED"
        or str(state.get("action", "")).upper() == "CIRCUIT_BREAKER"
    ):
        print("[Scalper] Regime SUSPENDED (CIRCUIT_BREAKER) | Bypassing entries, emitting 0 intents")
        return {
            **state,
            "trade_intents": [],
            "action": "CIRCUIT_BREAKER",
            "target_price": None,
            "rationale": "Trading SUSPENDED by macro circuit breaker. Emitting 0 intents.",
        }

    if str(state.get("market_regime", "")).upper() == "SESSION_OPEN_WATCH":
        return _evaluate_session_open_reversion(state)

    primary_symbol = str(state.get("asset_symbol", "BTC/USD"))
    symbols = list(state.get("asset_symbols") or [primary_symbol])
    strategy_intents: list[TradeIntent] = []

    for sym in symbols:
        for strategy_fn in (
            _evaluate_crt_rebreak,
            _evaluate_smc_golden_fvg,
            _evaluate_lvn_absorption_squeeze,
        ):
            strategy_intent = strategy_fn(state, symbol=sym)
            if strategy_intent is not None:
                strategy_intents.append(strategy_intent)
                break

    if strategy_intents:
        primary = strategy_intents[0]
        return {
            **state,
            "trade_intents": strategy_intents,
            "action": primary.action,
            "target_price": primary.target_price,
            "rationale": "; ".join(i.rationale for i in strategy_intents),
        }

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

    phase_summary = _summarize_active_symbol_phases(symbols, primary_symbol)
    print("[Scalper] Emitted 0 intents | Action: HOLD")
    return {
        **state,
        "trade_intents": [],
        "action": "HOLD",
        "target_price": None,
        "rationale": phase_summary or "Scalper hold: no tickers met fee-adjusted or grid criteria.",
    }
