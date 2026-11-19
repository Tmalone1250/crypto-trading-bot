"""Regime Supervisor Node: deterministic volatility, session-open, and circuit-breaker gate.

Inspects the most recent 5 candles in ``state["historical_data"]``, detects major
session opens (New York 08:30 CST, London, Asia) for J.J. Simon's Fair Price
Reversion state machine in DuckDB, enforces the Three-Loss Rule via
``session_metrics``, and gates LLM activation on BOTH:
  1. 5-bar range >= 0.25%
  2. Price movement >= 0.20% from ``state["last_alpha_eval_price"]`` (when > 0.0)
"""

from datetime import date, datetime, timezone
import uuid
from zoneinfo import ZoneInfo
import duckdb

from state import AgentState

DB_PATH: str = "agent_trades.duckdb"
LOOKBACK_CANDLES: int = 5
VOLATILITY_THRESHOLD_PCT: float = 0.25
ALPHA_REENTRY_MOVE_PCT: float = 0.20
FAIR_PRICE_STRATEGY: str = "FAIR_PRICE_REVERSION"
THREE_LOSS_LIMIT: int = 3
FLAT_RATIONALE: str = "Market is flat. Skipping LLM execution to preserve rate limits."
THREE_LOSS_RATIONALE: str = (
    "Three-Loss Rule active (consecutive_losses >= 3). "
    "Holding FLAT to prevent revenge trading."
)

_NY_TZ = ZoneInfo("America/Chicago")
_LONDON_TZ = ZoneInfo("Europe/London")
_ASIA_TZ = ZoneInfo("Asia/Tokyo")


def detect_session_open(now: datetime | None = None) -> str | None:
    """Detect the exact minute of New York (08:30 CST), London, or Asia session opens."""
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)

    ny_time = current.astimezone(_NY_TZ)
    if ny_time.hour == 8 and ny_time.minute == 30:
        return "NEW_YORK"

    london_time = current.astimezone(_LONDON_TZ)
    if london_time.hour == 8 and london_time.minute == 0:
        return "LONDON"

    asia_time = current.astimezone(_ASIA_TZ)
    if asia_time.hour == 9 and asia_time.minute == 0:
        return "ASIA"

    return None


def _ensure_strategy_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Ensure ``setup_state_machine`` and ``session_metrics`` tables exist in DuckDB."""
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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS session_metrics (
            date DATE PRIMARY KEY,
            realized_pnl DOUBLE DEFAULT 0.0,
            consecutive_losses INTEGER DEFAULT 0
        )
        """
    )


def _get_consecutive_losses(conn: duckdb.DuckDBPyConnection, current_date: date) -> int:
    """Query ``session_metrics`` for ``consecutive_losses`` on ``current_date``."""
    row = conn.execute(
        "SELECT consecutive_losses FROM session_metrics WHERE date = ?",
        [current_date],
    ).fetchone()
    if row is None or row[0] is None:
        return 0
    return int(row[0])


def _has_active_fair_price_setup(conn: duckdb.DuckDBPyConnection) -> bool:
    """Return True if an active ``FAIR_PRICE_REVERSION`` setup is awaiting Phase 1/2."""
    row = conn.execute(
        """
        SELECT setup_id
        FROM setup_state_machine
        WHERE strategy = ?
          AND phase IN ('PHASE_0_ANCHORED', 'PHASE_1_DISPLACED')
        ORDER BY timestamp DESC
        LIMIT 1
        """,
        [FAIR_PRICE_STRATEGY],
    ).fetchone()
    return row is not None


def _anchor_fair_price_setup(
    conn: duckdb.DuckDBPyConnection,
    session_name: str,
    fair_price: float,
    now_naive: datetime,
) -> str:
    """Insert a ``PHASE_0_ANCHORED`` Fair Price Reversion row into ``setup_state_machine``."""
    setup_id = f"{session_name}-{now_naive.strftime('%Y%m%d%H%M')}-{uuid.uuid4().hex[:6]}"
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
        [setup_id, FAIR_PRICE_STRATEGY, "PHASE_0_ANCHORED", None, float(fair_price), now_naive],
    )
    return setup_id


def _extract_opening_price(state: AgentState) -> float:
    """Resolve the exact session opening price from state or latest candle."""
    current_price = float(state.get("current_price", 0.0))
    if current_price > 0.0:
        return current_price

    df = state.get("historical_data")
    if df is not None and not df.empty:
        if "open" in df.columns:
            return float(df["open"].iloc[-1])
        if "close" in df.columns:
            return float(df["close"].iloc[-1])
    return 0.0


def supervisor_node(state: AgentState, now: datetime | None = None) -> AgentState:
    """Classify market regime as ``FLAT``, ``ACTIVE``, ``SESSION_OPEN_WATCH``, or ``SUSPENDED``."""
    if (
        str(state.get("action", "")).upper() == "CIRCUIT_BREAKER"
        or str(state.get("market_regime", "")).upper() == "SUSPENDED"
    ):
        print("[Supervisor] CIRCUIT_BREAKER active | Regime: SUSPENDED")
        return {
            **state,
            "market_regime": "SUSPENDED",
            "action": "CIRCUIT_BREAKER",
            "rationale": state.get("rationale")
            or "Macro circuit breaker active. Trading SUSPENDED.",
        }

    current_utc = now or datetime.now(timezone.utc)
    if current_utc.tzinfo is None:
        current_utc = current_utc.replace(tzinfo=timezone.utc)
    now_naive = current_utc.astimezone(timezone.utc).replace(tzinfo=None)
    current_date = current_utc.astimezone(_NY_TZ).date()

    session_open = detect_session_open(current_utc)
    conn = duckdb.connect(DB_PATH)
    try:
        _ensure_strategy_tables(conn)
        consecutive_losses = _get_consecutive_losses(conn, current_date)
        has_active_setup = _has_active_fair_price_setup(conn)

        if session_open is not None or has_active_setup:
            if consecutive_losses >= THREE_LOSS_LIMIT:
                print(
                    f"[Supervisor] Three-Loss Rule triggered (consecutive_losses={consecutive_losses} >= "
                    f"{THREE_LOSS_LIMIT}) | Regime: FLAT | Action: HOLD"
                )
                return {
                    **state,
                    "market_regime": "FLAT",
                    "action": "HOLD",
                    "rationale": THREE_LOSS_RATIONALE,
                }

            if session_open is not None:
                opening_price = _extract_opening_price(state)
                if opening_price > 0.0:
                    setup_id = _anchor_fair_price_setup(
                        conn=conn,
                        session_name=session_open,
                        fair_price=opening_price,
                        now_naive=now_naive,
                    )
                    print(
                        f"[Supervisor] {session_open} Open Detected | "
                        f"Anchored Fair Price ${opening_price:.2f} ({setup_id}) | "
                        "Regime: SESSION_OPEN_WATCH"
                    )
                    return {
                        **state,
                        "market_regime": "SESSION_OPEN_WATCH",
                        "rationale": (
                            f"{session_open} open anchored Fair Price ${opening_price:.2f} "
                            f"({setup_id})."
                        ),
                    }

            if has_active_setup:
                print("[Supervisor] Active Fair Price Reversion setup | Regime: SESSION_OPEN_WATCH")
                return {
                    **state,
                    "market_regime": "SESSION_OPEN_WATCH",
                }
    finally:
        conn.close()

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
