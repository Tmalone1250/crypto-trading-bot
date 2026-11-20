"""DuckDB Action Ledger for real Alpaca bot actions (never routine polling ticks)."""

import csv
from datetime import datetime, timezone
import io
import os
from typing import Any
import duckdb

DB_PATH: str = "agent_trades.duckdb"
BOT_NAME: str = "alpaca_crypto"
DEFAULT_UNIT: str = "USD"

ACTION_LOG_COLUMNS: tuple[str, ...] = (
    "timestamp_utc",
    "bot",
    "mode",
    "action",
    "symbol",
    "asset_id",
    "qty",
    "notional",
    "unit",
    "price_or_value",
    "pnl_amount",
    "pnl_pct",
    "strategy",
    "reason",
)


def _current_mode() -> str:
    """Return ``'live'`` only when ``TRADING_MODE=production``, else ``'paper'``."""
    return (
        "live"
        if os.environ.get("TRADING_MODE", "dry_run").lower() == "production"
        else "paper"
    )


def ensure_action_log_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Ensure the 14-column ``action_log`` table exists in DuckDB."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS action_log (
            timestamp_utc TIMESTAMP NOT NULL,
            bot VARCHAR NOT NULL DEFAULT 'alpaca_crypto',
            mode VARCHAR NOT NULL,
            action VARCHAR NOT NULL,
            symbol VARCHAR,
            asset_id VARCHAR,
            qty DOUBLE,
            notional DOUBLE,
            unit VARCHAR NOT NULL DEFAULT 'USD',
            price_or_value DOUBLE,
            pnl_amount DOUBLE,
            pnl_pct DOUBLE,
            strategy VARCHAR,
            reason VARCHAR
        )
        """
    )


def log_action(
    action: str,
    symbol: str = "",
    asset_id: str = "",
    qty: float | None = None,
    notional: float | None = None,
    price_or_value: float | None = None,
    pnl_amount: float | None = None,
    pnl_pct: float | None = None,
    strategy: str = "",
    reason: str = "",
    mode: str | None = None,
    db_path: str = DB_PATH,
    timestamp_utc: datetime | None = None,
) -> None:
    """Record a single real bot action into DuckDB ``action_log``."""
    ts = (
        timestamp_utc
        if timestamp_utc is not None
        else datetime.now(timezone.utc).replace(tzinfo=None)
    )
    resolved_mode = mode or _current_mode()
    try:
        conn = duckdb.connect(db_path)
        try:
            ensure_action_log_table(conn)
            conn.execute(
                """
                INSERT INTO action_log (
                    timestamp_utc,
                    bot,
                    mode,
                    action,
                    symbol,
                    asset_id,
                    qty,
                    notional,
                    unit,
                    price_or_value,
                    pnl_amount,
                    pnl_pct,
                    strategy,
                    reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    ts,
                    BOT_NAME,
                    resolved_mode,
                    str(action).upper(),
                    str(symbol) if symbol else "",
                    str(asset_id) if asset_id else "",
                    float(qty) if qty is not None else None,
                    float(notional) if notional is not None else None,
                    DEFAULT_UNIT,
                    float(price_or_value) if price_or_value is not None else None,
                    float(pnl_amount) if pnl_amount is not None else None,
                    float(pnl_pct) if pnl_pct is not None else None,
                    str(strategy) if strategy else "",
                    str(reason) if reason else "",
                ],
            )
        finally:
            conn.close()
    except Exception:
        # Action logging is best-effort and must never interrupt live order execution.
        pass


def _expand_action_filters(actions: list[str] | None) -> list[str]:
    """Expand shorthand action filters (e.g. ``BUY`` -> ``BUY, LIMIT_BUY, MARKET_BUY``)."""
    if not actions:
        return []
    expanded: set[str] = set()
    for raw in actions:
        token = raw.strip().upper()
        if not token:
            continue
        expanded.add(token)
        if token == "BUY":
            expanded.update({"LIMIT_BUY", "MARKET_BUY"})
        elif token == "SELL":
            expanded.update({"LIMIT_SELL", "MARKET_SELL"})
    return sorted(expanded)


def export_actions_csv(
    db_path: str = DB_PATH,
    limit: int = 1000,
    actions: list[str] | None = None,
) -> str:
    """Query ``action_log`` and serialize rows into the 14-column CSV format."""
    safe_limit = max(1, min(int(limit), 10000))
    filter_actions = _expand_action_filters(actions)

    conn = duckdb.connect(db_path)
    try:
        ensure_action_log_table(conn)
        if filter_actions:
            placeholders = ", ".join("?" for _ in filter_actions)
            rows = conn.execute(
                f"""
                SELECT {", ".join(ACTION_LOG_COLUMNS)}
                FROM action_log
                WHERE UPPER(action) IN ({placeholders})
                ORDER BY timestamp_utc DESC
                LIMIT ?
                """,
                [*filter_actions, safe_limit],
            ).fetchall()
        else:
            rows = conn.execute(
                f"""
                SELECT {", ".join(ACTION_LOG_COLUMNS)}
                FROM action_log
                ORDER BY timestamp_utc DESC
                LIMIT ?
                """,
                [safe_limit],
            ).fetchall()
    finally:
        conn.close()

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(ACTION_LOG_COLUMNS)
    for row in rows:
        formatted_row: list[Any] = []
        for idx, val in enumerate(row):
            if val is None:
                formatted_row.append("")
            elif idx == 0 and isinstance(val, datetime):
                formatted_row.append(val.isoformat() + "Z")
            else:
                formatted_row.append(val)
        writer.writerow(formatted_row)
    return buf.getvalue()
