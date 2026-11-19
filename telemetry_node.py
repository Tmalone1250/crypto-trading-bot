"""Telemetry Logger Node: persists per-cycle agent state to DuckDB for dashboarding.

Runs as the final convergence node before ``END`` across all graph paths
(including ``HOLD`` bypasses, risk vetoes, and executed trades), writing a row
to the ``agent_telemetry`` table in ``agent_trades.duckdb`` including serialized
``scalper_intents`` JSON for the Next.js monitoring dashboard.
"""

from datetime import datetime, timezone
import json
from typing import Any
import duckdb

from state import AgentState

DB_PATH: str = "agent_trades.duckdb"


def _parse_optional_float(value: object) -> float | None:
    """Safely convert a numeric value or numeric string to ``float``."""
    if value is None or value == "":
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _serialize_trade_intents(trade_intents: list[Any]) -> str:
    """Serialize a list of ``TradeIntent`` Pydantic objects (or dicts) to a JSON string."""
    serialized: list[dict[str, Any]] = []
    for intent in trade_intents:
        if hasattr(intent, "model_dump"):
            serialized.append(intent.model_dump())
        elif hasattr(intent, "dict"):
            serialized.append(intent.dict())
        elif isinstance(intent, dict):
            serialized.append(dict(intent))
        else:
            serialized.append({"value": str(intent)})
    return json.dumps(serialized)


def _serialize_watched_assets(state: AgentState) -> str:
    """Serialize a map of each symbol in ``state['asset_symbols']`` to its latest price."""
    primary_symbol = str(state.get("asset_symbol", "BTC/USD"))
    symbols = list(state.get("asset_symbols") or [primary_symbol])
    market_map = state.get("market_data_by_symbol") or {}
    current_price = _parse_optional_float(state.get("current_price"))

    watched: dict[str, float | None] = {}
    for sym in symbols:
        price: float | None = None
        if sym == primary_symbol and current_price is not None and current_price > 0.0:
            price = round(current_price, 6)
        else:
            sym_data = market_map.get(sym)
            if sym_data is None:
                sym_data = market_map.get(sym.replace("/", ""))
            if isinstance(sym_data, (int, float)):
                price = round(float(sym_data), 6)
            elif hasattr(sym_data, "empty") and not sym_data.empty and "close" in sym_data.columns:
                closed_val = _parse_optional_float(sym_data["close"].iloc[-1])
                if closed_val is not None and closed_val > 0.0:
                    price = round(closed_val, 6)
        watched[str(sym)] = price
    return json.dumps(watched)


def _serialize_open_positions(state: AgentState) -> str:
    """Serialize active open positions (symbol, qty, avg_entry_price, unrealized_pl) to JSON."""
    positions = state.get("open_positions") or []
    cleaned: list[dict[str, Any]] = []
    for pos in positions:
        if not isinstance(pos, dict):
            continue
        cleaned.append(
            {
                "symbol": str(pos.get("symbol", "")),
                "qty": _parse_optional_float(pos.get("qty")),
                "avg_entry_price": _parse_optional_float(pos.get("avg_entry_price")),
                "unrealized_pl": _parse_optional_float(
                    pos.get("unrealized_pl", pos.get("unrealized_pnl"))
                ),
            }
        )
    return json.dumps(cleaned)


def _resolve_asset_phases(
    state: AgentState,
    conn: duckdb.DuckDBPyConnection,
) -> str:
    """Resolve current strategy phase (or HOLD) per watched symbol and serialize to JSON."""
    primary_symbol = str(state.get("asset_symbol", "BTC/USD"))
    symbols = list(state.get("asset_symbols") or [primary_symbol])
    phases: dict[str, str] = {str(sym): "HOLD" for sym in symbols}

    # 1. Check DuckDB setup_state_machine if available.
    try:
        for sym in symbols:
            row = conn.execute(
                """
                SELECT phase
                FROM setup_state_machine
                WHERE (
                    setup_id LIKE ?
                    OR setup_id LIKE ?
                    OR (? = ? AND setup_id NOT LIKE '%:%')
                )
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                [f"%:{sym}:%", f"%:{sym}", sym, primary_symbol],
            ).fetchone()
            if row is not None and row[0]:
                phases[str(sym)] = str(row[0])
    except Exception:
        pass

    # 2. Check cycle rationale summary (e.g. "BTC: PHASE_1_SWEPT | ETH: HOLD").
    rationale = str(state.get("rationale", ""))
    for part in rationale.split("|"):
        segment = part.strip()
        if ":" in segment:
            prefix, val = segment.split(":", 1)
            prefix_clean = prefix.strip().upper()
            val_clean = val.strip()
            for sym in symbols:
                if sym.upper() == prefix_clean or sym.split("/")[0].upper() == prefix_clean:
                    if val_clean.startswith("PHASE_") or val_clean == "HOLD":
                        phases[str(sym)] = val_clean

    # 3. Override with any emitted TradeIntent action for the symbol this cycle.
    for intent in state.get("trade_intents") or []:
        intent_sym = (
            intent.get("symbol")
            if isinstance(intent, dict)
            else getattr(intent, "symbol", None)
        )
        intent_act = (
            intent.get("action")
            if isinstance(intent, dict)
            else getattr(intent, "action", None)
        )
        if intent_sym and intent_act and str(intent_sym) in phases:
            phases[str(intent_sym)] = str(intent_act)

    return json.dumps(phases)


def _log_telemetry_to_duckdb(
    timestamp: datetime,
    portfolio_value: float | None,
    market_regime: str,
    action: str,
    target_price: float | None,
    rationale: str,
    scalper_intents: str,
    watched_assets: str,
    open_positions: str,
    state: AgentState,
) -> None:
    """Ensure ``agent_telemetry`` exists (with all dashboard columns) and insert a snapshot."""
    conn = duckdb.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_telemetry (
                timestamp TIMESTAMP NOT NULL,
                portfolio_value DOUBLE,
                market_regime VARCHAR,
                action VARCHAR,
                target_price DOUBLE,
                rationale VARCHAR,
                scalper_intents VARCHAR,
                watched_assets VARCHAR,
                open_positions VARCHAR,
                asset_phases VARCHAR
            )
            """
        )
        # Failsafe schema upgrades for pre-existing agent_telemetry tables.
        for col_name in (
            "scalper_intents",
            "watched_assets",
            "open_positions",
            "asset_phases",
        ):
            conn.execute(
                f"ALTER TABLE agent_telemetry ADD COLUMN IF NOT EXISTS {col_name} VARCHAR"
            )

        asset_phases = _resolve_asset_phases(state, conn)
        conn.execute(
            """
            INSERT INTO agent_telemetry (
                timestamp,
                portfolio_value,
                market_regime,
                action,
                target_price,
                rationale,
                scalper_intents,
                watched_assets,
                open_positions,
                asset_phases
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                timestamp,
                portfolio_value,
                market_regime,
                action,
                target_price,
                rationale,
                scalper_intents,
                watched_assets,
                open_positions,
                asset_phases,
            ],
        )
    finally:
        conn.close()


def telemetry_node(state: AgentState) -> AgentState:
    """Persist the final cycle telemetry snapshot to DuckDB ``agent_telemetry``."""
    account_data = state.get("account_data") or {}
    portfolio_value = _parse_optional_float(account_data.get("portfolio_value"))
    market_regime = str(state.get("market_regime", ""))
    action = str(state.get("action", "HOLD"))
    target_price = _parse_optional_float(state.get("target_price"))
    rationale = str(state.get("rationale", ""))
    trade_intents = state.get("trade_intents") or []
    scalper_intents_json = _serialize_trade_intents(trade_intents)
    watched_assets_json = _serialize_watched_assets(state)
    open_positions_json = _serialize_open_positions(state)
    timestamp = datetime.now(timezone.utc).replace(tzinfo=None)

    if market_regime.upper() == "FLAT" and trade_intents:
        action = "SCALP_ACTIVE"
        rationale = f"Scalper executing {len(trade_intents)} grid actions."

    try:
        _log_telemetry_to_duckdb(
            timestamp=timestamp,
            portfolio_value=portfolio_value,
            market_regime=market_regime,
            action=action,
            target_price=target_price,
            rationale=rationale,
            scalper_intents=scalper_intents_json,
            watched_assets=watched_assets_json,
            open_positions=open_positions_json,
            state=state,
        )
        print(
            f"[Telemetry] Logged cycle: regime={market_regime} | action={action} "
            f"| portfolio_value={portfolio_value} | intents={len(trade_intents)}"
        )
        return state
    except Exception as exc:
        existing_err = state.get("error_message")
        err_msg = (
            f"{existing_err}; Telemetry error: {exc}"
            if existing_err
            else f"Telemetry error: {exc}"
        )
        return {
            **state,
            "error_message": err_msg,
        }
