"""Embedded HTTP Telemetry Bridge serving DuckDB agent_telemetry over JSON."""

from datetime import date, datetime
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
from typing import Any
from urllib.parse import parse_qs, urlparse
from dotenv import load_dotenv
import duckdb

from action_ledger import export_actions_csv

load_dotenv()

DB_PATH: str = "agent_trades.duckdb"
DEFAULT_HOST: str = "0.0.0.0"
DEFAULT_PORT: int = 8000


def _to_json_scalar(value: Any) -> Any:
    """Convert DuckDB timestamps and Decimals into JSON-safe strings or floats."""
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def _to_nullable_float(value: Any) -> float | None:
    """Safely convert numeric or Decimal values to ``float | None``."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_scalper_intents(raw: Any) -> list[dict[str, Any]]:
    """Safely parse ``scalper_intents`` JSON string into a list of dicts."""
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


DEFAULT_WATCHED_ASSETS: dict[str, float | None] = {
    "BTC/USD": None,
    "ETH/USD": None,
    "SOL/USD": None,
}
DEFAULT_ASSET_PHASES: dict[str, str] = {
    "BTC/USD": "HOLD",
    "ETH/USD": "HOLD",
    "SOL/USD": "HOLD",
}


def _parse_watched_assets(raw: Any) -> dict[str, float | None]:
    """Safely parse ``watched_assets`` JSON string into a ``symbol -> price`` dict."""
    if isinstance(raw, dict):
        return {str(k): _to_nullable_float(v) for k, v in raw.items()}
    if not isinstance(raw, str) or not raw.strip():
        return dict(DEFAULT_WATCHED_ASSETS)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and parsed:
            return {str(k): _to_nullable_float(v) for k, v in parsed.items()}
    except json.JSONDecodeError:
        pass
    return dict(DEFAULT_WATCHED_ASSETS)


def _parse_open_positions(raw: Any) -> list[dict[str, Any]]:
    """Safely parse ``open_positions`` JSON string into a list of position dicts."""
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def _parse_asset_phases(raw: Any) -> dict[str, str]:
    """Safely parse ``asset_phases`` JSON string into a ``symbol -> phase`` dict."""
    if isinstance(raw, dict) and raw:
        return {str(k): str(v or "HOLD") for k, v in raw.items()}
    if not isinstance(raw, str) or not raw.strip():
        return dict(DEFAULT_ASSET_PHASES)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and parsed:
            return {str(k): str(v or "HOLD") for k, v in parsed.items()}
    except json.JSONDecodeError:
        pass
    return dict(DEFAULT_ASSET_PHASES)


def query_telemetry_payload(db_path: str = DB_PATH, limit: int = 100) -> dict[str, Any]:
    """Query ``agent_telemetry`` from DuckDB and format JSON payload for the dashboard."""
    conn = duckdb.connect(db_path)
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
        for col_name in (
            "scalper_intents",
            "watched_assets",
            "open_positions",
            "asset_phases",
        ):
            conn.execute(
                f"ALTER TABLE agent_telemetry ADD COLUMN IF NOT EXISTS {col_name} VARCHAR"
            )
        raw_rows = conn.execute(
            """
            SELECT
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
            FROM agent_telemetry
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()
    finally:
        conn.close()

    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        rows.append(
            {
                "timestamp": str(_to_json_scalar(row[0]) or ""),
                "portfolio_value": _to_nullable_float(row[1]),
                "market_regime": str(row[2] or "FLAT"),
                "action": str(row[3] or "HOLD"),
                "target_price": _to_nullable_float(row[4]),
                "rationale": str(row[5] or ""),
                "scalper_intents": _parse_scalper_intents(row[6]),
                "watched_assets": _parse_watched_assets(row[7]),
                "open_positions": _parse_open_positions(row[8]),
                "asset_phases": _parse_asset_phases(row[9]),
            }
        )

    latest = rows[0] if rows else None
    equity_history = [
        {
            "timestamp": r["timestamp"],
            "portfolio_value": r["portfolio_value"],
        }
        for r in reversed(rows)
        if r["portfolio_value"] is not None
    ]

    return {
        "data": rows,
        "watched_assets": (
            latest["watched_assets"] if latest else dict(DEFAULT_WATCHED_ASSETS)
        ),
        "open_positions": latest["open_positions"] if latest else [],
        "asset_phases": (
            latest["asset_phases"] if latest else dict(DEFAULT_ASSET_PHASES)
        ),
        "latest_portfolio_equity": latest["portfolio_value"] if latest else None,
        "market_regime": latest["market_regime"] if latest else "FLAT",
        "latest_sync_timestamp": latest["timestamp"] if latest else None,
        "sample_count": len(rows),
        "equity_history": equity_history,
    }


class TelemetryRequestHandler(BaseHTTPRequestHandler):
    """Lightweight HTTP handler exposing ``GET /api/telemetry`` with optional API-key auth."""

    db_path: str = DB_PATH

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_csv(self, status_code: int, csv_text: str, filename: str = "alpaca_actions.csv") -> None:
        body = csv_text.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        normalized_path = parsed.path.rstrip("/")
        is_csv_export = normalized_path == "/api/export/actions.csv"
        if not is_csv_export and normalized_path not in ("", "/api/telemetry", "/telemetry"):
            self._send_json(404, {"data": [], "error": f"Not found: {parsed.path}"})
            return

        query_params = parse_qs(parsed.query)
        expected_key = os.environ.get("TELEMETRY_API_KEY", "").strip()
        is_direct_loopback = (
            self.client_address[0] in ("127.0.0.1", "::1")
            and not self.headers.get("CF-Connecting-IP")
            and not self.headers.get("X-Forwarded-For")
        )
        if expected_key and not is_direct_loopback:
            provided_key = (
                self.headers.get("X-API-Key", "").strip()
                or (query_params.get("api_key") or [""])[0].strip()
            )
            if provided_key != expected_key:
                self._send_json(
                    401,
                    {"data": [], "error": "Unauthorized: invalid or missing X-API-Key"},
                )
                return

        if is_csv_export:
            try:
                raw_limit = (query_params.get("limit") or ["1000"])[0]
                limit_val = int(raw_limit)
            except ValueError:
                limit_val = 1000
            raw_actions = (query_params.get("actions") or [""])[0]
            action_list = [a.strip() for a in raw_actions.split(",") if a.strip()] or None
            try:
                csv_text = export_actions_csv(
                    db_path=self.db_path,
                    limit=limit_val,
                    actions=action_list,
                )
                self._send_csv(200, csv_text, filename="alpaca_actions.csv")
            except Exception as exc:
                self._send_json(500, {"data": [], "error": f"CSV export failed: {exc}"})
            return

        try:
            payload = query_telemetry_payload(db_path=self.db_path)
            self._send_json(200, payload)
        except Exception as exc:
            self._send_json(
                500,
                {"data": [], "error": f"Telemetry query failed: {exc}"},
            )

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence noisy per-poll HTTP access logs in systemd stdout."""
        return


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_telemetry_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    db_path: str = DB_PATH,
) -> tuple[ReusableThreadingHTTPServer, threading.Thread]:
    """Start the embedded telemetry HTTP bridge on a background daemon thread."""

    class _BoundHandler(TelemetryRequestHandler):
        pass

    _BoundHandler.db_path = db_path
    server = ReusableThreadingHTTPServer((host, port), _BoundHandler)
    thread = threading.Thread(
        target=server.serve_forever,
        name="telemetry-api-server",
        daemon=True,
    )
    thread.start()
    print(f"[Telemetry API] Listening on http://{host}:{port}/api/telemetry")
    return server, thread
