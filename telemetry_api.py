"""Embedded HTTP Telemetry Bridge serving DuckDB agent_telemetry over JSON."""

from datetime import date, datetime
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
from typing import Any
from urllib.parse import urlparse
from dotenv import load_dotenv
import duckdb

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
                scalper_intents VARCHAR
            )
            """
        )
        conn.execute(
            "ALTER TABLE agent_telemetry ADD COLUMN IF NOT EXISTS scalper_intents VARCHAR"
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
                scalper_intents
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

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/telemetry":
            self._send_json(404, {"data": [], "error": "Not found"})
            return

        expected_key = os.environ.get("TELEMETRY_API_KEY", "").strip()
        if expected_key:
            provided_key = self.headers.get("X-API-Key", "")
            if provided_key != expected_key:
                self._send_json(
                    401,
                    {"data": [], "error": "Unauthorized: invalid or missing X-API-Key"},
                )
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
