import fs from "node:fs";
import path from "node:path";
import { connection, NextResponse } from "next/server";
import { DuckDBInstance, type DuckDBConnection } from "@duckdb/node-api";

export interface ScalperIntent {
  symbol: string;
  action: "LIMIT_BUY" | "LIMIT_SELL" | string;
  target_price: number;
  qty: number;
  notional_usd?: number;
  rationale?: string;
}

export interface TelemetryRow {
  timestamp: string;
  portfolio_value: number | null;
  market_regime: string;
  action: string;
  target_price: number | null;
  rationale: string;
  scalper_intents: ScalperIntent[];
}

function parseScalperIntents(raw: unknown): ScalperIntent[] {
  if (Array.isArray(raw)) {
    return raw as ScalperIntent[];
  }
  if (typeof raw !== "string" || raw.trim() === "") {
    return [];
  }
  try {
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as ScalperIntent[]) : [];
  } catch {
    return [];
  }
}

function toNullableNumber(value: unknown): number | null {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

export async function GET() {
  await connection();
  const dbPath = path.resolve(process.cwd(), "..", "agent_trades.duckdb");

  if (!fs.existsSync(dbPath)) {
    return NextResponse.json(
      { data: [], warning: "Telemetry database not found yet." },
      { status: 200, headers: { "Cache-Control": "no-store" } }
    );
  }

  let instance: DuckDBInstance | null = null;
  let dbConnection: DuckDBConnection | null = null;

  try {
    instance = await DuckDBInstance.create(dbPath, {
      access_mode: "READ_ONLY",
    });
    dbConnection = await instance.connect();

    const reader = await dbConnection.runAndReadAll(`
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
      LIMIT 100
    `);

    const rawRows = reader.getRowObjectsJson();
    const data: TelemetryRow[] = rawRows.map((row) => ({
      timestamp: String(row.timestamp ?? ""),
      portfolio_value: toNullableNumber(row.portfolio_value),
      market_regime: String(row.market_regime ?? "UNKNOWN"),
      action: String(row.action ?? "HOLD"),
      target_price: toNullableNumber(row.target_price),
      rationale: String(row.rationale ?? ""),
      scalper_intents: parseScalperIntents(row.scalper_intents),
    }));

    return NextResponse.json(
      { data },
      { status: 200, headers: { "Cache-Control": "no-store" } }
    );
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Unknown DuckDB error";
    console.error("[Telemetry API] DuckDB query failed:", message);
    return NextResponse.json(
      { data: [], error: `Failed to read telemetry database: ${message}` },
      { status: 500, headers: { "Cache-Control": "no-store" } }
    );
  } finally {
    try {
      dbConnection?.closeSync();
    } catch {
      // Ignore cleanup errors on already-closed connections.
    }
    try {
      instance?.closeSync();
    } catch {
      // Ignore cleanup errors on already-closed instances.
    }
  }
}
