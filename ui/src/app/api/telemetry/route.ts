import { connection, NextResponse } from "next/server";

export interface ScalperIntent {
  symbol: string;
  action: "LIMIT_BUY" | "LIMIT_SELL" | "MARKET_BUY" | "MARKET_SELL" | string;
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
  watched_assets?: Record<string, number | null>;
}

export async function GET() {
  await connection();

  const upstreamUrl =
    process.env.TELEMETRY_API_URL || "http://127.0.0.1:8000/api/telemetry";
  const apiKey = process.env.TELEMETRY_API_KEY;

  const headers: HeadersInit = {
    Accept: "application/json",
  };
  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }

  try {
    const response = await fetch(upstreamUrl, {
      method: "GET",
      headers,
      cache: "no-store",
    });

    const payload = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        {
          data: [],
          error:
            payload?.error ??
            `Upstream telemetry bridge returned HTTP ${response.status}`,
        },
        { status: response.status, headers: { "Cache-Control": "no-store" } }
      );
    }

    return NextResponse.json(
      {
        data: Array.isArray(payload?.data) ? (payload.data as TelemetryRow[]) : [],
        watched_assets: payload?.watched_assets ?? {
          "BTC/USD": null,
          "ETH/USD": null,
          "SOL/USD": null,
        },
        latest_portfolio_equity: payload?.latest_portfolio_equity ?? null,
        market_regime: payload?.market_regime ?? "FLAT",
        latest_sync_timestamp: payload?.latest_sync_timestamp ?? null,
        sample_count: payload?.sample_count ?? 0,
        equity_history: Array.isArray(payload?.equity_history)
          ? payload.equity_history
          : [],
      },
      { status: 200, headers: { "Cache-Control": "no-store" } }
    );
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Unknown upstream fetch error";
    console.error("[Telemetry API] Upstream bridge request failed:", message);
    return NextResponse.json(
      {
        data: [],
        error: `Failed to reach telemetry bridge at ${upstreamUrl}: ${message}`,
      },
      { status: 502, headers: { "Cache-Control": "no-store" } }
    );
  }
}
