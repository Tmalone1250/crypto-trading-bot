"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  Clock,
  RefreshCw,
  Terminal,
  TrendingUp,
  Zap,
} from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface ScalperIntent {
  symbol: string;
  action: string;
  target_price: number;
  qty: number;
  notional_usd?: number;
  rationale?: string;
}

interface TelemetryRow {
  timestamp: string;
  portfolio_value: number | null;
  market_regime: string;
  action: string;
  target_price: number | null;
  rationale: string;
  scalper_intents: ScalperIntent[];
}

interface TelemetryApiResponse {
  data?: TelemetryRow[];
  error?: string;
  warning?: string;
}

interface ChartPoint {
  timeLabel: string;
  fullTimestamp: string;
  portfolio_value: number;
  action: string;
  market_regime: string;
}

function formatTimestamp(raw: string): string {
  if (!raw) return "--";
  const normalized = raw.includes("T") ? raw : raw.replace(" ", "T");
  const date = new Date(normalized.endsWith("Z") ? normalized : `${normalized}Z`);
  if (Number.isNaN(date.getTime())) {
    return raw;
  }
  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function formatCurrency(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "--";
  }
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

function getActionBadgeClasses(action: string): string {
  const upper = action.toUpperCase();
  if (upper === "SCALP_ACTIVE" || upper.includes("BUY")) {
    return "bg-orange-500 text-black border border-orange-400";
  }
  if (upper.includes("SELL")) {
    return "bg-white text-black border border-white";
  }
  if (upper === "CANCEL_ALL") {
    return "bg-red-600 text-white border border-red-500";
  }
  return "bg-neutral-800 text-neutral-300 border border-neutral-700";
}

export default function CommandCenterPage() {
  const [rows, setRows] = useState<TelemetryRow[]>([]);
  const [lastSync, setLastSync] = useState<string>("--");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    let isMounted = true;

    const fetchTelemetry = async () => {
      try {
        const res = await fetch("/api/telemetry", { cache: "no-store" });
        const payload: TelemetryApiResponse = await res.json();

        if (!isMounted) return;

        if (!res.ok) {
          setErrorMessage(
            payload.error ?? `Telemetry endpoint returned ${res.status}`
          );
        } else {
          setErrorMessage(payload.error ?? null);
          setRows(Array.isArray(payload.data) ? payload.data : []);
          setLastSync(
            new Date().toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
              second: "2-digit",
              hour12: false,
            })
          );
        }
      } catch (err) {
        if (!isMounted) return;
        setErrorMessage(
          err instanceof Error ? err.message : "Failed to reach telemetry API"
        );
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    void fetchTelemetry();
    const intervalId = setInterval(() => {
      void fetchTelemetry();
    }, 5000);

    return () => {
      isMounted = false;
      clearInterval(intervalId);
    };
  }, []);

  const latestRow = rows[0] ?? null;
  const currentRegime = (latestRow?.market_regime ?? "FLAT").toUpperCase();
  const isActiveRegime = currentRegime === "ACTIVE";

  const chartData = useMemo<ChartPoint[]>(() => {
    return [...rows]
      .reverse()
      .filter(
        (row): row is TelemetryRow & { portfolio_value: number } =>
          row.portfolio_value !== null && Number.isFinite(row.portfolio_value)
      )
      .map((row) => ({
        timeLabel: formatTimestamp(row.timestamp),
        fullTimestamp: row.timestamp,
        portfolio_value: row.portfolio_value,
        action: row.action,
        market_regime: row.market_regime,
      }));
  }, [rows]);

  return (
    <div className="min-h-screen bg-[#000000] text-white selection:bg-orange-500 selection:text-black">
      <div className="mx-auto flex max-w-7xl flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8">
        {/* Command Center Header */}
        <header className="flex flex-col justify-between gap-4 border border-neutral-800 bg-neutral-950 p-5 sm:flex-row sm:items-center">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center bg-orange-500 text-black">
              <Zap className="h-5 w-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-bold tracking-wider uppercase text-white">
                  LangGraph Command Center
                </h1>
                <span className="h-2 w-2 rounded-full bg-orange-500 animate-pulse" />
              </div>
              <p className="font-mono text-xs text-neutral-400">
                AUTONOMOUS CRYPTO SWING &amp; SCALP TELEMETRY // 5S POLL
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-4">
            <div className="border border-neutral-800 bg-black px-3 py-2">
              <div className="font-mono text-[10px] uppercase tracking-widest text-neutral-400">
                Portfolio Equity
              </div>
              <div className="font-mono text-base font-bold text-white">
                {formatCurrency(latestRow?.portfolio_value)}
              </div>
            </div>

            <div className="flex flex-col items-start gap-1">
              <span className="font-mono text-[10px] uppercase tracking-widest text-neutral-400">
                Market Regime
              </span>
              <span
                className={
                  isActiveRegime
                    ? "inline-flex items-center gap-1.5 bg-orange-500 px-3 py-1 font-mono text-xs font-extrabold uppercase tracking-wider text-black"
                    : "inline-flex items-center gap-1.5 bg-neutral-800 px-3 py-1 font-mono text-xs font-bold uppercase tracking-wider text-neutral-300"
                }
              >
                <Activity className="h-3.5 w-3.5" />
                {currentRegime}
              </span>
            </div>

            <div className="flex flex-col items-start gap-1 border-l border-neutral-800 pl-4">
              <span className="font-mono text-[10px] uppercase tracking-widest text-neutral-400">
                Latest Sync
              </span>
              <span className="inline-flex items-center gap-1.5 font-mono text-xs text-white">
                <Clock className="h-3.5 w-3.5 text-orange-500" />
                {lastSync}
              </span>
            </div>
          </div>
        </header>

        {errorMessage && (
          <div className="border border-orange-500 bg-neutral-950 px-4 py-3 font-mono text-xs text-orange-400">
            [TELEMETRY WARNING] {errorMessage}
          </div>
        )}

        {/* Portfolio Value Step Chart */}
        <section className="border border-neutral-800 bg-neutral-950 p-5">
          <div className="mb-4 flex items-center justify-between border-b border-neutral-800 pb-3">
            <div className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-orange-500" />
              <h2 className="font-mono text-sm font-bold uppercase tracking-wider text-white">
                Portfolio Value Over Time
              </h2>
            </div>
            <span className="font-mono text-xs text-neutral-400">
              {chartData.length} SAMPLES
            </span>
          </div>

          <div className="h-72 w-full min-w-0">
            {isLoading && chartData.length === 0 ? (
              <div className="flex h-full items-center justify-center font-mono text-xs text-neutral-400">
                <RefreshCw className="mr-2 h-4 w-4 animate-spin text-orange-500" />
                INITIALIZING TELEMETRY STREAM...
              </div>
            ) : chartData.length === 0 ? (
              <div className="flex h-full items-center justify-center font-mono text-xs text-neutral-500">
                NO TELEMETRY POINTS RECORDED IN DUCKDB YET
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={chartData}
                  margin={{ top: 10, right: 16, left: 8, bottom: 4 }}
                >
                  <CartesianGrid stroke="#262626" strokeDasharray="3 3" />
                  <XAxis
                    dataKey="timeLabel"
                    stroke="#737373"
                    tick={{ fill: "#a3a3a3", fontSize: 11, fontFamily: "monospace" }}
                    tickLine={false}
                  />
                  <YAxis
                    stroke="#737373"
                    domain={["auto", "auto"]}
                    tickFormatter={(val: number) => `$${val.toLocaleString()}`}
                    tick={{ fill: "#a3a3a3", fontSize: 11, fontFamily: "monospace" }}
                    tickLine={false}
                    width={84}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0a0a0a",
                      borderColor: "#f97316",
                      borderWidth: "1px",
                      color: "#ffffff",
                      fontFamily: "monospace",
                      fontSize: "12px",
                    }}
                    labelStyle={{ color: "#f97316", fontWeight: 700 }}
                    formatter={(value: unknown) => [
                      formatCurrency(Number(value)),
                      "Portfolio Value",
                    ]}
                  />
                  <Line
                    type="stepAfter"
                    dataKey="portfolio_value"
                    stroke="#f97316"
                    strokeWidth={2.5}
                    dot={false}
                    activeDot={{ r: 5, fill: "#f97316", stroke: "#000000" }}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </section>

        {/* Agent Intelligence Feed */}
        <section className="border border-neutral-800 bg-neutral-950 p-5">
          <div className="mb-4 flex items-center justify-between border-b border-neutral-800 pb-3">
            <div className="flex items-center gap-2">
              <Terminal className="h-4 w-4 text-orange-500" />
              <h2 className="font-mono text-sm font-bold uppercase tracking-wider text-white">
                Agent Intelligence Feed
              </h2>
            </div>
            <span className="font-mono text-xs text-neutral-400">
              LATEST {rows.length} CYCLES
            </span>
          </div>

          <div className="max-h-[32rem] space-y-3 overflow-y-auto pr-1 font-mono">
            {rows.length === 0 ? (
              <div className="border border-neutral-800 bg-black p-6 text-center text-xs text-neutral-500">
                WAITING FOR AGENT TELEMETRY EVENTS...
              </div>
            ) : (
              rows.map((row, idx) => (
                <article
                  key={`${row.timestamp}-${idx}`}
                  className="border border-neutral-800 bg-black p-4 transition-colors hover:border-orange-500/60"
                >
                  <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-xs text-neutral-400">
                        [{formatTimestamp(row.timestamp)}]
                      </span>
                      <span
                        className={`px-2 py-0.5 text-[11px] font-bold uppercase tracking-wider ${getActionBadgeClasses(
                          row.action
                        )}`}
                      >
                        {row.action}
                      </span>
                      <span className="border border-neutral-800 bg-neutral-950 px-2 py-0.5 text-[11px] uppercase text-neutral-300">
                        {row.market_regime}
                      </span>
                      {row.target_price !== null && (
                        <span className="text-xs text-orange-400">
                          TARGET: {formatCurrency(row.target_price)}
                        </span>
                      )}
                    </div>
                    <span className="text-xs text-neutral-400">
                      EQUITY: {formatCurrency(row.portfolio_value)}
                    </span>
                  </div>

                  <p className="text-xs leading-relaxed text-white">
                    {row.rationale || "No rationale recorded."}
                  </p>

                  {row.scalper_intents && row.scalper_intents.length > 0 && (
                    <div className="mt-3 space-y-2 border-l-2 border-orange-500 pl-3">
                      <div className="text-[10px] font-bold uppercase tracking-widest text-orange-500">
                        Scalper Grid Limit Orders ({row.scalper_intents.length})
                      </div>
                      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                        {row.scalper_intents.map((intent, intentIdx) => (
                          <div
                            key={`${intent.symbol}-${intent.action}-${intentIdx}`}
                            className="border border-orange-500/40 bg-neutral-950 p-2.5 text-xs"
                          >
                            <div className="mb-1 flex items-center justify-between">
                              <span className="font-bold text-white">
                                {intent.symbol}
                              </span>
                              <span className="bg-orange-500 px-1.5 py-0.5 text-[10px] font-extrabold uppercase text-black">
                                {intent.action}
                              </span>
                            </div>
                            <div className="flex items-center justify-between text-[11px] text-neutral-300">
                              <span>LIMIT: {formatCurrency(intent.target_price)}</span>
                              <span>QTY: {intent.qty}</span>
                            </div>
                            {intent.notional_usd !== undefined && (
                              <div className="mt-0.5 text-[11px] text-orange-400">
                                NOTIONAL: {formatCurrency(intent.notional_usd)}
                              </div>
                            )}
                            {intent.rationale && (
                              <p className="mt-1.5 border-t border-neutral-800 pt-1 text-[10px] leading-snug text-neutral-400">
                                {intent.rationale}
                              </p>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </article>
              ))
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
