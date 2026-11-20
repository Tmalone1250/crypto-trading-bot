import { connection, NextResponse } from "next/server";

function resolveExportUrl(rawUrl: string | undefined, search: string): string {
  const base = (rawUrl || "http://127.0.0.1:8000/api/telemetry")
    .trim()
    .replace(/\/+$/, "");
  const originBase = base.endsWith("/api/telemetry")
    ? base.slice(0, -"/api/telemetry".length)
    : base;
  return `${originBase}/api/export/actions.csv${search}`;
}

export async function GET(request: Request) {
  await connection();

  const reqUrl = new URL(request.url);
  const upstreamUrl = resolveExportUrl(
    process.env.TELEMETRY_API_URL,
    reqUrl.search
  );
  const apiKey = process.env.TELEMETRY_API_KEY?.trim();

  const headers: HeadersInit = {
    Accept: "text/csv",
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

    const csvText = await response.text();
    if (!response.ok) {
      return NextResponse.json(
        { error: `Upstream CSV export returned HTTP ${response.status}` },
        { status: response.status, headers: { "Cache-Control": "no-store" } }
      );
    }

    return new NextResponse(csvText, {
      status: 200,
      headers: {
        "Content-Type": "text/csv; charset=utf-8",
        "Content-Disposition": 'attachment; filename="alpaca_actions.csv"',
        "Cache-Control": "no-store",
        "Access-Control-Allow-Origin": "*",
      },
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Unknown upstream export error";
    return NextResponse.json(
      { error: `Failed to export actions CSV: ${message}` },
      { status: 502, headers: { "Cache-Control": "no-store" } }
    );
  }
}
