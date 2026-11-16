"""Main daemon entry point for the continuous LangGraph crypto trading swarm."""

import os
import time
import pandas as pd

from agent_graph import app
from config import get_alpaca_paper_config
from mcp_client import PersistentAlpacaMCPClient
from state import AgentState

EXECUTION_INTERVAL_SECONDS: int = 60


def build_initial_state(
    target_symbol: str,
    realized_pnl: float = 0.0,
    last_alpha_eval_price: float = 0.0,
    mcp_client: PersistentAlpacaMCPClient | None = None,
) -> AgentState:
    """Construct the per-cycle ``AgentState`` payload for ``app.invoke``."""
    return {
        "asset_symbol": target_symbol,
        "asset_symbols": ["BTC/USD", "ETH/USD", "SOL/USD"],
        "market_data_by_symbol": {},
        "current_price": 0.0,
        "historical_data": pd.DataFrame(),
        "sentiment_score": 0.0,
        "risk_cleared": False,
        "order_details": {},
        "error_message": None,
        "account_data": {},
        "open_positions": [],
        "active_orders": [],
        "action": "HOLD",
        "target_price": None,
        "rationale": "",
        "market_regime": "FLAT",
        "realized_pnl": realized_pnl,
        "trade_intents": [],
        "last_alpha_eval_price": last_alpha_eval_price,
        "mcp_client": mcp_client,
    }


def run_daemon() -> None:
    """Boot the persistent Alpaca MCP client and run the 60s LangGraph loop."""
    get_alpaca_paper_config()
    target_symbol = os.environ.get("ASSET_SYMBOL", "BTC/USD")
    last_alpha_eval_price: float = 0.0
    realized_pnl: float = 0.0

    print("Quant Agent Online. Booting persistent Alpaca MCP server...")
    with PersistentAlpacaMCPClient() as mcp_client:
        print("Alpaca MCP server initialized. Entering 1-minute execution loop...")
        while True:
            initial_state = build_initial_state(
                target_symbol=target_symbol,
                realized_pnl=realized_pnl,
                last_alpha_eval_price=last_alpha_eval_price,
                mcp_client=mcp_client,
            )

            try:
                result = app.invoke(initial_state)
                last_alpha_eval_price = float(
                    result.get("last_alpha_eval_price", last_alpha_eval_price)
                )
                realized_pnl = float(result.get("realized_pnl", realized_pnl))
                if result.get("error_message"):
                    print(f"[Daemon] Cycle warning: {result['error_message']}")
            except KeyboardInterrupt:
                print("Shutdown signal received. Exiting Quant Agent daemon.")
                break
            except Exception as exc:
                print(f"[Daemon Error] Graph invocation failed (retrying in 60s): {exc}")

            time.sleep(EXECUTION_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_daemon()
