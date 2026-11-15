"""LangGraph Traffic Cop: compiles the 6-node Multi-Agent Swarm trading state machine.

Wires ``portfolio`` -> ``observer`` -> ``supervisor`` -> ``alpha`` -> ``risk`` -> ``executor``
with deterministic conditional routing gates after Observer, Supervisor, Alpha, and Risk nodes.
"""

import os
import time
import pandas as pd
from langgraph.graph import END, StateGraph

from alpha_node import alpha_logic_gate_node
from executor_node import execution_node
from observer_node import market_observer_node
from portfolio_node import portfolio_node
from risk_node import risk_manager_node
from scalper_expert_node import scalper_expert_node
from state import AgentState
from supervisor_node import supervisor_node

SENTIMENT_THRESHOLD: float = 0.5
SCAN_INTERVAL_SECONDS: int = 60
EXECUTABLE_ACTIONS: frozenset[str] = frozenset(
    {"MARKET_BUY", "LIMIT_BUY", "MARKET_SELL", "LIMIT_SELL", "CANCEL_ALL"}
)


def route_after_observer(state: AgentState) -> str:
    """Route to ``supervisor`` only when Node 1 populated ``historical_data`` without errors.

    Args:
        state: Current ``AgentState`` dictionary after Node 1 execution.

    Returns:
        ``END`` if ``error_message`` is set or ``historical_data`` is empty,
        otherwise ``"supervisor"``.
    """
    df = state["historical_data"]
    if state["error_message"] is not None or df is None or df.empty:
        return END
    return "supervisor"


def route_after_supervisor(state: AgentState) -> str:
    """Route ``ACTIVE`` regimes to ``alpha`` and ``FLAT`` regimes to ``scalper``.

    Args:
        state: Current ``AgentState`` dictionary after Supervisor execution.

    Returns:
        ``"alpha"`` if ``market_regime == "ACTIVE"``, otherwise ``"scalper"``.
    """
    regime = str(state.get("market_regime", "FLAT")).upper()
    if regime == "ACTIVE":
        return "alpha"
    return "scalper"


def route_after_alpha(state: AgentState) -> str:
    """Route buy, sell, and cancellation actions downstream, and ``HOLD`` to ``END``.

    Args:
        state: Current ``AgentState`` dictionary after Node 2 execution.

    Returns:
        ``"risk"`` if ``action`` is in
        ``{"MARKET_BUY", "LIMIT_BUY", "MARKET_SELL", "LIMIT_SELL", "CANCEL_ALL"}``,
        otherwise ``END``.
    """
    action = str(state.get("action", "HOLD")).upper()
    if action in EXECUTABLE_ACTIONS:
        return "risk"
    return END


def route_after_risk(state: AgentState) -> str:
    """Route to ``executor`` only when Node 3 clears micro-fund risk limits.

    Args:
        state: Current ``AgentState`` dictionary after Node 3 execution.

    Returns:
        ``"executor"`` if ``risk_cleared is True``, otherwise ``END``.
    """
    if state["risk_cleared"] is True:
        return "executor"
    return END


def build_agent_graph():
    """Construct and compile the deterministic LangGraph trading workflow."""
    graph = StateGraph(AgentState)

    # Register all seven deterministic swarm execution nodes.
    graph.add_node("portfolio", portfolio_node)
    graph.add_node("observer", market_observer_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("scalper", scalper_expert_node)
    graph.add_node("alpha", alpha_logic_gate_node)
    graph.add_node("risk", risk_manager_node)
    graph.add_node("executor", execution_node)

    # Wire entry point (START -> portfolio -> observer -> supervisor) and conditional transitions.
    graph.set_entry_point("portfolio")
    graph.add_edge("portfolio", "observer")
    graph.add_conditional_edges(
        "observer",
        route_after_observer,
        {
            "supervisor": "supervisor",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "alpha": "alpha",
            "scalper": "scalper",
        },
    )
    graph.add_conditional_edges(
        "scalper",
        route_after_alpha,
        {
            "risk": "risk",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "alpha",
        route_after_alpha,
        {
            "risk": "risk",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "risk",
        route_after_risk,
        {
            "executor": "executor",
            END: END,
        },
    )
    graph.add_edge("executor", END)

    return graph.compile()


app = build_agent_graph()


if __name__ == "__main__":
    print("🚀 Quant Agent Online. Beginning market scanning loop...")
    target_symbol = os.environ.get("ASSET_SYMBOL", "BTC/USD")

    while True:
        initial_state: AgentState = {
            "asset_symbol": target_symbol,
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
            "realized_pnl": 0.0,
        }

        try:
            result = app.invoke(initial_state)
            if result.get("error_message"):
                print(f"[Scan] {result['error_message']}")
        except KeyboardInterrupt:
            print("Shutting down Quant Agent scanning loop.")
            break
        except Exception as exc:
            print(f"[Error] Graph execution failed: {exc}")

        # Scan once per minute to respect API and LLM rate limits.
        time.sleep(SCAN_INTERVAL_SECONDS)
