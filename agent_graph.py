"""LangGraph Traffic Cop: compiles the 5-node autonomous trading state machine.

Wires ``portfolio`` -> ``observer`` -> ``alpha`` -> ``risk`` -> ``executor`` with
deterministic conditional routing gates after Observer, Alpha, and Risk nodes.
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
from state import AgentState

SENTIMENT_THRESHOLD: float = 0.5
SCAN_INTERVAL_SECONDS: int = 60


def route_after_observer(state: AgentState) -> str:
    """Route to ``alpha`` only when Node 1 populated ``historical_data`` without errors.

    Args:
        state: Current ``AgentState`` dictionary after Node 1 execution.

    Returns:
        ``END`` if ``error_message`` is set or ``historical_data`` is empty,
        otherwise ``"alpha"``.
    """
    df = state["historical_data"]
    if state["error_message"] is not None or df is None or df.empty:
        return END
    return "alpha"


def route_after_alpha(state: AgentState) -> str:
    """Route to ``risk`` for ``BUY``/``SELL`` decisions, or ``END`` when ``HOLD``.

    Args:
        state: Current ``AgentState`` dictionary after Node 2 execution.

    Returns:
        ``"risk"`` if ``decision`` is ``"BUY"`` or ``"SELL"``, otherwise ``END``.
    """
    decision = str(state.get("decision", "HOLD")).upper()
    if decision in {"BUY", "SELL"}:
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

    # Register all five deterministic execution nodes.
    graph.add_node("portfolio", portfolio_node)
    graph.add_node("observer", market_observer_node)
    graph.add_node("alpha", alpha_logic_gate_node)
    graph.add_node("risk", risk_manager_node)
    graph.add_node("executor", execution_node)

    # Wire entry point (START -> portfolio -> observer) and conditional transitions.
    graph.set_entry_point("portfolio")
    graph.add_edge("portfolio", "observer")
    graph.add_conditional_edges(
        "observer",
        route_after_observer,
        {
            "alpha": "alpha",
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
            "decision": "HOLD",
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
