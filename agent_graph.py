"""LangGraph Traffic Cop: compiles the 4-node autonomous trading state machine.

Wires ``observer`` -> ``alpha`` -> ``risk`` -> ``executor`` with deterministic
conditional routing gates between Alpha/Risk and Risk/Executor.
"""

from langgraph.graph import END, StateGraph

from alpha_node import alpha_logic_gate_node
from executor_node import execution_node
from observer_node import market_observer_node
from risk_node import risk_manager_node
from state import AgentState

SENTIMENT_THRESHOLD: float = 0.5


def route_after_alpha(state: AgentState) -> str:
    """Route to ``risk`` only when sentiment exceeds the bullish threshold.

    Args:
        state: Current ``AgentState`` dictionary after Node 2 execution.

    Returns:
        ``"risk"`` if ``sentiment_score > 0.5``, otherwise ``END``.
    """
    if state["sentiment_score"] > SENTIMENT_THRESHOLD:
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

    # Register all four deterministic execution nodes.
    graph.add_node("observer", market_observer_node)
    graph.add_node("alpha", alpha_logic_gate_node)
    graph.add_node("risk", risk_manager_node)
    graph.add_node("executor", execution_node)

    # Wire entry point and sequential/conditional transitions.
    graph.set_entry_point("observer")
    graph.add_edge("observer", "alpha")
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
