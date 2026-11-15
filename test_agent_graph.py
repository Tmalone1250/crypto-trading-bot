"""Unit tests for the compiled LangGraph workflow and conditional routers."""

import pandas as pd
import pytest
from langgraph.graph import END

import agent_graph
from agent_graph import (
    app,
    build_agent_graph,
    route_after_alpha,
    route_after_observer,
    route_after_risk,
)
from state import AgentState


def _make_state(**overrides) -> AgentState:
    """Return a valid synthetic ``AgentState`` with optional field overrides."""
    base: AgentState = {
        "asset_symbol": "BTC/USD",
        "current_price": 90000.0,
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
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    ("has_data", "error_message", "expected"),
    [
        (True, None, "alpha"),
        (True, "Market observer error", END),
        (False, None, END),
        (False, "No market data returned", END),
    ],
)
def test_route_after_observer(
    has_data: bool,
    error_message: str | None,
    expected: str,
) -> None:
    """Verify ``route_after_observer`` short-circuits to ``END`` on errors or empty data."""
    df = pd.DataFrame({"close": [90000.0]}) if has_data else pd.DataFrame()
    state = _make_state(historical_data=df, error_message=error_message)
    assert route_after_observer(state) == expected


@pytest.mark.parametrize(
    ("action", "target_price", "expected"),
    [
        ("MARKET_BUY", None, "risk"),
        ("LIMIT_BUY", 89500.0, "risk"),
        ("CANCEL_ALL", None, "risk"),
        ("HOLD", None, END),
    ],
)
def test_route_after_alpha(
    action: str,
    target_price: float | None,
    expected: str,
) -> None:
    """Verify ``route_after_alpha`` routes ``MARKET_BUY``, ``LIMIT_BUY``, ``CANCEL_ALL`` to ``risk`` and ``HOLD`` to ``END``."""
    state = _make_state(action=action, target_price=target_price)
    assert route_after_alpha(state) == expected


@pytest.mark.parametrize(
    ("risk_cleared", "expected"),
    [
        (True, "executor"),
        (False, END),
    ],
)
def test_route_after_risk(risk_cleared: bool, expected: str) -> None:
    """Verify ``route_after_risk`` routes to ``executor`` iff ``risk_cleared is True``."""
    state = _make_state(risk_cleared=risk_cleared)
    assert route_after_risk(state) == expected


def test_graph_compiles_with_expected_nodes() -> None:
    """Verify the LangGraph workflow compiles cleanly, starts at ``portfolio``, and routes to ``observer``."""
    compiled = build_agent_graph()
    assert compiled is not None
    assert app is not None
    assert hasattr(agent_graph, "app")

    graph_view = compiled.get_graph()
    node_names = set(graph_view.nodes.keys())
    assert {"portfolio", "observer", "alpha", "risk", "executor"}.issubset(node_names)

    edges = {(edge.source, edge.target) for edge in graph_view.edges}
    assert ("__start__", "portfolio") in edges
    assert ("portfolio", "observer") in edges
