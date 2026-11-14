"""Unit tests for the compiled LangGraph workflow and conditional routers."""

import pandas as pd
import pytest
from langgraph.graph import END

import agent_graph
from agent_graph import app, build_agent_graph, route_after_alpha, route_after_risk
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
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    ("sentiment_score", "expected"),
    [
        (0.51, "risk"),
        (0.95, "risk"),
        (0.5, END),
        (0.0, END),
        (-0.8, END),
    ],
)
def test_route_after_alpha(sentiment_score: float, expected: str) -> None:
    """Verify ``route_after_alpha`` routes to ``risk`` iff ``sentiment_score > 0.5``."""
    state = _make_state(sentiment_score=sentiment_score)
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
    """Verify the LangGraph workflow compiles cleanly and contains all four nodes."""
    compiled = build_agent_graph()
    assert compiled is not None
    assert app is not None
    assert hasattr(agent_graph, "app")

    node_names = set(compiled.get_graph().nodes.keys())
    assert {"observer", "alpha", "risk", "executor"}.issubset(node_names)
