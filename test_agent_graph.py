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
    route_after_supervisor,
)
from state import AgentState


def _make_state(**overrides) -> AgentState:
    """Return a valid synthetic ``AgentState`` with optional field overrides."""
    base: AgentState = {
        "asset_symbol": "BTC/USD",
        "asset_symbols": ["BTC/USD", "ETH/USD", "SOL/USD"],
        "market_data_by_symbol": {},
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
        "market_regime": "FLAT",
        "realized_pnl": 0.0,
        "trade_intents": [],
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    ("has_data", "error_message", "expected"),
    [
        (True, None, "supervisor"),
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
    """Verify ``route_after_observer`` routes valid data to ``supervisor`` and errors to ``END``."""
    df = pd.DataFrame({"close": [90000.0]}) if has_data else pd.DataFrame()
    state = _make_state(historical_data=df, error_message=error_message)
    assert route_after_observer(state) == expected


@pytest.mark.parametrize(
    ("market_regime", "expected"),
    [
        ("ACTIVE", "alpha"),
        ("FLAT", "scalper"),
    ],
)
def test_route_after_supervisor(market_regime: str, expected: str) -> None:
    """Verify ``route_after_supervisor`` routes ``ACTIVE`` to ``alpha`` and ``FLAT`` to ``scalper``."""
    state = _make_state(market_regime=market_regime)
    assert route_after_supervisor(state) == expected


@pytest.mark.parametrize(
    ("action", "target_price", "expected"),
    [
        ("MARKET_BUY", None, "risk"),
        ("LIMIT_BUY", 89500.0, "risk"),
        ("MARKET_SELL", None, "risk"),
        ("LIMIT_SELL", 93500.0, "risk"),
        ("CANCEL_ALL", None, "risk"),
        ("HOLD", None, END),
    ],
)
def test_route_after_alpha(
    action: str,
    target_price: float | None,
    expected: str,
) -> None:
    """Verify ``route_after_alpha`` routes buy, sell, and cancel actions to ``risk`` and ``HOLD`` to ``END``."""
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
    assert {
        "portfolio",
        "observer",
        "supervisor",
        "scalper",
        "alpha",
        "risk",
        "executor",
    }.issubset(node_names)

    edges = {(edge.source, edge.target) for edge in graph_view.edges}
    assert ("__start__", "portfolio") in edges
    assert ("portfolio", "observer") in edges
    assert ("observer", "supervisor") in edges
    assert ("supervisor", "alpha") in edges
    assert ("supervisor", "scalper") in edges
    assert ("scalper", "risk") in edges
