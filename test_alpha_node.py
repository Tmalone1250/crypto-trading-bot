"""Unit tests for Node 2 (Alpha Macro Circuit Breaker via Gemini Decisions API)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from alpha_node import (
    CIRCUIT_BREAKER_INSTRUCTION,
    CIRCUIT_BREAKER_QUESTION_ID,
    DECISION_MODEL,
    alpha_logic_gate_node,
)
from state import AgentState


REQUIRED_STATE_KEYS = {
    "asset_symbol",
    "asset_symbols",
    "market_data_by_symbol",
    "current_price",
    "historical_data",
    "sentiment_score",
    "risk_cleared",
    "order_details",
    "error_message",
    "account_data",
    "open_positions",
    "active_orders",
    "action",
    "target_price",
    "rationale",
    "market_regime",
    "realized_pnl",
    "trade_intents",
    "last_alpha_eval_price",
    "macro_context",
    "mcp_client",
}


def _assert_valid_agent_state(state: AgentState) -> None:
    """Validate that ``state`` conforms to the ``AgentState`` schema and types."""
    assert set(state.keys()) == REQUIRED_STATE_KEYS
    assert isinstance(state["asset_symbol"], str)
    assert isinstance(state["asset_symbols"], list)
    assert isinstance(state["market_data_by_symbol"], dict)
    assert isinstance(state["current_price"], float)
    assert isinstance(state["historical_data"], pd.DataFrame)
    assert isinstance(state["sentiment_score"], float)
    assert isinstance(state["risk_cleared"], bool)
    assert isinstance(state["order_details"], dict)
    assert state["error_message"] is None or isinstance(state["error_message"], str)
    assert isinstance(state["account_data"], dict)
    assert isinstance(state["open_positions"], list)
    assert isinstance(state["active_orders"], list)
    assert isinstance(state["action"], str)
    assert state["target_price"] is None or isinstance(state["target_price"], float)
    assert isinstance(state["rationale"], str)
    assert isinstance(state["market_regime"], str)
    assert isinstance(state["realized_pnl"], float)
    assert isinstance(state["trade_intents"], list)
    assert isinstance(state["last_alpha_eval_price"], float)
    assert isinstance(state["macro_context"], str)


@pytest.fixture
def populated_state() -> AgentState:
    """Return a valid ``AgentState`` populated with OHLCV and macro calendar context."""
    df = pd.DataFrame(
        {
            "open": [90000.0, 90500.0, 91000.0],
            "high": [90600.0, 91200.0, 91800.0],
            "low": [89900.0, 90400.0, 90950.0],
            "close": [90500.0, 91050.0, 91650.0],
            "volume": [12.5, 18.2, 24.0],
        }
    )
    return {
        "asset_symbol": "BTC/USD",
        "asset_symbols": ["BTC/USD", "ETH/USD", "SOL/USD"],
        "market_data_by_symbol": {},
        "current_price": 91650.0,
        "historical_data": df,
        "sentiment_score": 0.0,
        "risk_cleared": False,
        "order_details": {},
        "error_message": None,
        "account_data": {
            "buying_power": "399916.80",
            "cash": "99979.20",
            "portfolio_value": "99999.89",
        },
        "open_positions": [],
        "active_orders": [],
        "action": "HOLD",
        "target_price": None,
        "rationale": "",
        "market_regime": "ACTIVE",
        "realized_pnl": 0.0,
        "trade_intents": [],
        "last_alpha_eval_price": 0.0,
        "macro_context": (
            "Current UTC Time: 2026-11-18T18:30:00Z\n"
            "High-Impact USD Macro Calendar (Next 24h):\n"
            "- [2026-11-18T19:00:00Z] USD (high): FOMC Rate Decision (in 30 minutes)"
        ),
        "mcp_client": None,
    }


def test_alpha_fomc_30_minutes_away_triggers_circuit_breaker(
    populated_state: AgentState,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify mocked FOMC rate decision 30m away with noul > 0.85 triggers CIRCUIT_BREAKER."""
    _assert_valid_agent_state(populated_state)

    mock_decision_response = SimpleNamespace(
        answers={CIRCUIT_BREAKER_QUESTION_ID: SimpleNamespace(noul=0.94)}
    )
    mock_client = MagicMock()
    mock_client.decisions.create.return_value = mock_decision_response

    with patch("alpha_node.genai.Client", return_value=mock_client) as mock_client_cls:
        updated_state = alpha_logic_gate_node(populated_state)

    _assert_valid_agent_state(updated_state)
    assert mock_client_cls.call_args.kwargs["http_options"] == {"api_version": "v1beta"}
    mock_client.decisions.create.assert_called_once_with(
        model=DECISION_MODEL,
        input=populated_state["macro_context"],
        questions={
            CIRCUIT_BREAKER_QUESTION_ID: {
                "type": "noul",
                "instructions": CIRCUIT_BREAKER_INSTRUCTION,
            }
        },
    )
    assert updated_state["action"] == "CIRCUIT_BREAKER"
    assert updated_state["target_price"] is None
    assert updated_state["sentiment_score"] == pytest.approx(-0.94)
    assert updated_state["last_alpha_eval_price"] == pytest.approx(91650.0)
    assert updated_state["error_message"] is None

    captured = capsys.readouterr().out
    assert "[Alpha] Decision: CIRCUIT_BREAKER" in captured


def test_alpha_below_threshold_outputs_hold(populated_state: AgentState) -> None:
    """Verify noul probability <= 0.85 outputs HOLD."""
    populated_state["macro_context"] = (
        "Current UTC Time: 2026-11-18T18:30:00Z\n"
        "High-Impact USD Macro Calendar (Next 24h): No high-impact events scheduled."
    )
    mock_decision_response = SimpleNamespace(
        answers={CIRCUIT_BREAKER_QUESTION_ID: SimpleNamespace(noul=0.12)}
    )
    mock_client = MagicMock()
    mock_client.decisions.create.return_value = mock_decision_response

    with patch("alpha_node.genai.Client", return_value=mock_client) as mock_client_cls:
        updated_state = alpha_logic_gate_node(populated_state)

    _assert_valid_agent_state(updated_state)
    assert mock_client_cls.call_args.kwargs["http_options"] == {"api_version": "v1beta"}
    assert updated_state["action"] == "HOLD"
    assert updated_state["sentiment_score"] == pytest.approx(0.0)
    assert updated_state["error_message"] is None


def test_alpha_decisions_api_failure_defaults_hold(populated_state: AgentState) -> None:
    """Verify Decisions API failures safely default to HOLD with zero sentiment."""
    mock_client = MagicMock()
    mock_client.decisions.create.side_effect = RuntimeError("Decisions API 503")

    with patch("alpha_node.genai.Client", return_value=mock_client) as mock_client_cls:
        updated_state = alpha_logic_gate_node(populated_state)

    _assert_valid_agent_state(updated_state)
    assert mock_client_cls.call_args.kwargs["http_options"] == {"api_version": "v1beta"}
    assert updated_state["action"] == "HOLD"
    assert updated_state["target_price"] is None
    assert updated_state["sentiment_score"] == pytest.approx(0.0)
    assert updated_state["error_message"] is not None
    assert "Decisions API 503" in updated_state["error_message"]
