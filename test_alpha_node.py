"""Unit tests for Node 2 (Alpha Logic Gate) following TDD and .ai_rules standards."""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from alpha_node import GEMINI_MODEL, SentimentResponse, alpha_logic_gate_node
from state import AgentState


REQUIRED_STATE_KEYS = {
    "asset_symbol",
    "current_price",
    "historical_data",
    "sentiment_score",
    "risk_cleared",
    "order_details",
    "error_message",
}


def _assert_valid_agent_state(state: AgentState) -> None:
    """Validate that ``state`` conforms to the ``AgentState`` schema and types."""
    assert set(state.keys()) == REQUIRED_STATE_KEYS
    assert isinstance(state["asset_symbol"], str)
    assert isinstance(state["current_price"], float)
    assert isinstance(state["historical_data"], pd.DataFrame)
    assert isinstance(state["sentiment_score"], float)
    assert isinstance(state["risk_cleared"], bool)
    assert isinstance(state["order_details"], dict)
    assert state["error_message"] is None or isinstance(state["error_message"], str)


@pytest.fixture
def populated_state() -> AgentState:
    """Return a valid ``AgentState`` populated with sample OHLCV data."""
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
        "current_price": 91650.0,
        "historical_data": df,
        "sentiment_score": 0.0,
        "risk_cleared": False,
        "order_details": {},
        "error_message": None,
    }


def test_alpha_logic_gate_node_success(populated_state: AgentState) -> None:
    """Verify a valid structured JSON response updates ``sentiment_score``."""
    _assert_valid_agent_state(populated_state)

    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = SentimentResponse(sentiment_score=0.78)

    mock_llm_instance = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_structured_llm

    with patch("alpha_node.ChatGoogleGenerativeAI", return_value=mock_llm_instance) as mock_cls:
        updated_state = alpha_logic_gate_node(populated_state)

    mock_cls.assert_called_once()
    assert mock_cls.call_args.kwargs["model"] == GEMINI_MODEL
    mock_llm_instance.with_structured_output.assert_called_once_with(SentimentResponse)
    mock_structured_llm.invoke.assert_called_once()

    _assert_valid_agent_state(updated_state)
    assert updated_state["sentiment_score"] == pytest.approx(0.78)
    assert updated_state["error_message"] is None


def test_alpha_logic_gate_node_api_failure_defaults_score(populated_state: AgentState) -> None:
    """Verify API failure/rate limit defaults ``sentiment_score`` to 0.0 and sets ``error_message``."""
    _assert_valid_agent_state(populated_state)
    populated_state["sentiment_score"] = 0.65

    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.side_effect = RuntimeError("429 Resource Exhausted: rate limit exceeded")

    mock_llm_instance = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_structured_llm

    with patch("alpha_node.ChatGoogleGenerativeAI", return_value=mock_llm_instance):
        updated_state = alpha_logic_gate_node(populated_state)

    _assert_valid_agent_state(updated_state)
    assert updated_state["sentiment_score"] == 0.0
    assert updated_state["error_message"] is not None
    assert "rate limit exceeded" in updated_state["error_message"]
