"""Unit tests for Node 2 (Alpha Logic Gate / Autonomous Quant Manager)."""

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
    "account_data",
    "open_positions",
    "active_orders",
    "action",
    "target_price",
    "rationale",
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
    assert isinstance(state["account_data"], dict)
    assert isinstance(state["open_positions"], list)
    assert isinstance(state["active_orders"], list)
    assert isinstance(state["action"], str)
    assert state["target_price"] is None or isinstance(state["target_price"], float)
    assert isinstance(state["rationale"], str)


@pytest.fixture
def populated_state() -> AgentState:
    """Return a valid ``AgentState`` populated with portfolio state and sample OHLCV data."""
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
        "account_data": {
            "buying_power": "399916.80",
            "cash": "99979.20",
            "portfolio_value": "99999.89",
        },
        "open_positions": [
            {
                "symbol": "BTCUSD",
                "side": "long",
                "qty": "0.000332",
                "avg_entry_price": "90100.0",
                "market_value": "30.43",
                "unrealized_pl": "0.51",
            }
        ],
        "active_orders": [
            {
                "id": "ord-open-001",
                "symbol": "BTC/USD",
                "side": "buy",
                "type": "limit",
                "qty": "0.00025",
                "limit_price": "89500.0",
            }
        ],
        "action": "HOLD",
        "target_price": None,
        "rationale": "",
    }


def test_alpha_logic_gate_node_injects_portfolio_state_and_returns_limit_buy(
    populated_state: AgentState,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify prompt injects portfolio state & condensed bars, and updates action/target_price/rationale."""
    _assert_valid_agent_state(populated_state)

    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = SentimentResponse(
        action="LIMIT_BUY",
        target_price=90400.0,
        rationale="Place resting limit bid at hourly structural support near $90,400.",
        sentiment_score=0.78,
    )

    mock_llm_instance = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_structured_llm

    with patch("alpha_node.ChatGoogleGenerativeAI", return_value=mock_llm_instance) as mock_cls:
        updated_state = alpha_logic_gate_node(populated_state)

    mock_cls.assert_called_once()
    assert mock_cls.call_args.kwargs["model"] == GEMINI_MODEL
    mock_llm_instance.with_structured_output.assert_called_once_with(SentimentResponse)
    mock_structured_llm.invoke.assert_called_once()

    messages = mock_structured_llm.invoke.call_args.args[0]
    assert len(messages) == 2
    system_content = messages[0].content
    human_content = messages[1].content

    assert "399916.80" in system_content
    assert "BTCUSD" in system_content
    assert "LIMIT_BUY" in system_content
    assert "CANCEL_ALL" in system_content
    assert "Window Summary (3 bars)" in human_content
    assert "BTC/USD" in human_content

    _assert_valid_agent_state(updated_state)
    assert updated_state["action"] == "LIMIT_BUY"
    assert updated_state["target_price"] == pytest.approx(90400.0)
    assert "structural support" in updated_state["rationale"]
    assert updated_state["sentiment_score"] == pytest.approx(0.78)
    assert updated_state["error_message"] is None

    captured = capsys.readouterr().out
    assert "[Alpha] Action: LIMIT_BUY" in captured
    assert "90400.0" in captured
    assert "structural support" in captured


def test_alpha_logic_gate_node_api_failure_defaults_hold_and_zero_score(
    populated_state: AgentState,
) -> None:
    """Verify API failure/rate limit defaults ``action`` to ``HOLD`` and ``sentiment_score`` to ``0.0``."""
    _assert_valid_agent_state(populated_state)
    populated_state["sentiment_score"] = 0.65
    populated_state["action"] = "MARKET_BUY"
    populated_state["target_price"] = 91000.0
    populated_state["rationale"] = "Previous rationale"

    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.side_effect = RuntimeError("429 Resource Exhausted: rate limit exceeded")

    mock_llm_instance = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_structured_llm

    with patch("alpha_node.ChatGoogleGenerativeAI", return_value=mock_llm_instance):
        updated_state = alpha_logic_gate_node(populated_state)

    _assert_valid_agent_state(updated_state)
    assert updated_state["action"] == "HOLD"
    assert updated_state["target_price"] is None
    assert updated_state["rationale"] == ""
    assert updated_state["sentiment_score"] == 0.0
    assert updated_state["error_message"] is not None
    assert "rate limit exceeded" in updated_state["error_message"]
