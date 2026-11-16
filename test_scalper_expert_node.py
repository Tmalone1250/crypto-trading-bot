"""Unit tests for the Multi-Asset Maker-Only Scalper Expert Node."""

import pandas as pd
import pytest

from scalper_expert_node import TRANCHE_NOTIONAL_USD, scalper_expert_node
from state import AgentState, TradeIntent


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
}


def _assert_valid_agent_state(state: AgentState) -> None:
    """Validate that ``state`` conforms to the ``AgentState`` schema and types."""
    assert set(state.keys()) == REQUIRED_STATE_KEYS
    assert isinstance(state["asset_symbols"], list)
    assert isinstance(state["market_data_by_symbol"], dict)
    assert isinstance(state["trade_intents"], list)
    for intent in state["trade_intents"]:
        assert isinstance(intent, TradeIntent)
        assert intent.action in {"LIMIT_BUY", "LIMIT_SELL"}


def _make_df(highs: list[float], lows: list[float], closes: list[float]) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame."""
    return pd.DataFrame({"high": highs, "low": lows, "close": closes})


def _base_state(
    market_data_by_symbol: dict[str, pd.DataFrame] | None = None,
    open_positions: list[dict] | None = None,
    asset_symbols: list[str] | None = None,
) -> AgentState:
    """Build a valid FLAT-regime ``AgentState`` for multi-asset scalper tests."""
    return {
        "asset_symbol": "BTC/USD",
        "asset_symbols": asset_symbols or ["BTC/USD", "ETH/USD", "SOL/USD"],
        "market_data_by_symbol": market_data_by_symbol or {},
        "current_price": 0.0,
        "historical_data": pd.DataFrame(),
        "sentiment_score": 0.0,
        "risk_cleared": False,
        "order_details": {},
        "error_message": None,
        "account_data": {"buying_power": "2000.00", "cash": "1000.00"},
        "open_positions": open_positions or [],
        "active_orders": [],
        "action": "HOLD",
        "target_price": None,
        "rationale": "",
        "market_regime": "FLAT",
        "realized_pnl": 0.0,
        "trade_intents": [],
    }


def test_scalper_fee_adjusted_sell_requires_1_5_percent_gain() -> None:
    """Verify ``LIMIT_SELL`` triggers at >= +1.5% (1.015x) and ignores +0.8% below fee target."""
    # avg_entry = 100.0 -> 100.80 (+0.8%) must HOLD; 101.60 (+1.6%) must LIMIT_SELL.
    df_below_target = _make_df([101.0] * 5, [100.0] * 5, [100.80] * 5)
    state_hold = _base_state(
        asset_symbols=["SOL/USD"],
        market_data_by_symbol={"SOL/USD": df_below_target},
        open_positions=[{"symbol": "SOLUSD", "avg_entry_price": "100.0", "qty": "2.0"}],
    )
    res_hold = scalper_expert_node(state_hold)
    _assert_valid_agent_state(res_hold)
    assert res_hold["trade_intents"] == []
    assert res_hold["action"] == "HOLD"

    df_above_target = _make_df([102.0] * 5, [100.5] * 5, [101.60] * 5)
    state_sell = _base_state(
        asset_symbols=["SOL/USD"],
        market_data_by_symbol={"SOL/USD": df_above_target},
        open_positions=[{"symbol": "SOLUSD", "avg_entry_price": "100.0", "qty": "2.0"}],
    )
    res_sell = scalper_expert_node(state_sell)
    _assert_valid_agent_state(res_sell)
    assert len(res_sell["trade_intents"]) == 1
    intent = res_sell["trade_intents"][0]
    assert intent.symbol == "SOL/USD"
    assert intent.action == "LIMIT_SELL"
    assert intent.target_price == pytest.approx(101.60)


def test_scalper_multi_tranche_grid_buys_200_usd_on_2_percent_drop() -> None:
    """Verify open position allows $200 ``LIMIT_BUY`` tranche when price drops >= 2% below avg_entry."""
    # avg_entry = 3000.0 -> 2935.0 (-2.17%) triggers $200 LIMIT_BUY tranche.
    df_eth_drop = _make_df([2960.0] * 5, [2930.0] * 5, [2935.0] * 5)
    state = _base_state(
        asset_symbols=["ETH/USD"],
        market_data_by_symbol={"ETH/USD": df_eth_drop},
        open_positions=[{"symbol": "ETHUSD", "avg_entry_price": "3000.0", "qty": "0.066"}],
    )
    updated = scalper_expert_node(state)

    _assert_valid_agent_state(updated)
    assert len(updated["trade_intents"]) == 1
    intent = updated["trade_intents"][0]
    assert intent.symbol == "ETH/USD"
    assert intent.action == "LIMIT_BUY"
    assert intent.target_price == pytest.approx(2935.0)
    assert intent.notional_usd == pytest.approx(TRANCHE_NOTIONAL_USD)
    assert intent.qty == pytest.approx(round(200.0 / 2935.0, 6))


def test_scalper_multi_asset_loop_emits_maker_only_intents_list() -> None:
    """Verify multi-asset loop over BTC/ETH/SOL emits a ``list[TradeIntent]`` with maker-only actions."""
    # BTC/USD: flat (no position), price in lower 20% of 90000..90200 -> LIMIT_BUY
    btc_df = _make_df(
        [90180.0, 90200.0, 90150.0, 90120.0, 90100.0],
        [90050.0, 90080.0, 90020.0, 90000.0, 90010.0],
        [90100.0, 90140.0, 90090.0, 90030.0, 90020.0],
    )
    # ETH/USD: open position @ 3000.0, price 2990.0 (-0.33%) -> no intent
    eth_df = _make_df([3010.0] * 5, [2985.0] * 5, [2990.0] * 5)
    # SOL/USD: open position @ 100.0, price 102.0 (+2.0% >= 1.5%) -> LIMIT_SELL
    sol_df = _make_df([102.5] * 5, [100.5] * 5, [102.0] * 5)

    state = _base_state(
        asset_symbols=["BTC/USD", "ETH/USD", "SOL/USD"],
        market_data_by_symbol={
            "BTC/USD": btc_df,
            "ETH/USD": eth_df,
            "SOL/USD": sol_df,
        },
        open_positions=[
            {"symbol": "ETHUSD", "avg_entry_price": "3000.0", "qty": "0.066"},
            {"symbol": "SOLUSD", "avg_entry_price": "100.0", "qty": "2.0"},
        ],
    )
    updated = scalper_expert_node(state)

    _assert_valid_agent_state(updated)
    intents = updated["trade_intents"]
    assert len(intents) == 2
    assert {i.symbol: i.action for i in intents} == {
        "BTC/USD": "LIMIT_BUY",
        "SOL/USD": "LIMIT_SELL",
    }
    assert all(i.action in {"LIMIT_BUY", "LIMIT_SELL"} for i in intents)
