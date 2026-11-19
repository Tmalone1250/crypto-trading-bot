"""Unit tests for the Multi-Asset Maker-Only Scalper Expert Node."""

from datetime import datetime
import duckdb
import pandas as pd
import pytest

from risk_node import risk_manager_node
import scalper_expert_node as scalper_mod
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
    "last_alpha_eval_price",
    "mcp_client",
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
        "last_alpha_eval_price": 0.0,
        "mcp_client": None,
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


def test_scalper_session_open_watch_phase_1_displacement_and_phase_2_entry(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify SESSION_OPEN_WATCH transitions PHASE_0 -> PHASE_1 -> PHASE_2 and routes 1.5R risk."""
    db_file = str(tmp_path / "test_reversion.duckdb")
    monkeypatch.setattr(scalper_mod, "DB_PATH", db_file)

    conn = duckdb.connect(db_file)
    try:
        conn.execute(
            """
            CREATE TABLE setup_state_machine (
                setup_id VARCHAR PRIMARY KEY,
                strategy VARCHAR NOT NULL,
                phase VARCHAR NOT NULL,
                sweep_low DOUBLE,
                target_price DOUBLE,
                timestamp TIMESTAMP NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO setup_state_machine (setup_id, strategy, phase, sweep_low, target_price, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ["NEW_YORK-202611160830-abc123", "FAIR_PRICE_REVERSION", "PHASE_0_ANCHORED", None, 100000.0, datetime(2026, 11, 16, 14, 30)],
        )
    finally:
        conn.close()

    # Phase 1: price drops -0.30% (99700 vs 100000 Fair Price >= 0.20%) -> PHASE_1_DISPLACED
    df_phase1 = pd.DataFrame(
        {
            "open": [100000.0, 99850.0],
            "high": [100020.0, 99880.0],
            "low": [99800.0, 99650.0],
            "close": [99850.0, 99700.0],
        }
    )
    state_p1 = _base_state(
        asset_symbols=["BTC/USD"],
        market_data_by_symbol={"BTC/USD": df_phase1},
    )
    state_p1["market_regime"] = "SESSION_OPEN_WATCH"
    state_p1["current_price"] = 99700.0

    res_p1 = scalper_expert_node(state_p1)
    assert res_p1["action"] == "HOLD"
    assert res_p1["trade_intents"] == []

    conn = duckdb.connect(db_file)
    try:
        phase_after_p1 = conn.execute(
            "SELECT phase, sweep_low FROM setup_state_machine WHERE setup_id = 'NEW_YORK-202611160830-abc123'"
        ).fetchone()
        assert phase_after_p1 == ("PHASE_1_DISPLACED", 99650.0)
    finally:
        conn.close()

    # Phase 2: 1m bullish displacement candle (body > prev body, closes past prev high)
    df_phase2 = pd.DataFrame(
        {
            "open": [99700.0, 99690.0],
            "high": [99740.0, 99860.0],
            "low": [99650.0, 99680.0],
            "close": [99690.0, 99850.0],
        }
    )
    state_p2 = _base_state(
        asset_symbols=["BTC/USD"],
        market_data_by_symbol={"BTC/USD": df_phase2},
    )
    state_p2["market_regime"] = "SESSION_OPEN_WATCH"
    state_p2["current_price"] = 99850.0

    res_p2 = scalper_expert_node(state_p2)
    assert len(res_p2["trade_intents"]) == 1
    intent = res_p2["trade_intents"][0]
    assert intent.action == "LIMIT_BUY"
    assert intent.target_price == pytest.approx(99850.0)
    assert intent.raw_stop_loss == pytest.approx(99650.0)

    # Route through risk_manager_node to verify automatic 1.5R math
    risk_out = risk_manager_node(res_p2)
    assert risk_out["risk_cleared"] is True
    assert risk_out["order_details"]["risk_reward_ratio"] == pytest.approx(1.5)
    assert risk_out["order_details"]["stop_loss_price"] == pytest.approx(99650.0 * 0.9995)


def test_scalper_crt_rebreak_full_multi_phase_state_transitions(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify CRT Re-break PHASE_1_SWEPT -> PHASE_2 -> PHASE_3 -> PHASE_4 and 1.5R handoff."""
    db_file = str(tmp_path / "test_crt.duckdb")
    monkeypatch.setattr(scalper_mod, "DB_PATH", db_file)

    # Candle 1: H1=101000, L1=99000
    # Candle 2: Sweeps L1 (L2=98600 < 99000), H2=99800, closes inside C1 (99400)
    # Midpoint M = (99800 + 98600) / 2 = 99200; Target = H1 = 101000
    df_1h = pd.DataFrame(
        {
            "open": [100000.0, 99500.0],
            "high": [101000.0, 99800.0],
            "low": [99000.0, 98600.0],
            "close": [99600.0, 99400.0],
        }
    )
    df_1m_neutral = pd.DataFrame(
        {
            "open": [99100.0, 99100.0, 99100.0, 99100.0, 99100.0],
            "high": [99150.0, 99150.0, 99150.0, 99150.0, 99150.0],
            "low": [99050.0, 99050.0, 99050.0, 99050.0, 99050.0],
            "close": [99100.0, 99100.0, 99100.0, 99100.0, 99100.0],
        }
    )

    def _make_crt_state(price: float) -> AgentState:
        st = _base_state(
            asset_symbols=["BTC/USD"],
            market_data_by_symbol={
                "BTC/USD": df_1m_neutral,
                "BTC/USD_1H": df_1h,
            },
        )
        st["market_regime"] = "FLAT"
        st["current_price"] = price
        return st

    # Phase 1: Sweep & Midpoint Lock (price at 99100 < M=99200)
    res1 = scalper_expert_node(_make_crt_state(99100.0))
    assert res1["trade_intents"] == []

    conn = duckdb.connect(db_file)
    try:
        row1 = conn.execute(
            "SELECT strategy, phase, sweep_low, target_price FROM setup_state_machine LIMIT 1"
        ).fetchone()
        assert row1 == ("CRT_REBREAK", "PHASE_1_SWEPT", 99200.0, 101000.0)
    finally:
        conn.close()

    # Phase 2: First Break above M (99250 > 99200) -> PHASE_2_FIRST_BREAK, no entry
    res2 = scalper_expert_node(_make_crt_state(99250.0))
    assert res2["trade_intents"] == []

    conn = duckdb.connect(db_file)
    try:
        phase2 = conn.execute("SELECT phase FROM setup_state_machine LIMIT 1").fetchone()[0]
        assert phase2 == "PHASE_2_FIRST_BREAK"
    finally:
        conn.close()

    # Phase 3: Pullback below M (99150 < 99200) -> PHASE_3_PULLBACK
    res3 = scalper_expert_node(_make_crt_state(99150.0))
    assert res3["trade_intents"] == []

    conn = duckdb.connect(db_file)
    try:
        phase3 = conn.execute("SELECT phase FROM setup_state_machine LIMIT 1").fetchone()[0]
        assert phase3 == "PHASE_3_PULLBACK"
    finally:
        conn.close()

    # Phase 4: Re-break above M (99220 >= 99200) -> emits LIMIT_BUY at M=99200
    # Distance M to H1 = 101000 - 99200 = 1800 -> SL distance = 1800 / 1.5 = 1200 -> raw SL = 98000
    res4 = scalper_expert_node(_make_crt_state(99220.0))
    assert len(res4["trade_intents"]) == 1
    intent = res4["trade_intents"][0]
    assert intent.action == "LIMIT_BUY"
    assert intent.target_price == pytest.approx(99200.0)
    assert intent.raw_stop_loss == pytest.approx(98000.0)

    conn = duckdb.connect(db_file)
    try:
        phase4 = conn.execute("SELECT phase FROM setup_state_machine LIMIT 1").fetchone()[0]
        assert phase4 == "PHASE_4_EXECUTED"
    finally:
        conn.close()

    # Verify risk_node.py consumes raw_stop_loss and computes 1.5R constraints
    risk_out = risk_manager_node(res4)
    assert risk_out["risk_cleared"] is True
    assert risk_out["order_details"]["stop_loss_price"] == pytest.approx(98000.0 * 0.9995)
    assert risk_out["order_details"]["risk_reward_ratio"] == pytest.approx(1.5)


def test_scalper_smc_golden_fvg_multi_phase_transitions(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify Strategy 3 (SMC_GOLDEN_FVG) PHASE_1_POI_TAPPED -> PHASE_2_CHOCH -> PHASE_3_LIMIT_ARMED."""
    db_file = str(tmp_path / "test_smc.duckdb")
    monkeypatch.setattr(scalper_mod, "DB_PATH", db_file)

    # 15m unmitigated Bullish Demand FVG: c0.high=99500 < c2.low=99700 -> [99500, 99700]
    df_15m = pd.DataFrame(
        {
            "open": [99300.0, 99450.0, 99750.0],
            "high": [99500.0, 99780.0, 99950.0],
            "low": [99200.0, 99420.0, 99700.0],
            "close": [99450.0, 99760.0, 99900.0],
        }
    )
    # 1m Phase 1: price taps 15m FVG at 99600
    df_1m_tap = pd.DataFrame(
        {
            "open": [99650.0, 99620.0, 99610.0, 99605.0, 99610.0],
            "high": [99680.0, 99640.0, 99630.0, 99625.0, 99620.0],
            "low": [99600.0, 99590.0, 99580.0, 99575.0, 99590.0],
            "close": [99620.0, 99610.0, 99605.0, 99610.0, 99600.0],
        }
    )

    st_p1 = _base_state(
        asset_symbols=["BTC/USD"],
        market_data_by_symbol={"BTC/USD": df_1m_tap, "BTC/USD_15M": df_15m},
    )
    st_p1["market_regime"] = "FLAT"
    st_p1["current_price"] = 99600.0
    assert scalper_expert_node(st_p1)["trade_intents"] == []

    conn = duckdb.connect(db_file)
    try:
        row1 = conn.execute(
            "SELECT strategy, phase, sweep_low, target_price FROM setup_state_machine LIMIT 1"
        ).fetchone()
        assert row1 == ("SMC_GOLDEN_FVG", "PHASE_1_POI_TAPPED", 99500.0, 99700.0)
    finally:
        conn.close()

    # 1m Phase 2 & 3: sweep low=99500, full-body CHoCH up to disp_high=100100, leaving 1m FVG
    # Leg: 99500 -> 100100 (span=600). Golden Zone [50%, 61.8%] = [99729.2, 99800.0]
    # 1m FVG: bar[-3].high=99720 < bar[-1].low=99820 -> [99720, 99820] overlaps Golden Zone!
    df_1m_choch = pd.DataFrame(
        {
            "open": [99620.0, 99600.0, 99540.0, 99620.0, 99850.0],
            "high": [99680.0, 99640.0, 99720.0, 99880.0, 100100.0],
            "low": [99580.0, 99560.0, 99500.0, 99610.0, 99820.0],
            "close": [99600.0, 99570.0, 99620.0, 99850.0, 100050.0],
        }
    )
    st_p2 = _base_state(
        asset_symbols=["BTC/USD"],
        market_data_by_symbol={"BTC/USD": df_1m_choch, "BTC/USD_15M": df_15m},
    )
    st_p2["market_regime"] = "FLAT"
    st_p2["current_price"] = 100050.0
    assert scalper_expert_node(st_p2)["trade_intents"] == []

    conn = duckdb.connect(db_file)
    try:
        phase2 = conn.execute("SELECT phase FROM setup_state_machine LIMIT 1").fetchone()[0]
        assert phase2 == "PHASE_2_CHOCH"
    finally:
        conn.close()

    # Phase 3: arms LIMIT_BUY in Golden Zone [99729.2, 99800.0] -> entry=99764.6, raw_sl < 99500
    res_p3 = scalper_expert_node(st_p2)
    assert len(res_p3["trade_intents"]) == 1
    intent = res_p3["trade_intents"][0]
    assert intent.action == "LIMIT_BUY"
    assert 99729.2 <= intent.target_price <= 99800.0
    assert intent.raw_stop_loss is not None
    assert intent.raw_stop_loss < 99500.0

    risk_out = risk_manager_node(res_p3)
    assert risk_out["risk_cleared"] is True
    assert risk_out["order_details"]["risk_reward_ratio"] == pytest.approx(1.5)


def test_scalper_lvn_absorption_squeeze_multi_phase(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify Strategy 4 (LVN_ABSORPTION_SQUEEZE) PHASE_1 -> PHASE_2 -> PHASE_3 MARKET_BUY."""
    db_file = str(tmp_path / "test_lvn.duckdb")
    monkeypatch.setattr(scalper_mod, "DB_PATH", db_file)

    # Build 60 1m bars: heavy volume around 100000..100500 (POC/VAL), sparse LVN around 99100..99200
    closes = [100200.0] * 55 + [99150.0, 100100.0, 100150.0, 100200.0, 100250.0]
    highs = [c + 50.0 for c in closes]
    lows = [c - 50.0 for c in closes]
    opens = [c - 10.0 for c in closes]
    vols = [100.0] * 55 + [10.0, 100.0, 100.0, 100.0, 100.0]
    df_60 = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols}
    )

    st_p1 = _base_state(
        asset_symbols=["BTC/USD"],
        market_data_by_symbol={"BTC/USD": df_60},
    )
    st_p1["market_regime"] = "FLAT"
    st_p1["current_price"] = 100250.0
    assert scalper_expert_node(st_p1)["trade_intents"] == []

    conn = duckdb.connect(db_file)
    try:
        row1 = conn.execute(
            "SELECT strategy, phase, sweep_low, target_price FROM setup_state_machine LIMIT 1"
        ).fetchone()
        assert row1 is not None
        assert row1[0] == "LVN_ABSORPTION_SQUEEZE"
        assert row1[1] == "PHASE_1_LVN_IDENTIFIED"
        lvn_low, lvn_high = float(row1[2]), float(row1[3])
    finally:
        conn.close()

    # Phase 2: candle drops into [lvn_low, lvn_high] with 3x volume and narrow body + lower wick
    lvn_mid = (lvn_low + lvn_high) / 2.0
    df_abs = df_60.copy()
    df_abs.iloc[-1] = {
        "open": lvn_mid + 5.0,
        "high": lvn_mid + 15.0,
        "low": lvn_mid - 40.0,
        "close": lvn_mid + 4.0,
        "volume": 350.0,
    }
    st_p2 = _base_state(
        asset_symbols=["BTC/USD"],
        market_data_by_symbol={"BTC/USD": df_abs},
    )
    st_p2["market_regime"] = "FLAT"
    st_p2["current_price"] = lvn_mid + 4.0
    assert scalper_expert_node(st_p2)["trade_intents"] == []

    conn = duckdb.connect(db_file)
    try:
        phase2 = conn.execute("SELECT phase FROM setup_state_machine LIMIT 1").fetchone()[0]
        assert phase2 == "PHASE_2_ABSORPTION"
    finally:
        conn.close()

    # Phase 3: price breaks above absorption candle high (lvn_mid + 15.0) -> MARKET_BUY
    st_p3 = _base_state(
        asset_symbols=["BTC/USD"],
        market_data_by_symbol={"BTC/USD": df_abs},
    )
    st_p3["market_regime"] = "FLAT"
    st_p3["current_price"] = lvn_mid + 25.0
    res_p3 = scalper_expert_node(st_p3)

    assert len(res_p3["trade_intents"]) == 1
    intent = res_p3["trade_intents"][0]
    assert intent.action == "MARKET_BUY"
    assert intent.raw_stop_loss == pytest.approx(lvn_mid - 40.0)

    risk_out = risk_manager_node(res_p3)
    assert risk_out["risk_cleared"] is True
    assert risk_out["order_details"]["order_type"] == "market"
    assert "extended_hours" not in risk_out["order_details"]


def test_scalper_bypasses_all_entries_when_suspended() -> None:
    """Verify SUSPENDED regime bypasses all entry logic and emits 0 intents."""
    btc_df = _make_df(
        [90180.0, 90200.0, 90150.0, 90120.0, 90100.0],
        [90050.0, 90080.0, 90020.0, 90000.0, 90010.0],
        [90100.0, 90140.0, 90090.0, 90030.0, 90020.0],
    )
    state = _base_state(
        asset_symbols=["BTC/USD"],
        market_data_by_symbol={"BTC/USD": btc_df},
    )
    state["market_regime"] = "SUSPENDED"

    updated = scalper_expert_node(state)
    assert updated["trade_intents"] == []
    assert updated["action"] == "CIRCUIT_BREAKER"


def test_scalper_multi_symbol_crt_isolation_for_eth_and_sol(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify ETH/USD and SOL/USD maintain isolated setup_state_machine rows and trigger independently."""
    db_file = str(tmp_path / "test_multi_symbol.duckdb")
    monkeypatch.setattr(scalper_mod, "DB_PATH", db_file)

    # ETH/USD 1H sweep: H1=3100, L1=3000; C2 sweeps L1 (L2=2960, H2=3040, close=3020) -> M=3000
    eth_1h = pd.DataFrame(
        {
            "open": [3050.0, 3010.0],
            "high": [3100.0, 3040.0],
            "low": [3000.0, 2960.0],
            "close": [3040.0, 3020.0],
        }
    )
    # BTC/USD neutral (no sweep)
    btc_1h = pd.DataFrame(
        {
            "open": [90100.0, 90110.0],
            "high": [90200.0, 90180.0],
            "low": [90000.0, 90050.0],
            "close": [90120.0, 90120.0],
        }
    )
    btc_1m = _make_df([90200.0] * 5, [90000.0] * 5, [90120.0] * 5)

    def _build_multi_state(eth_price: float) -> AgentState:
        eth_1m = _make_df([eth_price + 5.0] * 5, [eth_price - 5.0] * 5, [eth_price] * 5)
        st = _base_state(
            asset_symbols=["BTC/USD", "ETH/USD", "SOL/USD"],
            market_data_by_symbol={
                "BTC/USD": btc_1m,
                "BTC/USD_1H": btc_1h,
                "ETH/USD": eth_1m,
                "ETH/USD_1M": eth_1m,
                "ETH/USD_1H": eth_1h,
            },
        )
        st["market_regime"] = "FLAT"
        st["current_price"] = 90120.0
        return st

    # Cycle 1: ETH/USD locks PHASE_1_SWEPT (M=3000, Target=3100) while BTC/SOL stay HOLD
    res1 = scalper_expert_node(_build_multi_state(2990.0))
    assert res1["trade_intents"] == []
    assert "BTC: HOLD | ETH: PHASE_1_SWEPT | SOL: HOLD" in res1["rationale"]

    # Cycle 2: ETH crosses above M=3000 (3010) -> PHASE_2_FIRST_BREAK
    res2 = scalper_expert_node(_build_multi_state(3010.0))
    assert res2["trade_intents"] == []
    assert "ETH: PHASE_2_FIRST_BREAK" in res2["rationale"]

    # Cycle 3: ETH pulls back below M=3000 (2995) -> PHASE_3_PULLBACK
    res3 = scalper_expert_node(_build_multi_state(2995.0))
    assert res3["trade_intents"] == []
    assert "ETH: PHASE_3_PULLBACK" in res3["rationale"]

    # Cycle 4: ETH re-breaks M=3000 (3005) -> emits LIMIT_BUY for ETH/USD at M=3000
    res4 = scalper_expert_node(_build_multi_state(3005.0))
    assert len(res4["trade_intents"]) == 1
    intent = res4["trade_intents"][0]
    assert intent.symbol == "ETH/USD"
    assert intent.action == "LIMIT_BUY"
    assert intent.target_price == pytest.approx(3000.0)
    assert intent.raw_stop_loss == pytest.approx(3000.0 - (100.0 / 1.5))
