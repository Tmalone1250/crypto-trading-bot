"""Phase 4 Handoff Module: Structural Entry Filter mined in Epic 4 (Iteration 2).

Integration Instructions for ``observer_node.py`` (Rate-Limit Preservation):
---------------------------------------------------------------------------
To prevent Node 2 (``alpha_node.py``) from burning the 15 RPM ``gemini-3.5-flash-lite``
free-tier quota during choppy or bearish market regimes, wire ``check_entry_signal``
directly into ``observer_node.py`` (and the post-observer router in ``agent_graph.py``):

1. Inside ``market_observer_node(state: AgentState)`` in ``observer_node.py``:
   - Import ``check_entry_signal`` from ``strategy_filter``.
   - After fetching the latest OHLCV DataFrame ``df``, evaluate
     ``signal_triggered = check_entry_signal(df)``.
   - If ``not signal_triggered``, record an informational message in
     ``error_message`` (e.g., ``"No structural entry setup on latest closed candle."``)
     or a dedicated state flag so the graph routes immediately to ``END``
     without invoking ``alpha_logic_gate_node``.
2. Only when ``check_entry_signal(df)`` returns ``True`` should the workflow
   transition from ``observer`` -> ``alpha`` to request LLM sentiment confirmation.
"""

import numpy as np
import pandas as pd
import talib

EMA_PERIOD: int = 200
MACD_FAST_PERIOD: int = 48
MACD_SLOW_PERIOD: int = 104
MACD_SIGNAL_PERIOD: int = 36
STOCHRSI_PERIOD: int = 14
STOCHRSI_FASTK_PERIOD: int = 5
STOCHRSI_FASTD_PERIOD: int = 3
STOCHRSI_FASTD_MATYPE: int = 0
ROC_PERIOD: int = 72


def check_entry_signal(df: pd.DataFrame) -> bool:
    """Evaluate the Iteration 2 TA-Lib swing-trading formula on the latest closed candle.

    Computes the exact indicator stack discovered in Epic 4 (Sharpe 1.6451):
      - 200-period EMA macro trend filter (``close > ema_200``)
      - Multi-day MACD 48/104/36 momentum (``macd_2 > macd_signal_2`` and ``macd_2 > 0.0``)
      - 72-period Rate of Change drift confirmation (``roc_72 > 0.0``)
      - Stochastic RSI 14/5/3 bullish crossover (``stoch_k > stoch_d``)

    Args:
        df: OHLCV Pandas DataFrame containing at least a ``"close"`` column
            ordered chronologically with the most recent closed candle at ``df.iloc[-1]``.

    Returns:
        ``True`` if all five Iteration 2 entry conditions are satisfied on
        ``df.iloc[-1]``, otherwise ``False``.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False

    if "close" not in df.columns:
        return False

    work_df = df.copy()
    close_arr = work_df["close"].to_numpy(dtype=np.float64)

    # 1. Macro Trend Filter: 200-period Exponential Moving Average
    work_df["ema_200"] = talib.EMA(close_arr, timeperiod=EMA_PERIOD)

    # 2. Multi-Horizon MACD (48 / 104 / 36)
    macd_2, macd_signal_2, _ = talib.MACD(
        close_arr,
        fastperiod=MACD_FAST_PERIOD,
        slowperiod=MACD_SLOW_PERIOD,
        signalperiod=MACD_SIGNAL_PERIOD,
    )
    work_df["macd_2"] = macd_2
    work_df["macd_signal_2"] = macd_signal_2

    # 3. Stochastic RSI (14 / 5 / 3)
    stoch_k, stoch_d = talib.STOCHRSI(
        close_arr,
        timeperiod=STOCHRSI_PERIOD,
        fastk_period=STOCHRSI_FASTK_PERIOD,
        fastd_period=STOCHRSI_FASTD_PERIOD,
        fastd_matype=STOCHRSI_FASTD_MATYPE,
    )
    work_df["stoch_k"] = stoch_k
    work_df["stoch_d"] = stoch_d

    # 4. 72-Period Rate of Change
    work_df["roc_72"] = talib.ROC(close_arr, timeperiod=ROC_PERIOD)

    # Evaluate strictly on the most recent fully closed candle
    latest = work_df.iloc[-1]
    required_fields = (
        latest["close"],
        latest["ema_200"],
        latest["macd_2"],
        latest["macd_signal_2"],
        latest["roc_72"],
        latest["stoch_k"],
        latest["stoch_d"],
    )
    if any(pd.isna(val) for val in required_fields):
        return False

    is_entry = (
        (latest["close"] > latest["ema_200"])
        and (latest["macd_2"] > latest["macd_signal_2"])
        and (latest["macd_2"] > 0.0)
        and (latest["roc_72"] > 0.0)
        and (latest["stoch_k"] > latest["stoch_d"])
    )

    return bool(is_entry)
