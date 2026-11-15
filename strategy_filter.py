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

    NOTE: Temporary TEST MODE active — immediately returns ``True`` to bypass
    structural checks and force a test trade through the execution pipeline.
    """
    _ = df
    return True
