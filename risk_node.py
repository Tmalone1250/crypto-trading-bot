"""Node 3: Risk Manager for the LangGraph execution engine.

Uses ``TA-Lib`` to compute the 14-period Average True Range (ATR), derives a
1.5x ATR structural stop-loss, and enforces the $10.00 micro-fund veto rule
(maximum $0.80 / 8% projected dollar risk per trade).
"""

import os
import numpy as np
import talib

from state import AgentState

ACCOUNT_SIZE_USD: float = 10.00
MAX_RISK_USD: float = 0.80
ATR_PERIOD: int = 14
ATR_STOP_MULTIPLIER: float = 1.5
RISK_REWARD_RATIO: float = 2.0


def _is_dry_run() -> bool:
    """Default to paper-trading/dry-run unless explicitly toggled for production."""
    return os.environ.get("TRADING_MODE", "dry_run").lower() != "production"


def risk_manager_node(state: AgentState) -> AgentState:
    """Evaluate volatility risk using TA-Lib ATR and populate bracket order parameters.

    Computes the 14-period ATR on ``state["historical_data"]``, calculates a
    ``1.5x`` ATR stop-loss for a ``$10.00`` notional position, and vetoes the
    trade if projected dollar risk exceeds ``$0.80`` (8% of the account).

    Args:
        state: Current ``AgentState`` dictionary.

    Returns:
        Updated ``AgentState`` dictionary with ``risk_cleared`` and
        ``order_details`` (or ``error_message`` on veto/failure).
    """
    symbol = state["asset_symbol"]
    df = state["historical_data"]
    current_price = float(state["current_price"])

    if df is None or df.empty:
        return {
            **state,
            "risk_cleared": False,
            "error_message": f"Risk veto for {symbol}: historical_data is empty.",
        }

    required_cols = {"high", "low", "close"}
    if not required_cols.issubset(df.columns):
        missing = sorted(required_cols - set(df.columns))
        return {
            **state,
            "risk_cleared": False,
            "error_message": f"Risk veto for {symbol}: missing columns {missing}.",
        }

    if current_price <= 0.0:
        return {
            **state,
            "risk_cleared": False,
            "error_message": f"Risk veto for {symbol}: current_price must be positive.",
        }

    try:
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        close = df["close"].to_numpy(dtype=np.float64)

        atr_series = talib.ATR(high, low, close, timeperiod=ATR_PERIOD)
        latest_atr = float(atr_series[-1])

        if np.isnan(latest_atr) or latest_atr <= 0.0:
            return {
                **state,
                "risk_cleared": False,
                "error_message": (
                    f"Risk veto for {symbol}: insufficient data for {ATR_PERIOD}-period ATR."
                ),
            }

        stop_distance = ATR_STOP_MULTIPLIER * latest_atr
        position_size = ACCOUNT_SIZE_USD / current_price
        projected_dollar_risk = position_size * stop_distance

        if projected_dollar_risk > MAX_RISK_USD:
            return {
                **state,
                "risk_cleared": False,
                "error_message": (
                    f"Risk veto for {symbol}: projected dollar risk "
                    f"${projected_dollar_risk:.4f} exceeds max ${MAX_RISK_USD:.2f} "
                    f"(1.5x ATR = {stop_distance:.4f})."
                ),
            }

        stop_loss_price = current_price - stop_distance
        if stop_loss_price <= 0.0:
            return {
                **state,
                "risk_cleared": False,
                "error_message": f"Risk veto for {symbol}: computed stop_loss_price <= 0.",
            }

        target_price = current_price + (RISK_REWARD_RATIO * stop_distance)

        return {
            **state,
            "risk_cleared": True,
            "order_details": {
                "symbol": symbol,
                "side": "buy",
                "order_type": "bracket",
                "entry_price": float(current_price),
                "stop_loss_price": float(round(stop_loss_price, 6)),
                "target_price": float(round(target_price, 6)),
                "position_size": float(round(position_size, 8)),
                "atr_14": float(round(latest_atr, 6)),
                "projected_dollar_risk": float(round(projected_dollar_risk, 4)),
                "risk_reward_ratio": RISK_REWARD_RATIO,
                "dry_run": _is_dry_run(),
            },
            "error_message": None,
        }
    except Exception as exc:
        return {
            **state,
            "risk_cleared": False,
            "error_message": f"Risk manager error for {symbol}: {exc}",
        }
