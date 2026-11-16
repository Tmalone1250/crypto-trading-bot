"""Node 3: Risk Manager for the LangGraph execution engine.

Uses ``TA-Lib`` to compute the 14-period Average True Range (ATR), derives a
1.5x ATR structural stop-loss, sizes orders on a $25.00 base notional to clear
Alpaca's ``0.0001`` BTC ``min_order_size``, and enforces the 8% ($2.00) max risk rule.
"""

import os
import numpy as np
import talib

from state import AgentState

ACCOUNT_SIZE_USD: float = 25.00
MAX_RISK_USD: float = 2.00  # 8% of $25.00 base trade size
MIN_CRYPTO_ORDER_SIZE: float = 0.0001
QTY_PRECISION: int = 6
ATR_PERIOD: int = 14
ATR_STOP_MULTIPLIER: float = 1.5
RISK_REWARD_RATIO: float = 2.0
DEFAULT_ORDER_TYPE: str = "limit"
DEFAULT_TIME_IN_FORCE: str = "gtc"


def _is_dry_run() -> bool:
    """Default to paper-trading/dry-run unless explicitly toggled for production."""
    return os.environ.get("TRADING_MODE", "dry_run").lower() != "production"


def risk_manager_node(state: AgentState) -> AgentState:
    """Evaluate volatility risk using TA-Lib ATR and populate crypto order parameters.

    Computes the 14-period ATR on ``state["historical_data"]``, calculates a
    ``1.5x`` ATR stop-loss for a ``$25.00`` notional position (clearing Alpaca's
    ``0.0001`` BTC minimum order size up to $250,000/BTC), and vetoes the trade
    if projected dollar risk exceeds ``$2.00`` (8% of the base size).

    Args:
        state: Current ``AgentState`` dictionary.

    Returns:
        Updated ``AgentState`` dictionary with ``risk_cleared`` and
        ``order_details`` (or ``error_message`` on veto/failure).
    """
    symbol = state["asset_symbol"]
    df = state["historical_data"]
    current_price = float(state["current_price"])
    action = str(state.get("action", "LIMIT_BUY")).upper()

    if state.get("trade_intents"):
        return {
            **state,
            "risk_cleared": True,
            "error_message": None,
        }

    if action == "CANCEL_ALL":
        return {
            **state,
            "risk_cleared": True,
            "order_details": {
                "symbol": symbol,
                "action": "CANCEL_ALL",
                "dry_run": _is_dry_run(),
            },
            "error_message": None,
        }

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
        position_size = float(round(ACCOUNT_SIZE_USD / current_price, QTY_PRECISION))

        if position_size < MIN_CRYPTO_ORDER_SIZE:
            return {
                **state,
                "risk_cleared": False,
                "error_message": (
                    f"Risk veto for {symbol}: rounded position_size {position_size} "
                    f"is below Alpaca min_order_size {MIN_CRYPTO_ORDER_SIZE}."
                ),
            }

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
        side = "sell" if action in {"MARKET_SELL", "LIMIT_SELL"} else "buy"
        order_type = "market" if action in {"MARKET_BUY", "MARKET_SELL"} else DEFAULT_ORDER_TYPE
        limit_entry_price = (
            float(state["target_price"])
            if action in {"LIMIT_BUY", "LIMIT_SELL"} and state.get("target_price") is not None
            else float(round(current_price, 2))
        )

        return {
            **state,
            "risk_cleared": True,
            "order_details": {
                "symbol": symbol,
                "side": side,
                "order_type": order_type,
                "time_in_force": DEFAULT_TIME_IN_FORCE,
                "limit_price": limit_entry_price,
                "entry_price": float(current_price),
                "stop_loss_price": float(round(stop_loss_price, 6)),
                "target_price": float(round(target_price, 6)),
                "position_size": position_size,
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
