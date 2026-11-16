"""Centralized Alpaca Paper Trading configuration and client factories."""

import os
from typing import Any
from dotenv import load_dotenv

load_dotenv()

ALPACA_PAPER_URL: str = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL: str = "https://data.alpaca.markets"


def get_alpaca_paper_config() -> dict[str, Any]:
    """Resolve and synchronize Alpaca paper credentials and endpoints."""
    api_key = (
        os.environ.get("APCA_API_KEY_ID")
        or os.environ.get("ALPACA_API_KEY")
        or ""
    )
    secret_key = (
        os.environ.get("APCA_API_SECRET_KEY")
        or os.environ.get("ALPACA_SECRET_KEY")
        or ""
    )
    base_url = os.environ.get("APCA_API_BASE_URL", ALPACA_PAPER_URL)

    if api_key:
        os.environ.setdefault("APCA_API_KEY_ID", api_key)
        os.environ.setdefault("ALPACA_API_KEY", api_key)
    if secret_key:
        os.environ.setdefault("APCA_API_SECRET_KEY", secret_key)
        os.environ.setdefault("ALPACA_SECRET_KEY", secret_key)
    os.environ.setdefault("APCA_API_BASE_URL", base_url)
    os.environ.setdefault("ALPACA_PAPER_TRADE", "true")

    return {
        "api_key": api_key,
        "secret_key": secret_key,
        "base_url": base_url,
        "data_url": os.environ.get("APCA_DATA_URL", ALPACA_DATA_URL),
        "paper": True,
    }


def get_alpaca_trading_client() -> Any:
    """Instantiate an Alpaca ``TradingClient`` pinned to Paper Trading."""
    from alpaca.trading.client import TradingClient

    cfg = get_alpaca_paper_config()
    return TradingClient(
        api_key=cfg["api_key"],
        secret_key=cfg["secret_key"],
        paper=True,
        url_override=cfg["base_url"],
    )


def get_alpaca_crypto_data_client() -> Any:
    """Instantiate an Alpaca ``CryptoHistoricalDataClient`` with paper credentials."""
    from alpaca.data.historical import CryptoHistoricalDataClient

    cfg = get_alpaca_paper_config()
    return CryptoHistoricalDataClient(
        api_key=cfg["api_key"] or None,
        secret_key=cfg["secret_key"] or None,
        url_override=cfg["data_url"],
    )
