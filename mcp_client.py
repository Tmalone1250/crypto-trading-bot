"""Persistent Alpaca MCP stdio client manager for long-running swarm loops.

Boots ``alpaca-mcp-server`` once on a dedicated background asyncio thread before
the main scanning loop starts, allowing synchronous LangGraph nodes to reuse
the initialized ``ClientSession`` across cycles without respawning FastMCP.
"""

import asyncio
import os
import shlex
import threading
from typing import Any

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()


def build_alpaca_server_params(paper_trade_default: str | None = None) -> StdioServerParameters:
    """Construct ``StdioServerParameters`` for ``alpaca-mcp-server``."""
    command = os.environ.get("ALPACA_MCP_COMMAND", "uvx")
    args = shlex.split(os.environ.get("ALPACA_MCP_ARGS", "alpaca-mcp-server"))

    env = dict(os.environ)
    for key_name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
        val = os.environ.get(key_name)
        if val is not None:
            env[key_name] = val
    if paper_trade_default is not None:
        env.setdefault("ALPACA_PAPER_TRADE", paper_trade_default)

    return StdioServerParameters(
        command=command,
        args=args,
        env=env,
    )


class PersistentAlpacaMCPClient:
    """Thread-safe context manager holding a single initialized Alpaca ``ClientSession``."""

    def __init__(self, server_params: StdioServerParameters | None = None) -> None:
        self._server_params = server_params or build_alpaca_server_params()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: ClientSession | None = None
        self._ready_event = threading.Event()
        self._stop_future: asyncio.Future[None] | None = None
        self._startup_error: BaseException | None = None

    def start(self, timeout: float = 30.0) -> "PersistentAlpacaMCPClient":
        """Boot the MCP server once on a background event-loop thread."""
        if self._thread is not None and self._thread.is_alive():
            return self

        self._ready_event.clear()
        self._startup_error = None
        self._thread = threading.Thread(
            target=self._run_loop,
            name="alpaca-mcp-client",
            daemon=True,
        )
        self._thread.start()

        if not self._ready_event.wait(timeout=timeout):
            raise TimeoutError("Timed out waiting for Alpaca MCP server initialization.")
        if self._startup_error is not None:
            raise RuntimeError(
                f"Failed to initialize persistent Alpaca MCP client: {self._startup_error}"
            ) from self._startup_error
        return self

    def _run_loop(self) -> None:
        """Run the persistent ``stdio_client`` and ``ClientSession`` lifecycle."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        async def _lifecycle() -> None:
            self._stop_future = loop.create_future()
            try:
                async with stdio_client(self._server_params) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        self._session = session
                        self._ready_event.set()
                        await self._stop_future
            except BaseException as exc:
                self._startup_error = exc
                self._ready_event.set()
            finally:
                self._session = None

        try:
            loop.run_until_complete(_lifecycle())
        finally:
            loop.close()

    def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> Any:
        """Invoke an MCP tool on the persistent ``ClientSession``."""
        if self._loop is None or self._session is None:
            raise RuntimeError("PersistentAlpacaMCPClient is not initialized.")
        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(tool_name, arguments=arguments or {}),
            self._loop,
        )
        return future.result(timeout=timeout)

    def close(self, timeout: float = 10.0) -> None:
        """Gracefully close the persistent MCP session and background thread."""
        if self._loop is not None and self._stop_future is not None and not self._stop_future.done():
            self._loop.call_soon_threadsafe(self._stop_future.set_result, None)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self._loop = None
        self._session = None

    def __enter__(self) -> "PersistentAlpacaMCPClient":
        return self.start()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
