"""Node 0: Self-Awareness Portfolio Node for the LangGraph execution engine.

Connects to ``alpaca-mcp-server`` over MCP stdio transport to query the agent's
live account balances, open positions, and active open orders before market
scanning begins.

Cross-referenced ``alpaca-mcp-server`` v2 (FastMCP 3.4.7) tool schemas:
- Account tool: ``get_account_info`` (alias ``get_account``)
- Open positions tool: ``get_all_positions`` (alias ``get_open_positions``)
- Open orders tool: ``get_orders`` with ``{"status": "open"}`` (alias ``get_open_orders``)

Each tool wraps its response in the FastMCP ``_alpaca_mcp_security`` JSON envelope:
- ``get_account_info`` -> ``{"_alpaca_mcp_security": {...}, "data": {...}}``
- ``get_all_positions`` -> ``{"_alpaca_mcp_security": {...}, "data": {"result": [...]}}``
- ``get_orders`` -> ``{"_alpaca_mcp_security": {...}, "data": {"result": [...]}}``
"""

import asyncio
import json
import os
import shlex
from typing import Any

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from state import AgentState

load_dotenv()

ACCOUNT_TOOL_NAME: str = "get_account_info"
POSITIONS_TOOL_NAME: str = "get_all_positions"
ORDERS_TOOL_NAME: str = "get_orders"


def _build_server_params() -> StdioServerParameters:
    """Construct ``StdioServerParameters`` for ``alpaca-mcp-server``."""
    command = os.environ.get("ALPACA_MCP_COMMAND", "uvx")
    args = shlex.split(os.environ.get("ALPACA_MCP_ARGS", "alpaca-mcp-server"))

    env = dict(os.environ)
    for key_name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
        val = os.environ.get(key_name)
        if val is not None:
            env[key_name] = val

    return StdioServerParameters(
        command=command,
        args=args,
        env=env,
    )


def _extract_mcp_result(result: Any) -> Any:
    """Extract structured or JSON/text payload from an MCP ``CallToolResult``."""
    is_error = bool(
        getattr(result, "is_error", False) or getattr(result, "isError", False)
    )
    content = getattr(result, "content", None) or []

    if is_error:
        error_parts = [
            str(getattr(item, "text", item))
            for item in content
        ]
        raise RuntimeError(" ".join(error_parts) or "Alpaca MCP tool returned an error.")

    structured = getattr(result, "structured_content", None)
    if structured is None:
        structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured

    text_chunks = [
        text_val
        for item in content
        if isinstance((text_val := getattr(item, "text", None)), str)
    ]
    if len(text_chunks) == 1:
        return text_chunks[0]
    if text_chunks:
        return "\n".join(text_chunks)
    return {}


def _parse_json_payload(raw_payload: Any) -> Any:
    """Parse a string MCP payload as JSON if needed."""
    if not isinstance(raw_payload, str):
        return raw_payload

    stripped = raw_payload.strip()
    if not stripped:
        return {}

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    for start_char, end_char in (("{", "}"), ("[", "]")):
        start_idx = stripped.find(start_char)
        end_idx = stripped.rfind(end_char)
        if start_idx != -1 and end_idx > start_idx:
            try:
                return json.loads(stripped[start_idx : end_idx + 1])
            except json.JSONDecodeError:
                pass

    return {}


def _unwrap_security_envelope(raw_payload: Any) -> Any:
    """Unpack the FastMCP ``_alpaca_mcp_security`` envelope and return ``data``."""
    parsed = _parse_json_payload(raw_payload)
    if isinstance(parsed, dict) and (
        "_alpaca_mcp_security" in parsed or "data" in parsed
    ):
        data_section = parsed.get("data", parsed)
        if isinstance(data_section, str):
            return _parse_json_payload(data_section)
        return data_section
    return parsed


def _unpack_account_payload(raw_payload: Any) -> dict[str, Any]:
    """Unpack the ``get_account_info`` FastMCP security envelope into a ``dict``."""
    data = _unwrap_security_envelope(raw_payload)
    if isinstance(data, dict):
        if "account" in data and isinstance(data["account"], dict):
            return dict(data["account"])
        return dict(data)
    return {}


def _unpack_list_payload(raw_payload: Any, fallback_key: str) -> list[dict[str, Any]]:
    """Unpack ``get_all_positions`` or ``get_orders`` FastMCP security envelope into a ``list``."""
    data = _unwrap_security_envelope(raw_payload)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if isinstance(data, dict):
        for key in ("result", fallback_key, "items", "positions", "orders"):
            candidate = data.get(key)
            if isinstance(candidate, list):
                return [item for item in candidate if isinstance(item, dict)]

    return []


async def _call_single_mcp_tool(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> Any:
    """Open an MCP stdio session and invoke a single tool on ``alpaca-mcp-server``."""
    server_params = _build_server_params()
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(
                tool_name,
                arguments=arguments or {},
            )
            return _extract_mcp_result(result)


async def _fetch_portfolio_state_mcp() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch account info, open positions, and active open orders in a single MCP session."""
    server_params = _build_server_params()
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            account_res = await session.call_tool(
                ACCOUNT_TOOL_NAME,
                arguments={},
            )
            positions_res = await session.call_tool(
                POSITIONS_TOOL_NAME,
                arguments={},
            )
            orders_res = await session.call_tool(
                ORDERS_TOOL_NAME,
                arguments={"status": "open"},
            )

            account_data = _unpack_account_payload(_extract_mcp_result(account_res))
            open_positions = _unpack_list_payload(
                _extract_mcp_result(positions_res),
                fallback_key="positions",
            )
            active_orders = _unpack_list_payload(
                _extract_mcp_result(orders_res),
                fallback_key="orders",
            )
            return account_data, open_positions, active_orders


def get_account_info() -> dict[str, Any]:
    """Invoke ``get_account_info`` on ``alpaca-mcp-server`` and unpack its envelope."""
    raw = asyncio.run(_call_single_mcp_tool(ACCOUNT_TOOL_NAME, {}))
    return _unpack_account_payload(raw)


def get_account() -> dict[str, Any]:
    """Alias for ``get_account_info``."""
    return get_account_info()


def get_all_positions() -> list[dict[str, Any]]:
    """Invoke ``get_all_positions`` on ``alpaca-mcp-server`` and unpack its envelope."""
    raw = asyncio.run(_call_single_mcp_tool(POSITIONS_TOOL_NAME, {}))
    return _unpack_list_payload(raw, fallback_key="positions")


def get_open_positions() -> list[dict[str, Any]]:
    """Alias for ``get_all_positions``."""
    return get_all_positions()


def get_orders(status: str = "open") -> list[dict[str, Any]]:
    """Invoke ``get_orders`` on ``alpaca-mcp-server`` and unpack its envelope."""
    raw = asyncio.run(_call_single_mcp_tool(ORDERS_TOOL_NAME, {"status": status}))
    return _unpack_list_payload(raw, fallback_key="orders")


def get_open_orders() -> list[dict[str, Any]]:
    """Alias for ``get_orders(status='open')``."""
    return get_orders(status="open")


def portfolio_node(state: AgentState) -> AgentState:
    """Populate ``account_data``, ``open_positions``, and ``active_orders`` via Alpaca MCP."""
    try:
        account_data, open_positions, active_orders = asyncio.run(
            _fetch_portfolio_state_mcp()
        )
    except Exception as exc:
        return {
            **state,
            "account_data": state.get("account_data", {}) or {},
            "open_positions": state.get("open_positions", []) or [],
            "active_orders": state.get("active_orders", []) or [],
            "error_message": f"Portfolio awareness error: {exc}",
        }

    print(
        f"[Portfolio] Value: ${account_data.get('portfolio_value', 'N/A')} "
        f"| Open Positions: {len(open_positions)}"
    )

    return {
        **state,
        "account_data": account_data,
        "open_positions": open_positions,
        "active_orders": active_orders,
        "error_message": None,
    }
