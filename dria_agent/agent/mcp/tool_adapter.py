import asyncio
from typing import Any, List, Dict, Optional
from functools import partial

from .client import MCPClient
from dria_agent.agent.tool import tool, ToolCall
import nest_asyncio

nest_asyncio.apply()


def create_mcp_tool_executor(
    client: MCPClient, tool_name: str, tool_info: Dict[str, Any]
) -> ToolCall:
    """Create a ToolCall instance that wraps an MCP tool

    Args:
        client: MCPClient instance
        tool_name: Name of the MCP tool
        tool_info: Tool information from MCP server

    Returns:
        ToolCall instance that wraps the MCP tool
    """

    async def _execute(**kwargs) -> Any:
        """Async executor for the MCP tool"""
        return await client.execute_tool(tool_name, **kwargs)

    def sync_execute(*args, **kwargs) -> Any:
        """Synchronous wrapper for the async executor"""
        if args and isinstance(args[0], dict):
            kwargs = args[0]
        elif args:
            required_params = tool_info.get("input_schema", {}).get("required", [])
            if not required_params:
                raise ValueError(f"Given parameters undefined for tool {tool_name}")
            args = args[:len(required_params)]
            kwargs = {param: arg for param, arg in zip(required_params, args)}
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_execute(**kwargs))

    # Create a function with the tool's name and docstring
    tool_func = partial(sync_execute)
    tool_func.__name__ = tool_name
    tool_func.__doc__ = tool_info.get("description", "")

    # Wrap it with @tool decorator
    return tool(tool_func)


class MCPToolAdapter:
    """Adapter that converts MCP tools into Dria Agent compatible tools"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize MCP tool adapter

        Args:
            config_path: Optional path to MCP config file
        """
        self.config_path = config_path
        self.clients: Dict[str, MCPClient] = {}
        self._tools: List[ToolCall] = []

    async def connect_server(self, server_name: str) -> None:
        """Connect to an MCP server and load its tools

        Args:
            server_name: Name of the MCP server to connect to
        """
        client = MCPClient(server_name, self.config_path)
        await client.connect()
        self.clients[server_name] = client

        # Convert MCP tools to Dria Agent tools
        for tool_info in client.available_tools:
            tool_call = create_mcp_tool_executor(client, tool_info["name"], tool_info)
            self._tools.append(tool_call)

    async def connect_servers(self, server_names: List[str]) -> None:
        """Connect to multiple MCP servers and load their tools

        Args:
            server_names: List of MCP server names to connect to
        """
        for server_name in server_names:
            await self.connect_server(server_name)

    @property
    def tools(self) -> List[ToolCall]:
        """Get all tools from connected MCP servers"""
        return self._tools

    async def close(self) -> None:
        """Close all MCP client connections"""
        for client in self.clients.values():
            await client.close()
        self.clients.clear()
        self._tools.clear()
