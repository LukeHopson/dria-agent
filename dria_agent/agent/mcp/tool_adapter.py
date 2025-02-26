from functools import partial
from typing import Any, List, Dict, Optional

from dria_agent.agent.tool import tool, ToolCall
from .client import MCPClient
from .config import MCPConfigManager


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

    async def async_execute(*args, **kwargs) -> Any:
        """Synchronous wrapper for the async executor"""
        if args and isinstance(args[0], dict):
            kwargs = args[0]
        elif args:
            required_params = tool_info.get("input_schema", {}).get("required", [])
            if not required_params:
                raise ValueError(f"Given parameters undefined for tool {tool_name}")
            args = args[: len(required_params)]
            kwargs = {param: arg for param, arg in zip(required_params, args)}
        execution_result = await _execute(**kwargs)
        return execution_result.content[0].text

    # Create a function with the tool's name and docstring
    tool_func = partial(async_execute)
    tool_func.__name__ = tool_name
    tool_func.__doc__ = tool_info.get("description", "")
    tool_func.input_schema = tool_info.get("input_schema", None)

    # Wrap it with @tool decorator
    return tool(tool_func)


class MCPToolAdapter:
    """Adapter that converts MCP tools into Dria Agent compatible tools"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize MCP tool adapter

        Args:
            config_path: Optional path to MCP config file
        """
        self.config_manager = MCPConfigManager(config_path)
        self.clients: Dict[str, MCPClient] = {}
        self._tools: List[ToolCall] = []

    async def connect_server(self, server_name: str) -> None:
        """Connect to an MCP server and load its tools

        Args:
            server_name: Name of the MCP server to connect to
        """
        client = MCPClient(server_name, self.config_manager)
        await client.connect()
        self.clients[server_name] = client

        # Convert MCP tools to Dria Agent tools
        for tool_info in client.available_tools:
            tool_call = create_mcp_tool_executor(client, tool_info["name"], tool_info)
            self._tools.append(tool_call)

    async def connect_servers(self) -> None:
        """Connect to multiple MCP servers and load their tools"""
        server_names = list(self.config_manager.servers.keys())
        for server_name in server_names:
            await self.connect_server(server_name)

    async def close_servers(self) -> None:
        """Close all connected MCP servers"""
        try:
            for client in self.clients.values():
                try:
                    await client.cleanup()
                except GeneratorExit:
                    # Handle GeneratorExit during individual client cleanup
                    continue
                except Exception as e:
                    # Log but continue with other clients
                    print(f"Error during cleanup of server '{client.server_name}': {e}")
                    continue
        except GeneratorExit:
            # Handle GeneratorExit for the entire close operation
            # Don't re-raise GeneratorExit as it's a control flow exception
            return

    @property
    def tools(self) -> List[ToolCall]:
        """Get all tools from connected MCP servers"""
        return self._tools
