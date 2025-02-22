from contextlib import AsyncExitStack
from typing import Optional, List, Dict, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import MCPConfigManager


class MCPClient:
    def __init__(self, server_name: str, config_path: Optional[str] = None):
        """Initialize MCP client

        Args:
            server_name: Name of the MCP server to connect to
            config_path: Optional path to MCP config file
        """
        self.server_name = server_name
        self.config_manager = MCPConfigManager(config_path)
        self.server_config = self.config_manager.get_server_config(server_name)

        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.stdio = None
        self.write = None
        self._available_tools: List[Dict[str, Any]] = []

    async def connect(self):
        """Connect to the MCP server"""
        server_params = StdioServerParameters(
            command=self.server_config.command,
            args=self.server_config.args,
            env=self.server_config.env,
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()
        await self._update_available_tools()

    async def _update_available_tools(self):
        """Update the list of available tools from the server"""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        response = await self.session.list_tools()
        self._available_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in response.tools
        ]

    @property
    def available_tools(self) -> List[Dict[str, Any]]:
        """Get the list of available tools"""
        return self._available_tools

    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool with the given parameters

        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool parameters

        Returns:
            Tool execution result
        """
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        return await self.session.call_tool(tool_name, kwargs)

    async def close(self):
        """Close the MCP client connection"""
        if self.exit_stack:
            await self.exit_stack.aclose()
