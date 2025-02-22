import asyncio
from dria_agent import ToolCallingAgent
from dria_agent.agent.mcp.tool_adapter import MCPToolAdapter


async def setup_mcp_tools(config_path: str = None) -> list:
    """Set up MCP tools and convert them to Dria Agent format

    Args:
        config_path: Optional path to MCP config file

    Returns:
        List of tools from MCP servers
    """
    # Initialize MCP tool adapter
    adapter = MCPToolAdapter(config_path)

    # Connect to MCP servers defined in config
    await adapter.connect_servers(["fetch"])

    # Get tools in Dria Agent format
    return adapter.tools


async def main():

    mcp_tools = await setup_mcp_tools("mcp/mcp.json")

    # Create Dria Agent with combined tools
    agent = ToolCallingAgent(tools=mcp_tools, mode="fast", backend="ollama")

    print("\n--- Example: Using MCP fetch tool ---")
    query = "fetch google.com"
    agent.run(query, print_results=True)


if __name__ == "__main__":
    asyncio.run(main())
