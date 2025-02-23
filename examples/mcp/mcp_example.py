import asyncio
from dria_agent import ToolCallingAgent


async def main():

    # Create Dria Agent with MCP
    agent = ToolCallingAgent(mcp_file="mcp/mcp.json", backend="ollama")
    await agent.initialize_servers()

    print("\n--- Example: Using MCP fetch tool ---")
    query = "fetch https://www.google.com"
    await agent.run(query, print_results=True)


if __name__ == "__main__":
    asyncio.run(main())
