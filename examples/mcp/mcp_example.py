import asyncio
from dria_agent import ToolCallingAgent


async def main():

    # Create Dria Agent with MCP
    agent = ToolCallingAgent(mcp_file="mcp/mcp.json", backend="ollama")

    print("\n--- Example: Using MCP fetch tool ---")
    query = "fetch https://www.google.com"
    try:
        await agent.initialize_servers()
        await agent.async_run(query, print_results=True)
    finally:
        await agent.close_servers()


if __name__ == "__main__":
    asyncio.run(main())
