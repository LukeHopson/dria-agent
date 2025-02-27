import asyncio

from dria_agent import ToolCallingAgent


async def main():
    # Create Dria Agent with MCP
    agent = ToolCallingAgent(mcp_file="mcp/obsidian/mcp.json", backend="ollama")

    try:
        await agent.initialize_servers()

        # Search notes for specific term
        search_query = "search my notes term 'synthetic data' on vault 'my vault'"
        result = await agent.async_run_feedback(
            search_query, print_results=False, num_tools=5
        )
        print(f"\n{result.final_answer()}")

        if result.errors:
            print(f"\nErrors encountered: {result.errors}")
    finally:
        await agent.close_servers()


if __name__ == "__main__":
    asyncio.run(main())
