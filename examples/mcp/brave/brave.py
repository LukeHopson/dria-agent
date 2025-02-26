import asyncio

from dria_agent import ToolCallingAgent


async def main():
    # Create Dria Agent with MCP
    agent = ToolCallingAgent(mcp_file="mcp/brave/mcp.json", backend="ollama")

    try:
        await agent.initialize_servers()
        
        # Run a specific search query
        result = await agent.async_run_feedback("synthetic data", print_results=False, num_tools=5)
        print(f"\n{result.final_answer()}")

        if result.errors:
            print(f"\nErrors encountered: {result.errors}")
    finally:
        await agent.close_servers()


if __name__ == "__main__":
    asyncio.run(main())
