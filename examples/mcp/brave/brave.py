import asyncio

from dria_agent import ToolCallingAgent


async def run_agent_query(agent, query):
    try:
        result = await agent.async_run_feedback(query, print_results=False, num_tools=5)
        return result
    except Exception as e:
        return f"Error: {e}"


async def main():
    # Create Dria Agent with MCP
    agent = ToolCallingAgent(mcp_file="mcp/brave/mcp.json", backend="ollama")

    print("🤖 Brave Assistant Terminal")
    print("Type 'exit' to quit")

    try:
        await agent.initialize_servers()

        while True:
            user_input = input("\n> ")
            if user_input.lower() == 'exit':
                break

            print("\nProcessing your request...")
            result = await run_agent_query(agent, user_input)
            print(f"\n{result.final_answer()}")

            if result.errors:
                print(f"\nErrors encountered: {result.errors}")
    finally:
        await agent.close_servers()


if __name__ == "__main__":
    asyncio.run(main())
