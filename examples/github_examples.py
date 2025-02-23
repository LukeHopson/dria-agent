from dria_agent import ToolCallingAgent
from dria_agent.tools.library import GITHUB_TOOLS
import asyncio
# Create an inference engine with the available tool(s).
agent = ToolCallingAgent(tools=GITHUB_TOOLS, backend="mlx")

async def main():
    # --- Example 1: Simple tool usage ---
    query = "List the repositories of the user 'firstbatchxyz'"
    execution = await agent.run_feedback(query, print_results=True)


    # --- Example 2: Simple tool usage ---
    query = "List issues with the repository 'firstbatchxyz/dkn-compute-node'"
    execution = await agent.run_feedback(query, print_results=True)

if __name__ == "__main__":
    asyncio.run(main())
