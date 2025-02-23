from dria_agent import ToolCallingAgent
from dria_agent.tools.library import DOCKER_TOOLS
import asyncio
# Create an inference engine with the available tool(s).
agent = ToolCallingAgent(tools=DOCKER_TOOLS, backend="ollama")

async def main():
    # --- Example 1: Simple tool usage ---
    query = (
        "Stop all the running docker and show there are no running containers afterwards."
    )
    execution = await agent.run_feedback(query, print_results=True)

if __name__ == "__main__":
    asyncio.run(main())
