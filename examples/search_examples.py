from dria_agent import ToolCallingAgent
from dria_agent.tools.library import SEARCH_TOOLS, API_TOOLS, APPLE_TOOLS
import asyncio
# Create an inference engine with the available tool(s).
agent = ToolCallingAgent(tools=SEARCH_TOOLS + API_TOOLS + APPLE_TOOLS, backend="ollama")

async def main():
    # --- Example 1: Parallel calls ---
    query = "Search for the best restaurants in Istanbul in google. Export urls of the results and save to istanbul.json."
    execution = await agent.run_feedback(query, print_results=True, num_tools=4)


    # --- Example 2: Simple Call ---
    query = "What is the weather in Istanbul?"
    execution = await agent.run_feedback(query, print_results=True)

if __name__ == "__main__":
    asyncio.run(main())
