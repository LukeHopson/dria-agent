from dria_agent.agent import ToolCallingAgent
from dria_agent.tools import SEARCH_TOOLS, API_TOOLS, APPLE_TOOLS

# Create an inference engine with the available tool(s).
agent = ToolCallingAgent(tools=SEARCH_TOOLS + API_TOOLS + APPLE_TOOLS, backend="ollama")

# --- Example 1: Parallel calls ---
query = "Search for the best restaurants in Istanbul in google. Export urls of the results and save to istanbul.json."
execution = agent.run(query, print_results=True, num_tools=4)


# --- Example 2: Simple Call ---
query = "What is the weather in Istanbul?"
execution = agent.run(query, print_results=True)
