from dria_agent.agent import ToolCallingAgent
from dria_agent.tools import APPLE_TOOLS

# Create an inference engine with the available tool(s).
agent = ToolCallingAgent(tools=APPLE_TOOLS, backend="ollama")


# --- Example 1: Simple tool usage ---
query = "Show me how can I get to barbaros bulvarı by motorbike"
execution = agent.run(query, print_results=True)
