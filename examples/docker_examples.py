from dria_agent import ToolCallingAgent
from dria_agent.tools.library import DOCKER_TOOLS

# Create an inference engine with the available tool(s).
agent = ToolCallingAgent(tools=DOCKER_TOOLS, backend="ollama")

# --- Example 1: Simple tool usage ---
query = (
    "Stop all the running docker and show there are no running containers afterwards."
)
agent.run_feedback(query, print_results=True)
