from dria_agent import ToolCallingAgent
from dria_agent.tools.library import GITHUB_TOOLS

# Create an inference engine with the available tool(s).
agent = ToolCallingAgent(tools=GITHUB_TOOLS, backend="mlx")

# --- Example 1: Simple tool usage ---
query = "List the repositories of the user 'firstbatchxyz'"
agent.run_feedback(query, print_results=True)

# --- Example 2: Simple tool usage ---
query = "List issues with the repository 'firstbatchxyz/dkn-compute-node'"
agent.run_feedback(query, print_results=True)
