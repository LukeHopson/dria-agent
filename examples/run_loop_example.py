from dria_agent.agent import ToolCallingAgent
from dria_agent.tools import APPLE_TOOLS

# Create an inference engine with the available tool(s).
agent = ToolCallingAgent(tools=APPLE_TOOLS, backend="ollama")

query = "Find the file called dota2"
execution = agent.run_loop(query, print_results=True)
