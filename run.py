from agent import ToolCallingAgentFactory
from tools.tool import tool
from rich.console import Console
from rich.panel import Panel
from tools.library.math_tools import ALL_TOOLS

# Create an inference engine with the available tool(s).
agent = ToolCallingAgentFactory.create(tools=ALL_TOOLS, backend="mlx")

console = Console()
# --- Example 1: Using a query string ---
query = "Calculate the area of a triangle with 3 sides: a = 3 and b = 4 and c = 5"
execution = agent.run(query)
panel = Panel(
    query, title="Execution Result", expand=True, subtitle=str(execution.final_answer())
)
console.print(panel)

# --- Example 1: Reasoning ---
query = "Calculate the area of all possible triangles given 4 sides: a = 4 and b = 8, c = 2 and d = 6"
execution = agent.run(query)
panel = Panel(
    query,
    title="Execution Result",
    subtitle=str(execution.final_answer()),
    expand=False,
)
console.print(panel)
