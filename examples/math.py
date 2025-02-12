from dria_agent.agent import ToolCallingAgentFactory
from rich.console import Console
from rich.panel import Panel
from dria_agent.tools.library.math_tools import MATH_TOOLS

# Create an inference engine with the available tool(s).
agent = ToolCallingAgentFactory.create(tools=MATH_TOOLS, backend="ollama")

console = Console()

# --- Example 1: Simple tool usage ---
query = "Calculate the area of a triangle with 3 sides: a = 3 and b = 4 and c = 5"
execution = agent.run(query)
panel = Panel(
    query, title="Execution Result", expand=True, subtitle=str(execution.final_answer())
)
console.print(panel)

# --- Example 2: Reasoning with tools ---
query = "Calculate the area of all possible triangles given 4 sides: a = 4 and b = 8, c = 2 and d = 6"
execution = agent.run(query)
panel = Panel(
    query,
    title="Execution Result",
    subtitle=str(execution.final_answer()),
    expand=False,
)
console.print(panel)

# --- Example 3: Use single tools for multiple task ---
query = "Please solve 5x^2 + 8x + 9 = 0 and 4x^2 + 11x - 3 = 0"
execution = agent.run(query)
panel = Panel(
    query,
    title="Execution Result",
    subtitle=str(execution.final_answer()),
    expand=False,
)
console.print(panel)
