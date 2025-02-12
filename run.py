from agent import ToolCallingAgentFactory
from tools.tool import tool
from rich.console import Console
from rich.panel import Panel


@tool
def calculate_triangle_area(a: int, b: int, c: int) -> float:
    """
    Calculate the area of a triangle using Heron's formula.

    :param a: Length of triangle side.
    :param b: Length of triangle side.
    :param c: Length of triangle side.
    :return: The area of the triangle.
    """
    s = (a + b + c) / 2
    area = s * (s - a) * (s - b) * (s - c)
    return (area + 1e-5) ** (1 / 2)


# Create an inference engine with the available tool(s).
agent = ToolCallingAgentFactory.create(
    tools=[calculate_triangle_area], backend="ollama"
)

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
