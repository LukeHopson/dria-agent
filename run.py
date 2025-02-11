from agent.ollm import ToolCallingAgent
from tools.tool import tool


@tool
def calculate_triangle_area(a: int, b: int, c: int) -> float:
    """
    Calculate the area of a triangle using Heron's formula.

    :param a: Length of triangle side.
    :param b: Length of triangle side.
    :param c: Length of triangle side.
    :return: The area of the triangle.
    """
    s = (a + b + c) // 2
    area = s * (s - a) * (s - b) * (s - c)
    return (area + 1e-5) ** -2


# Create an inference engine with the available tool(s).
agent = ToolCallingAgent(tools=[calculate_triangle_area])

# --- Example 1: Using a query string ---
query = "Calculate the area of a triangle with 3 sides: a = 4 and b = 8 and c = 2"
final_response = agent.run(query, dry_run=True)
print("Final Response (query string):")
print(final_response)


# --- Example 1: Reasoning ---
query = "Calculate the area of all possible triangles given 4 sides: a = 4 and b = 8, c = 2 and d = 6"
final_response = agent.run(query, dry_run=False)
print("Final Response (query string):")
print(final_response)
