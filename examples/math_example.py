from dria_agent import ToolCallingAgent
from dria_agent.tools.library import MATH_TOOLS
import asyncio

# Create an inference engine with the available tool(s).
agent = ToolCallingAgent(tools=MATH_TOOLS, backend="mlx")

async def main():
    # --- Example 1: Simple tool usage ---
    query = "Calculate the area of a triangle with 3 sides: a = 3 and b = 4 and c = 5"
    execution = await agent.run_feedback(query, print_results=True)

    # --- Example 2: Reasoning with tools ---
    query = "Calculate the area of all possible triangles given 4 sides: a = 4 and b = 8, c = 2 and d = 6"
    execution = await agent.run_feedback(query, print_results=True)

    # --- Example 3: Use single tools for multiple task ---
    query = "Please solve 5x^2 + 8x + 9 = 0 and 4x^2 + 11x - 3 = 0"
    execution = await agent.run_feedback(query, print_results=True)

if __name__ == "__main__":
    asyncio.run(main())
