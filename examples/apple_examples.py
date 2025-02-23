import asyncio

from dria_agent import ToolCallingAgent
from dria_agent.tools import APPLE_TOOLS

# Create an inference engine with the available tool(s).
agent = ToolCallingAgent(tools=APPLE_TOOLS, backend="ollama")


async def main():
    # --- Example 1: Simple tool usage ---
    query = "Show me how can I get to Barbaros bulvarı"
    execution = await agent.run(query, print_results=True)

    # --- Example 2: Create calendar event ---
    query = "Add new event to my calendar for tomorrow at 3PM to meetup with Batu? His mail is batuhan@firstbatch.xyz. We gonna talk about tiny-agents. My mail is omer@firstbatch.xyz. We said we'll meet around Dün moda"
    execution = await agent.run(query, print_results=True)

    # --- Example 3: Open up a file ---
    query = "Find the file dota2.pdf."
    execution = await agent.run(query, print_results=True)

    # --- Example 4: Find the contact, and send an SMS ---
    contact_name = "John Doe"
    message = "Have you heard about tiny agents?"
    query = f"Find {contact_name} and send an SMS saying {message}"
    execution = await agent.run(query, print_results=True, num_tools=3)


if __name__ == "__main__":
    asyncio.run(main())
