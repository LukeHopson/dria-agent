import argparse
import asyncio
import sys

from rich.console import Console

from dria_agent import ToolCallingAgent
from dria_agent.tools import (
    APPLE_TOOLS,
    API_TOOLS,
    MATH_TOOLS,
    SLACK_TOOLS,
    DOCKER_TOOLS,
    SEARCH_TOOLS,
)

console = Console()


async def async_query_mode(agent, query):
    try:
        await agent.initialize_servers()
        result = await agent.async_run_feedback(query, print_results=True)
        return result
    finally:
        await agent.close_servers()


async def async_chat_mode(agent):
    console.print(
        "Chat mode. Type 'exit' to quit. Type 'clear' to clear the screen.",
        style="bold green",
    )
    try:
        await agent.initialize_servers()
        await agent.async_run_chat()
    finally:
        await agent.close_servers()


def chat_mode(agent):
    console.print(
        "Chat mode. Type 'exit' to quit. Type 'clear' to clear the screen.",
        style="bold green",
    )
    agent.run_chat()


def main():
    parser = argparse.ArgumentParser(description="dria_agent CLI tool.")
    parser.add_argument("query", nargs="*", help="Query string")
    parser.add_argument("--chat", action="store_true", help="Enable chat mode")
    parser.add_argument(
        "--mcp_path",
        type=str,
        help="Path to MCP (Model Control Protocol) JSON file",
    )
    parser.add_argument(
        "--backend",
        choices=["mlx", "ollama", "huggingface"],
        default="ollama",
        help="Select backend",
    )
    parser.add_argument(
        "--agent_mode",
        choices=["ultra_light", "fast", "balanced", "performant"],
        default="performant",
        help="Select agent mode",
    )
    args = parser.parse_args()

    all_tools = (
        APPLE_TOOLS + API_TOOLS + MATH_TOOLS + SLACK_TOOLS + DOCKER_TOOLS + SEARCH_TOOLS
    )

    try:
        if args.mcp_path:
            agent = ToolCallingAgent(
                mcp_file=args.mcp_path, backend=args.backend, mode=args.agent_mode
            )
        else:
            agent = ToolCallingAgent(
                tools=all_tools, backend=args.backend, mode=args.agent_mode
            )

        if args.mcp_path and args.chat:
            asyncio.run(async_chat_mode(agent))
        elif args.mcp_path and args.query:
            query = " ".join(args.query)
            asyncio.run(async_query_mode(agent, query))
        elif args.chat:
            chat_mode(agent)
        elif args.query:
            query = " ".join(args.query)
            agent.run_feedback(query, print_results=True)
        else:
            parser.print_help()
    except Exception as e:
        console.print(f"Error: {str(e)}", style="bold red")
        return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("Interrupted by user. Exiting...", style="bold red")
        sys.exit(130)
