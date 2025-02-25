import argparse
import asyncio

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


def chat_mode(agent):
    console = Console()
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
    agent = ToolCallingAgent(
        tools=all_tools, backend=args.backend, mode=args.agent_mode
    )

    if args.chat:
        chat_mode(agent)
    elif args.query:
        query = " ".join(args.query)
        agent.run_feedback(query, print_results=True)
    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console = Console()
        console.print("Interrupted by user. Exiting...", style="bold red")
