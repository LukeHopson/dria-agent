from typing import List, Literal
import logging

import copy
from dria_agent.agent.settings.providers import PROVIDER_URLS
from dria_agent.pythonic.schemas import ExecutionResults
from .mcp import MCPToolAdapter
from .utils import *
from .checkers import check_and_install_ollama
from rich.logging import RichHandler
from rich.console import Console

console_handler = RichHandler(rich_tracebacks=True)
file_handler = logging.FileHandler("app.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[console_handler, file_handler],
    # level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logging.getLogger("httpx").setLevel(logging.WARNING)


class ToolCallingAgent(object):
    BACKENDS = {
        "huggingface": HuggingfaceToolCallingAgent,
        "mlx": MLXToolCallingAgent,
        "ollama": OllamaToolCallingAgent,
        "api": ApiToolCallingAgent,
    }

    EMBEDDING_MAP = {
        "huggingface": HuggingFaceEmbedding,
        "mlx": HuggingFaceEmbedding,
        "ollama": OllamaEmbedding,
        "api": HuggingFaceEmbedding,
    }

    MODE_MAP = {
        "fast": {
            "ollama": ["driaforall/tiny-agent-a:1.5b", "snowflake-arctic-embed:s"],
            "huggingface": [
                "driaforall/Tiny-Agent-a-3B",
                "Snowflake/snowflake-arctic-embed-m",
            ],
            "mlx": [
                "driaforall/Tiny-Agent-a-1.5B-Q8-mlx",
                "Snowflake/snowflake-arctic-embed-s",
            ],
            "api": ["driaforall/Tiny-Agent-a-3B", "Snowflake/snowflake-arctic-embed-m"],
        },
        "balanced": {
            "ollama": ["driaforall/tiny-agent-a:3b-q4_K_M", "snowflake-arctic-embed:m"],
            "huggingface": [
                "driaforall/Tiny-Agent-a-3B",
                "Snowflake/snowflake-arctic-embed-m",
            ],
            "mlx": [
                "driaforall/Tiny-Agent-a-1.5B-Q8-mlx",
                "Snowflake/snowflake-arctic-embed-m",
            ],
            "api": ["driaforall/Tiny-Agent-a-3B", "Snowflake/snowflake-arctic-embed-m"],
        },
        "performant": {
            "ollama": ["driaforall/tiny-agent-a:3b", "snowflake-arctic-embed:m"],
            "huggingface": [
                "driaforall/Tiny-Agent-a-3B",
                "Snowflake/snowflake-arctic-embed-l",
            ],
            "mlx": [
                "driaforall/Tiny-Agent-a-3B-Q8-mlx",
                "Snowflake/snowflake-arctic-embed-m",
            ],
            "api": ["driaforall/Tiny-Agent-a-3B", "Snowflake/snowflake-arctic-embed-l"],
        },
        "ultra_light": {
            "ollama": ["driaforall/tiny-agent-a:0.5b", "snowflake-arctic-embed:xs"],
            "huggingface": [
                "driaforall/Tiny-Agent-a-0.5B",
                "Snowflake/snowflake-arctic-embed-xs",
            ],
            "mlx": [
                "driaforall/Tiny-Agent-a-0.5B-Q8-mlx",
                "Snowflake/snowflake-arctic-embed-xs",
            ],
            "api": [
                "driaforall/Tiny-Agent-a-0.5B",
                "Snowflake/snowflake-arctic-embed-xs",
            ],
        },
    }

    embedding_dims = {
        "snowflake-arctic-embed:xs": 384,
        "snowflake-arctic-embed:s": 384,
        "snowflake-arctic-embed:m": 768,
        "snowflake-arctic-embed-l": 1024,
        "Snowflake/snowflake-arctic-embed-xs": 384,
        "Snowflake/snowflake-arctic-embed-s": 384,
        "Snowflake/snowflake-arctic-embed-m": 768,
        "Snowflake/snowflake-arctic-embed-l": 1024,
    }

    def __init__(
        self,
        mcp_file: Optional[str] = None,
        tools: Optional[List] = None,
        backend: str = "ollama",
        mode: Literal["ultra_light", "fast", "balanced", "performant"] = "performant",
        **kwargs,
    ):
        if mcp_file is None and tools is None:
            raise ValueError(
                "Either mcp_file or tools must be provided. "
                "For MCP tools, provide path to MCP config JSON file. "
                "For regular tools, provide a list of tool functions decorated with @tool."
            )

        self.adapter = MCPToolAdapter(mcp_file) if mcp_file is not None else None
        tools = self.adapter.tools if self.adapter else tools

        agent_cls = self.BACKENDS.get(backend)
        embedding_cls = self.EMBEDDING_MAP.get(backend)
        if not agent_cls or not embedding_cls:
            raise ValueError(f"Unknown agent type: {backend}")
        if backend == "api":
            if "provider" not in kwargs:
                raise ValueError("API provider not provided")
            provider = kwargs["provider"]
            logging.warning("Using %s API as backend", provider)
            if provider not in list(PROVIDER_URLS.keys()):
                raise ValueError(f"Unknown provider: {provider}")

            if provider == "ollama":
                embedding_cls = OllamaEmbedding

        model_pairs = self.MODE_MAP[mode][backend]
        if backend == "ollama":
            check_and_install_ollama(model_pairs[0], model_pairs[1])

        self.agent = agent_cls(
            model=model_pairs[0],
            embedding=embedding_cls(
                model_name=model_pairs[1], dim=self.embedding_dims[model_pairs[1]]
            ),
            tools=tools,
            **kwargs,
        )

    async def initialize_servers(self):
        """Asynchronously initialize the agent, including connecting to the MCP server."""
        if self.adapter:
            await self.adapter.connect_servers()
            self.agent.set_tools(self.adapter.tools)

    async def run(
        self,
        query: str,
        dry_run=False,
        show_completion=True,
        num_tools=2,
        print_results=True,
    ) -> ExecutionResults:
        execution = await self.agent.run(
            query, dry_run=dry_run, show_completion=show_completion, num_tools=num_tools
        )
        if print_results:
            console = Console()
            console.print(create_panel("Query", query, "End of Query"))
            console.print(
                create_panel(
                    title="Execution Result", content=str(execution.final_answer())
                )
            )

            if execution.errors:
                console.print(
                    create_panel(title="Errors", content=str(execution.errors))
                )

        return execution

    async def run_feedback(
        self,
        query: str,
        show_completion=True,
        num_tools=2,
        print_results=True,
        max_iterations=3,
    ) -> ExecutionResults:

        execution = await self.agent.run(
            query, dry_run=False, show_completion=show_completion, num_tools=num_tools
        )
        if print_results:
            console = Console()
            console.print(create_panel("Query", query, "End of Query"))
            console.print(
                create_panel(
                    title="Execution Result", content=str(execution.final_answer())
                )
            )

            if execution.errors:
                console.print(
                    create_panel(title="Errors", content=str(execution.errors))
                )

        history = [
            {"role": "user", "content": query},
        ]

        iterations = 0
        while execution.errors:
            history.append({"role": "assistant", "content": execution.content})
            history.append(
                {
                    "role": "user",
                    "content": f"Please re-think your code and fix errors. You got the following errors: {str(execution.errors)}",
                }
            )
            execution = self.agent.run(
                copy.deepcopy(history),
                dry_run=False,
                show_completion=show_completion,
                num_tools=num_tools,
            )
            if print_results:
                console = Console()
                console.print(create_panel("Query", query, "End of Query"))
                console.print(
                    create_panel(
                        title="Execution Result", content=str(execution.final_answer())
                    )
                )

                if execution.errors:
                    console.print(
                        create_panel(title="Errors", content=str(execution.errors))
                    )
            if iterations == max_iterations - 1:
                break
            iterations += 1
        return execution

    def run_chat(
        self, show_completion=True, num_tools=3, print_results=True, max_iterations=1
    ) -> None:
        history = []
        console = Console()
        while True:
            user_input = input(": ").strip()
            if user_input.lower() in ("exit", "quit"):
                break
            history.append({"role": "user", "content": user_input})
            history = compress_history(history, threshold=3500)

            execution = self.agent.run(
                copy.deepcopy(history),
                dry_run=False,
                show_completion=show_completion,
                num_tools=num_tools,
            )
            if print_results:
                console.print(create_panel("User Query", user_input, "End of Query"))
                console.print(
                    create_panel("Execution Result", str(execution.final_answer()))
                )
                if execution.errors:
                    console.print(create_panel("Errors", str(execution.errors)))
            iterations = 0
            while execution.errors and iterations < max_iterations:
                history.append({"role": "assistant", "content": execution.content})
                feedback = f"Please re-think your response and fix errors. Errors: {execution.errors}"
                history.append({"role": "user", "content": feedback})
                execution = self.agent.run(
                    copy.deepcopy(history),
                    dry_run=False,
                    show_completion=show_completion,
                    num_tools=num_tools,
                )
                if print_results:
                    console.print(
                        create_panel(
                            "Assistant Response", str(execution.final_answer())
                        )
                    )
                    if execution.errors:
                        console.print(create_panel("Errors", str(execution.errors)))
                iterations += 1
            history.append({"role": "assistant", "content": execution.content})
            history.append({"role": "tool", "content": str(execution.final_answer())})
