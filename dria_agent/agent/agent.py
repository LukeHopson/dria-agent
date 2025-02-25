import copy
import logging
from typing import List, Literal, Callable

from rich.console import Console
from rich.logging import RichHandler

from dria_agent.agent.settings.providers import PROVIDER_URLS
from dria_agent.pythonic.schemas import ExecutionResults
from .checkers import check_and_install_ollama
from .mcp import MCPToolAdapter
from .utils import *

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

        self._mcp_adapter = MCPToolAdapter(mcp_file) if mcp_file is not None else None
        tools = self._mcp_adapter.tools if self._mcp_adapter else tools

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
        if self._mcp_adapter:
            await self._mcp_adapter.connect_servers()
            self.agent.set_tools(self._mcp_adapter.tools)

    async def close_servers(self):
        """Close the MCP server connection."""
        await self._mcp_adapter.close_servers()

    @staticmethod
    def _print_execution_results(execution: ExecutionResults, query: str) -> None:
        """Helper method to print execution results in a consistent format"""
        console = Console()
        console.print(create_panel("Query", query, "End of Query"))
        console.print(
            create_panel(
                title="Execution Result", content=str(execution.final_answer())
            )
        )

        if execution.errors:
            console.print(create_panel(title="Errors", content=str(execution.errors)))

    def run(
        self,
        query: str,
        dry_run: bool = False,
        show_completion: bool = True,
        num_tools: int = 2,
        print_results: bool = True,
    ) -> ExecutionResults:
        """
        Run the agent synchronously with the given query.

        Args:
            query: The query string to process
            dry_run: If True, don't execute tools, just return planned execution
            show_completion: Whether to show the agent's completion
            num_tools: Number of tools to use for the query
            print_results: Whether to print execution results

        Returns:
            ExecutionResults containing the execution outcome
        """
        execution = self.agent.run(
            query, dry_run=dry_run, show_completion=show_completion, num_tools=num_tools
        )
        if print_results:
            self._print_execution_results(execution, query)
        return execution

    async def async_run(
        self,
        query: str,
        dry_run: bool = False,
        show_completion: bool = True,
        num_tools: int = 2,
        print_results: bool = True,
    ) -> ExecutionResults:
        """
        Run the agent asynchronously with the given query.

        Args:
            query: The query string to process
            dry_run: If True, don't execute tools, just return planned execution
            show_completion: Whether to show the agent's completion
            num_tools: Number of tools to use for the query
            print_results: Whether to print execution results

        Returns:
            ExecutionResults containing the execution outcome
        """

        execution = await self.agent.async_run(
            query, dry_run=dry_run, show_completion=show_completion, num_tools=num_tools
        )
        if print_results:
            self._print_execution_results(execution, query)
        return execution

    def run_feedback(
        self,
        query: str,
        show_completion: bool = True,
        num_tools: int = 2,
        print_results: bool = True,
        max_iterations: int = 3,
    ) -> ExecutionResults:
        """
        Run the agent with feedback loop to handle errors.

        Args:
            query: The query string to process
            show_completion: Whether to show the agent's completion
            num_tools: Number of tools to use for the query
            print_results: Whether to print execution results
            max_iterations: Maximum number of feedback iterations

        Returns:
            ExecutionResults containing the final execution outcome
        """
        execution = self._run_with_feedback(
            query=query,
            show_completion=show_completion,
            num_tools=num_tools,
            print_results=print_results,
            max_iterations=max_iterations,
            run_func=self.agent.run,
        )
        return execution

    async def async_run_feedback(
        self,
        query: str,
        show_completion: bool = True,
        num_tools: int = 2,
        print_results: bool = True,
        max_iterations: int = 3,
    ) -> ExecutionResults:
        """
        Run the agent asynchronously with feedback loop to handle errors.

        Args:
            query: The query string to process
            show_completion: Whether to show the agent's completion
            num_tools: Number of tools to use for the query
            print_results: Whether to print execution results
            max_iterations: Maximum number of feedback iterations

        Returns:
            ExecutionResults containing the final execution outcome
        """
        execution = await self._async_run_with_feedback(
            query=query,
            show_completion=show_completion,
            num_tools=num_tools,
            print_results=print_results,
            max_iterations=max_iterations,
        )
        return execution

    async def _async_run_with_feedback(
        self,
        query: str,
        show_completion: bool,
        num_tools: int,
        print_results: bool,
        max_iterations: int,
    ) -> ExecutionResults:
        """
        Helper method implementing the async feedback loop logic.

        Args:
            query: The query string to process
            show_completion: Whether to show the agent's completion
            num_tools: Number of tools to use for the query
            print_results: Whether to print execution results
            max_iterations: Maximum number of feedback iterations

        Returns:
            ExecutionResults containing the final execution outcome
        """
        execution = await self.agent.async_run(
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

        history = [{"role": "user", "content": query}]

        iterations = 0
        while execution.errors and iterations < max_iterations:
            history.extend(
                [
                    {"role": "assistant", "content": execution.content},
                    {
                        "role": "user",
                        "content": f"Please re-think your code and fix errors. You got the following errors: {str(execution.errors)}",
                    },
                ]
            )

            execution = await self.agent.async_run(
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

            iterations += 1

        return execution

    def _run_with_feedback(
        self,
        query: str,
        show_completion: bool,
        num_tools: int,
        print_results: bool,
        max_iterations: int,
        run_func: Callable,
    ) -> ExecutionResults:
        """
        Helper method implementing the feedback loop logic.

        Args:
            query: The query string to process
            show_completion: Whether to show the agent's completion
            num_tools: Number of tools to use for the query
            print_results: Whether to print execution results
            max_iterations: Maximum number of feedback iterations
            run_func: Function to use for running the agent (sync or async)

        Returns:
            ExecutionResults containing the final execution outcome
        """
        execution = run_func(
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

        history = [{"role": "user", "content": query}]

        iterations = 0
        while execution.errors and iterations < max_iterations:
            history.extend(
                [
                    {"role": "assistant", "content": execution.content},
                    {
                        "role": "user",
                        "content": f"Please re-think your code and fix errors. You got the following errors: {str(execution.errors)}",
                    },
                ]
            )

            execution = run_func(
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
