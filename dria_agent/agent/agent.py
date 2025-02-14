from typing import List, Literal
import logging

from dria_agent.agent.settings.providers import PROVIDER_URLS
from dria_agent.pythonic.schemas import ExecutionResults
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
        tools: List,
        backend: str = "ollama",
        mode: Literal["ultra_light", "fast", "balanced", "performant"] = "performant",
        **kwargs,
    ):
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
            embedding=embedding_cls(model_name=model_pairs[1]),
            tools=tools,
            **kwargs,
        )

    def run(
        self,
        query: str,
        dry_run=False,
        show_completion=True,
        num_tools=2,
        print_results=True,
    ) -> ExecutionResults:
        execution = self.agent.run(
            query, dry_run=dry_run, show_completion=show_completion, num_tools=num_tools
        )
        if print_results:
            console = Console()
            console.print(create_panel("Query", query, "End of Query"))
            console.print(create_panel(title="Execution Result", content=str(execution.final_answer())))

            if execution.errors:
                console.print(create_panel(title="Errors", content=str(execution.errors)))

        return execution
