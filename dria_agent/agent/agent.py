from typing import List
import logging

from dria_agent.agent.clients.base import ToolCallingAgentBase
from dria_agent.agent.settings.providers import PROVIDER_URLS
from dria_agent.agent.clients.hfc import HuggingfaceToolCallingAgent
from dria_agent.agent.clients.ollmc import OllamaToolCallingAgent
from dria_agent.agent.clients.mlxc import MLXToolCallingAgent
from dria_agent.agent.clients.apic import ApiToolCallingAgent
from dria_agent.pythonic.schemas import ExecutionResults
from dria_agent.tools.embedder import OllamaEmbedding, HuggingFaceEmbedding
from .checkers import check_and_install_ollama
from rich.logging import RichHandler

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
    def __init__(self, agent):
        self.agent: ToolCallingAgentBase = agent

    def run(self, query: str, dry_run=False, show_completion=True, num_tools=2) -> ExecutionResults:
        return self.agent.run(query, dry_run=dry_run, show_completion=show_completion, num_tools=num_tools)


class ToolCallingAgentFactory:
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

    @classmethod
    def create(cls, tools: List, backend: str = "ollama", **kwargs):
        agent_cls = cls.BACKENDS.get(backend)
        embedding_cls = cls.EMBEDDING_MAP.get(backend)
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

        if backend == "ollama":
            check_and_install_ollama()

        return ToolCallingAgent(
            agent=agent_cls(embedding=embedding_cls(), tools=tools, **kwargs)
        )
