from dria_agent.agent.clients.hfc import HuggingfaceToolCallingAgent
from dria_agent.agent.clients.ollmc import OllamaToolCallingAgent
from dria_agent.agent.clients.mlxc import MLXToolCallingAgent
from dria_agent.agent.clients.apic import ApiToolCallingAgent
from dria_agent.agent.embedder import OllamaEmbedding, HuggingFaceEmbedding
from typing import Optional
from rich.panel import Panel

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


def create_panel(title: str, content: str, subtitle: Optional[str]=None) -> Panel:
    if subtitle:
        return Panel(content, title=title, subtitle=subtitle, border_style="blue", expand=True)
    return Panel(content, title=title, border_style="blue", expand=True)
