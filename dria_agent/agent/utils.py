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


def create_panel(title: str, content: str, subtitle: Optional[str] = None) -> Panel:
    if subtitle:
        return Panel(
            content, title=title, subtitle=subtitle, border_style="blue", expand=True
        )
    return Panel(content, title=title, border_style="blue", expand=True)


def count_tokens(text: str) -> int:
    """
    Count tokens in a text using simple whitespace splitting.

    :param text: Input text.
    :return: Number of tokens.
    """
    return len(str(text).split())


def total_token_count(history: list) -> int:
    """
    Compute the total number of tokens in the conversation history.

    :param history: List of message dictionaries.
    :return: Total token count.
    """
    return sum(count_tokens(msg["content"]) for msg in history)


def compress_history(history: list, threshold: int = 500) -> list:
    """
    Compress the conversation history if the total token count exceeds a threshold.
    It prunes non-user messages (e.g., assistant responses) by replacing their content
    with a placeholder "[pruned]", starting with the longest turns.

    :param history: List of message dictionaries with 'role' and 'content' keys.
    :param threshold: Maximum allowed token count.
    :return: A potentially compressed history that meets the threshold.
    """
    current_tokens = total_token_count(history)
    if current_tokens <= threshold:
        return history

    # Build list of candidate messages (non-user) with their indices and token counts.
    candidates = [
        (idx, count_tokens(msg["content"]))
        for idx, msg in enumerate(history)
        if msg["role"] != "user"
    ]

    # Sort candidates by token count in descending order (longest first).
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Create a copy of history to prune.
    pruned_history = history.copy()

    # Prune messages until total tokens are under the threshold.
    for idx, token_count in candidates:
        if total_token_count(pruned_history) <= threshold:
            break
        pruned_history[idx]["content"] = "[pruned]"

    return pruned_history
