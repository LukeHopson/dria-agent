from typing import List, Union, Dict
import importlib.util
import logging

if importlib.util.find_spec("mlx_lm") is None:
    raise ImportError(
        "Optional dependency 'ollama' is not installed. Install it with: pip install 'dria-agent[mlx]'"
    )
else:
    from mlx_lm import load, generate
from agent.settings.prompt import system_prompt
from .base import ToolCallingAgentBase
from pythonic.schemas import ExecutionResults
from pythonic.engine import execute_tool_call
from rich.console import Console
from rich.panel import Panel

logger = logging.getLogger(__name__)


class MLXToolCallingAgent(ToolCallingAgentBase):
    def __init__(
        self, embedding, tools: List, model: str = "driaforall/Tiny-Agent-a-3B-Q8-mlx"
    ):
        super().__init__(embedding, tools, model)
        self.model, self.tokenizer = load(model)

    def run(
        self, query: Union[str, List[Dict]], dry_run=False, show_completion=True
    ) -> ExecutionResults:
        messages = (
            [{"role": "user", "content": query}]
            if isinstance(query, str)
            else query.copy()
        )
        tool_info = "\n".join(str(tool) for tool in self.tools.values())
        system_message = {
            "role": "system",
            "content": system_prompt.replace("{{functions_schema}}", tool_info),
        }
        messages.insert(0, system_message)

        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

        content = generate(self.model, self.tokenizer, prompt=prompt, verbose=False)

        if show_completion:
            console = Console()
            console.rule("[bold blue]Agent Response")
            panel = Panel(
                content, title="Agent", subtitle="End of Response", expand=False
            )
            console.print(panel)
            console.rule()

        if dry_run:
            return ExecutionResults(
                content=content, results={}, data={}, errors=[], is_dry=True
            )
        return execute_tool_call(
            completion=content, functions=[t.func for t in self.tools.values()]
        )
