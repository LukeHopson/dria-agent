from typing import List, Union, Dict
import importlib.util
import logging

from dria_agent.agent.settings.prompt import system_prompt
from .base import ToolCallingAgentBase
from dria_agent.pythonic.schemas import ExecutionResults
from dria_agent.pythonic.engine import execute_tool_call
from rich.console import Console
from rich.panel import Panel

logger = logging.getLogger(__name__)


class MLXToolCallingAgent(ToolCallingAgentBase):
    def __init__(
        self, embedding, tools: List, model: str = "driaforall/Tiny-Agent-a-3B-Q8-mlx"
    ):
        super().__init__(embedding, tools, model)
        if importlib.util.find_spec("mlx_lm") is None:
            raise ImportError(
                "Optional dependency 'ollama' is not installed. Install it with: pip install 'dria-agent[mlx]'"
            )
        else:
            from mlx_lm import load, generate
        self.model, self.tokenizer = load(model)
        self.generate = generate

    def run(
        self,
        query: Union[str, List[Dict]],
        dry_run=False,
        show_completion=True,
        num_tools=2,
    ) -> ExecutionResults:

        if num_tools <= 0 and num_tools > 3:
            raise RuntimeError(
                "Number of tools cannot be less than 0 or greater than 3 for optimal performance"
            )

        messages = (
            [{"role": "user", "content": query}]
            if isinstance(query, str)
            else query.copy()
        )

        inds = self.db.nearest(query, k=num_tools)
        tools = [list(self.tools.values())[ind] for ind in inds]

        tool_info = "\n".join(str(tool) for tool in tools)
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

        content = self.generate(
            self.model, self.tokenizer, prompt=prompt, verbose=False, max_tokens=750
        )

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
        return execute_tool_call(completion=content, functions=[t.func for t in tools])
