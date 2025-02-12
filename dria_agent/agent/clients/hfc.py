from typing import List, Union, Dict
import logging
import importlib.util

from dria_agent.agent.settings.prompt import system_prompt
from .base import ToolCallingAgentBase
from dria_agent.pythonic.schemas import ExecutionResults
from dria_agent.pythonic.engine import execute_tool_call
from rich.console import Console
from rich.panel import Panel

logger = logging.getLogger(__name__)


class HuggingfaceToolCallingAgent(ToolCallingAgentBase):
    def __init__(
        self,
        embedding,
        tools: List,
        model: str = "driaforall/Tiny-Agent-a-3b",
        tokenizer: str = "driaforall/Tiny-Agent-a-3b",
    ):
        super().__init__(embedding, tools, model)
        if importlib.util.find_spec("transformers") is None:
            raise ImportError(
                "Optional dependency 'transformers' is not installed. Install it with: pip install 'dria-agent[huggingface]'"
            )
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.temperature = 0.1
        self.top_p = 0.95

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
        tool_info = "\n".join(str(tool) for tool in self.tools.values())
        system_message = {
            "role": "system",
            "content": system_prompt.replace("{{functions_schema}}", tool_info),
        }
        messages.insert(0, system_message)
        prompt = (
            "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages) + "\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        content = self.tokenizer.decode(outputs[0], skip_special_tokens=True)[
            len(prompt) :
        ].strip()

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
