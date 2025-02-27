from typing import List, Union, Dict, Callable, Tuple
import logging
import importlib.util

from dria_agent.agent.settings.prompt import system_prompt
from .base import ToolCallingAgentBase
from dria_agent.pythonic.schemas import ExecutionResults
from dria_agent.pythonic.engine import execute_tool_call, async_execute_tool_call
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
        self.temperature = 0.5
        self.min_p = 0.95

    def _prepare_messages(
        self, query: Union[str, List[Dict]], num_tools: int
    ) -> Tuple[str, List[Callable]]:
        """Prepare messages and tools for execution"""
        if num_tools <= 0 or num_tools > 5:
            raise RuntimeError(
                "Number of tools cannot be less than 0 or greater than 3 for optimal performance"
            )

        messages = (
            [{"role": "user", "content": query}]
            if isinstance(query, str)
            else query.copy()
        )

        # Get search query from user messages
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        search_query = (
            user_msgs[-2]
            if "Please re-think your response and fix errors" in user_msgs[-1]
            else user_msgs[-1]
        )

        # Get relevant tools
        inds = self.db.nearest(search_query, k=num_tools)
        tools = [list(self.tools.values())[ind] for ind in inds]
        tool_info = "\n".join(str(tool) for tool in tools)

        # Add system message
        messages.insert(
            0,
            {
                "role": "system",
                "content": system_prompt.replace("{{functions_schema}}", tool_info),
            },
        )

        # Generate prompt
        prompt = (
            "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages) + "\n"
        )

        return prompt, [t.func for t in tools]

    def _generate_content(self, prompt: str) -> str:
        """Generate content from prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=self.temperature,
            min_p=self.min_p,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)[
            len(prompt) :
        ].strip()

    def _display_completion(self, content: str) -> None:
        """Display completion in console"""
        console = Console()
        console.rule("[bold blue]Agent Response")
        panel = Panel(content, title="Agent", subtitle="End of Response", expand=False)
        console.print(panel)
        console.rule()

    def run(
        self,
        query: Union[str, List[Dict]],
        dry_run: bool = False,
        show_completion: bool = True,
        num_tools: int = 2,
    ) -> ExecutionResults:
        """Run agent synchronously"""
        prompt, tools = self._prepare_messages(query, num_tools)
        content = self._generate_content(prompt)

        if show_completion:
            self._display_completion(content)

        if dry_run:
            return ExecutionResults(
                content=content, results={}, data={}, errors=[], is_dry=True
            )

        return execute_tool_call(completion=content, functions=tools)

    async def async_run(
        self,
        query: Union[str, List[Dict]],
        dry_run: bool = False,
        show_completion: bool = True,
        num_tools: int = 2,
    ) -> ExecutionResults:
        """Run agent asynchronously"""
        prompt, tools = self._prepare_messages(query, num_tools)
        content = self._generate_content(prompt)

        if show_completion:
            self._display_completion(content)

        if dry_run:
            return ExecutionResults(
                content=content, results={}, data={}, errors=[], is_dry=True
            )

        return await async_execute_tool_call(completion=content, functions=tools)

    def instruct(self, query: Union[str, List[Dict]], show_completion: bool = False):

        messages = (
            [{"role": "user", "content": query}]
            if isinstance(query, str)
            else query.copy()
        )

        prompt = (
                "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages) + "\n"
        )

        content = self._generate_content(prompt)
        if show_completion:
            self._display_completion("Instruct Mode: \n\n" + content)

        return content
