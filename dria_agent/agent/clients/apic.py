from typing import List, Union, Dict, Tuple, Callable

from rich.console import Console
from rich.panel import Panel

from dria_agent.agent.clients.base import ToolCallingAgentBase
from dria_agent.agent.settings.prompt import system_prompt
from dria_agent.pythonic.engine import execute_tool_call, async_execute_tool_call
from dria_agent.pythonic.schemas import ExecutionResults
from .api import OpenAICompatible


class ApiToolCallingAgent(ToolCallingAgentBase):
    def __init__(
        self,
        embedding,
        tools: List,
        model: str = "driaforall/Tiny-Agent-a-3B",
        **kwargs
    ):
        super().__init__(embedding, tools, model)
        self.provider = kwargs["provider"]
        self.client = OpenAICompatible()

    def _prepare_messages(
        self, query: Union[str, List[Dict]], num_tools: int
    ) -> Tuple[List, List[Callable]]:
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

        return messages, [t.func for t in tools]

    def _generate_content(self, messages: List[Dict]) -> str:
        """Generate content from messages"""
        return self.client.get_completion(
            model_name=self.model,
            provider=self.provider,
            messages=messages,
            options={"temperature": 0.0},
        )

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
        messages, tools = self._prepare_messages(query, num_tools)
        content = self._generate_content(messages)

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
        messages, tools = self._prepare_messages(query, num_tools)
        content = self._generate_content(messages)

        if show_completion:
            self._display_completion(content)

        if dry_run:
            return ExecutionResults(
                content=content, results={}, data={}, errors=[], is_dry=True
            )

        return await async_execute_tool_call(completion=content, functions=tools)
