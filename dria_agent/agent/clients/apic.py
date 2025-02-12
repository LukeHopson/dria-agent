from dria_agent.agent.settings.prompt import system_prompt
from dria_agent.agent.clients.base import ToolCallingAgentBase
from dria_agent.pythonic.schemas import ExecutionResults
from dria_agent.pythonic.engine import execute_tool_call
from typing import List, Union, Dict
from rich.console import Console
from rich.panel import Panel
from .api import get_completion


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

    def run(
        self,
        query: Union[str, List[Dict]],
        dry_run=False,
        show_completion=True,
        num_tools=2,
    ) -> ExecutionResults:
        """
        Performs an inference given a query string or a list of message dicts.

        :param query: A string (query) or a list of message dicts for a conversation.
        :param dry_run: If True, returns the final response as a string instead of executing the tool.
        :return: The final ExecutionResults
        """

        if num_tools <= 0 and num_tools > 3:
            raise RuntimeError(
                "Number of tools cannot be less than 0 or greater than 3 for optimal performance"
            )

        # If query is a string, convert it to a list of messages.
        if isinstance(query, str):
            messages = [{"role": "user", "content": query}]
        else:
            messages = query.copy()

        inds = self.db.nearest(query, k=2)
        tools = [list(self.tools.values())[ind] for ind in inds]

        # Create a system message listing the available tools.
        tool_info = "\n".join(str(tool) for tool in tools)
        system_message = {
            "role": "system",
            "content": system_prompt.replace("{{functions_schema}}", tool_info),
        }
        messages.insert(0, system_message)

        # Make the initial call to the chat model.
        response = get_completion(
            model_name=self.model,
            provider=self.provider,
            messages=messages,
            options={"temperature": 0.0},
        )
        content = response

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
        else:
            return execute_tool_call(
                completion=content, functions=[t.func for t in tools]
            )
