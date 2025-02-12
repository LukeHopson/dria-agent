from typing import List, Union, Dict
import importlib.util
import logging

if importlib.util.find_spec("ollama") is None:
    raise ImportError(
        "Optional dependency 'ollama' is not installed. Install it with: pip install 'dria-agent[ollama]'"
    )
else:
    from ollama import chat, ChatResponse
from agent.settings.prompt import system_prompt
from .base import ToolCallingAgentBase
from pythonic.schemas import ExecutionResults
from pythonic.engine import execute_tool_call
from rich.console import Console
from rich.panel import Panel

logger = logging.getLogger(__name__)


class OllamaToolCallingAgent(ToolCallingAgentBase):
    def __init__(self, embedding, tools: List, model: str = "dria-agent-a-3b:q8_0"):
        super().__init__(embedding, tools, model)

    def run(
        self, query: Union[str, List[Dict]], dry_run=False, show_completion=True
    ) -> ExecutionResults:
        """
        Performs an inference given a query string or a list of message dicts.

        If the chat model's response starts with 'CALL:', it will extract the tool name and
        arguments (in JSON), invoke the corresponding tool, and then (optionally) pass the tool
        result back to the model.

        :param query: A string (query) or a list of message dicts for a conversation.
        :param dry_run: If True, returns the final response as a string instead of executing the tool.
        :return: The final ChatResponse from the model.
        """
        # If query is a string, convert it to a list of messages.
        if isinstance(query, str):
            messages = [{"role": "user", "content": query}]
        else:
            messages = query.copy()

        inds = self.db.nearest(query)
        tools = [list(self.tools.values())[ind] for ind in inds]

        # Create a system message listing the available tools.
        tool_info = "\n".join(str(tool) for tool in tools)
        system_message = {
            "role": "system",
            "content": system_prompt.replace("{{functions_schema}}", tool_info),
        }
        messages.insert(0, system_message)

        # Make the initial call to the chat model.
        response: ChatResponse = chat(
            model=self.model, messages=messages, options={"temperature": 0.0}
        )
        content = response.message.content

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
