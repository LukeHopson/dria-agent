from abc import ABC, abstractmethod
from typing import List, Union, Dict, Callable, Tuple
from dria_agent.pythonic.engine import ExecutionResults
from dria_agent.agent.vdb import ToolDB


class ToolCallingAgentBase(ABC):

    def __init__(self, embedding, tools: List, model: str):
        """
        :param tools: A list of tool objects. Each tool should have a .name attribute and be callable.
        :param model: The name of the model to use for chat inference.
        """
        # Build a mapping from tool names to tool objects.
        self.tools = {tool.name: tool for tool in tools}
        self.db = ToolDB(embedding=embedding)
        schemas = [str(tool) for name, tool in self.tools.items()]
        self.db.add(schemas)
        self.model = model

    @abstractmethod
    def _prepare_messages(
        self, query: Union[str, List[Dict]], num_tools: int
    ) -> Tuple[str, List[Callable]]:
        """Prepare messages and tools for execution"""
        pass

    @abstractmethod
    def _generate_content(self, messages: Union[List[Dict], str]) -> str:
        """Generate content from messages"""
        pass

    @abstractmethod
    def _display_completion(self, content: str) -> None:
        """Display completion in console"""
        pass

    @abstractmethod
    def run(
        self,
        query: Union[str, List[Dict]],
        dry_run=False,
        show_completion=True,
        num_tools=3,
    ) -> ExecutionResults:
        """
        Performs an inference given a query string or a list of message dicts.

        :param query: A string (query) or a list of message dicts for a conversation.
        :param dry_run: If True, returns the final response as a string instead of executing the tool.
        :param show_completion: If True, displays the completion in the console.
        :param num_tools: The number of tools to use for the inference.
        :return: The final response from the model.
        """
        pass

    @abstractmethod
    async def async_run(
        self,
        query: Union[str, List[Dict]],
        dry_run=False,
        show_completion=True,
        num_tools=3,
    ) -> ExecutionResults:
        """
        Asynchronously performs an inference given a query string or a list of message dicts.

        :param query: A string (query) or a list of message dicts for a conversation.
        :param dry_run: If True, returns the final response as a string instead of executing the tool.
        :param show_completion: If True, displays the completion in the console.
        :param num_tools: The number of tools to use for the inference.
        :return: The final response from the model.
        """
        pass

    def set_tools(self, tools: List):
        """
        Set the tools for the agent.
        """
        self.tools = {tool.name: tool for tool in tools}
        schemas = [str(tool) for name, tool in self.tools.items()]
        self.db.add(schemas)
