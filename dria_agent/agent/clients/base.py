from abc import ABC, abstractmethod
from typing import List, Union, Dict
from dria_agent.pythonic.engine import ExecutionResults
from dria_agent.tools.vdb import ToolDB


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
        :return: The final response from the model.
        """
        pass
