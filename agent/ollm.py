import json
from typing import List, Union, Dict
from ollama import chat, ChatResponse
from .prompt import system_prompt

class ToolCallingAgent:
    def __init__(self, tools: List, model: str = 'driaforall/Dria-Agent-a-1.5B'):
        """
        :param tools: A list of tool objects. Each tool should have a .name attribute and be callable.
        :param model: The name of the model to use for chat inference.
        """
        # Build a mapping from tool names to tool objects.
        self.tools = {tool.name: tool for tool in tools}
        self.model = model

    def run(self, query: Union[str, List[Dict]], dry_run=False) -> str:
        """
        Performs an inference given a query string or a list of message dicts.

        If the chat model's response starts with 'CALL:', it will extract the tool name and
        arguments (in JSON), invoke the corresponding tool, and then (optionally) pass the tool
        result back to the model.

        :param query: A string (query) or a list of message dicts for a conversation.
        :return: The final ChatResponse from the model.
        """
        # If query is a string, convert it to a list of messages.
        if isinstance(query, str):
            messages = [{'role': 'user', 'content': query}]
        else:
            messages = query.copy()

        # Create a system message listing the available tools.
        tool_info = "\n".join(str(tool) for tool in self.tools.values())
        system_message = {
            "role": "system",
            "content": system_prompt.replace("{{functions_schema}}", tool_info),
        }
        messages.insert(0, system_message)

        # Make the initial call to the chat model.
        response: ChatResponse = chat(model=self.model, messages=messages)
        content = response.message.content

        if dry_run:
            return content
        else:
            pass


