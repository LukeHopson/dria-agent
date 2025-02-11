import json
from typing import List, Union, Dict
from vllm import LLM, SamplingParams
from .prompt import system_prompt
from .base import ToolCallingAgentBase
from pythonic.schemas import ExecutionResults
from pythonic.engine import execute_tool_call


class ToolCallingAgent(ToolCallingAgentBase):
    def __init__(
        self,
        tools: List,
        model: str = "driaforall/Tiny-Agent-a-3b",
        tokenizer: str = "driaforall/Tiny-Agent-a-3b",
    ):
        super().__init__(tools, model)
        self.llm = LLM(model=model, tokenizer=tokenizer)
        self.sampling_params = SamplingParams(temperature=0.1, top_p=0.95)

    def run(self, query: Union[str, List[Dict]], dry_run=False) -> ExecutionResults:
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
        outputs = self.llm.chat(messages, self.sampling_params)
        content = outputs[0].outputs[0].text
        if dry_run:
            return ExecutionResults(
                content=content, results={}, data={}, errors=[], is_dry=True
            )
        return execute_tool_call(
            completion=content, functions=[t.func for t in self.tools.values()]
        )
