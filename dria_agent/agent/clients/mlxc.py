import importlib.util
import logging
import math
from functools import partial
from typing import List, Union, Callable, Dict, Tuple

from rich.console import Console
from rich.panel import Panel

from dria_agent.agent.settings.prompt import system_prompt
from dria_agent.pythonic.engine import execute_tool_call, async_execute_tool_call
from dria_agent.pythonic.schemas import ExecutionResults
from .base import ToolCallingAgentBase

logger = logging.getLogger(__name__)


class MLXToolCallingAgent(ToolCallingAgentBase):
    def __init__(
        self, embedding, tools: List, model: str = "driaforall/Tiny-Agent-a-3B-Q8-mlx"
    ):
        super().__init__(embedding, tools, model)
        if importlib.util.find_spec("mlx_lm") is None:
            raise ImportError(
                "Optional dependency 'mlx_lm' is not installed. Install it with: pip install 'dria-agent[mlx]'"
            )
        else:
            from mlx_lm import load, generate
            import mlx.core as mx

        # link [https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/sample_utils.py]
        # Copyright © 2023-2024 Apple Inc.
        def make_sampler(
            temp: float = 0.5,
            min_p: float = 0.9,
            min_tokens_to_keep: int = 1,
            top_k: int = -1,
        ):
            if temp == 0:
                return lambda x: mx.argmax(x, axis=-1)
            elif min_p != 0.0:
                return lambda x: min_p_sampling(x, min_p, min_tokens_to_keep, temp)
            elif top_k > 0:
                return lambda x: top_k_sampling(x, top_k, temp)
            else:
                return lambda x: categorical_sampling(x, temp)

        @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
        def categorical_sampling(logits, temp):
            return mx.random.categorical(logits * (1 / temp))

        @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
        def top_k_sampling(
            logprobs: mx.array,
            top_k: int,
            temperature=1.0,
        ) -> mx.array:
            """
            Sample from only the top K tokens ranked by probability.

            Args:
                logprobs: A vector of log probabilities.
                top_k (int): Top k tokens to sample from.
            """
            vocab_size = logprobs.shape[-1]
            if not isinstance(top_k, int) or not (0 < top_k < vocab_size):
                raise ValueError(
                    f"`top_k` has to be an integer in the (0, {vocab_size}] interval,"
                    f" but is {top_k}."
                )
            logprobs = logprobs * (1 / temperature)
            mask_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]
            masked_logprobs = mx.put_along_axis(
                logprobs, mask_idx, mx.array(-float("inf"), logprobs.dtype), axis=-1
            )
            return mx.random.categorical(masked_logprobs, axis=-1)

        @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
        def min_p_sampling(
            logprobs: mx.array,
            min_p: float,
            min_tokens_to_keep: int = 1,
            temperature=1.0,
        ) -> mx.array:

            if not (0 <= min_p <= 1.0):
                raise ValueError(
                    f"`min_p` has to be a float in the [0, 1] interval, but is {min_p}"
                )
            if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
                raise ValueError(
                    f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}"
                )
            logprobs = logprobs * (1 / temperature)

            # Indices sorted in decreasing order
            sorted_indices = mx.argsort(-logprobs, axis=-1)
            sorted_logprobs = mx.take_along_axis(logprobs, sorted_indices, axis=-1)

            # Top probability
            top_logprobs = sorted_logprobs[:, 0:1]

            # Calculate the min_p threshold
            scaled_min_p = top_logprobs + math.log(min_p)

            # Mask tokens that have a probability less than the scaled min_p
            tokens_to_remove = sorted_logprobs < scaled_min_p
            tokens_to_remove[..., :min_tokens_to_keep] = False

            # Create pool of tokens with probability less than scaled min_p
            selected_logprobs = mx.where(
                tokens_to_remove, -float("inf"), sorted_logprobs
            )

            # Return sampled tokens
            sorted_tokens = mx.random.categorical(selected_logprobs, axis=-1)[:, None]
            return mx.take_along_axis(sorted_indices, sorted_tokens, axis=-1).squeeze(1)

        self.sampler = make_sampler(0.5, 0.9)
        self.model, self.tokenizer = load(model)
        self.generate = generate

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
            self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            if getattr(self.tokenizer, "chat_template", None)
            else "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        )

        return prompt, [t.func for t in tools]

    def _generate_content(self, prompt: str) -> str:
        """Generate content from prompt"""
        content = self.generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            verbose=False,
            max_tokens=750,
            sampler=self.sampler,
        )
        return content.split("<|endoftext|>")[0].strip()

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
