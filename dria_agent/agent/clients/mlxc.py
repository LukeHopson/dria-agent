from typing import List, Union, Callable, Dict, Optional
import importlib.util
import logging

import math
from functools import partial


from dria_agent.agent.settings.prompt import system_prompt
from .base import ToolCallingAgentBase
from dria_agent.pythonic.schemas import ExecutionResults
from dria_agent.pythonic.engine import execute_tool_call
from rich.console import Console
from rich.panel import Panel

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

    def run(
        self,
        query: Union[str, List[Dict]],
        dry_run=False,
        show_completion=True,
        num_tools=2,
    ) -> ExecutionResults:

        if num_tools <= 0 or num_tools > 5:
            raise RuntimeError(
                "Number of tools cannot be less than 0 or greater than 3 for optimal performance"
            )

        messages = (
            [{"role": "user", "content": query}]
            if isinstance(query, str)
            else query.copy()
        )

        # Use the last three messages from user for search query.
        # This is to keep a balance between context size and relevance.
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        search_query = (
            user_msgs[-2]
            if "Please re-think your response and fix errors" in user_msgs[-1]
            else user_msgs[-1]
        )

        inds = self.db.nearest(search_query, k=num_tools)
        tools = [list(self.tools.values())[ind] for ind in inds]

        tool_info = "\n".join(str(tool) for tool in tools)
        system_message = {
            "role": "system",
            "content": system_prompt.replace("{{functions_schema}}", tool_info),
        }
        messages.insert(0, system_message)

        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

        content = self.generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            verbose=False,
            max_tokens=750,
            sampler=self.sampler,
        )

        content = content.split("<|endoftext|>")[0].strip()

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
        return execute_tool_call(completion=content, functions=[t.func for t in tools])
