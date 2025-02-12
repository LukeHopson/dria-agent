import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List
from dria_agent.agent.clients.api import embed
from .tool import ToolCall


class BaseEmbedding(ABC):
    def __init__(self, model_name: str, dim: int):
        self.model_name = model_name
        self.dim = dim

    @abstractmethod
    def batch_embed(self, texts: List[Union[ToolCall, str]]) -> np.ndarray:
        pass

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        pass

    def embed(self, text: Union[ToolCall, str]) -> np.ndarray:
        return self.batch_embed([text])[0]


class OllamaEmbedding(BaseEmbedding):
    def __init__(self, model_name: str = "snowflake-arctic-embed:m", dim: int = 768):
        super().__init__(model_name, dim)
        self.ollama = __import__("ollama")

    def batch_embed(self, texts: List[Union[ToolCall, str]]) -> np.ndarray:
        results = self.ollama.embed(model=self.model_name, input=texts)
        return np.array(results.embeddings, dtype=np.float16)

    def embed_query(self, text: str) -> np.ndarray:
        text = "Represent this sentence for searching relevant passages: " + text
        results = self.ollama.embed(model=self.model_name, input=text)
        return np.array(results.embeddings, dtype=np.float16)


class HuggingFaceEmbedding(BaseEmbedding):
    def __init__(self, dim: int = 768, model_name="Snowflake/snowflake-arctic-embed-m"):
        super().__init__(model_name, dim)
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def batch_embed(self, texts: List[Union[ToolCall, str]]) -> np.ndarray:
        return self.model.encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        text = "Represent this sentence for searching relevant passages: " + text
        return self.model.encode(text)
