import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List
from .tool import ToolCall


class BaseEmbedding(ABC):
    def __init__(self, model_name: str, dim: int):
        self.model_name = model_name
        self.dim = dim

    @abstractmethod
    def batch_embed(self, texts: List[Union[ToolCall, str]]) -> np.ndarray:
        pass

    def embed(self, text: Union[ToolCall, str]) -> np.ndarray:
        return self.batch_embed([text])[0]


class OllamaEmbedding(BaseEmbedding):
    def __init__(self, model_name: str, dim: int):
        super().__init__(model_name, dim)
        self.ollama = __import__("ollama")

    def batch_embed(self, texts: List[Union[ToolCall, str]]) -> np.ndarray:
        results = self.ollama.embed(model=self.model_name, input=texts)
        return np.array(results, dtype=np.float32)


class VLLMEmbedding(BaseEmbedding):
    def __init__(self, model_name: str, dim: int):
        super().__init__(model_name, dim)
        from vllm import LLM

        self.model = LLM(model=model_name, enforce_eager=True)

    def batch_embed(self, texts: List[Union[ToolCall, str]]) -> np.ndarray:
        outputs = self.model.encode(texts)
        embeddings = [
            np.array(out.outputs.embedding, dtype=np.float32) for out in outputs
        ]
        return np.stack(embeddings)


class HuggingFaceEmbedding(BaseEmbedding):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def batch_embed(self, texts: List[Union[ToolCall, str]]) -> np.ndarray:
        return self.model.encode(texts)
