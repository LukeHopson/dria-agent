"""
A simple, lightweight vector database for handling tools larger/longer than context size.
"""

import logging

import numpy as np
from ollama import ResponseError

from .embedder import BaseEmbedding

logger = logging.getLogger(__name__)


class ToolDB:
    def __init__(self, embedding: BaseEmbedding, max_size=1000):

        self.embedding = embedding
        self.vectors = np.full((max_size, self.embedding.dim), np.inf, dtype=np.float16)
        self.count = 0

    def add(self, texts: list[str]):
        try:
            embeddings = self.embedding.batch_embed(texts)
        except ResponseError:
            embeddings = []
            for t in texts:
                try:
                    embeddings.append(self.embedding.embed(t))
                except ResponseError:
                    logger.info(
                        f"Doc string is too long for function {t.split()[1].split('(')[0]}"
                    )
                    continue

        for t, e in zip(texts, embeddings):
            self.vectors[self.count] = e
            self.count += 1

    def nearest(self, query, k=1):
        q = self.embedding.embed_query(query)
        q = np.array(q, dtype=self.vectors.dtype)
        dists = np.linalg.norm(self.vectors[: self.count] - q, axis=1)
        return np.argsort(dists)[:k]
