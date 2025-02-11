"""
A simple, lightweight vector database for handling tools larger/longer than context size.
"""

import numpy as np
from .embedder import BaseEmbedding


class ToolDB:
    def __init__(self, embedding: BaseEmbedding, max_size=1000):

        self.embedding = embedding
        self.embedding.embed()
        self.vectors = np.zeros((max_size, dim), dtype=np.float32)
        self.meta = [None] * max_size
        self.count = 0

    def add(self, vec, meta=None):
        self.vectors[self.count] = vec
        self.meta[self.count] = meta
        self.count += 1

    def nearest(self, query, k=1):
        dists = np.linalg.norm(self.vectors[: self.count] - query, axis=1)
        inds = np.argsort(dists)[:k]
        return [(self.meta[i], self.vectors[i]) for i in inds]
