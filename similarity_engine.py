"""
FAISS-based approximate nearest neighbor search engine.

Uses HNSW (Hierarchical Navigable Small World) graphs as the primary index,
with an exact brute-force index available for recall benchmarking.

References:
    Malkov & Yashunin, "Efficient and robust approximate nearest neighbor
    search using Hierarchical Navigable Small World graphs",
    IEEE TPAMI 2020 — arXiv:1603.09320

    Johnson et al., "Billion-scale similarity search with GPUs",
    IEEE Transactions on Big Data, 2019
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

import config

logger = logging.getLogger(__name__)


class FAISSEngine:
    """
    Vector similarity search backed by FAISS.

    Two index types are supported:

    - HNSW (IndexHNSWFlat): Graph-based approximate nearest neighbor search.
      Query complexity is O(log n), making it highly scalable. The graph
      is layered — upper layers hold fewer nodes with long-range connections,
      lower layers are dense with fine-grained connections. Search starts at
      the top and greedily descends until layer 0.

    - FlatIP (IndexFlatIP): Exact inner-product search, equivalent to cosine
      similarity when vectors are L2-normalised. Used as the ground-truth
      baseline for measuring HNSW recall.

    All vectors must be float32 and L2-normalised before being added to the
    index. Under L2 normalisation, inner product equals cosine similarity,
    so both index types return cosine-based rankings.
    """

    def __init__(
        self,
        dimension: int,
        use_hnsw: bool = True,
        hnsw_m: int = None,
        hnsw_ef_construction: int = None,
        hnsw_ef_search: int = None,
    ):
        """
        Args:
            dimension: Dimensionality of the vectors to be indexed.
            use_hnsw: Use HNSW if True, exact FlatIP if False.
            hnsw_m: Number of bidirectional links per node in the HNSW graph.
                    Higher values improve recall at the cost of memory and
                    build time. Typical range: 16–64.
            hnsw_ef_construction: Size of the dynamic candidate list during
                                  graph construction. Higher values yield
                                  a better-connected graph but slower builds.
            hnsw_ef_search: Size of the candidate list during queries.
                            Can be tuned at query time without rebuilding.
        """
        self.dimension = dimension
        self.use_hnsw = use_hnsw
        self.hnsw_m = hnsw_m or config.HNSW_M
        self.hnsw_ef_construction = hnsw_ef_construction or config.HNSW_EF_CONSTRUCTION
        self.hnsw_ef_search = hnsw_ef_search or config.HNSW_EF_SEARCH

        self.index = None
        self.n_vectors = 0
        self._build_time = 0.0

    def build(self, vectors: np.ndarray) -> "FAISSEngine":
        """
        Build the FAISS index from a set of L2-normalised float32 vectors.

        Args:
            vectors: Shape (n, dimension), dtype float32, L2-normalised.

        Returns:
            Self, to allow method chaining.
        """
        assert vectors.shape[1] == self.dimension, (
            f"Vector dimension mismatch: expected {self.dimension}, "
            f"got {vectors.shape[1]}"
        )
        assert vectors.dtype == np.float32, "Vectors must be float32"

        self.n_vectors = vectors.shape[0]
        start = time.perf_counter()

        if self.use_hnsw:
            logger.info(
                "Building HNSW index: %d vectors × %d dims  "
                "(M=%d, efConstruction=%d)",
                self.n_vectors, self.dimension,
                self.hnsw_m, self.hnsw_ef_construction,
            )
            self.index = faiss.IndexHNSWFlat(self.dimension, self.hnsw_m)
            self.index.hnsw.efConstruction = self.hnsw_ef_construction
            self.index.hnsw.efSearch = self.hnsw_ef_search
        else:
            logger.info(
                "Building exact index: %d vectors × %d dims",
                self.n_vectors, self.dimension,
            )
            self.index = faiss.IndexFlatIP(self.dimension)

        self.index.add(vectors)
        self._build_time = time.perf_counter() - start
        logger.info(
            "Index ready in %.2fs  (total vectors: %d)",
            self._build_time, self.index.ntotal,
        )
        return self

    def search(
        self,
        query_vectors: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the k nearest neighbours for each query vector.

        Args:
            query_vectors: Shape (n_queries, dimension), float32, L2-normalised.
            k: Number of neighbours to retrieve.

        Returns:
            distances: Shape (n_queries, k) — similarity scores (higher = closer).
            indices:   Shape (n_queries, k) — positions in the original vector array.
        """
        if self.index is None:
            raise RuntimeError("Index has not been built yet. Call build() first.")

        k = min(k, self.n_vectors)

        if self.use_hnsw:
            self.index.hnsw.efSearch = self.hnsw_ef_search

        return self.index.search(query_vectors, k)

    def search_single(
        self,
        query_vector: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convenience wrapper for querying a single vector.

        Returns:
            distances: Shape (k,)
            indices:   Shape (k,)
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        distances, indices = self.search(query_vector, k)
        return distances[0], indices[0]

    def save(self, path: str):
        """Serialise the index to disk."""
        if self.index is None:
            raise RuntimeError("No index to save.")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        faiss.write_index(self.index, path)
        logger.info(
            "Index saved to %s  (%.1f MB)",
            path, os.path.getsize(path) / 1024 / 1024,
        )

    def load(self, path: str) -> "FAISSEngine":
        """Load a previously serialised index from disk."""
        self.index = faiss.read_index(path)
        self.n_vectors = self.index.ntotal

        if self.use_hnsw and hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = self.hnsw_ef_search

        logger.info("Index loaded from %s  (%d vectors)", path, self.n_vectors)
        return self

    def benchmark(
        self,
        query_vectors: np.ndarray,
        k: int = 10,
        n_queries: int = 100,
    ) -> Dict:
        """
        Measure query latency and, for HNSW indices, Recall@k against the
        exact FlatIP baseline.

        Args:
            query_vectors: Pool of vectors from which random queries are drawn.
            k: Neighbourhood size.
            n_queries: Number of queries to run.

        Returns:
            Dictionary of benchmark metrics.
        """
        n_queries = min(n_queries, query_vectors.shape[0])
        sample_idx = np.random.choice(query_vectors.shape[0], n_queries, replace=False)
        queries = query_vectors[sample_idx]

        # Warm-up pass to prime caches
        self.search(queries[:5], k)

        latencies = []
        for i in range(n_queries):
            q = queries[i : i + 1]
            t0 = time.perf_counter()
            self.search(q, k)
            latencies.append((time.perf_counter() - t0) * 1_000)

        latencies = np.array(latencies)

        results = {
            "index_type": "HNSW" if self.use_hnsw else "FlatIP (exact)",
            "n_vectors": self.n_vectors,
            "dimension": self.dimension,
            "n_queries": n_queries,
            "k": k,
            "build_time_s": self._build_time,
            "latency_p50_ms": float(np.percentile(latencies, 50)),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
            "latency_p99_ms": float(np.percentile(latencies, 99)),
            "latency_mean_ms": float(np.mean(latencies)),
        }

        if self.use_hnsw:
            flat = faiss.IndexFlatIP(self.dimension)
            flat.add(query_vectors)
            recall_sum = 0
            for q in queries[:min(100, n_queries)]:
                q = q.reshape(1, -1)
                _, gt = flat.search(q, k)
                _, approx = self.search(q, k)
                recall_sum += len(set(gt[0]) & set(approx[0])) / k
            results["recall_at_k"] = recall_sum / min(100, n_queries)

        logger.info("Benchmark: %s", results)
        return results
