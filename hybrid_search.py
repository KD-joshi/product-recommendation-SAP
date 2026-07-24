"""
Hybrid Search Engine Module

Implements Sparse (BM25) search and Reciprocal Rank Fusion (RRF) 
to combine Dense (FAISS) and Sparse scores.
"""
import os
import pickle
import logging
import numpy as np
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

def tokenize(text: str) -> List[str]:
    """Simple whitespace and lowercasing tokenizer."""
    if not isinstance(text, str):
        return []
    return text.lower().split()

class BM25Engine:
    def __init__(self):
        self.bm25 = None
        self.corpus_size = 0
        
    def build(self, documents: List[str]):
        """
        Build the BM25 index from a list of document strings.
        Usually this is the product title + brand + categories.
        """
        logger.info(f"Tokenizing {len(documents)} documents for BM25...")
        tokenized_corpus = [tokenize(doc) for doc in documents]
        self.corpus_size = len(documents)
        
        logger.info("Building BM25 index...")
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built successfully.")
        
    def save(self, filepath: str):
        """Save the BM25 model to disk."""
        if self.bm25 is None:
            raise RuntimeError("BM25 model not built yet.")
        with open(filepath, 'wb') as f:
            pickle.dump(self.bm25, f)
        logger.info(f"Saved BM25 index to {filepath}")
        
    def load(self, filepath: str):
        """Load the BM25 model from disk."""
        with open(filepath, 'rb') as f:
            self.bm25 = pickle.load(f)
        self.corpus_size = self.bm25.corpus_size
        logger.info(f"Loaded BM25 index from {filepath}")
        
    def search(self, query: str, top_k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the BM25 index.
        Returns: (scores, indices)
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 model not loaded.")
            
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_n = min(top_k, self.corpus_size)
        indices = np.argsort(scores)[::-1][:top_n]
        top_scores = scores[indices]
        
        return top_scores, indices

def reciprocal_rank_fusion(
    dense_results: Tuple[np.ndarray, np.ndarray], 
    sparse_results: Tuple[np.ndarray, np.ndarray],
    k: int = 60,
    top_k: int = 100
) -> List[int]:
    """
    Combine Dense (FAISS) and Sparse (BM25) results using RRF.
    RRF Score = 1 / (k + rank_dense) + 1 / (k + rank_sparse)
    
    dense_results: (distances, indices)
    sparse_results: (scores, indices)
    """
    dense_dists, dense_idx = dense_results
    sparse_scores, sparse_idx = sparse_results
    
    rrf_scores: Dict[int, float] = {}
    
    # Process Dense Ranks
    for rank, doc_idx in enumerate(dense_idx):
        if doc_idx < 0: # FAISS padding
            continue
        # FAISS returns distances (lower is better). They are already sorted.
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
        
    # Process Sparse Ranks
    for rank, doc_idx in enumerate(sparse_idx):
        if doc_idx < 0 or sparse_scores[rank] == 0.0:
            continue
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
        
    # Sort by RRF score descending
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return just the indices
    return [doc_idx for doc_idx, score in sorted_results[:top_k]]
