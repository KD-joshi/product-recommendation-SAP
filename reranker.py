"""
Reranker Module

Implements a Stage-2 Cross-Encoder reranker.
Takes the top-K candidates from the fast retriever (FAISS or Hybrid)
and scores them with full cross-attention for maximum accuracy.
"""
import logging
from typing import List, Tuple
import pandas as pd
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class Stage2Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        
    def load(self):
        """Lazy load the CrossEncoder model to save memory until needed."""
        if self.model is None:
            logger.info(f"Loading CrossEncoder: {self.model_name}")
            self.model = CrossEncoder(self.model_name, max_length=512)
            logger.info("CrossEncoder loaded successfully.")
            
    def rerank(self, query_text: str, candidate_texts: List[str], candidate_ids: List[str], top_k: int = 10) -> List[str]:
        """
        Rerank a list of candidates against the query text.
        Returns the top-K product IDs.
        """
        if not candidate_texts or not candidate_ids:
            return []
            
        self.load()
        
        # Create pairs: (query, candidate1), (query, candidate2)...
        cross_inp = [[query_text, doc] for doc in candidate_texts]
        
        # Score the pairs
        scores = self.model.predict(cross_inp)
        
        # Sort by score descending
        # Zip scores with IDs
        scored_candidates = list(zip(scores, candidate_ids))
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Return only the IDs of the top_k
        return [candidate_id for score, candidate_id in scored_candidates[:top_k]]
