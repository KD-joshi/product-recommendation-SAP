"""
Product Similarity Search Interface.

Provides the core `ProductSimilaritySearch` class for querying similar products.
Coordinates data loading, feature extraction, and FAISS-based vector search.

Usage:
    from similarity_search import ProductSimilaritySearch

    search = ProductSimilaritySearch()
    search.initialize() # or search.load(index_dir)
    results = search.find_similar_products("product_id", num_similar=5)
"""

import logging
import os
import pickle
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from data_loader import load_and_clean_data
from feature_engine import (
    StructuredFeatureEncoder,
    TextEmbedder,
    ImageEmbedder,
    build_combined_features,
    normalize_l2,
)
from similarity_engine import FAISSEngine
from hybrid_search import BM25Engine, reciprocal_rank_fusion
from reranker import Stage2Reranker

logger = logging.getLogger(__name__)


class ProductSimilaritySearch:
    """
    End-to-end product similarity search system.
    
    Supports three search modes:
    1. "text_structured" — text embeddings + structured features (fast, no images)
    2. "image" — CLIP image embeddings only (visual similarity)
    3. "combined" — weighted fusion of all features (best quality, needs images)
    """
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.product_ids: Optional[np.ndarray] = None
        self.id_to_idx: Optional[Dict[str, int]] = None
        
        # Feature vectors
        self.structured_features: Optional[np.ndarray] = None
        self.text_embeddings: Optional[np.ndarray] = None
        self.image_embeddings: Optional[np.ndarray] = None
        self.combined_features: Optional[np.ndarray] = None
        
        # FAISS engines (one per mode)
        self.engines: Dict[str, FAISSEngine] = {}
        
        # Encoders
        self.struct_encoder: Optional[StructuredFeatureEncoder] = None
        self.text_embedder: Optional[TextEmbedder] = None
        
        # Advanced IR Components
        self.bm25_engine: Optional[BM25Engine] = None
        self.reranker: Optional[Stage2Reranker] = None
        
        self._initialized = False
    
    def initialize(
        self,
        data_path: str = None,
        build_image_index: bool = False,
        use_hnsw: bool = True,
    ):
        start = time.perf_counter()
        
        logger.info("=" * 60)
        logger.info("STEP 1: Loading data...")
        logger.info("=" * 60)
        self.df = load_and_clean_data(data_path)
        self.product_ids = self.df['uniq_id'].values
        self.id_to_idx = {pid: idx for idx, pid in enumerate(self.product_ids)}
        
        logger.info("=" * 60)
        logger.info("STEP 2: Building structured features...")
        logger.info("=" * 60)
        self.struct_encoder = StructuredFeatureEncoder()
        self.structured_features = self.struct_encoder.fit_transform(self.df)
        
        logger.info("=" * 60)
        logger.info("STEP 3: Building text embeddings...")
        logger.info("=" * 60)
        self.text_embedder = TextEmbedder()
        self.text_embeddings = self.text_embedder.encode(self.df)
        
        logger.info("=" * 60)
        logger.info("STEP 4: Combining features...")
        logger.info("=" * 60)
        text_struct = build_combined_features(
            structured=self.structured_features,
            text_embeddings=self.text_embeddings,
            image_embeddings=None,
        )
        self.combined_features = text_struct
        
        logger.info("=" * 60)
        logger.info("STEP 5: Building FAISS index...")
        logger.info("=" * 60)
        dim = self.combined_features.shape[1]
        engine = FAISSEngine(dimension=dim, use_hnsw=use_hnsw)
        engine.build(self.combined_features)
        self.engines["text_structured"] = engine
        
        if build_image_index:
            logger.info("=" * 60)
            logger.info("STEP 6: Building image embeddings (this takes a while)...")
            logger.info("=" * 60)
            image_embedder = ImageEmbedder()
            image_urls = self.df['image_url'].tolist()
            self.image_embeddings = image_embedder.encode_from_urls(image_urls)
            
            img_engine = FAISSEngine(
                dimension=config.IMAGE_EMBEDDING_DIM, use_hnsw=use_hnsw
            )
            img_engine.build(normalize_l2(self.image_embeddings))
            self.engines["image"] = img_engine
            
            combined_all = build_combined_features(
                structured=self.structured_features,
                text_embeddings=self.text_embeddings,
                image_embeddings=self.image_embeddings,
            )
            combined_engine = FAISSEngine(
                dimension=combined_all.shape[1], use_hnsw=use_hnsw
            )
            combined_engine.build(combined_all)
            self.engines["combined"] = combined_engine

        logger.info("=" * 60)
        logger.info("STEP 7: Building BM25 index...")
        logger.info("=" * 60)
        bm25_docs = (self.df['product_name'] + " " + self.df['brand'] + " " + self.df['categories']).tolist()
        self.bm25_engine = BM25Engine()
        self.bm25_engine.build(bm25_docs)
        
        # Initialize Reranker
        self.reranker = Stage2Reranker()
        
        elapsed = time.perf_counter() - start
        self._initialized = True
        logger.info(f"✅ Initialization complete in {elapsed:.1f}s")
        logger.info(f"   Products indexed: {len(self.product_ids)}")
        logger.info(f"   Available modes: {list(self.engines.keys())}")
        
    def _get_candidates(self, query_idx: int, mode: str, top_k: int) -> Tuple[List[int], List[float]]:
        """Fetch candidates and their scores using Hybrid or Dense search."""
        engine = self.engines[mode]
        
        if mode == "text_structured":
            query_vector = build_combined_features(
                structured=self.structured_features[query_idx:query_idx+1],
                text_embeddings=self.text_embeddings[query_idx:query_idx+1],
                image_embeddings=None,
            )[0]
        elif mode == "image":
            query_vector = normalize_l2(self.image_embeddings[query_idx:query_idx+1])[0]
        elif mode == "combined":
            query_vector = build_combined_features(
                structured=self.structured_features[query_idx:query_idx+1],
                text_embeddings=self.text_embeddings[query_idx:query_idx+1],
                image_embeddings=self.image_embeddings[query_idx:query_idx+1] if self.image_embeddings is not None else None,
            )[0]
        else:
            query_vector = self.combined_features[query_idx]
            
        # Dense Search
        distances, indices = engine.search_single(query_vector, k=min(100, len(self.product_ids)))
        dense_results = (distances, indices)
        
        if mode in ["text_structured", "combined"] and self.bm25_engine:
            # Sparse Search
            row = self.df.iloc[query_idx]
            bm25_query = f"{row['product_name']} {row['brand']} {row['categories']}"
            sparse_scores, sparse_idx = self.bm25_engine.search(bm25_query, top_k=min(100, len(self.product_ids)))
            sparse_results = (sparse_scores, sparse_idx)
            
            # Hybrid RRF
            final_indices = reciprocal_rank_fusion(dense_results, sparse_results, top_k=min(100, len(self.product_ids)))
            # We don't have exact distances for RRF, so we yield dummy scores for compatibility
            final_scores = [0.0] * len(final_indices)
        else:
            final_indices = indices.tolist()
            final_scores = distances.tolist()
            
        # Exclude query product
        clean_indices = []
        clean_scores = []
        for idx, dist in zip(final_indices, final_scores):
            if idx == query_idx or idx < 0:
                continue
            clean_indices.append(idx)
            clean_scores.append(dist)
            
        return clean_indices, clean_scores
    
    @lru_cache(maxsize=config.LRU_CACHE_SIZE)
    def find_similar_products(
        self,
        product_id: str,
        num_similar: int,
        mode: str = "text_structured",
    ) -> List[str]:
        if not self._initialized:
            raise RuntimeError("Must call initialize() first")
        if product_id not in self.id_to_idx:
            raise ValueError(f"Product ID '{product_id}' not found in dataset")
        
        query_idx = self.id_to_idx[product_id]
        
        # Fetch Top-50 candidates
        candidate_indices, _ = self._get_candidates(query_idx, mode, top_k=50)
        
        # Stage-2 Reranking for text/combined
        if mode in ["text_structured", "combined"] and self.reranker:
            row = self.df.iloc[query_idx]
            query_text = f"{row['product_name']} {row['brand']} {row['categories']}"
            
            candidate_texts = []
            candidate_ids = []
            rerank_count = max(config.RERANKER_TOP_K, num_similar)
            for idx in candidate_indices[:rerank_count]:
                c_row = self.df.iloc[idx]
                candidate_texts.append(f"{c_row['product_name']} {c_row['brand']} {c_row['categories']}")
                candidate_ids.append(self.product_ids[idx])
                
            results = self.reranker.rerank(query_text, candidate_texts, candidate_ids, top_k=num_similar)
            return results
        else:
            return [self.product_ids[idx] for idx in candidate_indices[:num_similar]]
    
    @lru_cache(maxsize=config.LRU_CACHE_SIZE)
    def calculate_similarity(
        self,
        product_id: str,
        mode: str = "text_structured",
        top_k: int = 100,
    ) -> pd.DataFrame:
        if not self._initialized:
            raise RuntimeError("Must call initialize() first")
            
        query_idx = self.id_to_idx[product_id]
        candidate_indices, candidate_scores = self._get_candidates(query_idx, mode, top_k=top_k)
        
        # If reranking, we override the scores
        if mode in ["text_structured", "combined"] and self.reranker:
            def get_audience(cat, name):
                text = (str(cat) + " " + str(name)).lower()
                if any(w in text for w in ['women', 'woman', 'girls', 'girl', 'lady', 'ladies', 'female']):
                    return "Women"
                elif any(w in text for w in ['men ', "men'", 'mens', ' man ', 'boys', 'boy', 'male']):
                    return "Men"
                return "Unisex"

            row = self.df.iloc[query_idx]
            q_aud = get_audience(row['categories'], row['product_name'])
            query_text = f"Audience: {q_aud} | Category: {row['categories']} | Item: {row['product_name']} | Color: {row['color']} | Brand: {row['brand']}"
            
            candidate_texts = []
            candidate_ids = []
            rerank_count = max(config.RERANKER_TOP_K, top_k)
            for idx in candidate_indices[:rerank_count]:
                c_row = self.df.iloc[idx]
                c_aud = get_audience(c_row['categories'], c_row['product_name'])
                candidate_texts.append(f"Audience: {c_aud} | Category: {c_row['categories']} | Item: {c_row['product_name']} | Color: {c_row['color']} | Brand: {c_row['brand']}")
                candidate_ids.append(self.product_ids[idx])
                
            self.reranker.load()
            cross_inp = [[query_text, doc] for doc in candidate_texts]
            scores = self.reranker.model.predict(cross_inp)
            
            # Sort by reranker score
            scored_candidates = list(zip(candidate_indices[:rerank_count], scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            candidate_indices = [x[0] for x in scored_candidates]
            candidate_scores = [x[1] for x in scored_candidates]
            
        rows = []
        for idx, dist in zip(candidate_indices[:top_k], candidate_scores[:top_k]):
            rows.append({
                'uniq_id': self.product_ids[idx],
                'similarity_score': float(dist),
                'product_name': self.df.iloc[idx]['product_name'],
                'brand': self.df.iloc[idx]['brand'],
                'sales_price': float(self.df.iloc[idx]['sales_price']) if pd.notna(self.df.iloc[idx]['sales_price']) else None,
                'rating': float(self.df.iloc[idx]['rating']) if pd.notna(self.df.iloc[idx]['rating']) else None,
                'image_url': self.df.iloc[idx]['image_url'] if pd.notna(self.df.iloc[idx]['image_url']) else None
            })
        
        return pd.DataFrame(rows)
    
    def save(self, index_dir: str = None):
        index_dir = index_dir or config.INDEX_DIR
        os.makedirs(index_dir, exist_ok=True)
        
        for mode, engine in self.engines.items():
            engine.save(os.path.join(index_dir, f"faiss_{mode}.index"))
        
        self.struct_encoder.save(os.path.join(index_dir, "struct_encoder.pkl"))
        
        np.save(os.path.join(index_dir, "product_ids.npy"), self.product_ids)
        np.save(os.path.join(index_dir, "combined_features.npy"), self.combined_features)
        np.save(os.path.join(index_dir, "text_embeddings.npy"), self.text_embeddings)
        np.save(os.path.join(index_dir, "structured_features.npy"), self.structured_features)
        
        if self.image_embeddings is not None:
            np.save(os.path.join(index_dir, "image_embeddings.npy"), self.image_embeddings)
            
        if self.bm25_engine:
            self.bm25_engine.save(os.path.join(index_dir, "bm25.pkl"))
        
        self.df.to_parquet(os.path.join(index_dir, "products.parquet"))
        logger.info(f"✅ All indices and data saved to {index_dir}/")
    
    def load(self, index_dir: str = None):
        index_dir = index_dir or config.INDEX_DIR
        
        self.df = pd.read_parquet(os.path.join(index_dir, "products.parquet"))
        self.product_ids = np.load(os.path.join(index_dir, "product_ids.npy"), allow_pickle=True)
        self.id_to_idx = {pid: idx for idx, pid in enumerate(self.product_ids)}
        
        self.combined_features = np.load(os.path.join(index_dir, "combined_features.npy"))
        self.text_embeddings = np.load(os.path.join(index_dir, "text_embeddings.npy"))
        self.structured_features = np.load(os.path.join(index_dir, "structured_features.npy"))
        
        img_path = os.path.join(index_dir, "image_embeddings.npy")
        if os.path.exists(img_path):
            self.image_embeddings = np.load(img_path)
        
        self.struct_encoder = StructuredFeatureEncoder.load(
            os.path.join(index_dir, "struct_encoder.pkl")
        )
        
        for mode_file in Path(index_dir).glob("faiss_*.index"):
            mode = mode_file.stem.replace("faiss_", "")
            engine = FAISSEngine(dimension=1, use_hnsw=True) 
            engine.load(str(mode_file))
            self.engines[mode] = engine
            
        bm25_path = os.path.join(index_dir, "bm25.pkl")
        if os.path.exists(bm25_path):
            self.bm25_engine = BM25Engine()
            self.bm25_engine.load(bm25_path)
            
        self.reranker = Stage2Reranker()
        
        self._initialized = True
        logger.info(f"✅ Loaded from {index_dir}/")

# Global instance
_search_instance: Optional[ProductSimilaritySearch] = None

def _get_instance() -> ProductSimilaritySearch:
    global _search_instance
    if _search_instance is None:
        _search_instance = ProductSimilaritySearch()
        index_dir = config.INDEX_DIR
        if os.path.exists(os.path.join(index_dir, "faiss_text_structured.index")):
            _search_instance.load(index_dir)
        else:
            _search_instance.initialize()
            _search_instance.save(index_dir)
    return _search_instance

def find_similar_products(product_id: str, num_similar: int) -> List[str]:
    instance = _get_instance()
    return instance.find_similar_products(product_id, num_similar)

def calculate_similarity() -> None:
    pass

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    search = ProductSimilaritySearch()
    search.initialize()
    search.save()
