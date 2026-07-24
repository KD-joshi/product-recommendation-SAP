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

logger = logging.getLogger(__name__)


class ProductSimilaritySearch:
    """
    End-to-end product similarity search system.
    
    Supports three search modes:
    1. "text_structured" — text embeddings + structured features (fast, no images)
    2. "image" — CLIP image embeddings only (visual similarity)
    3. "combined" — weighted fusion of all features (best quality, needs images)
    
    The default mode is "text_structured" since it doesn't require downloading
    30k images and gives excellent results for fashion products.
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
        
        # Encoders (needed for transform at query time if needed)
        self.struct_encoder: Optional[StructuredFeatureEncoder] = None
        self.text_embedder: Optional[TextEmbedder] = None
        
        self._initialized = False
    
    def initialize(
        self,
        data_path: str = None,
        build_image_index: bool = False,
        force_image_rebuild: bool = False,
        use_hnsw: bool = True,
    ):
        """
        Initialize the search system: load data, build embeddings, build indices.
        
        This is the "offline" step. In production, you'd run this once and save
        the results. The FastAPI app then loads the pre-built indices on startup.
        
        Args:
            data_path: Path to the LDJSON file
            build_image_index: Whether to download images and build CLIP embeddings
            use_hnsw: If True, use HNSW index. If False, use brute-force.
        """
        start = time.perf_counter()
        
        # Step 1: Load and clean data
        logger.info("=" * 60)
        logger.info("STEP 1: Loading data...")
        logger.info("=" * 60)
        self.df = load_and_clean_data(data_path)
        self.product_ids = self.df['uniq_id'].values
        self.id_to_idx = {pid: idx for idx, pid in enumerate(self.product_ids)}
        
        # Step 2: Build structured features
        logger.info("=" * 60)
        logger.info("STEP 2: Building structured features...")
        logger.info("=" * 60)
        self.struct_encoder = StructuredFeatureEncoder()
        self.structured_features = self.struct_encoder.fit_transform(self.df)
        
        # Step 3: Build text embeddings
        logger.info("=" * 60)
        logger.info("STEP 3: Building text embeddings...")
        logger.info("=" * 60)
        self.text_embedder = TextEmbedder()
        self.text_embeddings = self.text_embedder.encode(self.df)
        
        # Step 4: Build combined text + structured features
        logger.info("=" * 60)
        logger.info("STEP 4: Combining features...")
        logger.info("=" * 60)
        text_struct = build_combined_features(
            structured=self.structured_features,
            text_embeddings=self.text_embeddings,
            image_embeddings=None,  # No images for now
        )
        self.combined_features = text_struct
        
        # Step 5: Build FAISS index
        logger.info("=" * 60)
        logger.info("STEP 5: Building FAISS index...")
        logger.info("=" * 60)
        dim = self.combined_features.shape[1]
        engine = FAISSEngine(dimension=dim, use_hnsw=use_hnsw)
        engine.build(self.combined_features)
        self.engines["text_structured"] = engine
        
        # Step 6: Optionally build image index
        if build_image_index:
            img_path = os.path.join(config.INDEX_DIR, "image_embeddings.npy")
            if not force_image_rebuild and os.path.exists(img_path):
                logger.info("=" * 60)
                logger.info("STEP 6: Loading existing image embeddings from disk (skipping rebuild)...")
                logger.info("=" * 60)
                self.image_embeddings = np.load(img_path)
            else:
                logger.info("=" * 60)
                logger.info("STEP 6: Building image embeddings (this takes a while)...")
                logger.info("=" * 60)
                image_embedder = ImageEmbedder()
                image_urls = self.df['image_url'].tolist()
                self.image_embeddings = image_embedder.encode_from_urls(image_urls)
            
            # Image-only index
            img_engine = FAISSEngine(
                dimension=config.IMAGE_EMBEDDING_DIM, use_hnsw=use_hnsw
            )
            img_engine.build(normalize_l2(self.image_embeddings))
            self.engines["image"] = img_engine
            
            # Combined (text + structured + image) index
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
        
        elapsed = time.perf_counter() - start
        self._initialized = True
        logger.info(f"✅ Initialization complete in {elapsed:.1f}s")
        logger.info(f"   Products indexed: {len(self.product_ids)}")
        logger.info(f"   Available modes: {list(self.engines.keys())}")
    
    @lru_cache(maxsize=config.LRU_CACHE_SIZE)
    def find_similar_products(
        self,
        product_id: str,
        num_similar: int,
        mode: str = "text_structured",
    ) -> List[str]:
        """
        Find the most similar products to the given product_id.

        Args:
            product_id: The uniq_id of the query product.
            num_similar: Number of similar products to return.
            mode: Search mode — "text_structured", "image", or "combined".

        Returns:
            List of product IDs sorted by descending similarity.
        """
        if not self._initialized:
            raise RuntimeError("Must call initialize() first")
        
        if product_id not in self.id_to_idx:
            raise ValueError(f"Product ID '{product_id}' not found in dataset")
        
        if mode not in self.engines:
            available = list(self.engines.keys())
            raise ValueError(f"Mode '{mode}' not available. Available modes: {available}")
        
        # Get the query product's index
        query_idx = self.id_to_idx[product_id]
        
        # Get the query vector from the appropriate feature set
        engine = self.engines[mode]
        
        # We need to get the right feature vector for this mode
        if mode == "text_structured":
            query_vector = build_combined_features(
                structured=self.structured_features[query_idx:query_idx+1],
                text_embeddings=self.text_embeddings[query_idx:query_idx+1],
                image_embeddings=None,
            )[0]
        elif mode == "image":
            query_vector = normalize_l2(self.image_embeddings[query_idx:query_idx+1])[0]
        elif mode == "combined":
            # The combined engine has its own feature set
            query_vector = build_combined_features(
                structured=self.structured_features[query_idx:query_idx+1],
                text_embeddings=self.text_embeddings[query_idx:query_idx+1],
                image_embeddings=self.image_embeddings[query_idx:query_idx+1] if self.image_embeddings is not None else None,
            )[0]
        else:
            query_vector = self.combined_features[query_idx]
        
        # Search for k+1 neighbors (to exclude the query product itself)
        distances, indices = engine.search_single(query_vector, k=num_similar + 1)
        
        # Exclude the query product and collect results
        results = []
        for idx, dist in zip(indices, distances):
            if idx == query_idx:
                continue
            if idx < 0:  # FAISS returns -1 for invalid results
                continue
            results.append(self.product_ids[idx])
            if len(results) >= num_similar:
                break
        
        return results
    
    @lru_cache(maxsize=config.LRU_CACHE_SIZE)
    def calculate_similarity(
        self,
        product_id: str,
        mode: str = "text_structured",
        top_k: int = 100,
    ) -> pd.DataFrame:
        """
        Calculate similarity scores between a product and its nearest neighbors.
        
        Returns a DataFrame with columns: [uniq_id, similarity_score, product_name, brand]
        Useful for inspecting and debugging results.
        """
        if not self._initialized:
            raise RuntimeError("Must call initialize() first")
        
        query_idx = self.id_to_idx[product_id]
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
        
        distances, indices = engine.search_single(query_vector, k=top_k + 1)
        
        rows = []
        for idx, dist in zip(indices, distances):
            if idx == query_idx or idx < 0:
                continue
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
        """Save all indices and encoders to disk for later loading."""
        index_dir = index_dir or config.INDEX_DIR
        os.makedirs(index_dir, exist_ok=True)
        
        # Save FAISS indices
        for mode, engine in self.engines.items():
            engine.save(os.path.join(index_dir, f"faiss_{mode}.index"))
        
        # Save structured encoder
        self.struct_encoder.save(os.path.join(index_dir, "struct_encoder.pkl"))
        
        # Save feature vectors and product IDs
        np.save(os.path.join(index_dir, "product_ids.npy"), self.product_ids)
        np.save(os.path.join(index_dir, "combined_features.npy"), self.combined_features)
        np.save(os.path.join(index_dir, "text_embeddings.npy"), self.text_embeddings)
        np.save(os.path.join(index_dir, "structured_features.npy"), self.structured_features)
        
        if self.image_embeddings is not None:
            np.save(os.path.join(index_dir, "image_embeddings.npy"), self.image_embeddings)
        
        # Save DataFrame (for serving product details)
        self.df.to_parquet(os.path.join(index_dir, "products.parquet"))
        
        logger.info(f"✅ All indices and data saved to {index_dir}/")
    
    def load(self, index_dir: str = None):
        """Load pre-built indices from disk (fast startup for API)."""
        index_dir = index_dir or config.INDEX_DIR
        
        # Load product data
        self.df = pd.read_parquet(os.path.join(index_dir, "products.parquet"))
        self.product_ids = np.load(os.path.join(index_dir, "product_ids.npy"), allow_pickle=True)
        self.id_to_idx = {pid: idx for idx, pid in enumerate(self.product_ids)}
        
        # Load feature vectors
        self.combined_features = np.load(os.path.join(index_dir, "combined_features.npy"))
        self.text_embeddings = np.load(os.path.join(index_dir, "text_embeddings.npy"))
        self.structured_features = np.load(os.path.join(index_dir, "structured_features.npy"))
        
        img_path = os.path.join(index_dir, "image_embeddings.npy")
        if os.path.exists(img_path):
            self.image_embeddings = np.load(img_path)
        
        # Load structured encoder
        self.struct_encoder = StructuredFeatureEncoder.load(
            os.path.join(index_dir, "struct_encoder.pkl")
        )
        
        # Load FAISS indices
        for mode_file in Path(index_dir).glob("faiss_*.index"):
            mode = mode_file.stem.replace("faiss_", "")
            dim = self.combined_features.shape[1]
            
            # Detect dimension from the index file
            engine = FAISSEngine(dimension=1, use_hnsw=True)  # dim corrected on load
            engine.load(str(mode_file))
            self.engines[mode] = engine
        
        self._initialized = True
        logger.info(f"✅ Loaded from {index_dir}/")
        logger.info(f"   Products: {len(self.product_ids)}")
        logger.info(f"   Available modes: {list(self.engines.keys())}")


# ==============================================================================
# Module-level convenience functions
# ==============================================================================

# Global instance (initialized once, shared across API calls)
_search_instance: Optional[ProductSimilaritySearch] = None


def _get_instance() -> ProductSimilaritySearch:
    """Get or create the global search instance."""
    global _search_instance
    if _search_instance is None:
        _search_instance = ProductSimilaritySearch()
        
        # Try loading pre-built indices first (fast)
        index_dir = config.INDEX_DIR
        if os.path.exists(os.path.join(index_dir, "faiss_text_structured.index")):
            logger.info("Loading pre-built indices...")
            _search_instance.load(index_dir)
        else:
            logger.info("No pre-built indices found, building from scratch...")
            _search_instance.initialize()
            _search_instance.save(index_dir)
    
    return _search_instance


def find_similar_products(product_id: str, num_similar: int) -> List[str]:
    """
    Find products similar to the given product.

    Args:
        product_id: The uniq_id of the query product.
        num_similar: Number of similar products to return.

    Returns:
        List of product IDs, most similar first.
    """
    instance = _get_instance()
    return instance.find_similar_products(product_id, num_similar)


def calculate_similarity() -> None:
    """
    Placeholder for explicit pairwise similarity computation.

    In this implementation, similarity is computed implicitly via
    FAISS vector search inside find_similar_products(). For detailed
    similarity scores, use ProductSimilaritySearch.calculate_similarity().
    """
    pass


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    
    print("Initializing product similarity search...")
    search = ProductSimilaritySearch()
    search.initialize()
    
    # Get a random product ID for testing
    test_id = search.product_ids[0]
    print(f"\n🔍 Finding products similar to: {test_id}")
    print(f"   Product: {search.df.iloc[0]['product_name']}")
    print(f"   Brand: {search.df.iloc[0]['brand']}")
    
    similar = search.find_similar_products(test_id, num_similar=5)
    print(f"\n📋 Top 5 similar products:")
    for i, pid in enumerate(similar, 1):
        idx = search.id_to_idx[pid]
        row = search.df.iloc[idx]
        print(f"   {i}. [{pid[:8]}...] {row['product_name'][:60]}")
        print(f"      Brand: {row['brand']}, Price: ₹{row['sales_price']:.0f}, Rating: {row['rating']}")
    
    # Benchmark
    print(f"\n⚡ Benchmarking...")
    results = search.engines["text_structured"].benchmark(
        search.combined_features, k=10, n_queries=100
    )
    print(f"   Latency (p50): {results['latency_p50_ms']:.2f}ms")
    print(f"   Latency (p95): {results['latency_p95_ms']:.2f}ms")
    if 'recall_at_k' in results:
        print(f"   Recall@10: {results['recall_at_k']:.4f}")
    
    # Save indices
    search.save()
    print(f"\n✅ Indices saved to {config.INDEX_DIR}/")
