"""
Offline Index Builder.

This script pre-computes product embeddings and FAISS indices,
saving them to disk for fast loading by the API service.

Pipeline:
1. Load and clean the product dataset.
2. Generate text embeddings via Sentence-BERT.
3. Generate structured features (MinMax scaling + OneHot encoding).
4. Combine features into unified product vectors.
5. Build and benchmark FAISS indices (HNSW and Brute-force).
6. Save indices and artifacts to disk.

Usage:
    python build_index.py
"""

import json
import logging
import os
import time

import numpy as np

import config
from similarity_search import ProductSimilaritySearch
from similarity_engine import FAISSEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    print("=" * 70)
    print("  PRODUCT SIMILARITY SEARCH — INDEX BUILDER")
    print("=" * 70)
    
    total_start = time.perf_counter()
    
    # ====================================================================
    # STEP 1: Build the main index (HNSW)
    # ====================================================================
    print("\n📦 Building main index with FAISS HNSW...")
    search = ProductSimilaritySearch()
    search.initialize(use_hnsw=True, build_image_index=True)
    
    # ====================================================================
    # STEP 2: Benchmark HNSW vs Brute-Force
    # ====================================================================
    # This demonstrates Part 3 (Bonus) — we show that HNSW gives the same
    # results as brute-force but faster (especially at scale).
    print("\n⚡ Benchmarking HNSW vs Brute-Force...")
    
    vectors = search.combined_features
    dim = vectors.shape[1]
    
    # Benchmark HNSW
    hnsw_results = search.engines["text_structured"].benchmark(
        vectors, k=10, n_queries=200
    )
    
    # Benchmark brute-force
    print("\n  Building brute-force index for comparison...")
    bf_engine = FAISSEngine(dimension=dim, use_hnsw=False)
    bf_engine.build(vectors)
    bf_results = bf_engine.benchmark(vectors, k=10, n_queries=200)
    
    # Print comparison
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS")
    print("=" * 70)
    print(f"\n  {'Metric':<25} {'HNSW':>15} {'Brute-Force':>15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15}")
    print(f"  {'Build time (s)':<25} {hnsw_results['build_time_s']:>15.3f} {bf_results['build_time_s']:>15.3f}")
    print(f"  {'Latency p50 (ms)':<25} {hnsw_results['latency_p50_ms']:>15.3f} {bf_results['latency_p50_ms']:>15.3f}")
    print(f"  {'Latency p95 (ms)':<25} {hnsw_results['latency_p95_ms']:>15.3f} {bf_results['latency_p95_ms']:>15.3f}")
    print(f"  {'Latency p99 (ms)':<25} {hnsw_results['latency_p99_ms']:>15.3f} {bf_results['latency_p99_ms']:>15.3f}")
    if 'recall_at_k' in hnsw_results:
        print(f"  {'Recall@10':<25} {hnsw_results['recall_at_k']:>15.4f} {'1.0000':>15}")
    print()
    
    # ====================================================================
    # STEP 3: Save benchmark results
    # ====================================================================
    os.makedirs(config.INDEX_DIR, exist_ok=True)
    benchmark_path = os.path.join(config.INDEX_DIR, "benchmark_results.json")
    with open(benchmark_path, 'w') as f:
        json.dump({
            "hnsw": hnsw_results,
            "brute_force": bf_results,
            "hnsw_params": {
                "M": config.HNSW_M,
                "efConstruction": config.HNSW_EF_CONSTRUCTION,
                "efSearch": config.HNSW_EF_SEARCH,
            }
        }, f, indent=2)
    print(f"  Benchmark results saved to {benchmark_path}")
    
    # ====================================================================
    # STEP 4: Save indices
    # ====================================================================
    search.save()
    
    # ====================================================================
    # STEP 5: Validate with sample queries
    # ====================================================================
    print("\n" + "=" * 70)
    print("  SAMPLE RESULTS (Sanity Check)")
    print("=" * 70)
    
    # Pick 3 diverse products for validation
    test_indices = [0, len(search.product_ids)//2, len(search.product_ids)-1]
    
    for test_idx in test_indices:
        test_id = search.product_ids[test_idx]
        row = search.df.iloc[test_idx]
        
        print(f"\n  🔍 Query: {row['product_name'][:60]}")
        print(f"     Brand: {row['brand']}, Price: ₹{row.get('sales_price', 'N/A')}, "
              f"Rating: {row.get('rating', 'N/A')}")
        
        similar = search.find_similar_products(test_id, num_similar=3)
        
        for i, pid in enumerate(similar, 1):
            idx = search.id_to_idx[pid]
            s_row = search.df.iloc[idx]
            print(f"     {i}. {s_row['product_name'][:55]}")
            print(f"        Brand: {s_row['brand']}, Price: ₹{s_row.get('sales_price', 'N/A')}, "
                  f"Rating: {s_row.get('rating', 'N/A')}")
    
    # ====================================================================
    # DONE
    # ====================================================================
    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'=' * 70}")
    print(f"  ✅ ALL DONE in {total_elapsed:.1f}s")
    print(f"  Index directory: {config.INDEX_DIR}/")
    print(f"  Start the API: uvicorn app:app --host 0.0.0.0 --port 8000")
    print(f"  Swagger docs: http://localhost:8000/docs")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
