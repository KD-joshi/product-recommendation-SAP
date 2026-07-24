import time
import random
import numpy as np
from similarity_search import ProductSimilaritySearch
import config
import logging

# Disable verbose logging from the main app for the benchmark
logging.getLogger("similarity_search").setLevel(logging.WARNING)
logging.getLogger("similarity_engine").setLevel(logging.WARNING)

def run_benchmark(num_queries=50):
    print("=" * 60)
    print("  MULTIMODAL SEARCH BENCHMARK (LATENCY)")
    print("=" * 60)
    
    print("Loading indices into memory... (this takes a few seconds)")
    search = ProductSimilaritySearch()
    search.load(config.INDEX_DIR)
    
    modes = ['text_structured', 'image', 'combined']
    top_k = 10
    
    # We will pick random products from the dataset as query targets
    random.seed(42)
    sample_ids = random.sample(list(search.product_ids), num_queries)
    
    print(f"Running {num_queries} random queries per mode for Top-10 results...")
    print("-" * 60)
    print(f"{'Mode':<20} | {'p50 (ms)':<10} | {'p95 (ms)':<10} | {'p99 (ms)':<10}")
    print("-" * 60)
    
    results = {}
    
    for mode in modes:
        latencies = []
        
        # Warm-up phase (10 queries) to populate CPU caches
        for pid in sample_ids[:10]:
            search.find_similar_products(pid, top_k, mode=mode)
            
        # Actual benchmark
        for pid in sample_ids:
            start = time.perf_counter()
            search.find_similar_products(pid, top_k, mode=mode)
            end = time.perf_counter()
            latencies.append((end - start) * 1000) # Convert to ms
            
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        results[mode] = {
            'p50': p50,
            'p95': p95,
            'p99': p99
        }
        
        print(f"{mode:<20} | {p50:<10.3f} | {p95:<10.3f} | {p99:<10.3f}")
        
    print("=" * 60)
    print("Benchmark complete!")

if __name__ == "__main__":
    run_benchmark()
