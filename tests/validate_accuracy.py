import time
import numpy as np
import sys
import os

# Add parent directory to path so we can import from the main project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from similarity_search import ProductSimilaritySearch

def validate_cross_modal_agreement(num_samples=100, top_k=10):
    print(f"Loading search engine to validate {num_samples} random products...")
    search = ProductSimilaritySearch()
    search.load("indices")
    
    # Pick random products
    np.random.seed(42)
    sample_indices = np.random.choice(len(search.product_ids), num_samples, replace=False)
    sample_ids = [search.product_ids[idx] for idx in sample_indices]
    
    total_jaccard = 0.0
    
    print("\nStarting Cross-Modal Agreement Validation...")
    print(f"Comparing Top-{top_k} results of Image Engine vs Text Engine")
    print("-" * 50)
    
    for pid in sample_ids:
        text_results = search.find_similar_products(pid, top_k, mode="text_structured")
        image_results = search.find_similar_products(pid, top_k, mode="image")
        
        set_text = set(text_results)
        set_image = set(image_results)
        
        intersection = len(set_text.intersection(set_image))
        union = len(set_text.union(set_image))
        
        jaccard = intersection / union if union > 0 else 0
        total_jaccard += jaccard
        
    avg_jaccard = total_jaccard / num_samples
    
    print(f"\nValidation Complete!")
    print(f"Average Cross-Modal Agreement (Jaccard Index): {avg_jaccard:.4f}")
    print("\nWhat does this mean?")
    print("- 0.0 means the Image and Text engines find completely different items.")
    print("- 1.0 means both engines find the exact same items.")
    print("- In unsupervised E-commerce, an agreement of 0.1 to 0.3 is considered excellent because text and images capture distinctly different properties (visual vs semantic).")
    
if __name__ == "__main__":
    validate_cross_modal_agreement()
