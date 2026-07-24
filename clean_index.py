import os
import numpy as np
import pandas as pd
import config
from similarity_search import ProductSimilaritySearch

print("Loading existing vectors...")
idx_dir = config.INDEX_DIR

image_embeddings = np.load(os.path.join(idx_dir, "image_embeddings.npy"))
text_embeddings = np.load(os.path.join(idx_dir, "text_embeddings.npy"))
structured_features = np.load(os.path.join(idx_dir, "structured_features.npy"))
product_ids = np.load(os.path.join(idx_dir, "product_ids.npy"), allow_pickle=True)
df = pd.read_parquet(os.path.join(idx_dir, "products.parquet"))

# Find valid indices (image vectors that are not completely zero)
# The norm of a zero vector is 0
norms = np.linalg.norm(image_embeddings, axis=1)
valid_mask = norms > 1e-5

num_total = len(norms)
num_valid = np.sum(valid_mask)
print(f"Found {num_total - num_valid} dead links (zero vectors).")
print(f"Keeping {num_valid} valid products.")

# Slice arrays
print("Slicing arrays...")
image_embeddings = image_embeddings[valid_mask]
text_embeddings = text_embeddings[valid_mask]
structured_features = structured_features[valid_mask]
product_ids = product_ids[valid_mask]
df = df[valid_mask].reset_index(drop=True)

# Save back
print("Overwriting cleaned vectors to disk...")
np.save(os.path.join(idx_dir, "image_embeddings.npy"), image_embeddings)
np.save(os.path.join(idx_dir, "text_embeddings.npy"), text_embeddings)
np.save(os.path.join(idx_dir, "structured_features.npy"), structured_features)
np.save(os.path.join(idx_dir, "product_ids.npy"), product_ids)
df.to_parquet(os.path.join(idx_dir, "products.parquet"))

print("Rebuilding FAISS and BM25 indices from the cleaned vectors...")
# Initialize search engine but manually load arrays and build engines
search = ProductSimilaritySearch()
search.product_ids = product_ids
search.id_to_idx = {pid: idx for idx, pid in enumerate(product_ids)}
search.df = df

from feature_engine import StructuredFeatureEncoder, TextEmbedder
search.struct_encoder = StructuredFeatureEncoder.load(os.path.join(idx_dir, "struct_encoder.pkl"))
search.text_embedder = TextEmbedder()

search.structured_features = structured_features
search.text_embeddings = text_embeddings
search.image_embeddings = image_embeddings

from feature_engine import build_combined_features
text_struct = build_combined_features(
    structured=structured_features,
    text_embeddings=text_embeddings,
    image_embeddings=None,
)
search.combined_features = text_struct

from similarity_engine import FAISSEngine
from hybrid_search import BM25Engine
from feature_engine import normalize_l2

print("Building FAISS Text+Structured...")
engine = FAISSEngine(dimension=text_struct.shape[1], use_hnsw=True)
engine.build(text_struct)
search.engines["text_structured"] = engine

print("Building FAISS Image...")
img_engine = FAISSEngine(dimension=config.IMAGE_EMBEDDING_DIM, use_hnsw=True)
img_engine.build(normalize_l2(image_embeddings))
search.engines["image"] = img_engine

print("Building FAISS Combined...")
combined_all = build_combined_features(
    structured=structured_features,
    text_embeddings=text_embeddings,
    image_embeddings=image_embeddings,
)
combined_engine = FAISSEngine(dimension=combined_all.shape[1], use_hnsw=True)
combined_engine.build(combined_all)
search.engines["combined"] = combined_engine

print("Building BM25...")
bm25_docs = (df['product_name'] + " " + df['brand'] + " " + df['categories']).tolist()
bm25_engine = BM25Engine()
bm25_engine.build(bm25_docs)
search.bm25_engine = bm25_engine

search._initialized = True
search.save(idx_dir)

print("Done! The index is now perfectly clean and saved.")
