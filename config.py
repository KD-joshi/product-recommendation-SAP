"""
System Configuration.

Centralized configuration for hyperparameters, file paths, and environment settings.
Values can be overridden via environment variables to support different environments
(development, staging, production) without requiring code modifications.
"""

import os
from pathlib import Path

# ==============================================================================
# PATHS
# ==============================================================================
BASE_DIR = Path(__file__).parent
DATA_PATH = os.getenv(
    "DATA_PATH",
    str(BASE_DIR / "data" / "marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson")
)
INDEX_DIR = os.getenv("INDEX_DIR", str(BASE_DIR / "indices"))
IMAGE_CACHE_DIR = os.getenv("IMAGE_CACHE_DIR", str(BASE_DIR / "data" / "images"))

# ==============================================================================
# FAISS HNSW PARAMETERS
# ==============================================================================
# M: bidirectional links per node. Higher = better recall, more memory.
# efConstruction: candidate list size during graph build. Higher = better graph.
# efSearch: candidate list size during query. Tunable at runtime.
HNSW_M = int(os.getenv("HNSW_M", "32"))
HNSW_EF_CONSTRUCTION = int(os.getenv("HNSW_EF_CONSTRUCTION", "200"))
HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "64"))

# ==============================================================================
# EMBEDDING MODELS
# ==============================================================================
# Text: Sentence-BERT (all-MiniLM-L6-v2), 384-dimensional embeddings.
TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
TEXT_EMBEDDING_DIM = 384

# Image: CLIP (ViT-B/32), 512-dimensional embeddings.
IMAGE_MODEL_NAME = os.getenv("IMAGE_MODEL_NAME", "openai/clip-vit-base-patch32")
IMAGE_EMBEDDING_DIM = 512

# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================
# Keep top-N brands/colors as individual features; bucket the rest as "Other".
TOP_N_BRANDS = int(os.getenv("TOP_N_BRANDS", "100"))
TOP_N_COLORS = int(os.getenv("TOP_N_COLORS", "30"))

# Sentinel value for unknown weight in the dataset.
WEIGHT_SENTINEL = 999999999

# ==============================================================================
# SIMILARITY SEARCH
# ==============================================================================
# Late-fusion weights for combining text+structured and image scores.
TEXT_STRUCT_WEIGHT = float(os.getenv("TEXT_STRUCT_WEIGHT", "0.4"))
IMAGE_WEIGHT = float(os.getenv("IMAGE_WEIGHT", "0.6"))

# ==============================================================================
# CACHING
# ==============================================================================
# LRU cache size for find_similar_products results.
LRU_CACHE_SIZE = int(os.getenv("LRU_CACHE_SIZE", "1024"))

# ==============================================================================
# DIMENSIONALITY REDUCTION
# ==============================================================================
# Optional PCA compression of the combined feature vector.
USE_PCA = os.getenv("USE_PCA", "false").lower() == "true"
PCA_COMPONENTS = int(os.getenv("PCA_COMPONENTS", "256"))

# ==============================================================================
# API
# ==============================================================================
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
