"""
Feature Engineering Module.

Converts raw product data into numerical vectors suitable for similarity search.
Combines multiple feature modalities into a unified representation:

1. Structured Features:
   - Numerical data (price, rating, discount_percentage) normalized using MinMaxScaler.
   - Categorical data (brand, color, delivery type) processed via OneHotEncoder.
   - Binary flags (is_prime, is_bestseller).

2. Text Embeddings:
   - Semantic representations of product descriptions generated using Sentence-BERT
     (all-MiniLM-L6-v2) for robust text matching.

Vectors are L2-normalized to ensure compatibility with inner-product search
engines (e.g., FAISS IndexFlatIP or IndexHNSWFlat) for cosine similarity.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import config

logger = logging.getLogger(__name__)


class StructuredFeatureEncoder:
    """
    Encodes structured product attributes into a numerical vector.
    
    Handles three types of features:
    1. Numerical (price, rating, weight) → MinMaxScaler
    2. Categorical (brand, color, delivery_type) → OneHotEncoder (top-N + "Other")
    3. Binary (is_prime, is_bestseller) → pass-through
    """
    
    # Note: we exclude 'weight' as it's all NaN in this dataset (sentinel 999999999)
    NUMERICAL_COLS = ['sales_price', 'rating', 'discount_percentage', 'no_of_reviews']
    BINARY_COLS = ['is_prime', 'is_bestseller']
    
    def __init__(self, top_n_brands: int = None, top_n_colors: int = None):
        self.top_n_brands = top_n_brands or config.TOP_N_BRANDS
        self.top_n_colors = top_n_colors or config.TOP_N_COLORS
        
        self.num_scaler = MinMaxScaler()
        self.brand_encoder = OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist')
        self.color_encoder = OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist')
        self.delivery_encoder = OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist')
        
        self._top_brands = None
        self._top_colors = None
        self._is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'StructuredFeatureEncoder':
        """
        Learn the encoding parameters from the data.
        
        This must be called ONCE on the full dataset before transform().
        We learn: min/max for numerical features, top-N brands/colors.
        """
        logger.info("Fitting structured feature encoder...")
        
        # Numerical: fill NaN with median, then fit scaler
        num_data = df[self.NUMERICAL_COLS].copy()
        for col in self.NUMERICAL_COLS:
            median_val = num_data[col].median()
            num_data[col] = num_data[col].fillna(median_val)
        self.num_scaler.fit(num_data)
        self._numerical_medians = {col: num_data[col].median() for col in self.NUMERICAL_COLS}
        
        # Brands: keep top N, bucket rest as "Other"
        brand_counts = df['brand'].value_counts()
        self._top_brands = set(brand_counts.head(self.top_n_brands).index)
        brands_bucketed = df['brand'].astype(str).apply(
            lambda x: x if x in self._top_brands else "Other"
        ).to_numpy().reshape(-1, 1)
        self.brand_encoder.fit(brands_bucketed)
        
        # Colors: keep top N, bucket rest as "Other"
        color_counts = df['color'].value_counts()
        self._top_colors = set(color_counts.head(self.top_n_colors).index)
        colors_bucketed = df['color'].astype(str).apply(
            lambda x: x if x in self._top_colors else "Other"
        ).to_numpy().reshape(-1, 1)
        self.color_encoder.fit(colors_bucketed)
        
        # Delivery type: few unique values, encode all
        self.delivery_encoder.fit(df[['delivery_type']].astype(str))
        
        self._is_fitted = True
        logger.info(f"Fitted: {len(self._top_brands)} brands, {len(self._top_colors)} colors")
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform product data into a structured feature vector.
        
        Returns: numpy array of shape (n_products, n_structured_features)
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before transform()")
        
        # Numerical features
        num_data = df[self.NUMERICAL_COLS].copy()
        for col in self.NUMERICAL_COLS:
            num_data[col] = num_data[col].fillna(self._numerical_medians[col])
        num_features = self.num_scaler.transform(num_data)
        
        # Brand (one-hot)
        brands_bucketed = df['brand'].astype(str).apply(
            lambda x: x if x in self._top_brands else "Other"
        ).to_numpy().reshape(-1, 1)
        brand_features = self.brand_encoder.transform(brands_bucketed)
        
        # Color (one-hot)
        colors_bucketed = df['color'].astype(str).apply(
            lambda x: x if x in self._top_colors else "Other"
        ).to_numpy().reshape(-1, 1)
        color_features = self.color_encoder.transform(colors_bucketed)
        
        # Delivery type (one-hot)
        delivery_features = self.delivery_encoder.transform(df[['delivery_type']].astype(str))
        
        # Binary features
        binary_features = df[self.BINARY_COLS].values.astype(np.float32)
        
        # Concatenate all structured features
        structured = np.hstack([
            num_features,
            brand_features,
            color_features,
            delivery_features,
            binary_features
        ]).astype(np.float32)
        
        logger.info(f"Structured features shape: {structured.shape}")
        return structured
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Convenience method: fit + transform in one call."""
        return self.fit(df).transform(df)
    
    def save(self, path: str):
        """Save the fitted encoder to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Saved structured encoder to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'StructuredFeatureEncoder':
        """Load a fitted encoder from disk."""
        with open(path, 'rb') as f:
            encoder = pickle.load(f)
        logger.info(f"Loaded structured encoder from {path}")
        return encoder


class TextEmbedder:
    """
    Generates semantic text embeddings using Sentence-Transformers.

    Combines multiple product text fields (name, brand, color, categories,
    meta_keywords) into a single string per product, then encodes it into
    a 384-dimensional L2-normalised vector via all-MiniLM-L6-v2.

    References:
        Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using
        Siamese BERT-Networks", EMNLP 2019.
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.TEXT_MODEL_NAME
        self._model = None
    
    def _load_model(self):
        """Lazy-load the model (only when first needed)."""
        if self._model is None:
            import os
            import torch
            # Set number of threads for parallel CPU inference
            # Must be set before first use of torch ops
            n_cores = int(os.getenv("OMP_NUM_THREADS", str(os.cpu_count() or 4)))
            torch.set_num_threads(n_cores)
            
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading text model: {self.model_name} ({n_cores} threads)")
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Text model loaded (dim={self._model.get_embedding_dimension()})")
    
    def _build_text(self, df: pd.DataFrame) -> List[str]:
        """
        Combine product fields into a single text string per product.

        Uses explicit prompt tuning (Audience, Category, Item, Color, Brand)
        to maximize Sentence-BERT's semantic separation.
        """
        texts = []
        
        def get_audience(cat_str, name_str):
            text = (str(cat_str) + " " + str(name_str)).lower()
            if any(w in text for w in ['women', 'woman', 'girls', 'girl', 'lady', 'ladies', 'female']):
                return "Women"
            elif any(w in text for w in ['men ', "men'", 'mens', ' man ', 'boys', 'boy', 'male']):
                return "Men"
            return "Unisex"
            
        for _, row in df.iterrows():
            name = str(row.get('product_name', '')).strip()
            brand = str(row.get('brand', '')).strip()
            
            # Color: key fashion signal
            color = str(row.get('color', ''))
            color = '' if color == 'unknown' else color
            
            cats_raw = str(row.get('categories', '')).strip()
            keywords = str(row.get('meta_keywords', ''))[:150].strip()
            
            audience = get_audience(cats_raw, name)
            
            # Formatted prompt tuning structure
            text = f"Audience: {audience} | Category: {cats_raw} | Item: {name} | Color: {color} | Brand: {brand} | Keywords: {keywords}"
            texts.append(text)
        return texts
    
    def encode(self, df: pd.DataFrame, batch_size: int = 512) -> np.ndarray:
        """
        Generate text embeddings for all products.
        
        Args:
            df: DataFrame with text columns
            batch_size: Process this many products at once (memory control)
        
        Returns:
            numpy array of shape (n_products, 384), L2-normalized
        """
        self._load_model()
        
        texts = self._build_text(df)
        logger.info(f"Encoding {len(texts)} product texts (batch_size={batch_size})...")
        
        # SentenceTransformer handles batching internally
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2-normalize so cosine sim = dot product
        )
        
        logger.info(f"Text embeddings shape: {embeddings.shape}")
        return embeddings.astype(np.float32)


class ImageEmbedder:
    """
    Generates image embeddings using OpenAI CLIP.

    CLIP maps images into the same vector space as text, enabling
    cross-modal similarity. Products with visually similar appearances
    produce close vectors regardless of their textual descriptions.

    References:
        Radford et al., "Learning Transferable Visual Models From
        Natural Language Supervision", ICML 2021.
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.IMAGE_MODEL_NAME
        self._model = None
        self._processor = None
    
    def _load_model(self):
        """Lazy-load the CLIP model."""
        if self._model is None:
            from transformers import CLIPModel, CLIPProcessor
            import torch
            
            logger.info(f"Loading image model: {self.model_name}")
            self._model = CLIPModel.from_pretrained(self.model_name)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(self._device)
            self._model.eval()
            logger.info(f"Image model loaded on {self._device}")
    
    def encode_from_urls(
        self, 
        image_urls: List[str], 
        batch_size: int = 32,
        timeout: int = 5
    ) -> np.ndarray:
        """
        Download images from URLs and generate CLIP embeddings.
        
        Products with failed downloads get zero vectors.
        """
        import torch
        import requests
        from PIL import Image
        from io import BytesIO
        from tqdm import tqdm
        
        self._load_model()
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(image_urls), batch_size), desc="Encoding images"):
            batch_urls = image_urls[i:i + batch_size]
            batch_images = []
            batch_valid_indices = []
            
            # Download images concurrently
            def download_image(idx_url):
                j, url = idx_url
                if not url or not url.startswith("http"):
                    return j, None
                try:
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    response = requests.get(url, headers=headers, timeout=timeout, stream=True)
                    img = Image.open(BytesIO(response.content)).convert('RGB')
                    return j, img
                except Exception:
                    return j, None

            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=20) as executor:
                results = executor.map(download_image, enumerate(batch_urls))
                
            for j, img in results:
                if img is not None:
                    batch_images.append(img)
                    batch_valid_indices.append(j)
            
            # Initialize batch embeddings as zeros
            batch_embeddings = np.zeros((len(batch_urls), config.IMAGE_EMBEDDING_DIM), dtype=np.float32)
            
            if batch_images:
                # Process valid images through CLIP
                inputs = self._processor(images=batch_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    features = self._model.get_image_features(**inputs)
                    # Handle different transformers versions (sometimes returns BaseModelOutputWithPooling)
                    if not isinstance(features, torch.Tensor):
                        if hasattr(features, "image_embeds"):
                            features = features.image_embeds
                        elif hasattr(features, "pooler_output"):
                            features = features.pooler_output
                        else:
                            features = features[0]
                    # L2-normalize
                    features = features / features.norm(dim=-1, keepdim=True)
                    features = features.cpu().numpy()
                
                for idx, valid_idx in enumerate(batch_valid_indices):
                    batch_embeddings[valid_idx] = features[idx]
            
            all_embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(all_embeddings)
        
        n_valid = (np.linalg.norm(embeddings, axis=1) > 0).sum()
        logger.info(f"Image embeddings shape: {embeddings.shape}")
        logger.info(f"Valid image embeddings: {n_valid}/{len(embeddings)}")
        
        return embeddings


def normalize_l2(vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalize vectors to unit length.

    Under L2 normalisation, inner product equals cosine similarity,
    which is required by FAISS IndexFlatIP / IndexHNSWFlat.
    Zero-norm vectors (e.g., missing images) are left unchanged.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return (vectors / norms).astype(np.float32)


def build_combined_features(
    structured: np.ndarray,
    text_embeddings: np.ndarray,
    image_embeddings: Optional[np.ndarray] = None,
    text_struct_weight: float = None,
    image_weight: float = None,
) -> np.ndarray:
    """
    Combine feature modalities into a single vector per product via weighted
    late fusion.

    Each feature group is L2-normalised independently before concatenation
    to prevent higher-dimensional groups from dominating. When image
    embeddings are provided, configurable weights control the relative
    contribution of text+structured vs. image features. Products with
    zero-norm image vectors fall back to text+structured similarity.
    """
    if text_struct_weight is None:
        text_struct_weight = config.TEXT_STRUCT_WEIGHT
    if image_weight is None:
        image_weight = config.IMAGE_WEIGHT
    
    # Normalize each group separately
    structured_norm = normalize_l2(structured)
    text_norm = text_embeddings  # Already L2-normalized by SentenceTransformer
    
    # Concatenate text + structured
    text_struct = np.hstack([text_norm, structured_norm])
    text_struct = normalize_l2(text_struct)
    
    if image_embeddings is not None:
        image_norm = normalize_l2(image_embeddings)
        
        # Weighted combination
        combined = (text_struct_weight * text_struct)
        
        # For image: project to same dim as text_struct if needed, or pad/truncate
        # Since we're concatenating (not adding), just concat with weight scaling
        combined = np.hstack([
            text_struct_weight * text_struct,
            image_weight * image_norm
        ])
    else:
        combined = text_struct
    
    # Final L2 normalization
    combined = normalize_l2(combined)
    
    logger.info(f"Combined features shape: {combined.shape}")
    return combined
