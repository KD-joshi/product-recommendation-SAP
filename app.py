"""
FastAPI Microservice for Product Similarity Search.

This module exposes the find_similar_products functionality as a REST API.
It leverages FastAPI for automatic OpenAPI documentation, request validation,
and high performance asynchronous request handling.

Endpoints:
    GET /find_similar_products?product_id=xxx&num_similar=10
        Returns a list of similar product IDs.

    GET /health
        Returns the service health status.

Lifecycle:
    On startup, the application loads pre-built FAISS indices into memory.
    If indices are not found, it initializes and builds them before serving requests.
    Subsequent queries are served from in-memory indices for low latency.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from similarity_search import ProductSimilaritySearch
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s"
)
logger = logging.getLogger(__name__)

# Global search instance
search: Optional[ProductSimilaritySearch] = None
purged_ids: set = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.

    Loads pre-built FAISS indices from disk on startup. If no indices
    are found, initialises the full pipeline (data loading, embedding,
    index building) before the server starts accepting requests.
    """
    global search, purged_ids
    
    logger.info("🚀 Starting Product Similarity Search API...")
    
    import os, json
    purged_path = os.path.join(config.INDEX_DIR, "purged_ids.json")
    if os.path.exists(purged_path):
        with open(purged_path, 'r') as f:
            purged_ids = set(json.load(f))
            logger.info(f"Loaded {len(purged_ids)} purged product IDs.")
    
    search = ProductSimilaritySearch()
    
    index_path = os.path.join(config.INDEX_DIR, "faiss_text_structured.index")
    
    if os.path.exists(index_path):
        logger.info("📂 Loading pre-built indices from disk...")
        search.load(config.INDEX_DIR)
    else:
        logger.info("🔨 No pre-built indices found. Building from scratch...")
        logger.info("   (This takes ~30s the first time. Future starts will be fast.)")
        search.initialize()
        search.save(config.INDEX_DIR)
    
    logger.info(f"✅ API ready! {len(search.product_ids)} products indexed.")
    
    yield  # Server is running
    
    # Shutdown
    logger.info("👋 Shutting down API...")
    search = None


# Create the FastAPI app
app = FastAPI(
    title="Product Similarity Search API",
    description=(
        "Multimodal product similarity search for Amazon Fashion products. "
        "Uses FAISS HNSW for fast approximate nearest neighbor search with "
        "Sentence-BERT text embeddings, CLIP visual embeddings, and structured feature encoding."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS (Cross-Origin Resource Sharing) for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
)

@app.get("/health")
def health_check():
    """
    Health check endpoint for Kubernetes liveness/readiness probes.
    
    Returns 200 if the service is ready to handle requests.
    Used by K8s to know if the pod is alive and ready.
    """
    if search is None or not search._initialized:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {
        "status": "healthy",
        "products_indexed": len(search.product_ids),
        "available_modes": list(search.engines.keys()),
    }


@app.get("/find_similar_products")
def get_similar_products(
    product_id: str = Query(..., description="The unique ID of the product to find similar products for"),
    num_similar: int = Query(..., gt=0, description="Number of similar products to return"),
    mode: str = Query("text_structured", description="Search mode: text_structured, image, or combined"),
) -> List[str]:
    """
    Find products similar to the given product_id.

    Parameters:
    - **product_id**: The uniq_id of the query product.
    - **num_similar**: How many similar products to return (must be > 0).
    - **mode**: Feature set to use for similarity (default: text_structured).

    Returns:
    - List of product IDs sorted by descending similarity.

    Error codes:
    - 404: Product ID not found in dataset
    - 400: Invalid num_similar value
    - 422: Invalid mode
    - 503: Service not ready
    """
    if search is None or not search._initialized:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # Validate product_id exists
    if product_id not in search.id_to_idx:
        if product_id in purged_ids:
            raise HTTPException(
                status_code=404,
                detail=f"Product ID '{product_id}' not found. Note: This product was intentionally purged from the FAISS index during data cleaning due to a dead image link."
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Product ID '{product_id}' not found in the dataset."
            )
    
    # Validate num_similar
    max_similar = len(search.product_ids) - 1
    if num_similar > max_similar:
        raise HTTPException(
            status_code=400,
            detail=f"num_similar ({num_similar}) exceeds maximum ({max_similar})"
        )
    
    # Validate mode
    if mode not in search.engines:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid mode '{mode}'. Available: {list(search.engines.keys())}"
        )
    
    try:
        start = time.perf_counter()
        similar_products = search.find_similar_products(product_id, num_similar, mode=mode)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        logger.info(
            f"Found {len(similar_products)} similar products for {product_id[:8]}... "
            f"in {elapsed_ms:.1f}ms (mode={mode})"
        )
        
        return similar_products
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error finding similar products: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/product/{product_id}")
def get_product_details(product_id: str):
    """
    Retrieve metadata for a specific product.
    """
    if search is None or not search._initialized:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    if product_id not in search.id_to_idx:
        if product_id in purged_ids:
            raise HTTPException(
                status_code=404, 
                detail=f"Product ID '{product_id}' not found. Note: This product was intentionally purged during data cleaning due to a dead image link."
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Product ID '{product_id}' not found in the dataset."
            )
    
    idx = search.id_to_idx[product_id]
    row = search.df.iloc[idx]
    
    return {
        "uniq_id": product_id,
        "product_name": row['product_name'],
        "brand": row['brand'],
        "sales_price": float(row['sales_price']) if pd.notna(row['sales_price']) else None,
        "rating": float(row['rating']) if pd.notna(row['rating']) else None,
        "color": row['color'],
        "categories": row['categories'],
        "image_url": row['image_url'],
    }


@app.get("/find_similar_products_detailed")
def get_similar_products_detailed(
    product_id: str = Query(..., description="The unique ID of the product"),
    num_similar: int = Query(10, gt=0, description="Number of similar products"),
    mode: str = Query("text_structured", description="Search mode"),
):
    """
    Like find_similar_products, but returns full product details and similarity scores.
    """
    if search is None or not search._initialized:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    if product_id not in search.id_to_idx:
        if product_id in purged_ids:
            raise HTTPException(
                status_code=404, 
                detail=f"Product ID '{product_id}' not found. Note: This product was intentionally purged during data cleaning due to a dead image link."
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Product ID '{product_id}' not found in the dataset."
            )
    
    try:
        # Get query product details
        query_idx = search.id_to_idx[product_id]
        query_row = search.df.iloc[query_idx]
        
        # Get similar products with scores
        df_results = search.calculate_similarity(product_id, mode=mode, top_k=num_similar)
        
        return {
            "query_product": {
                "uniq_id": product_id,
                "product_name": query_row['product_name'],
                "brand": query_row['brand'],
                "sales_price": float(query_row['sales_price']) if pd.notna(query_row['sales_price']) else None,
                "image_url": query_row['image_url'] if pd.notna(query_row['image_url']) else None
            },
            "similar_products": df_results.head(num_similar).replace({float('nan'): None}).to_dict(orient='records'),
            "mode": mode,
        }
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.responses import HTMLResponse
import os

@app.get("/demo", response_class=HTMLResponse, summary="Demo UI")
def serve_demo_ui():
    """Serves the frontend visual demo HTML."""
    try:
        with open("demo.html", "r") as f:
            return f.read()
    except Exception as e:
        raise HTTPException(status_code=404, detail="demo.html not found")



# Required import for pandas in product endpoint
import pandas as pd


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
