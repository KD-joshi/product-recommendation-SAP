"""
Data Loader Module.

Responsible for loading and cleaning the product dataset.
Leverages Polars for high-performance, multi-threaded NDJSON parsing
and memory-efficient columnar data processing.

The module cleans and normalizes fields such as prices, weights, categories,
and handles missing or sentinel values before converting the final dataset
to a format suitable for feature extraction.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def _try_polars_load(path: str) -> Optional[pd.DataFrame]:
    """
    Try loading with Polars first (faster).
    Falls back gracefully if Polars isn't installed.
    """
    try:
        import polars as pl

        logger.info("Loading data with Polars (fast path)...")
        df_pl = pl.read_ndjson(path)
        logger.info(f"Polars loaded {len(df_pl)} rows")

        # Convert to pandas for downstream compatibility
        return df_pl.to_pandas()

    except ImportError:
        logger.info("Polars not installed, falling back to pandas...")
        return None
    except Exception as e:
        logger.warning(f"Polars failed ({e}), falling back to pandas...")
        return None


def _pandas_load(path: str) -> pd.DataFrame:
    """Fallback: load with pandas (slower but always works)."""
    logger.info("Loading data with pandas...")
    df = pd.read_json(path, lines=True)
    logger.info(f"Pandas loaded {len(df)} rows")
    return df


def _extract_color(details: object) -> str:
    """
    Extract color from the product_details__k_v_pairs field.

    The field is a dict like: {'Colour': 'Blue', 'Material': 'Cotton', ...}
    But sometimes it's a string, None, or has different key names.
    """
    if isinstance(details, dict):
        # Try common key names for color
        for key in ['Colour', 'Color', 'colour', 'color']:
            if key in details:
                return str(details[key]).strip().lower()
    return "unknown"


def _extract_child_categories(categories: object) -> str:
    """
    Extract child category names from parent___child_category__all.

    The field looks like: {'ClothingAccessories': '#19,259', 'WomensKurtasKurtis': '#1793'}
    We want: "ClothingAccessories WomensKurtasKurtis"
    """
    if isinstance(categories, dict):
        return " ".join(categories.keys())
    return "unknown"


def _extract_first_image_url(medium_field: object) -> str:
    """
    Extract the first image URL from the pipe-delimited 'medium' field.

    The field looks like: "url1.jpg|url2.jpg|url3.jpg"
    We take the first one.
    """
    if isinstance(medium_field, str) and medium_field.strip():
        urls = medium_field.split("|")
        if urls and urls[0].startswith("http"):
            return urls[0].strip()
    return ""


def load_and_clean_data(path: str = None) -> pd.DataFrame:
    """
    Load the Amazon Fashion dataset and clean it for feature engineering using pure Polars.
    """
    if path is None:
        path = config.DATA_PATH

    import polars as pl
    logger.info("Loading data with pure Polars pipeline...")
    
    # Read NDJSON (inferring schema)
    # Some columns might be missing or varying, so we use strict=False if possible, 
    # but read_ndjson usually handles it.
    df = pl.read_ndjson(path)
    logger.info(f"Raw data shape: {df.shape}")

    # Helper function for colors
    def extract_color(val):
        if isinstance(val, dict):
            for key in ['Colour', 'Color', 'colour', 'color']:
                if key in val:
                    return str(val[key]).strip().lower()
        elif isinstance(val, str):
            # rudimentary fallback
            if "color" in val.lower():
                import re
                m = re.search(r"'(?:Colour|Color|colour|color)': '([^']+)'", val)
                if m:
                    return m.group(1).strip().lower()
        return "unknown"

    def extract_categories(val):
        if isinstance(val, dict):
            return " ".join(val.keys())
        elif isinstance(val, str) and val.startswith("{"):
            import json
            try:
                d = json.loads(val.replace("'", '"'))
                return " ".join(d.keys())
            except:
                pass
        return "unknown"

    def extract_image(val):
        if isinstance(val, str) and val.strip():
            urls = val.split("|")
            if urls and urls[0].startswith("http"):
                return urls[0].strip()
        return ""
        
    # We will build a list of column expressions to construct the clean dataframe in one pass
    exprs = [
        pl.col('uniq_id').cast(pl.Utf8).fill_null(""),
        pl.col('product_name').cast(pl.Utf8).fill_null(""),
        pl.col('brand').cast(pl.Utf8).fill_null("Unknown"),
        
        # Cast prices, weights, ratings to Float64
        # Since they are strings like "200.00", we can try casting
        pl.col('sales_price').cast(pl.Utf8).str.replace(r"[^\d.]", "").cast(pl.Float64, strict=False),
        
        pl.col('weight').cast(pl.Utf8).str.replace(r"[^\d.]", "").cast(pl.Float64, strict=False).map_elements(
            lambda x: None if (x is not None and x >= config.WEIGHT_SENTINEL) else x, return_dtype=pl.Float64
        ),
        
        pl.col('rating').cast(pl.Utf8).str.replace(r"[^\d.]", "").cast(pl.Float64, strict=False),
        
        # Extract Categories
        pl.col('parent___child_category__all').map_elements(extract_categories, return_dtype=pl.Utf8).alias("categories"),
        
        # Extract Image URL
        pl.col('medium').map_elements(extract_image, return_dtype=pl.Utf8).alias("image_url"),
        
        pl.col('meta_keywords').cast(pl.Utf8).fill_null(""),
        pl.col('delivery_type').cast(pl.Utf8).fill_null("unknown"),
        
        # Booleans
        (pl.col('amazon_prime__y_or_n').cast(pl.Utf8).str.to_uppercase() == 'Y').cast(pl.Int32).fill_null(0).alias("is_prime"),
        (pl.col('best_seller_tag__y_or_n').cast(pl.Utf8).str.to_uppercase() == 'Y').cast(pl.Int32).fill_null(0).alias("is_bestseller"),
    ]
    
    # Handle optional columns
    if 'colour' in df.columns:
        exprs.append(
            pl.col('colour').cast(pl.Utf8).fill_null('').map_elements(
                lambda x: x.strip().lower() if x and x.strip() else 'unknown', return_dtype=pl.Utf8
            ).alias("color")
        )
    else:
        exprs.append(
            pl.col('product_details__k_v_pairs').map_elements(extract_color, return_dtype=pl.Utf8).alias("color")
        )
        
    if 'discount_percentage' in df.columns:
        exprs.append(pl.col('discount_percentage').cast(pl.Utf8).str.replace(r"[^\d.]", "").cast(pl.Float64, strict=False).fill_null(0.0))
    else:
        exprs.append(pl.lit(0.0).alias('discount_percentage'))
        
    if 'no__of_reviews' in df.columns:
        exprs.append(pl.col('no__of_reviews').cast(pl.Utf8).str.replace(r"[^\d.]", "").cast(pl.Float64, strict=False).fill_null(0.0).alias("no_of_reviews"))
    else:
        exprs.append(pl.lit(0.0).alias('no_of_reviews'))

    # Execute the Polars query
    df_clean = df.select(exprs)
    
    # Convert to Pandas only at the very end for compatibility with sklearn/transformers
    df_pandas = df_clean.to_pandas()
    
    logger.info(f"Clean data shape: {df_pandas.shape}")
    return df_pandas


if __name__ == "__main__":
    # Quick test: load and show stats
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = load_and_clean_data()
    print(f"\n✅ Loaded {len(df)} products")
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())
