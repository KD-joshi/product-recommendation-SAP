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
    Load the Amazon Fashion dataset and clean it for feature engineering.

    Returns a clean pandas DataFrame with these columns:
        - uniq_id (str): Product identifier
        - product_name (str): Full product name
        - brand (str): Brand name, "Unknown" if missing
        - sales_price (float): Price in INR, NaN if unparseable
        - weight (float): Weight, NaN if sentinel value
        - rating (float): Star rating 0-5, NaN if unparseable
        - color (str): Extracted color, "unknown" if missing
        - categories (str): Space-separated category names
        - image_url (str): First product image URL
        - meta_keywords (str): SEO keywords / description text
        - delivery_type (str): Fulfillment method
        - is_prime (int): 1 if Amazon Prime, 0 otherwise
        - is_bestseller (int): 1 if best seller, 0 otherwise
    """
    if path is None:
        path = config.DATA_PATH

    # Step 1: Load raw data (try Polars first, fallback to pandas)
    df = _try_polars_load(path)
    if df is None:
        df = _pandas_load(path)

    logger.info(f"Raw data shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    # Step 2: Clean each field
    # ---- uniq_id: our primary key, must be string
    df['uniq_id'] = df['uniq_id'].astype(str)

    # ---- product_name: fill missing with empty string
    df['product_name'] = df['product_name'].fillna("").astype(str)

    # ---- brand: many are None/NaN, replace with "Unknown"
    df['brand'] = df['brand'].fillna("Unknown").astype(str)

    # ---- sales_price: stored as string "200.00", convert to float
    df['sales_price'] = pd.to_numeric(df['sales_price'], errors='coerce')

    # ---- weight: sentinel value 999999999 means "unknown"
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
    df.loc[df['weight'] >= config.WEIGHT_SENTINEL, 'weight'] = np.nan

    # ---- rating: stored as string "5.0", convert to float
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # ---- color: the dataset has a direct 'colour' column — use it first!
    #      Fall back to extracting from product_details__k_v_pairs dict.
    if 'colour' in df.columns:
        df['color'] = df['colour'].fillna('').apply(
            lambda x: str(x).strip().lower() if x and str(x).strip() else 'unknown'
        )
    else:
        df['color'] = df['product_details__k_v_pairs'].apply(_extract_color)

    # ---- discount_percentage: extra signal if available
    if 'discount_percentage' in df.columns:
        df['discount_percentage'] = pd.to_numeric(df['discount_percentage'], errors='coerce').fillna(0.0)
    else:
        df['discount_percentage'] = 0.0

    # ---- no_of_reviews: social proof signal
    if 'no__of_reviews' in df.columns:
        df['no_of_reviews'] = pd.to_numeric(df['no__of_reviews'], errors='coerce').fillna(0.0)
    else:
        df['no_of_reviews'] = 0.0

    # ---- categories: extract from parent___child_category__all dict
    df['categories'] = df['parent___child_category__all'].apply(_extract_child_categories)

    # ---- image_url: first URL from pipe-delimited medium field
    df['image_url'] = df['medium'].apply(_extract_first_image_url)

    # ---- meta_keywords: fill missing
    df['meta_keywords'] = df['meta_keywords'].fillna("").astype(str)

    # ---- delivery_type: fill missing
    df['delivery_type'] = df['delivery_type'].fillna("unknown").astype(str)

    # ---- is_prime: convert Y/N to 1/0
    df['is_prime'] = (df['amazon_prime__y_or_n'].str.upper() == 'Y').astype(int)

    # ---- is_bestseller: convert Y/N to 1/0
    df['is_bestseller'] = (df['best_seller_tag__y_or_n'].str.upper() == 'Y').astype(int)

    # Step 3: Select only the columns we need
    clean_columns = [
        'uniq_id', 'product_name', 'brand', 'sales_price', 'weight',
        'rating', 'color', 'categories', 'image_url', 'meta_keywords',
        'delivery_type', 'is_prime', 'is_bestseller',
        'discount_percentage', 'no_of_reviews'
    ]
    df_clean = df[clean_columns].copy()

    # Step 4: Report data quality
    logger.info(f"Clean data shape: {df_clean.shape}")
    logger.info(f"Missing values:\n{df_clean.isnull().sum()}")
    logger.info(f"Sample brands: {df_clean['brand'].value_counts().head(5).to_dict()}")
    logger.info(f"Sample colors: {df_clean['color'].value_counts().head(5).to_dict()}")
    logger.info(f"Price range: {df_clean['sales_price'].min():.0f} - {df_clean['sales_price'].max():.0f}")
    logger.info(f"Products with images: {(df_clean['image_url'] != '').sum()}")

    return df_clean


if __name__ == "__main__":
    # Quick test: load and show stats
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = load_and_clean_data()
    print(f"\n✅ Loaded {len(df)} products")
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())
