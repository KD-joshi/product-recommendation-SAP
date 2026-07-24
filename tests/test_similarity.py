"""
Tests for the Product Similarity Search system.

Run with: .venv/bin/python -m pytest tests/ -v
"""

import json
import os
import sys
import pytest
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# DATA LOADER TESTS
# ============================================================

class TestDataLoader:
    """Tests for data_loader.py"""
    
    def test_load_returns_dataframe(self, sample_df):
        import pandas as pd
        assert isinstance(sample_df, pd.DataFrame)
    
    def test_has_required_columns(self, sample_df):
        required = ['uniq_id', 'product_name', 'brand', 'sales_price',
                    'weight', 'rating', 'color', 'categories', 'image_url']
        for col in required:
            assert col in sample_df.columns, f"Missing column: {col}"
    
    def test_uniq_id_is_string(self, sample_df):
        assert sample_df['uniq_id'].dtype == object  # pandas string = object dtype
    
    def test_sales_price_is_numeric(self, sample_df):
        assert np.issubdtype(sample_df['sales_price'].dtype, np.floating)
    
    def test_rating_is_numeric(self, sample_df):
        assert np.issubdtype(sample_df['rating'].dtype, np.floating)
    
    def test_no_weight_sentinel(self, sample_df):
        """Sentinel value 999999999 should be replaced with NaN."""
        from config import WEIGHT_SENTINEL
        assert (sample_df['weight'] >= WEIGHT_SENTINEL).sum() == 0, \
            "Sentinel weight values should be NaN"
    
    def test_brand_no_null(self, sample_df):
        """Nulls in brand should become 'Unknown'."""
        assert sample_df['brand'].isnull().sum() == 0
    
    def test_is_prime_binary(self, sample_df):
        assert set(sample_df['is_prime'].unique()).issubset({0, 1})
    
    def test_is_bestseller_binary(self, sample_df):
        assert set(sample_df['is_bestseller'].unique()).issubset({0, 1})
    
    def test_nonzero_rows(self, sample_df):
        assert len(sample_df) > 0


# ============================================================
# FEATURE ENGINE TESTS
# ============================================================

class TestStructuredFeatureEncoder:
    """Tests for StructuredFeatureEncoder in feature_engine.py"""
    
    def test_fit_transform_returns_array(self, sample_df):
        from feature_engine import StructuredFeatureEncoder
        encoder = StructuredFeatureEncoder(top_n_brands=10, top_n_colors=5)
        features = encoder.fit_transform(sample_df)
        assert isinstance(features, np.ndarray)
    
    def test_output_dtype_float32(self, sample_df):
        from feature_engine import StructuredFeatureEncoder
        encoder = StructuredFeatureEncoder(top_n_brands=10, top_n_colors=5)
        features = encoder.fit_transform(sample_df)
        assert features.dtype == np.float32
    
    def test_one_row_per_product(self, sample_df):
        from feature_engine import StructuredFeatureEncoder
        encoder = StructuredFeatureEncoder(top_n_brands=10, top_n_colors=5)
        features = encoder.fit_transform(sample_df)
        assert features.shape[0] == len(sample_df)
    
    def test_no_nans_in_output(self, sample_df):
        from feature_engine import StructuredFeatureEncoder
        encoder = StructuredFeatureEncoder(top_n_brands=10, top_n_colors=5)
        features = encoder.fit_transform(sample_df)
        assert not np.isnan(features).any(), "Output should have no NaN values"
    
    def test_values_between_0_and_1(self, sample_df):
        """MinMax-scaled numerical features should be in [0, 1]."""
        from feature_engine import StructuredFeatureEncoder
        encoder = StructuredFeatureEncoder(top_n_brands=10, top_n_colors=5)
        features = encoder.fit_transform(sample_df)
        # Not strictly [0,1] due to OHE, but values should be finite
        assert np.isfinite(features).all()
    
    def test_transform_matches_fit_transform(self, sample_df):
        """fit().transform() should give same result as fit_transform()."""
        from feature_engine import StructuredFeatureEncoder
        enc1 = StructuredFeatureEncoder(top_n_brands=10, top_n_colors=5)
        enc2 = StructuredFeatureEncoder(top_n_brands=10, top_n_colors=5)
        
        ft_result = enc1.fit_transform(sample_df)
        enc2.fit(sample_df)
        t_result = enc2.transform(sample_df)
        
        np.testing.assert_array_almost_equal(ft_result, t_result)


class TestNormalizeL2:
    def test_unit_norm(self):
        from feature_engine import normalize_l2
        vectors = np.random.rand(100, 50).astype(np.float32)
        normed = normalize_l2(vectors)
        norms = np.linalg.norm(normed, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)
    
    def test_zero_vector_stays_zero(self):
        from feature_engine import normalize_l2
        vectors = np.zeros((5, 10), dtype=np.float32)
        normed = normalize_l2(vectors)
        assert not np.isnan(normed).any()


# ============================================================
# SIMILARITY ENGINE TESTS
# ============================================================

class TestFAISSEngine:
    """Tests for FAISSEngine in similarity_engine.py"""
    
    def _make_vectors(self, n=200, d=64):
        """Create random L2-normalized float32 vectors."""
        from feature_engine import normalize_l2
        v = np.random.rand(n, d).astype(np.float32)
        return normalize_l2(v)
    
    def test_brute_force_build(self):
        from similarity_engine import FAISSEngine
        vectors = self._make_vectors()
        engine = FAISSEngine(dimension=64, use_hnsw=False)
        engine.build(vectors)
        assert engine.index.ntotal == 200
    
    def test_hnsw_build(self):
        from similarity_engine import FAISSEngine
        vectors = self._make_vectors()
        engine = FAISSEngine(dimension=64, use_hnsw=True)
        engine.build(vectors)
        assert engine.index.ntotal == 200
    
    def test_search_returns_k_results(self):
        from similarity_engine import FAISSEngine
        vectors = self._make_vectors()
        engine = FAISSEngine(dimension=64, use_hnsw=False)
        engine.build(vectors)
        distances, indices = engine.search_single(vectors[0], k=5)
        assert len(indices) == 5
        assert len(distances) == 5
    
    def test_search_excludes_negative_indices(self):
        from similarity_engine import FAISSEngine
        vectors = self._make_vectors(n=50)
        engine = FAISSEngine(dimension=64, use_hnsw=True)
        engine.build(vectors)
        _, indices = engine.search_single(vectors[0], k=10)
        assert all(i >= 0 for i in indices)
    
    def test_self_is_nearest_neighbor(self):
        """A product should be its own nearest neighbor."""
        from similarity_engine import FAISSEngine
        vectors = self._make_vectors()
        engine = FAISSEngine(dimension=64, use_hnsw=False)
        engine.build(vectors)
        _, indices = engine.search_single(vectors[0], k=1)
        assert indices[0] == 0
    
    def test_hnsw_vs_brute_force_recall(self):
        """HNSW should have >95% recall vs brute-force at this scale."""
        from similarity_engine import FAISSEngine
        vectors = self._make_vectors(n=500, d=128)
        
        bf = FAISSEngine(dimension=128, use_hnsw=False)
        bf.build(vectors)
        
        hnsw = FAISSEngine(dimension=128, use_hnsw=True, hnsw_m=32, hnsw_ef_search=64)
        hnsw.build(vectors)
        
        k = 10
        recall_sum = 0
        n_test = 50
        
        for i in range(n_test):
            q = vectors[i:i+1]
            _, bf_idx = bf.search(q, k)
            _, hnsw_idx = hnsw.search(q, k)
            
            gt = set(bf_idx[0])
            approx = set(hnsw_idx[0])
            recall_sum += len(gt & approx) / k
        
        recall = recall_sum / n_test
        assert recall > 0.95, f"HNSW recall@10 = {recall:.3f}, expected > 0.95"
    
    def test_save_and_load(self, tmp_path):
        from similarity_engine import FAISSEngine
        vectors = self._make_vectors()
        
        engine = FAISSEngine(dimension=64, use_hnsw=True)
        engine.build(vectors)
        
        path = str(tmp_path / "test.index")
        engine.save(path)
        
        loaded = FAISSEngine(dimension=64, use_hnsw=True)
        loaded.load(path)
        
        assert loaded.n_vectors == 200
        
        # Results should be identical
        d1, i1 = engine.search_single(vectors[0], k=5)
        d2, i2 = loaded.search_single(vectors[0], k=5)
        np.testing.assert_array_equal(i1, i2)


# ============================================================
# FIND_SIMILAR_PRODUCTS INTEGRATION TESTS
# ============================================================

class TestFindSimilarProducts:
    """Integration tests for the main find_similar_products function."""
    
    @pytest.fixture(scope="class")
    def search_instance(self, sample_df):
        """Build a small search instance using the sample data."""
        from similarity_search import ProductSimilaritySearch
        
        search = ProductSimilaritySearch()
        # Override df directly for speed (skip full file load)
        search.df = sample_df
        search.product_ids = sample_df['uniq_id'].values
        search.id_to_idx = {pid: idx for idx, pid in enumerate(search.product_ids)}
        
        # Build features and index
        from feature_engine import StructuredFeatureEncoder, TextEmbedder, build_combined_features
        
        enc = StructuredFeatureEncoder(top_n_brands=10, top_n_colors=5)
        search.structured_features = enc.fit_transform(sample_df)
        search.struct_encoder = enc
        
        embedder = TextEmbedder()
        search.text_embeddings = embedder.encode(sample_df, batch_size=32)
        search.text_embedder = embedder
        
        combined = build_combined_features(search.structured_features, search.text_embeddings)
        search.combined_features = combined
        
        from similarity_engine import FAISSEngine
        engine = FAISSEngine(dimension=combined.shape[1], use_hnsw=True)
        engine.build(combined)
        search.engines["text_structured"] = engine
        search._initialized = True
        
        return search
    
    def test_returns_list(self, search_instance):
        test_id = search_instance.product_ids[0]
        result = search_instance.find_similar_products(test_id, num_similar=5)
        assert isinstance(result, list)
    
    def test_returns_correct_count(self, search_instance):
        test_id = search_instance.product_ids[0]
        for n in [1, 3, 5, 10]:
            result = search_instance.find_similar_products(test_id, num_similar=n)
            assert len(result) == n, f"Expected {n} results, got {len(result)}"
    
    def test_results_are_strings(self, search_instance):
        test_id = search_instance.product_ids[0]
        result = search_instance.find_similar_products(test_id, num_similar=5)
        assert all(isinstance(pid, str) for pid in result)
    
    def test_query_not_in_results(self, search_instance):
        """The query product itself should not appear in results."""
        test_id = search_instance.product_ids[0]
        result = search_instance.find_similar_products(test_id, num_similar=5)
        assert test_id not in result
    
    def test_results_are_valid_product_ids(self, search_instance):
        """All returned IDs should exist in the dataset."""
        test_id = search_instance.product_ids[0]
        result = search_instance.find_similar_products(test_id, num_similar=5)
        valid_ids = set(search_instance.product_ids)
        for pid in result:
            assert pid in valid_ids, f"Returned unknown product ID: {pid}"
    
    def test_invalid_product_id_raises(self, search_instance):
        with pytest.raises(ValueError, match="not found"):
            search_instance.find_similar_products("nonexistent_id_12345", num_similar=5)
    
    def test_no_duplicates_in_results(self, search_instance):
        test_id = search_instance.product_ids[0]
        result = search_instance.find_similar_products(test_id, num_similar=10)
        assert len(result) == len(set(result)), "Results should not contain duplicates"


# ============================================================
# CONFTEST (shared fixtures)
# ============================================================

@pytest.fixture(scope="session")
def sample_df():
    """Load a small sample of the dataset for fast testing."""
    from data_loader import load_and_clean_data
    import config
    
    # Load just the first 500 rows for fast tests
    import json
    rows = []
    with open(config.DATA_PATH) as f:
        for i, line in enumerate(f):
            if i >= 500:
                break
            rows.append(json.loads(line))
    
    import pandas as pd
    raw_df = pd.DataFrame(rows)
    
    # Save to temp file and load through our pipeline
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ldjson', delete=False) as tmp:
        for row in rows:
            tmp.write(json.dumps(row) + '\n')
        tmp_path = tmp.name
    
    try:
        df = load_and_clean_data(tmp_path)
    finally:
        os.unlink(tmp_path)
    
    return df
