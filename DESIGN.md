# Design Document — Product Similarity Search

> **Author:** Kuldeep Joshi  
> **Assignment:** SAP CX II Technical Exercise — Product Similarity Search  
> **Date:** July 2026

---

## 1. Problem Statement

Given a product's `uniq_id`, the system must retrieve a ranked list of `num_similar` products that are most similar to it, searching across a dataset of approximately 30,000 Amazon Fashion products. The similarity must be computed across multiple modalities (text, structured attributes, and images) in milliseconds.

---

## 2. High-Level Architecture

I designed a **Hybrid Multimodal Search Engine** utilizing a two-stage retrieval pipeline, which is the current industry standard for scalable E-Commerce recommendation systems.

### Pipeline Overview
```text
┌───────────────────────────────────────────────────────────────────┐
│                      OFFLINE INDEXING                             │
│                                                                   │
│  Raw Dataset (LDJSON)                                             │
│      │                                                            │
│      ▼                                                            │
│  data_loader.py (Polars for fast NDJSON parsing)                  │
│      │                                                            │
│      ▼                                                            │
│  feature_engine.py                                                │
│      ├── Semantic Text: all-MiniLM-L6-v2 (384-dim)                │
│      ├── Visual Pixels: FashionCLIP (512-dim)                     │
│      ├── Structured: MinMax / OneHot (~50-dim)                    │
│      └── Lexical/Sparse: BM25 (Okapi)                             │
│                                                                   │
│  Saved to disk: indices/ (FAISS HNSW + BM25 PKL)                  │
└─────────────────────────────┬─────────────────────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────────────┐
│                      ONLINE API (FastAPI)                         │
│                                                                   │
│  1. STAGE 1: Fast Retrieval (FAISS + BM25)                        │
│       - Fetch Top-N using Reciprocal Rank Fusion (RRF)            │
│  2. STAGE 2: Deep Re-ranking (Cross-Encoder)                      │
│       - ms-marco-MiniLM-L-6-v2 strictly scores Top-15 pairs       │
│                                                                   │
│  Return: Final Top-K Product IDs                                  │
└───────────────────────────────────────────────────────────────────┘
```

---

## 3. Design Decisions & Trade-offs

### 3.1 Data Processing: Polars vs. Pandas
I chose **Polars** for the data loading pipeline. Polars relies on Apache Arrow (a columnar in-memory format), which allows for true multi-threaded execution and zero-copy interoperability with Numpy. For this 30k dataset, it parses the heavy NDJSON file roughly 4x faster than Pandas.

### 3.2 Feature Engineering

To accurately represent a fashion product, I extracted three distinct feature sets:

#### A. Semantic Text (Sentence-BERT)
I concatenated the `product_name`, `brand`, and `categories` and passed them through `all-MiniLM-L6-v2`. 
- **Why not TF-IDF alone?** TF-IDF looks for exact token matches. A transformer model understands semantic intent, recognizing that "running shoes" and "jogging sneakers" exist in the same vector space. `all-MiniLM-L6-v2` was selected because it generates compact 384-dimensional vectors at 14,000 sentences/sec on CPU.

#### B. Visual Embeddings (FashionCLIP)
The prompt suggested ResNet or EfficientNet. However, those models generate purely visual embeddings. I chose to implement **FashionCLIP** (`patrickjohncyh/fashion-clip`).
- **Why CLIP over ResNet?** CLIP projects images and text into the *same* vector space. Furthermore, FashionCLIP is fine-tuned specifically on apparel, allowing it to recognize domain-specific nuances (e.g., "A-line skirt" vs. "Pencil skirt") much better than a generic ResNet trained on ImageNet.

#### C. Structured Metadata
- **Numerics (`sales_price`, `rating`)**: Applied `MinMaxScaler`. Without scaling, a price of $150 would mathematically overpower a 4.5-star rating during distance calculations.
- **Categoricals (`brand`, `color`)**: Applied `OneHotEncoder` using a "Top-N Bucket" strategy. There is a massive long-tail of rare brands (see `eda/` visualizations). I bucketed rare brands into an "Other" category to prevent the creation of a massive, hyper-sparse matrix that would dilute the semantic embeddings.
- **Missing Values**: Nearly 100% of the dataset is missing `item_weight`. I used a sentinel value (`999999999`) to effectively nullify this feature in the vector space, rather than using median imputation which would imply false confidence.

---

## 4. Search Methodology

### Stage 1: Hybrid Retrieval (Dense + Sparse)
Dense Vectors (FAISS) capture semantic meaning well, but can sometimes fail on exact keyword matches. Sparse Vectors (BM25) excel at exact keyword matching but fail on synonyms. 

To get the best of both worlds, I implemented **Hybrid Search**:
1. **Dense Retrieval**: Querying the FAISS `IndexHNSWFlat` (Hierarchical Navigable Small World) index. HNSW provides $O(\log N)$ search time with >99% recall.
2. **Sparse Retrieval**: Querying an Okapi BM25 index for exact keyword matching.
3. **Fusion**: I fuse the result lists using **Reciprocal Rank Fusion (RRF)**, which mathematically combines the rankings without needing to normalize the wildly different score distributions of FAISS and BM25.

### Stage 2: Cross-Encoder Re-ranking
Once the Hybrid Search fetches the Top candidates, the top 15 results are passed to a **Cross-Encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
- A Bi-Encoder (FAISS) encodes the query and the document separately.
- A Cross-Encoder feeds both the query and document into the transformer *simultaneously*, allowing deep self-attention between the query words and document words. It is computationally heavy (hence why it only runs on the Top 15, configurable in `config.py`), but improves ranking accuracy.

---

## 5. API & Microservice Architecture

The solution is wrapped in a **FastAPI** microservice.

### Endpoints
The `GET /find_similar_products` endpoint exposes a flexible interface where the client can dictate the exact search mode:
- `mode=image`: Pure visual similarity using FashionCLIP. (~1-2ms latency).
- `mode=text_structured`: Relies on semantics and metadata. (~4-5ms latency).
- `mode=combined`: The Hybrid RRF pipeline. (~5-7ms latency).

### Data Integrity
During data ingestion, roughly 174 product image links were found to be completely dead (HTTP 404). These were aggressively purged from the FAISS index to ensure visual integrity. If a client queries one of these purged IDs, the API catches it and returns a custom 404 payload explaining the deliberate purge.

### In-Memory Caching
To optimize latency for viral/popular items, the `find_similar_products` method is decorated with a Python `lru_cache`. I avoided standing up a Redis container to keep the deployment architecture strictly self-contained, as memory usage at this dataset scale (30k) is nominal.

---

## 6. Deployment (Docker & Kubernetes)

The application is containerized utilizing Docker layer caching (copying `requirements.txt` before the codebase to prevent heavy pip installs on every rebuild). 

The Kubernetes manifests (`k8s/`) include:
- **Memory Limits**: Bounded to 4Gi to accommodate the in-memory FAISS indices safely.
- **Probes**: A `readinessProbe` blocks traffic until the 300MB indices are fully loaded from disk into RAM, ensuring zero-downtime rolling deployments.

---

## 7. References
- Malkov & Yashunin, *"Efficient and robust approximate nearest neighbor search using HNSW graphs"*, IEEE TPAMI 2020.
- Reimers & Gurevych, *"Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"*, EMNLP 2019.
- Radford et al. / Patrick John Chia et al., *"FashionCLIP"*, arXiv:2204.03972.
- Robertson et al., *"Okapi at TREC-3"*, TREC 1994.
- Cormack et al., *"Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"*, SIGIR 2009.
- Nogueira et al., *"Passage Re-ranking with BERT"*, arXiv:1901.04085.
- Johnson et al., *"Billion-scale similarity search with GPUs"*, IEEE Trans. Big Data 2019 (FAISS Foundation).
- [Polars Documentation](https://pola.rs/)
- [ANN-Benchmarks](http://ann-benchmarks.com/)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
