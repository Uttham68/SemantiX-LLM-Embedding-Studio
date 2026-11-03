"""
Module 2: Pretrained LLM Embedding Generation & Semantic Similarity Analysis
Responsible: Member 2 (Embedding Generation & Vector Space Analysis)

This module handles:
1. Loading pretrained Sentence-BERT Transformer models:
   - Primary Model: `all-MiniLM-L6-v2` (6-layer MiniLM bi-encoder, 384 dimensions)
   - Comparison Model: `paraphrase-MiniLM-L3-v2` (3-layer lightweight bi-encoder, 384 dimensions)
2. Generating L2-normalized dense vector representations for the corpus and caching them
   locally (`data/embeddings_cache.npz`) for instant offline reuse.
3. Building a classical lexical TF-IDF baseline (384/2000 sparse features) to contrast
   semantic embeddings against keyword matching.
4. Computing Cosine Similarity matrices and running the Paraphrase vs. Unrelated benchmark
   (demonstrating why Text A and Text B are close in LLM embedding space despite zero word overlap).
"""

import os
import time
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import BASE_DIR, DATA_DIR, clean_text_for_embeddings

EMBEDDINGS_CACHE_PATH = os.path.join(DATA_DIR, "embeddings_cache.npz")

PRIMARY_MODEL_NAME = "all-MiniLM-L6-v2"
SECONDARY_MODEL_NAME = "paraphrase-MiniLM-L3-v2"

# In-memory singleton cache so Streamlit does not reload models repeatedly
_MODEL_REGISTRY: Dict[str, object] = {}


def load_embedding_model(model_name: str = PRIMARY_MODEL_NAME):
    """
    Loads and caches a SentenceTransformer model in memory.
    Configured for minimal memory overhead (<180MB RAM) on cloud container runtimes.
    """
    if model_name not in _MODEL_REGISTRY:
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        from sentence_transformers import SentenceTransformer
        print(f"[Module 2] Loading pretrained Transformer model: {model_name} (CPU, 1 thread) ...")
        model = SentenceTransformer(model_name, device="cpu")
        model.eval()
        _MODEL_REGISTRY[model_name] = model
    return _MODEL_REGISTRY[model_name]


def encode_texts(
    texts: List[str],
    model_name: str = PRIMARY_MODEL_NAME,
    batch_size: int = 64,
    normalize: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Converts a list of texts into dense numerical vectors using a pretrained LLM embedding model.
    """
    import torch
    model = load_embedding_model(model_name)
    cleaned = [clean_text_for_embeddings(t) for t in texts]

    start_t = time.perf_counter()
    with torch.no_grad():
        embeddings = model.encode(
            cleaned,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
    elapsed = time.perf_counter() - start_t
    return embeddings.astype(np.float32), elapsed


def get_or_create_corpus_embeddings(
    df: pd.DataFrame, force_recompute: bool = False
) -> Dict[str, object]:
    """
    Loads precomputed embeddings for both pretrained LLM models from disk cache if available;
    otherwise computes them and saves `data/embeddings_cache.npz`.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    texts = df["text"].tolist()

    if os.path.exists(EMBEDDINGS_CACHE_PATH) and not force_recompute:
        data = np.load(EMBEDDINGS_CACHE_PATH, allow_pickle=True)
        if (
            "emb_minilm_l6" in data
            and len(data["emb_minilm_l6"]) == len(texts)
        ):
            return {
                "all-MiniLM-L6-v2": data["emb_minilm_l6"],
                "paraphrase-MiniLM-L3-v2": data["emb_minilm_l3"],
                "time_minilm_l6": float(data["time_l6"]),
                "time_minilm_l3": float(data["time_l3"]),
            }

    print(f"[Module 2] Encoding {len(texts)} documents with {PRIMARY_MODEL_NAME}...")
    emb_l6, time_l6 = encode_texts(texts, model_name=PRIMARY_MODEL_NAME)
    print(f"[Module 2] {PRIMARY_MODEL_NAME} finished in {time_l6:.2f}s (shape: {emb_l6.shape})")

    print(f"[Module 2] Encoding {len(texts)} documents with {SECONDARY_MODEL_NAME}...")
    emb_l3, time_l3 = encode_texts(texts, model_name=SECONDARY_MODEL_NAME)
    print(f"[Module 2] {SECONDARY_MODEL_NAME} finished in {time_l3:.2f}s (shape: {emb_l3.shape})")

    np.savez_compressed(
        EMBEDDINGS_CACHE_PATH,
        emb_minilm_l6=emb_l6,
        emb_minilm_l3=emb_l3,
        time_l6=np.array(time_l6),
        time_l3=np.array(time_l3),
    )
    print(f"[Module 2] Cached embeddings to {EMBEDDINGS_CACHE_PATH}")

    return {
        "all-MiniLM-L6-v2": emb_l6,
        "paraphrase-MiniLM-L3-v2": emb_l3,
        "time_minilm_l6": time_l6,
        "time_minilm_l3": time_l3,
    }


def compute_tfidf_features(
    texts: List[str], max_features: int = 1500
) -> Tuple[np.ndarray, TfidfVectorizer, float]:
    """
    Computes traditional lexical TF-IDF features for baseline comparison.
    """
    start_t = time.perf_counter()
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        norm="l2",
    )
    tfidf_matrix = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    elapsed = time.perf_counter() - start_t
    return tfidf_matrix, vectorizer, elapsed


def evaluate_semantic_similarity_demo(
    model_name: str = PRIMARY_MODEL_NAME,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Demonstrates the fundamental difference between Lexical (TF-IDF) similarity and
    Pretrained LLM Embedding similarity using concrete sentence pairs, starting with
    the exact Healthcare AI vs. Cricket example from Section 4 of the project specification.
    """
    demo_sentences = [
        "Artificial intelligence is transforming healthcare.",           # S1: AI + Healthcare
        "Machine learning is improving medical diagnosis.",             # S2: Paraphrase of S1 (0 shared words!)
        "India won the cricket championship.",                          # S3: Sports (Cricket)
        "The national cricket team celebrated their tournament victory.", # S4: Paraphrase of S3
        "The stock market recorded a significant increase.",            # S5: Finance
        "Equity indices surged as corporate earnings beat expectations.", # S6: Paraphrase of S5 (0 shared words!)
    ]

    # 1. Pretrained LLM Embeddings
    llm_emb, _ = encode_texts(demo_sentences, model_name=model_name)
    llm_sim_matrix = cosine_similarity(llm_emb)

    # 2. Lexical TF-IDF Baseline
    tfidf_vec = TfidfVectorizer(stop_words="english")
    tfidf_mat = tfidf_vec.fit_transform(demo_sentences).toarray()
    tfidf_sim_matrix = cosine_similarity(tfidf_mat)

    pairs_to_compare = [
        (0, 1, "Related (Paraphrase: AI in Healthcare)", "High (>0.60)"),
        (0, 2, "Unrelated (Healthcare AI vs. Cricket)", "Low (<0.15)"),
        (1, 2, "Unrelated (Medical ML vs. Cricket)", "Low (<0.15)"),
        (2, 3, "Related (Paraphrase: Cricket Victory)", "High (>0.65)"),
        (4, 5, "Related (Paraphrase: Stock Market Surge)", "High (>0.55)"),
        (2, 4, "Unrelated (Cricket vs. Stock Market)", "Low (<0.15)"),
    ]

    rows = []
    for i, j, relation, expected in pairs_to_compare:
        s_a = demo_sentences[i]
        s_b = demo_sentences[j]
        words_a = set(s_a.lower().replace(".", "").split()) - {"is", "the", "a", "in", "as", "their"}
        words_b = set(s_b.lower().replace(".", "").split()) - {"is", "the", "a", "in", "as", "their"}
        overlap = len(words_a.intersection(words_b))

        rows.append({
            "Text A": s_a,
            "Text B": s_b,
            "Relationship": relation,
            "Shared Keywords": overlap,
            "TF-IDF Cosine Sim": round(float(tfidf_sim_matrix[i, j]), 4),
            "LLM Embedding Sim": round(float(llm_sim_matrix[i, j]), 4),
            "Expected Behavior": expected,
        })

    comparison_df = pd.DataFrame(rows)
    return comparison_df, llm_sim_matrix, demo_sentences


def semantic_search_corpus(
    query: str,
    df: pd.DataFrame,
    corpus_embeddings: np.ndarray,
    model_name: str = PRIMARY_MODEL_NAME,
    top_k: int = 5,
    query_emb: np.ndarray = None,
) -> pd.DataFrame:
    """
    Finds the top-K most semantically similar DISTINCT documents in the corpus for any user query.
    Deduplicates results by the first 60 characters (`text[:60].lower()`) so synthetic template
    variations never appear as duplicate cards in the UI.
    Reuses precomputed `query_emb` if provided to avoid redundant model passes.
    """
    if query_emb is None:
        query_emb, _ = encode_texts([query], model_name=model_name)
    sims = np.dot(corpus_embeddings, query_emb[0] if query_emb.ndim > 1 else query_emb)
    sorted_indices = np.argsort(sims)[::-1]

    selected_indices = []
    seen_prefixes = set()

    for idx in sorted_indices:
        doc_text = str(df.iloc[idx]["text"]).strip()
        prefix = doc_text[:60].lower()
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)
        selected_indices.append(idx)
        if len(selected_indices) >= top_k:
            break

    results = df.iloc[selected_indices][["doc_id", "category", "subtopic", "text"]].copy()
    results["cosine_similarity"] = np.round(sims[selected_indices], 4)
    return results.reset_index(drop=True)

