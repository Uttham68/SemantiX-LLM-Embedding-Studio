"""
Public Interactive Web Application Server (`web_app.py`)
=========================================================
Note: The original Streamlit dashboard (`app.py`) remains completely untouched for
faculty evaluation and personal reference.

This standalone FastAPI web server powers a modern, public-facing interactive web application
at `http://localhost:8000` using the exact same backend modules (`src/`) and cached models.

Run with:
    python web_app.py
"""

import os
from typing import List
import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering, KMeans

from src.data_loader import BASE_DIR, CATEGORIES, RANDOM_SEED, load_or_create_dataset
from src.embeddings import (
    PRIMARY_MODEL_NAME,
    get_or_create_corpus_embeddings,
    encode_texts,
    semantic_search_corpus,
)
from src.models import (
    predict_single_text,
    run_kmeans_clustering,
)
from src.topic_visualizer import (
    compute_ctfidf_topics,
    discover_category_subtopics,
    compute_or_load_2d_projections,
    FIGURES_DIR,
)
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")
PIPELINE_PATH = os.path.join(RESULTS_DIR, "trained_pipeline.joblib")

WEB_DIR = os.path.join(BASE_DIR, "web")

app = FastAPI(
    title="SemantiX — Pretrained LLM Embedding Playground",
    description="Interactive Web Application for Text Classification, Clustering, and Topic Modeling Using Pretrained LLM Embeddings",
    version="1.1.0",
)

STATE = {}


def init_state():
    if STATE:
        return
    if not os.path.exists(PIPELINE_PATH):
        from run_experiments import run_all_experiments
        run_all_experiments(force_rebuild=False)
    STATE["bundle"] = joblib.load(PIPELINE_PATH)
    STATE["df"] = load_or_create_dataset(force_rebuild=False)
    STATE["emb_bundle"] = get_or_create_corpus_embeddings(STATE["df"], force_recompute=False)
    STATE["projections"] = compute_or_load_2d_projections(
        STATE["emb_bundle"][PRIMARY_MODEL_NAME], force_recompute=False
    )
    # Warm up bi-encoder model on CPU with single-thread lean memory footprint
    from src.embeddings import load_embedding_model
    load_embedding_model(PRIMARY_MODEL_NAME)
    import gc
    gc.collect()


@app.on_event("startup")
def on_startup():
    init_state()


class ClassifyRequest(BaseModel):
    text: str
    classifier_type: str = "Logistic Regression"


class SimilarityRequest(BaseModel):
    text_a: str
    text_b: str
    text_c: str


class CustomBatchRequest(BaseModel):
    texts: List[str]
    n_clusters: int = 3


@app.get("/api/overview")
def get_overview():
    init_state()
    bundle = STATE["bundle"]
    df = STATE["df"]

    return {
        "total_documents": len(df),
        "categories": CATEGORIES,
        "embedding_dim": 384,
        "primary_model": PRIMARY_MODEL_NAME,
        "test_accuracy": bundle["clf_eval_l6"]["accuracy"],
        "test_f1": bundle["clf_eval_l6"]["f1_score"],
        "clustering_nmi": bundle["clustering_res_k5"]["nmi_score"],
        "clustering_ari": bundle["clustering_res_k5"]["ari_score"],
        "topic_diversity": bundle["corpus_topics"]["topic_diversity"],
        "tech_topic_diversity": bundle["tech_subtopics"]["topic_diversity"],
        "model_comparison": bundle["model_comparison_df"].to_dict(orient="records"),
        "classifier_comparison": bundle["classifier_comparison_df"].to_dict(orient="records"),
        "per_class_metrics": bundle["clf_eval_l6"]["per_class_df"].to_dict(orient="records"),
        "k_sweep": bundle["k_sweep_df"].to_dict(orient="records"),
    }


@app.post("/api/classify")
def api_classify(req: ClassifyRequest):
    init_state()
    bundle = STATE["bundle"]
    df = STATE["df"]
    emb_l6 = STATE["emb_bundle"][PRIMARY_MODEL_NAME]

    text = req.text.strip()
    if not text:
        return JSONResponse({"error": "Please enter non-empty text."}, status_code=400)

    clf_name = req.classifier_type
    if clf_name not in bundle["trained_classifiers"]:
        clf_name = "Logistic Regression"

    clf_model = bundle["trained_classifiers"][clf_name]
    q_emb, elapsed_sec = encode_texts([text], model_name=PRIMARY_MODEL_NAME)
    pred_res = predict_single_text(
        text,
        clf_model,
        model_name=PRIMARY_MODEL_NAME,
        precomputed_emb=q_emb,
        infer_time_ms=elapsed_sec * 1000,
    )
    nearest_df = semantic_search_corpus(text, df, emb_l6, top_k=4, query_emb=q_emb)

    # Compute 2D coordinates (t-SNE & PCA) for the live query via temperature-scaled
    # cosine barycentric interpolation over its top-5 nearest corpus neighbors
    sims = np.dot(emb_l6, q_emb[0])
    top5_idx = np.argsort(sims)[::-1][:5]
    top5_sims = sims[top5_idx]
    weights = np.exp((top5_sims - np.max(top5_sims)) / 0.04)
    weights = weights / (np.sum(weights) + 1e-12)

    tsne_coords = STATE["projections"]["tsne_2d"]
    pca_coords = STATE["projections"]["pca_2d"]
    q_tsne = np.dot(weights, tsne_coords[top5_idx])
    q_pca = np.dot(weights, pca_coords[top5_idx])

    neighbor_coords = []
    for n_doc_id in nearest_df["doc_id"].tolist()[:3]:
        row_matches = np.where(df["doc_id"].values == n_doc_id)[0]
        if len(row_matches) > 0:
            r_idx = int(row_matches[0])
            neighbor_coords.append({
                "doc_id": n_doc_id,
                "tsne_x": round(float(tsne_coords[r_idx, 0]), 3),
                "tsne_y": round(float(tsne_coords[r_idx, 1]), 3),
                "pca_x": round(float(pca_coords[r_idx, 0]), 3),
                "pca_y": round(float(pca_coords[r_idx, 1]), 3),
            })

    return {
        "input_text": text,
        "classifier_used": clf_name,
        "predicted_category": pred_res["predicted_category"],
        "confidence": pred_res["confidence"],
        "confidence_pct": pred_res["confidence_pct"],
        "is_low_confidence": pred_res["is_low_confidence"],
        "confidence_status": pred_res["confidence_status"],
        "encoding_time_ms": pred_res["encoding_time_ms"],
        "probabilities": pred_res["probability_df"].to_dict(orient="records"),
        "nearest_documents": nearest_df.to_dict(orient="records"),
        "query_2d": {
            "tsne": {"x": round(float(q_tsne[0]), 3), "y": round(float(q_tsne[1]), 3)},
            "pca": {"x": round(float(q_pca[0]), 3), "y": round(float(q_pca[1]), 3)},
            "neighbors": neighbor_coords,
        },
    }


@app.post("/api/similarity")
def api_similarity(req: SimilarityRequest):
    init_state()
    texts = [req.text_a.strip(), req.text_b.strip(), req.text_c.strip()]
    if any(not t for t in texts):
        return JSONResponse({"error": "All three sentences must be non-empty."}, status_code=400)

    emb_abc, elapsed = encode_texts(texts, model_name=PRIMARY_MODEL_NAME)
    llm_sim = cosine_similarity(emb_abc)

    try:
        vec = TfidfVectorizer(stop_words="english")
        tfidf_mat = vec.fit_transform(texts).toarray()
        tfidf_sim = cosine_similarity(tfidf_mat)
    except Exception:
        tfidf_sim = np.zeros((3, 3))

    stop = {
        "is", "are", "was", "were", "the", "a", "an", "in", "on", "of",
        "to", "and", "for", "has", "have", "with", "by", "from", "that", "this",
    }

    def get_shared_words(s1: str, s2: str) -> List[str]:
        w1 = {w.strip(".,!?;:\"'()[]") for w in s1.lower().split()} - stop
        w2 = {w.strip(".,!?;:\"'()[]") for w in s2.lower().split()} - stop
        w1.discard("")
        w2.discard("")
        return sorted(list(w1.intersection(w2)))

    def verdict(score: float) -> dict:
        if score >= 0.40:
            return {"label": "Related (High Semantic Similarity)", "level": "high"}
        elif score >= 0.22:
            return {"label": "Partially Related (Moderate)", "level": "medium"}
        return {"label": "Unrelated (Different Topics)", "level": "low"}

    pairs_cfg = [
        ("Text A vs. Text B", 0, 1),
        ("Text A vs. Text C", 0, 2),
        ("Text B vs. Text C", 1, 2),
    ]
    pair_results = []
    for name, i, j in pairs_cfg:
        s_llm = round(float(llm_sim[i, j]), 4)
        s_tfidf = round(float(tfidf_sim[i, j]), 4)
        shared = get_shared_words(texts[i], texts[j])
        v = verdict(s_llm)
        pair_results.append({
            "pair": name,
            "shared_count": len(shared),
            "shared_words": shared,
            "tfidf_sim": s_tfidf,
            "llm_sim": s_llm,
            "verdict_label": v["label"],
            "verdict_level": v["level"],
        })

    best_pair = max(pair_results, key=lambda x: x["llm_sim"])

    return {
        "texts": {"A": texts[0], "B": texts[1], "C": texts[2]},
        "pairs": pair_results,
        "best_pair": best_pair,
        "matrix_llm": np.round(llm_sim, 4).tolist(),
        "matrix_tfidf": np.round(tfidf_sim, 4).tolist(),
        "encoding_time_ms": round(elapsed * 1000, 2),
    }


@app.get("/api/clustering")
def api_clustering(
    k: int = Query(5, ge=2, le=10),
    projection: str = Query("tsne"),
):
    init_state()
    bundle = STATE["bundle"]
    df = STATE["df"]
    emb_l6 = STATE["emb_bundle"][PRIMARY_MODEL_NAME]
    projections = STATE["projections"]

    if k == 5:
        clust_res = bundle["clustering_res_k5"]
    else:
        clust_res = run_kmeans_clustering(emb_l6, df, n_clusters=k)

    coords = projections["tsne_2d"] if projection == "tsne" else projections["pca_2d"]
    cluster_ids = clust_res["cluster_ids"]
    theme_map = clust_res["cluster_to_theme"]

    points = []
    for i in range(len(df)):
        c_id = int(cluster_ids[i])
        points.append({
            "doc_id": df.iloc[i]["doc_id"],
            "x": round(float(coords[i, 0]), 3),
            "y": round(float(coords[i, 1]), 3),
            "true_category": df.iloc[i]["category"],
            "subtopic": df.iloc[i]["subtopic"],
            "cluster_id": c_id,
            "cluster_label": theme_map[c_id],
            "snippet": df.iloc[i]["text"][:150] + "...",
        })

    return {
        "k": k,
        "projection": projection,
        "silhouette_score": clust_res["silhouette_score"],
        "ari_score": clust_res["ari_score"],
        "nmi_score": clust_res["nmi_score"],
        "inertia": clust_res["inertia"],
        "clusters": clust_res["cluster_summary_df"].to_dict(orient="records"),
        "points": points,
        "k_sweep": bundle["k_sweep_df"].to_dict(orient="records"),
    }


@app.get("/api/topics")
def api_topics(
    scope: str = Query("category"),
    category: str = Query("Technology"),
    n_topics: int = Query(4, ge=2, le=8),
):
    init_state()
    df = STATE["df"]
    emb_l6 = STATE["emb_bundle"][PRIMARY_MODEL_NAME]

    if scope == "category" and category in CATEGORIES:
        res = discover_category_subtopics(df, emb_l6, category=category, n_subtopics=n_topics)
    else:
        res = compute_ctfidf_topics(df, emb_l6, n_topics=n_topics, top_n_words=8)

    topics_list = []
    summary_records = res["topic_summary_df"].to_dict(orient="records")
    kw_dict = res["topic_keywords_dict"]

    for rec in summary_records:
        theme = rec["Discovered Theme"]
        kw_pairs = kw_dict.get(theme, [])
        topics_list.append({
            **rec,
            "keywords": [{"word": w, "weight": wt} for w, wt in kw_pairs],
        })

    return {
        "scope": scope,
        "category": category if scope == "category" else "Full Corpus (All 1,000 Docs)",
        "n_topics": len(topics_list),
        "topic_diversity": res["topic_diversity"],
        "mean_coherence": res["mean_coherence"],
        "topics": topics_list,
    }


@app.post("/api/analyze-custom-batch")
def api_analyze_custom_batch(req: CustomBatchRequest):
    """
    Allows any user to paste a custom list of sentences/paragraphs and runs:
    1. Pretrained LLM Embedding generation (`all-MiniLM-L6-v2`)
    2. Supervised Category Classification for each text (with low-confidence detection)
    3. Deterministic Agglomerative Cosine Clustering for small batches (N <= 15)
       or K-Means++ for larger batches.
    """
    init_state()
    bundle = STATE["bundle"]
    clf = bundle["trained_classifiers"]["Logistic Regression"]

    clean_list = [t.strip() for t in req.texts if t.strip()]
    if len(clean_list) < 2:
        return JSONResponse(
            {"error": "Please provide at least 2 non-empty sentences for batch clustering."},
            status_code=400,
        )

    k = min(max(2, req.n_clusters), len(clean_list))
    embs, elapsed = encode_texts(clean_list, model_name=PRIMARY_MODEL_NAME)

    preds = clf.predict(embs)
    probs = clf.predict_proba(embs).max(axis=1)

    if k == len(clean_list):
        c_ids = np.arange(len(clean_list))
    elif len(clean_list) <= 15:
        agg = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        c_ids = agg.fit_predict(embs)
    else:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_SEED)
        c_ids = km.fit_predict(embs)

    # Renumber cluster IDs sequentially (Group 1, Group 2, ...) in order of first appearance
    remap = {}
    ordered_c_ids = []
    for cid in c_ids:
        if int(cid) not in remap:
            remap[int(cid)] = len(remap) + 1
        ordered_c_ids.append(remap[int(cid)])

    items = []
    for idx, txt in enumerate(clean_list):
        conf_val = float(probs[idx])
        is_low = conf_val < 0.45
        items.append({
            "index": idx + 1,
            "text": txt,
            "predicted_category": str(preds[idx]),
            "confidence_pct": round(conf_val * 100, 1),
            "is_low_confidence": is_low,
            "cluster_id": f"Group {ordered_c_ids[idx]}",
        })

    return {
        "total_texts": len(clean_list),
        "n_clusters": k,
        "encoding_time_ms": round(elapsed * 1000, 2),
        "items": items,
    }


app.mount("/figures", StaticFiles(directory=FIGURES_DIR), name="figures")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
@app.head("/")
def serve_index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print("\n" + "=" * 70)
    print(" SemantiX Public Interactive Web Application")
    print(f" URL: http://localhost:{port}")
    print(" (Note: Original Streamlit app.py remains untouched for faculty review)")
    print("=" * 70 + "\n")
    uvicorn.run(app, host=host, port=port)
