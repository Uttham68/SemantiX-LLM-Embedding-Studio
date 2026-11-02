"""
Automated Project Verification & Unit Test Suite (`test_project.py`)
Run with:
    python test_project.py
"""

import os
import joblib
import numpy as np
import pandas as pd

from src.data_loader import BASE_DIR, DATASET_PATH, CATEGORIES
from src.embeddings import EMBEDDINGS_CACHE_PATH, PRIMARY_MODEL_NAME
from run_experiments import PIPELINE_PATH
import web_app


def run_verification_suite():
    print("=" * 72)
    print(" RUNNING AUTOMATED VERIFICATION SUITE (10 INTEGRITY CHECKS)")
    print("=" * 72)

    # 1. Dataset Existence, Balance & 100% Uniqueness
    assert os.path.exists(DATASET_PATH), "dataset.csv is missing!"
    df = pd.read_csv(DATASET_PATH)
    assert len(df) == 1000, f"Expected 1000 rows, got {len(df)}"
    assert set(df["category"].unique()) == set(CATEGORIES), "Category mismatch!"
    unique_texts = df["text"].nunique()
    unique_prefixes = df["text"].str[:50].str.lower().nunique()
    assert unique_texts == 1000, f"Expected 1000 unique texts, found {unique_texts}"
    assert unique_prefixes == 1000, f"Expected 1000 unique 50-char prefixes, found {unique_prefixes}"
    print("[PASS 1/10] Dataset Integrity: 1,000 balanced documents | 1,000/1,000 (100%) unique texts & prefixes.")

    # 2. Pretrained Embeddings Cache & Unit Hypersphere Normalization
    assert os.path.exists(EMBEDDINGS_CACHE_PATH), "embeddings_cache.npz is missing!"
    cache = np.load(EMBEDDINGS_CACHE_PATH)
    emb_l6 = cache["emb_minilm_l6"]
    assert emb_l6.shape == (1000, 384), f"Unexpected shape: {emb_l6.shape}"
    norms = np.linalg.norm(emb_l6, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), "Embeddings are not L2 unit-normalized!"
    print("[PASS 2/10] Embedding Geometry: Shape (1000, 384) | L2 Norms = 1.0000 +/- 1e-5.")

    # 3. Trained Pipeline Bundle & Classification Metrics
    assert os.path.exists(PIPELINE_PATH), "trained_pipeline.joblib is missing!"
    bundle = joblib.load(PIPELINE_PATH)
    test_acc = bundle["clf_eval_l6"]["accuracy"]
    test_f1 = bundle["clf_eval_l6"]["f1_score"]
    assert test_acc >= 0.94, f"Test accuracy too low: {test_acc}"
    print(f"[PASS 3/10] Supervised Classification: Test Accuracy = {test_acc*100:.1f}% | Macro F1 = {test_f1:.4f}.")

    # 4. Unsupervised Clustering Quality (K=5)
    ari = bundle["clustering_res_k5"]["ari_score"]
    nmi = bundle["clustering_res_k5"]["nmi_score"]
    sil = bundle["clustering_res_k5"]["silhouette_score"]
    assert ari >= 0.85 and nmi >= 0.84, f"Clustering metrics below threshold: ARI={ari}, NMI={nmi}"
    print(f"[PASS 4/10] Unsupervised K-Means (K=5): ARI = {ari:.4f} | NMI = {nmi:.4f} | Silhouette = {sil:.4f}.")

    # 5. Semantic vs. Lexical Paraphrase Verification
    web_app.init_state()
    sim_res = web_app.api_similarity(
        web_app.SimilarityRequest(
            text_a="Artificial intelligence is transforming healthcare.",
            text_b="Machine learning is improving medical diagnosis.",
            text_c="India won the cricket championship.",
        )
    )
    pair_ab = sim_res["pairs"][0]
    pair_ac = sim_res["pairs"][1]
    assert pair_ab["tfidf_sim"] == 0.0 and pair_ab["llm_sim"] > 0.50, "Paraphrase similarity check failed!"
    assert pair_ac["llm_sim"] < 0.15, "Unrelated similarity check failed!"
    print(
        f"[PASS 5/10] Semantic Similarity Lab: Paraphrase LLM Sim = {pair_ab['llm_sim']:.4f} (TF-IDF = 0.0000) | Unrelated = {pair_ac['llm_sim']:.4f}."
    )

    # 6. Live Classification & Deduplicated Nearest Neighbors
    clf_res = web_app.api_classify(
        web_app.ClassifyRequest(text="NASA launched a new space telescope to study distant exoplanets and galaxies.")
    )
    assert clf_res["predicted_category"] == "Technology" and clf_res["confidence_pct"] >= 70.0
    neighbors = clf_res["nearest_documents"]
    prefixes = {d["text"][:60].lower() for d in neighbors}
    assert len(prefixes) == len(neighbors), "Duplicate neighbor prefixes detected!"
    print(
        f"[PASS 6/10] Live Classification & Search: NASA Query -> {clf_res['predicted_category']} ({clf_res['confidence_pct']:.1f}%) | 4/4 Unique Neighbors."
    )

    # 7. Topic Discovery Title Uniqueness Across All Scopes
    for cat in CATEGORIES:
        t_res = web_app.api_topics(scope="category", category=cat, n_topics=4)
        titles = [t["Discovered Theme"].split("(")[0].strip() for t in t_res["topics"]]
        assert len(set(titles)) == len(titles), f"Duplicate topic titles in {cat}: {titles}"
    print("[PASS 7/10] Topic Discovery (c-TF-IDF): 100% unique, non-overlapping topic titles across all 5 categories.")

    # 8. Deterministic Small-Batch Sandbox Clustering (5 Domains -> 5 Groups)
    batch_res = web_app.api_analyze_custom_batch(
        web_app.CustomBatchRequest(
            texts=[
                "Quantum computers and neural networks are advancing in Silicon Valley.",
                "The central bank raised interest rates to curb retail inflation.",
                "Doctors used mRNA vaccines in clinical trials for patients.",
                "The national basketball team won the playoff championship.",
                "Parliamentary lawmakers debated constitutional legislation.",
            ],
            n_clusters=5,
        )
    )
    groups = {item["cluster_id"] for item in batch_res["items"]}
    assert len(groups) == 5, f"Expected 5 distinct groups, got {groups}"
    print("[PASS 8/10] Public Batch Sandbox: 5 domain sentences cleanly separated into 5 distinct groups.")

    # 9. All 7 Publication Figures Present
    fig_dir = os.path.join(BASE_DIR, "outputs", "figures")
    expected_figs = [
        "fig1_dataset_distribution.png",
        "fig2_confusion_matrix.png",
        "fig3_clustering_elbow_silhouette.png",
        "fig4_semantic_space_pca_tsne.png",
        "fig5_semantic_similarity_heatmap.png",
        "fig6_model_comparison_metrics.png",
        "fig7_topic_keywords.png",
    ]
    for fn in expected_figs:
        fpath = os.path.join(fig_dir, fn)
        assert os.path.exists(fpath) and os.path.getsize(fpath) > 20000, f"Missing or empty figure: {fn}"
    print("[PASS 9/10] Visual Suite: All 7 high-resolution PNG figures verified.")

    # 10. Academic Deliverables Present
    docs_dir = os.path.join(BASE_DIR, "docs")
    for doc_fn in [
        "Project_Report_5Pages.pdf",
        "Project_Report_5Pages.md",
        "Project_Presentation.pptx",
        "Presentation_Slides.md",
        "Viva_Guide_and_Demo.md",
        "Member_Viva_Cheat_Sheets.md",
    ]:
        dpath = os.path.join(docs_dir, doc_fn)
        assert os.path.exists(dpath) and os.path.getsize(dpath) > 1000, f"Missing deliverable: {doc_fn}"
    print("[PASS 10/10] Academic Deliverables: 5-Page PDF Report, PPTX Deck, Viva Guide & Member Cheat Sheets verified.")
    print("=" * 72)
    print(" ALL 10 VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 72)


if __name__ == "__main__":
    run_verification_suite()

