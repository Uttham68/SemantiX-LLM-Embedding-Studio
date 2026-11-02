"""
Master Experiment & Pipeline Runner (`run_experiments.py`)
Responsible: Member 5 (System Integration, Experimental Benchmarking & Artifact Generation)

Running this script (`python run_experiments.py`) executes all 4 project modules in sequence:
1. Loads/builds the 1,000-document 5-category benchmark dataset (`data/dataset.csv`).
2. Computes & caches pretrained LLM embeddings (`all-MiniLM-L6-v2` & `paraphrase-MiniLM-L3-v2`)
   alongside a classical lexical TF-IDF baseline.
3. Runs Experiments A, B, C, and D (Classification, Clustering, K-Sweep, Paraphrase Generalization,
   and c-TF-IDF Topic Discovery).
4. Saves all quantitative tables to `outputs/results/*.csv`, trained models to
   `outputs/results/trained_pipeline.joblib`, and all 7 figures to `outputs/figures/*.png`.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.data_loader import (
    BASE_DIR,
    CATEGORIES,
    load_or_create_dataset,
    get_dataset_summary,
    get_train_test_split,
)
from src.embeddings import (
    PRIMARY_MODEL_NAME,
    SECONDARY_MODEL_NAME,
    get_or_create_corpus_embeddings,
    compute_tfidf_features,
    encode_texts,
    evaluate_semantic_similarity_demo,
)
from src.models import (
    train_and_evaluate_classifier,
    get_paraphrase_generalization_test_set,
    run_kmeans_clustering,
    evaluate_k_selection_sweep,
)
from src.topic_visualizer import (
    compute_ctfidf_topics,
    discover_category_subtopics,
    compute_or_load_2d_projections,
    generate_all_visualizations,
)

RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")
PIPELINE_PATH = os.path.join(RESULTS_DIR, "trained_pipeline.joblib")


def run_all_experiments(force_rebuild: bool = False) -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 72)
    print(" SEMANTIC TEXT ANALYSIS & TOPIC DISCOVERY USING PRETRAINED LLM EMBEDDINGS")
    print("=" * 72)

    # ------------------------------------------------------------------
    # STEP 1: Dataset Loading & Preprocessing (Member 1)
    # ------------------------------------------------------------------
    df = load_or_create_dataset(force_rebuild=force_rebuild)
    summary = get_dataset_summary(df)
    summary["per_category_table"].to_csv(
        os.path.join(RESULTS_DIR, "dataset_category_stats.csv"), index=False
    )
    print(
        f"[Step 1] Loaded {summary['total_documents']} documents across "
        f"{summary['num_categories']} categories: {summary['categories']}"
    )

    # Stratified Train/Test indices (750 train, 250 test)
    train_df, test_df = get_train_test_split(df, test_size=0.25)
    train_idx = df["doc_id"].isin(set(train_df["doc_id"])).values
    test_idx = ~train_idx
    y_train = df.loc[train_idx, "category"].values
    y_test = df.loc[test_idx, "category"].values

    # ------------------------------------------------------------------
    # STEP 2: Pretrained LLM Embedding Generation & Similarity (Member 2)
    # ------------------------------------------------------------------
    emb_bundle = get_or_create_corpus_embeddings(df, force_recompute=force_rebuild)
    emb_l6 = emb_bundle[PRIMARY_MODEL_NAME]
    emb_l3 = emb_bundle[SECONDARY_MODEL_NAME]
    time_l6 = emb_bundle["time_minilm_l6"]
    time_l3 = emb_bundle["time_minilm_l3"]

    # Classical TF-IDF baseline fitted on training texts only (to prevent data leakage)
    from sklearn.feature_extraction.text import TfidfVectorizer
    import time

    t0 = time.perf_counter()
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1500, stop_words="english", ngram_range=(1, 2), norm="l2"
    )
    X_train_tfidf = tfidf_vectorizer.fit_transform(df.loc[train_idx, "text"]).toarray().astype(np.float32)
    X_test_tfidf = tfidf_vectorizer.transform(df.loc[test_idx, "text"]).toarray().astype(np.float32)
    X_all_tfidf = tfidf_vectorizer.transform(df["text"]).toarray().astype(np.float32)
    time_tfidf = time.perf_counter() - t0

    # Semantic Similarity Demo (Text A, Text B, Text C + Finance & Cricket paraphrases)
    sim_demo_df, sim_matrix, sim_sentences = evaluate_semantic_similarity_demo(PRIMARY_MODEL_NAME)
    sim_demo_df.to_csv(os.path.join(RESULTS_DIR, "semantic_similarity_demo.csv"), index=False)
    print("\n[Step 2] Semantic Similarity Benchmark (TF-IDF vs Pretrained LLM):")
    print(
        sim_demo_df[["Relationship", "Shared Keywords", "TF-IDF Cosine Sim", "LLM Embedding Sim"]].to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # STEP 3: Experiment A & B — Classification & Representation Benchmark (Member 3)
    # ------------------------------------------------------------------
    paraphrase_df = get_paraphrase_generalization_test_set()
    para_texts = paraphrase_df["text"].tolist()
    para_y = paraphrase_df["category"].values

    # Encode paraphrase test set with all 3 representations
    X_para_tfidf = tfidf_vectorizer.transform(para_texts).toarray().astype(np.float32)
    X_para_l3, _ = encode_texts(para_texts, model_name=SECONDARY_MODEL_NAME)
    X_para_l6, _ = encode_texts(para_texts, model_name=PRIMARY_MODEL_NAME)

    # Evaluate all 3 representations with Logistic Regression + K-Means (K=5)
    clf_tfidf = train_and_evaluate_classifier(X_train_tfidf, y_train, X_test_tfidf, y_test, "Logistic Regression")
    clf_l3 = train_and_evaluate_classifier(
        emb_l3[train_idx], y_train, emb_l3[test_idx], y_test, "Logistic Regression"
    )
    clf_l6 = train_and_evaluate_classifier(
        emb_l6[train_idx], y_train, emb_l6[test_idx], y_test, "Logistic Regression"
    )

    para_acc_tfidf = accuracy_score(para_y, clf_tfidf["model"].predict(X_para_tfidf))
    para_acc_l3 = accuracy_score(para_y, clf_l3["model"].predict(X_para_l3))
    para_acc_l6 = accuracy_score(para_y, clf_l6["model"].predict(X_para_l6))

    clust_tfidf = run_kmeans_clustering(X_all_tfidf, df, n_clusters=5)
    clust_l3 = run_kmeans_clustering(emb_l3, df, n_clusters=5)
    clust_l6 = run_kmeans_clustering(emb_l6, df, n_clusters=5)

    model_comparison_df = pd.DataFrame([
        {
            "Representation Model": "TF-IDF (Lexical Baseline)",
            "Vector Dim": X_all_tfidf.shape[1],
            "Test Accuracy": clf_tfidf["accuracy"],
            "Test Macro F1": clf_tfidf["f1_score"],
            "Paraphrase Generalization Acc": round(float(para_acc_tfidf), 4),
            "Clustering Silhouette": clust_tfidf["silhouette_score"],
            "Clustering ARI": clust_tfidf["ari_score"],
            "Clustering NMI": clust_tfidf["nmi_score"],
            "Encoding Time (s)": round(float(time_tfidf), 2),
        },
        {
            "Representation Model": "paraphrase-MiniLM-L3-v2 (3-Layer LLM)",
            "Vector Dim": emb_l3.shape[1],
            "Test Accuracy": clf_l3["accuracy"],
            "Test Macro F1": clf_l3["f1_score"],
            "Paraphrase Generalization Acc": round(float(para_acc_l3), 4),
            "Clustering Silhouette": clust_l3["silhouette_score"],
            "Clustering ARI": clust_l3["ari_score"],
            "Clustering NMI": clust_l3["nmi_score"],
            "Encoding Time (s)": round(float(time_l3), 2),
        },
        {
            "Representation Model": "all-MiniLM-L6-v2 (6-Layer LLM)",
            "Vector Dim": emb_l6.shape[1],
            "Test Accuracy": clf_l6["accuracy"],
            "Test Macro F1": clf_l6["f1_score"],
            "Paraphrase Generalization Acc": round(float(para_acc_l6), 4),
            "Clustering Silhouette": clust_l6["silhouette_score"],
            "Clustering ARI": clust_l6["ari_score"],
            "Clustering NMI": clust_l6["nmi_score"],
            "Encoding Time (s)": round(float(time_l6), 2),
        },
    ])
    model_comparison_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
    print("\n[Step 3A] Experiment A — Representation & Embedding Model Comparison:")
    print(model_comparison_df.to_string(index=False))

    # Experiment B: Classifier Head Comparison on `all-MiniLM-L6-v2`
    clf_rows = []
    trained_classifiers = {}
    for clf_name in ["Logistic Regression", "Linear SVM", "Random Forest"]:
        res = train_and_evaluate_classifier(
            emb_l6[train_idx], y_train, emb_l6[test_idx], y_test, clf_name
        )
        para_acc = accuracy_score(para_y, res["model"].predict(X_para_l6))
        trained_classifiers[clf_name] = res["model"]
        clf_rows.append({
            "Classifier Head": clf_name,
            "Test Accuracy": res["accuracy"],
            "Macro Precision": res["precision"],
            "Macro Recall": res["recall"],
            "Macro F1-Score": res["f1_score"],
            "Paraphrase Generalization Acc": round(float(para_acc), 4),
            "Train Time (s)": res["train_time_sec"],
        })

    classifier_comparison_df = pd.DataFrame(clf_rows)
    classifier_comparison_df.to_csv(os.path.join(RESULTS_DIR, "classifier_comparison.csv"), index=False)
    clf_l6["per_class_df"].to_csv(os.path.join(RESULTS_DIR, "per_class_metrics.csv"), index=False)
    print("\n[Step 3B] Experiment B — Classifier Head Comparison (on all-MiniLM-L6-v2):")
    print(classifier_comparison_df.to_string(index=False))

    # Experiment C: Unsupervised Clustering K-Selection Sweep (K = 2..10)
    k_sweep_df = evaluate_k_selection_sweep(emb_l6, df["category"])
    k_sweep_df.to_csv(os.path.join(RESULTS_DIR, "k_selection_sweep.csv"), index=False)
    clust_l6["cluster_summary_df"].to_csv(os.path.join(RESULTS_DIR, "cluster_summary.csv"), index=False)
    print("\n[Step 3C] Experiment C — Unsupervised K-Means Clustering (K=5):")
    print(
        clust_l6["cluster_summary_df"][
            ["Cluster ID", "Emerging Theme", "Document Count", "Cluster Purity (%)", "Avg Centroid Cosine Sim"]
        ].to_string(index=False)
    )

    # ------------------------------------------------------------------
    # STEP 4: Experiment D — Topic Discovery & Visualizations (Member 4)
    # ------------------------------------------------------------------
    corpus_topics = compute_ctfidf_topics(df, emb_l6, n_topics=5, top_n_words=8)
    corpus_topics["topic_summary_df"].to_csv(os.path.join(RESULTS_DIR, "corpus_topics.csv"), index=False)

    tech_subtopics = discover_category_subtopics(df, emb_l6, category="Technology", n_subtopics=4)
    tech_subtopics["topic_summary_df"].to_csv(
        os.path.join(RESULTS_DIR, "technology_subtopics.csv"), index=False
    )

    print("\n[Step 4A] Experiment D — Discovered Technology Sub-Topics (c-TF-IDF):")
    print(
        tech_subtopics["topic_summary_df"][
            ["Topic ID", "Discovered Theme", "Documents", "Top Keywords (c-TF-IDF)", "Semantic Coherence"]
        ].to_string(index=False)
    )
    print(
        f"Topic Diversity: {tech_subtopics['topic_diversity']:.3f} | "
        f"Mean Intra-Topic Coherence: {tech_subtopics['mean_coherence']:.3f}"
    )

    # 2D Projections (PCA & t-SNE) + All 7 Figures
    projections = compute_or_load_2d_projections(emb_l6, force_recompute=force_rebuild)
    fig_paths = generate_all_visualizations(
        df=df,
        projections=projections,
        clf_eval_l6=clf_l6,
        clf_eval_tfidf=clf_tfidf,
        clustering_res=clust_l6,
        k_sweep_df=k_sweep_df,
        sim_demo_df=sim_demo_df,
        sim_matrix=sim_matrix,
        sim_sentences=sim_sentences,
        model_comparison_df=model_comparison_df,
        corpus_topics=corpus_topics,
        tech_subtopics=tech_subtopics,
    )
    print(f"\n[Step 4B] Generated {len(fig_paths)} high-resolution figures in outputs/figures/")

    # Save complete pipeline state for instant loading in Streamlit
    pipeline_bundle = {
        "primary_model_name": PRIMARY_MODEL_NAME,
        "secondary_model_name": SECONDARY_MODEL_NAME,
        "categories": CATEGORIES,
        "dataset_summary": summary,
        "trained_classifiers": trained_classifiers,
        "tfidf_vectorizer": tfidf_vectorizer,
        "tfidf_classifier": clf_tfidf["model"],
        "clf_eval_l6": clf_l6,
        "clf_eval_tfidf": clf_tfidf,
        "model_comparison_df": model_comparison_df,
        "classifier_comparison_df": classifier_comparison_df,
        "sim_demo_df": sim_demo_df,
        "sim_matrix": sim_matrix,
        "sim_sentences": sim_sentences,
        "clustering_res_k5": clust_l6,
        "k_sweep_df": k_sweep_df,
        "corpus_topics": corpus_topics,
        "tech_subtopics": tech_subtopics,
        "fig_paths": fig_paths,
    }
    joblib.dump(pipeline_bundle, PIPELINE_PATH)
    print(f"[Step 5] Saved complete trained pipeline bundle to {PIPELINE_PATH}")
    print("=" * 72)
    return pipeline_bundle


if __name__ == "__main__":
    run_all_experiments(force_rebuild=False)

