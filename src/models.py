"""
Module 3: Supervised Text Classification & Unsupervised Document Clustering
Responsible: Member 3 (Classification, K-Means Clustering, and Quantitative Evaluation)

This module handles:
1. Supervised Text Classification on Pretrained LLM Embeddings:
   - Compares Logistic Regression (C=12.0 calibrated for unit-norm embeddings),
     Calibrated Linear SVM, and Random Forest.
   - Benchmarks Pretrained LLM Embeddings (`all-MiniLM-L6-v2`, `paraphrase-MiniLM-L3-v2`)
     against Traditional Lexical TF-IDF on both standard test data and a Synonym/Paraphrase
     Generalization benchmark.
   - Computes Accuracy, Precision, Recall, F1-Score, Per-Class metrics, and Confusion Matrix.
2. Unsupervised Text Clustering using K-Means:
   - Groups unlabeled documents in 384-dimensional embedding space.
   - Evaluates cluster count selection across K = 2..10 using the Elbow Method (Inertia)
     and Silhouette Score.
   - Evaluates cluster quality using Silhouette Score, Adjusted Rand Index (ARI), and
     Normalized Mutual Information (NMI).
"""

import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from src.data_loader import CATEGORIES, RANDOM_SEED
from src.embeddings import encode_texts, PRIMARY_MODEL_NAME


# =====================================================================
# PART A: SUPERVISED TEXT CLASSIFICATION
# =====================================================================

def build_classifier(classifier_type: str = "Logistic Regression"):
    """
    Creates a lightweight, well-calibrated supervised classifier head on top of
    frozen 384-dimensional unit-normalized pretrained LLM embeddings.
    Note: Because embeddings are L2-normalized (||x||_2 = 1), C=12.0 acts as an
    inverse temperature parameter (1/tau) so softmax probabilities are sharp and
    well-calibrated rather than squashed toward 1/K = 20%.
    """
    if classifier_type == "Logistic Regression":
        return LogisticRegression(C=12.0, max_iter=1000, random_state=RANDOM_SEED)
    elif classifier_type == "Linear SVM":
        base_svm = LinearSVC(C=2.0, random_state=RANDOM_SEED)
        return CalibratedClassifierCV(base_svm, cv=5)
    elif classifier_type == "Random Forest":
        base_rf = RandomForestClassifier(
            n_estimators=150, max_depth=16, random_state=RANDOM_SEED, n_jobs=-1
        )
        return CalibratedClassifierCV(base_rf, cv=3)
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")


def train_and_evaluate_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifier_type: str = "Logistic Regression",
    category_names: List[str] = CATEGORIES,
) -> Dict[str, object]:
    """
    Trains a classifier on embeddings and computes comprehensive evaluation metrics.
    """
    clf = build_classifier(classifier_type)

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = clf.predict(X_test)
    infer_time = time.perf_counter() - t1

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=category_names)

    p_cls, r_cls, f1_cls, sup_cls = precision_recall_fscore_support(
        y_test, y_pred, labels=category_names, zero_division=0
    )
    per_class_df = pd.DataFrame({
        "Category": category_names,
        "Precision": np.round(p_cls, 4),
        "Recall": np.round(r_cls, 4),
        "F1-Score": np.round(f1_cls, 4),
        "Support": sup_cls,
    })

    return {
        "model": clf,
        "classifier_type": classifier_type,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "train_time_sec": round(float(train_time), 4),
        "infer_time_sec": round(float(infer_time), 4),
        "confusion_matrix": cm,
        "per_class_df": per_class_df,
        "y_pred": y_pred,
    }


def get_paraphrase_generalization_test_set() -> pd.DataFrame:
    """
    Creates an out-of-distribution generalization benchmark of 25 sentences written with
    domain synonyms and varied phrasing (including the prompt's test examples) to test
    semantic generalization vs. keyword memorization.
    """
    samples = [
        # Sports (5)
        ("India defeated Australia in the final match.", "Sports"),
        ("The striker netted a stoppage-time header to clinch the championship cup.", "Sports"),
        ("Fast bowlers delivered yorkers during the decisive T20 cricket innings.", "Sports"),
        ("Olympic marathon runners paced themselves through the final stadium lap.", "Sports"),
        ("Grand slam tennis finalists rallied from two sets down on center court.", "Sports"),
        # Technology (5)
        ("New artificial intelligence technology was released.", "Technology"),
        ("Quantum computers and neural networks are advancing rapidly in Silicon Valley.", "Technology"),
        ("Zero-day cryptographic security patch mitigated remote ransomware flaws.", "Technology"),
        ("Serverless cloud microservices scaled automatically across data center regions.", "Technology"),
        ("NASA launched a new space telescope to study distant exoplanets and galaxies.", "Technology"),
        # Finance (5)
        ("The stock market recorded a significant increase.", "Finance"),
        ("Bullish Wall Street equities surged after strong quarterly corporate dividends.", "Finance"),
        ("Central bank policymakers hiked benchmark lending rates to cool retail inflation.", "Finance"),
        ("Institutional bondholders rebalanced sovereign fixed-income investment portfolios.", "Finance"),
        ("Venture capitalists underwrote a multibillion fintech IPO market valuation.", "Finance"),
        # Politics (5)
        ("Senators ratified the bipartisan appropriations legislation in parliament.", "Politics"),
        ("Parliamentary candidates debated electoral redistricting during the campaign.", "Politics"),
        ("Diplomatic ambassadors negotiated a multilateral international peace treaty.", "Politics"),
        ("Coalition lawmakers called a parliamentary confidence vote on constitutional reform.", "Politics"),
        ("Cabinet ministers reviewed national budget policy and regulatory mandates.", "Politics"),
        # Healthcare (5)
        ("Artificial intelligence is transforming healthcare.", "Healthcare"),
        ("Machine learning is improving medical diagnosis.", "Healthcare"),
        ("Oncologists prescribed targeted immunotherapy after clinical biopsy screening.", "Healthcare"),
        ("Randomized hospital cohorts demonstrated mRNA vaccine efficacy against viral pathogens.", "Healthcare"),
        ("Cardiovascular surgeons monitored postoperative intensive care patient vitals.", "Healthcare"),
    ]
    return pd.DataFrame(samples, columns=["text", "category"])


def predict_single_text(
    text: str,
    clf,
    model_name: str = PRIMARY_MODEL_NAME,
    precomputed_emb: np.ndarray = None,
    infer_time_ms: float = None,
) -> Dict[str, object]:
    """
    Predicts the category and calibrated confidence distribution for a single input sentence
    using the pretrained LLM embedding model and trained classifier.
    Flags borderline/OOD predictions when max probability < 45%.
    """
    if precomputed_emb is not None:
        emb = precomputed_emb if precomputed_emb.ndim > 1 else precomputed_emb.reshape(1, -1)
        infer_t = (infer_time_ms / 1000.0) if infer_time_ms is not None else 0.005
    else:
        emb, infer_t = encode_texts([text], model_name=model_name)
    pred_label = clf.predict(emb)[0]

    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(emb)[0]
        class_order = list(clf.classes_)
        prob_dict = {cls_name: round(float(p), 4) for cls_name, p in zip(class_order, probs)}
        confidence = float(np.max(probs))
    else:
        prob_dict = {c: (1.0 if c == pred_label else 0.0) for c in CATEGORIES}
        confidence = 1.0

    prob_df = (
        pd.DataFrame(
            [{"Category": k, "Confidence": v, "Percentage": round(v * 100, 2)} for k, v in prob_dict.items()]
        )
        .sort_values("Confidence", ascending=False)
        .reset_index(drop=True)
    )

    is_low_confidence = bool(confidence < 0.45)

    return {
        "input_text": text,
        "predicted_category": pred_label,
        "confidence": round(confidence, 4),
        "confidence_pct": round(confidence * 100, 2),
        "is_low_confidence": is_low_confidence,
        "confidence_status": "Low Confidence / Borderline Domain" if is_low_confidence else "High Confidence (In-Domain)",
        "probabilities": prob_dict,
        "probability_df": prob_df,
        "encoding_time_ms": round(infer_t * 1000, 2),
    }


# =====================================================================
# PART B: UNSUPERVISED TEXT CLUSTERING
# =====================================================================

def align_clusters_to_categories(
    cluster_ids: np.ndarray, true_categories: pd.Series
) -> Tuple[Dict[int, str], pd.DataFrame]:
    unique_clusters = sorted(np.unique(cluster_ids))
    unique_cats = CATEGORIES

    contingency = pd.crosstab(cluster_ids, true_categories)
    for c in unique_cats:
        if c not in contingency.columns:
            contingency[c] = 0
    contingency = contingency[unique_cats]

    cost_matrix = -contingency.values
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    cluster_to_theme: Dict[int, str] = {}
    for r, c in zip(row_ind, col_ind):
        c_id = int(contingency.index[r])
        cluster_to_theme[c_id] = f"Cluster {c_id + 1} ({unique_cats[c]}-related)"

    for c_id in unique_clusters:
        if int(c_id) not in cluster_to_theme:
            dominant_cat = contingency.loc[c_id].idxmax()
            cluster_to_theme[int(c_id)] = f"Cluster {int(c_id) + 1} ({dominant_cat} Sub-group)"

    return cluster_to_theme, contingency


def run_kmeans_clustering(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    n_clusters: int = 5,
    random_state: int = RANDOM_SEED,
) -> Dict[str, object]:
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=random_state,
    )
    cluster_ids = kmeans.fit_predict(embeddings)
    inertia = float(kmeans.inertia_)
    sil_score = float(silhouette_score(embeddings, cluster_ids, metric="cosine"))

    true_labels = df["category"].values
    ari_score = float(adjusted_rand_score(true_labels, cluster_ids))
    nmi_score = float(normalized_mutual_info_score(true_labels, cluster_ids))

    cluster_to_theme, contingency = align_clusters_to_categories(cluster_ids, df["category"])

    centroids = kmeans.cluster_centers_
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    norm_centroids = centroids / centroid_norms

    cluster_summary_rows = []
    for c_id in range(n_clusters):
        mask = cluster_ids == c_id
        c_indices = np.where(mask)[0]
        c_docs = df.iloc[c_indices]

        sims = np.dot(embeddings[c_indices], norm_centroids[c_id])
        best_local_idx = int(np.argmax(sims))
        exemplar_row = c_docs.iloc[best_local_idx]
        dominant_true_cat = c_docs["category"].mode().iloc[0] if len(c_docs) > 0 else "N/A"
        purity = float((c_docs["category"] == dominant_true_cat).mean()) if len(c_docs) > 0 else 0.0

        cluster_summary_rows.append({
            "Cluster ID": f"Cluster {c_id + 1}",
            "Emerging Theme": cluster_to_theme[c_id],
            "Document Count": int(len(c_docs)),
            "Dominant Ground-Truth": dominant_true_cat,
            "Cluster Purity (%)": round(purity * 100, 1),
            "Avg Centroid Cosine Sim": round(float(np.mean(sims)), 4),
            "Exemplar Document": exemplar_row["text"][:140] + "...",
        })

    cluster_summary_df = pd.DataFrame(cluster_summary_rows)

    return {
        "kmeans_model": kmeans,
        "cluster_ids": cluster_ids,
        "n_clusters": n_clusters,
        "inertia": round(inertia, 2),
        "silhouette_score": round(sil_score, 4),
        "ari_score": round(ari_score, 4),
        "nmi_score": round(nmi_score, 4),
        "cluster_to_theme": cluster_to_theme,
        "contingency_table": contingency,
        "cluster_summary_df": cluster_summary_df,
    }


def evaluate_k_selection_sweep(
    embeddings: np.ndarray,
    true_categories: pd.Series,
    k_values: List[int] = [2, 3, 4, 5, 6, 7, 8, 9, 10],
) -> pd.DataFrame:
    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_SEED)
        labels = km.fit_predict(embeddings)
        inertia = float(km.inertia_)
        sil = float(silhouette_score(embeddings, labels, metric="cosine"))
        ari = float(adjusted_rand_score(true_categories, labels))
        nmi = float(normalized_mutual_info_score(true_categories, labels))

        rows.append({
            "K": k,
            "Inertia (WCSS)": round(inertia, 2),
            "Silhouette Score": round(sil, 4),
            "Adjusted Rand Index (ARI)": round(ari, 4),
            "Normalized Mutual Info (NMI)": round(nmi, 4),
        })

    return pd.DataFrame(rows)
