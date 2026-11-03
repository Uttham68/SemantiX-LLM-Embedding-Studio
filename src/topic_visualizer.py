"""
Module 4: Semantic Topic Discovery (c-TF-IDF), Dimensionality Reduction & Visualization
Responsible: Member 4 (Topic Modeling, PCA/t-SNE Projections, and Experimental Plots)

This module handles:
1. Embedding-based Topic Discovery using K-Means + Class-based TF-IDF (c-TF-IDF):
   - Groups document embeddings in semantic space.
   - Treats each semantic cluster as a single macro-document and applies Class-based TF-IDF
     to extract discriminative topic keywords and weights.
   - Maps extracted c-TF-IDF keywords to accurate, non-overlapping human-readable Topic Titles
     with strict deduplication across all clusters.
   - Computes Topic Coherence (Intra-Topic Embedding Similarity) and Topic Diversity.
2. Dimensionality Reduction (384D -> 2D) using PCA and t-SNE.
3. Generating all 7 high-resolution academic visualizations in `outputs/figures/`.
"""

import os
from typing import Dict, List, Tuple, Set
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import BASE_DIR, DATA_DIR, CATEGORIES, RANDOM_SEED, clean_text_for_lexical

FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
PROJECTIONS_CACHE_PATH = os.path.join(DATA_DIR, "projections_cache.npz")

PALETTE = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#06B6D4", "#84CC16"]


# =====================================================================
# PART A: TOPIC DISCOVERY VIA EMBEDDINGS + CLASS-BASED TF-IDF (c-TF-IDF)
# =====================================================================

CUSTOM_STOPWORDS = {
    "said", "also", "would", "could", "one", "two", "new", "year", "years", "time",
    "people", "first", "last", "many", "make", "made", "use", "used", "using", "way",
    "like", "well", "get", "may", "even", "still", "much", "take", "good", "know",
    "think", "see", "come", "work", "day", "want", "look", "need", "back", "mr",
    "number", "part", "including", "high", "across", "based", "major", "furthermore",
    "told", "added", "uk", "us", "bbc", "news", "don", "just", "going", "say", "says",
}

# Keyword-driven semantic theme taxonomy (prevents mislabelled or duplicate cluster names)
TOPIC_THEME_RULES: List[Tuple[str, str, Set[str]]] = [
    # Technology themes
    ("Technology", "Gaming & Digital Entertainment", {
        "games", "game", "sony", "console", "nintendo", "xbox", "playstation", "gaming", "halo", "psp", "video"
    }),
    ("Technology", "Cybersecurity & Email Security", {
        "virus", "spam", "mail", "security", "worm", "hackers", "ransomware", "phishing", "encryption", "firewall", "malware", "patch", "attack", "attacks", "search"
    }),
    ("Technology", "Telecom, Mobile & Broadband", {
        "broadband", "tv", "service", "phone", "phones", "telecom", "wireless", "3g", "vodafone", "bt", "calls", "net"
    }),
    ("Technology", "Consumer Electronics & Hardware", {
        "gadgets", "gadget", "mobile", "handset", "ibm", "pc", "laptop", "hardware", "camera", "consumer", "device", "devices", "chips"
    }),
    ("Technology", "Digital Media & File Sharing", {
        "music", "files", "file", "apple", "itunes", "piracy", "mp3", "download", "dvd", "software", "copyright", "sharing", "peer"
    }),
    ("Technology", "AI, Space & Frontier Tech", {
        "ai", "artificial", "intelligence", "neural", "quantum", "learning", "deep", "space", "nasa", "telescope", "exoplanets", "galaxies", "robotics", "chip", "gpu", "cloud", "silicon", "valley"
    }),
    ("Technology", "Search Engines & Web Portals", {
        "web", "site", "sites", "google", "yahoo", "browser", "blog", "blogs", "portal", "internet", "users", "online"
    }),

    # Sports themes
    ("Sports", "Grand Slam Tennis & Tournaments", {
        "tennis", "open", "australian", "wimbledon", "roddick", "federer", "henman", "seed", "set", "court", "davis"
    }),
    ("Sports", "Athletics, Track & Olympics", {
        "olympic", "athletics", "indoor", "race", "medal", "holmes", "sprint", "marathon", "record", "world", "iaaf"
    }),
    ("Sports", "International Cricket", {
        "cricket", "batsman", "bowling", "wickets", "wicket", "innings", "test", "overs", "t20", "australia", "india", "pakistan"
    }),
    ("Sports", "Football & Premier League", {
        "chelsea", "arsenal", "united", "liverpool", "club", "cup", "league", "manager", "striker", "goal", "football", "coach", "player", "players", "milan", "real"
    }),
    ("Sports", "International Rugby & Six Nations", {
        "rugby", "nations", "wales", "england", "ireland", "france", "scotland", "half", "try", "lions", "robinson", "wilkinson", "bath"
    }),

    # Finance themes
    ("Finance", "Stock Markets & Equities", {
        "market", "stock", "shares", "wall", "street", "equities", "profits", "earnings", "index", "dividend", "investors", "exchange", "ftse", "dow"
    }),
    ("Finance", "Banking, Inflation & Monetary Policy", {
        "bank", "rates", "interest", "inflation", "economy", "growth", "gdp", "central", "monetary", "dollar", "euro", "imf", "deficit", "economic"
    }),
    ("Finance", "Global Trade, Energy & Aviation", {
        "oil", "yukos", "energy", "prices", "crude", "gas", "airlines", "boeing", "airbus", "trade", "exports", "manufacturing", "russian", "china"
    }),
    ("Finance", "Corporate Mergers & Retail Business", {
        "sales", "retail", "company", "firm", "deal", "takeover", "merger", "bid", "shareholders", "chief", "christmas", "stores", "lse", "deutsche"
    }),

    # Politics themes
    ("Politics", "Elections & Party Campaigns", {
        "election", "labour", "blair", "brown", "tory", "tories", "howard", "party", "voters", "campaign", "polls", "liberal", "kennedy", "general"
    }),
    ("Politics", "Legislation, Law & Civil Policy", {
        "law", "court", "lords", "parliament", "bill", "rights", "police", "prison", "terror", "detention", "home", "constitutional", "hunting", "ban"
    }),
    ("Politics", "Public Finance & Fiscal Governance", {
        "tax", "budget", "spending", "public", "council", "pension", "pensions", "welfare", "minister", "chancellor", "taxes", "local"
    }),
    ("Politics", "International Diplomacy & EU Affairs", {
        "eu", "european", "iraq", "foreign", "un", "diplomatic", "treaty", "international", "summit", "constitution", "referendum", "straw"
    }),

    # Healthcare themes
    ("Healthcare", "Medical Diagnostics & Clinical AI", {
        "imaging", "diagnosis", "radiology", "screening", "oncology", "cardiovascular", "biomarkers", "diagnostic", "machine", "learning", "clinical"
    }),
    ("Healthcare", "Clinical Trials & Therapeutics", {
        "vaccine", "trials", "drug", "drugs", "therapy", "antibody", "mrna", "treatment", "patients", "fda", "dose", "effects", "pharmaceutical"
    }),
    ("Healthcare", "Public Health & Epidemiology", {
        "infection", "disease", "diseases", "hospital", "epidemiology", "syndrome", "chronic", "pain", "symptom", "symptoms", "infectious", "health"
    }),
    ("Healthcare", "Primary Care & Clinical Medicine", {
        "doctor", "doctors", "medical", "surgery", "physician", "care", "medicine", "nutrition", "diet", "cancer", "blood", "patient", "vitamin"
    }),
]


def infer_unique_topic_title(
    top_words: List[str],
    dominant_category: str,
    used_titles: Set[str],
    is_corpus_wide: bool = False,
) -> str:
    """
    Determines an accurate human-readable theme name directly from the cluster's
    extracted c-TF-IDF keywords (`top_words`), enforcing strict uniqueness so two
    clusters in the same view never receive duplicate titles.
    """
    if is_corpus_wide:
        corpus_macro_map = {
            "Sports": "Sports & Athletics",
            "Technology": "Technology & Digital Systems",
            "Finance": "Finance, Markets & Economy",
            "Politics": "Politics, Law & Governance",
            "Healthcare": "Healthcare & Clinical Medicine",
        }
        macro_title = corpus_macro_map.get(dominant_category, dominant_category)
        if macro_title not in used_titles:
            used_titles.add(macro_title)
            return macro_title

    # Expand unigrams and bigrams in top_words into a weighted lookup
    kw_tokens = set()
    for w in top_words:
        kw_tokens.add(w.lower())
        for part in w.lower().split():
            kw_tokens.add(part)

    scored_candidates: List[Tuple[float, str]] = []
    for rule_cat, theme_name, rule_words in TOPIC_THEME_RULES:
        # Give higher priority to rules matching the cluster's dominant category
        cat_bonus = 1.5 if rule_cat == dominant_category else 0.0
        overlap = len(kw_tokens.intersection(rule_words))
        # Extra weight if top-3 keywords match
        top3_tokens = set(" ".join(top_words[:3]).lower().split())
        top3_overlap = len(top3_tokens.intersection(rule_words))
        score = overlap * 2.0 + top3_overlap * 3.0 + cat_bonus
        if overlap > 0:
            scored_candidates.append((score, theme_name))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    chosen_theme = None
    for _, candidate in scored_candidates:
        if candidate not in used_titles:
            chosen_theme = candidate
            break

    if chosen_theme is None:
        # Fallback using dominant category + top distinctive keyword
        distinct_kw = top_words[0].title() if top_words else "General"
        chosen_theme = f"{dominant_category}: {distinct_kw} & Related"
        if chosen_theme in used_titles and len(top_words) > 1:
            chosen_theme = f"{dominant_category}: {top_words[0].title()} & {top_words[1].title()}"

    used_titles.add(chosen_theme)
    return chosen_theme


def compute_ctfidf_topics(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    n_topics: int = 5,
    top_n_words: int = 8,
) -> Dict[str, object]:
    """
    Implements K-Means + Class-based TF-IDF (c-TF-IDF) topic discovery on pretrained
    LLM embeddings with keyword-driven theme naming and strict deduplication.
    """
    km = KMeans(n_clusters=n_topics, init="k-means++", n_init=10, random_state=RANDOM_SEED)
    topic_ids = km.fit_predict(embeddings)

    cleaned_texts = [clean_text_for_lexical(t) for t in df["text"].tolist()]
    class_docs = []
    for t_id in range(n_topics):
        docs_in_topic = [cleaned_texts[i] for i in range(len(cleaned_texts)) if topic_ids[i] == t_id]
        class_docs.append(" ".join(docs_in_topic))

    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stop_words = list(ENGLISH_STOP_WORDS.union(CUSTOM_STOPWORDS))

    vectorizer = CountVectorizer(
        stop_words=stop_words,
        ngram_range=(1, 2),
        min_df=2,
        max_features=2500,
    )
    count_matrix = vectorizer.fit_transform(class_docs).toarray().astype(np.float64)
    vocab = np.array(vectorizer.get_feature_names_out())

    words_per_class = count_matrix.sum(axis=1, keepdims=True) + 1e-12
    tf = count_matrix / words_per_class
    avg_words_per_class = float(count_matrix.sum(axis=1).mean())
    word_freq_all = count_matrix.sum(axis=0) + 1e-12
    idf = np.log(1.0 + (avg_words_per_class / word_freq_all))
    ctfidf_matrix = tf * idf

    centroids = km.cluster_centers_
    norm_centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)

    topic_records = []
    topic_keywords_dict: Dict[str, List[Tuple[str, float]]] = {}
    all_top_words: List[str] = []
    coherence_scores: List[float] = []
    used_titles: Set[str] = set()
    is_corpus_wide = df["category"].nunique() > 1

    for t_id in range(n_topics):
        scores = ctfidf_matrix[t_id]
        top_idx = np.argsort(scores)[::-1][:top_n_words]
        top_words = vocab[top_idx].tolist()
        top_weights = [round(float(scores[i]), 4) for i in top_idx]
        all_top_words.extend(top_words[:5])

        mask = topic_ids == t_id
        member_indices = np.where(mask)[0]
        member_df = df.iloc[member_indices]
        member_embs = embeddings[member_indices]

        sims_to_centroid = np.dot(member_embs, norm_centroids[t_id])
        top_doc_order = np.argsort(sims_to_centroid)[::-1]
        core_embs = member_embs[top_doc_order[: min(15, len(member_embs))]]
        if len(core_embs) > 1:
            sim_mat = cosine_similarity(core_embs)
            upper_tri = sim_mat[np.triu_indices(len(core_embs), k=1)]
            coherence = float(np.mean(upper_tri))
        else:
            coherence = 1.0
        coherence_scores.append(coherence)

        dom_cat = member_df["category"].mode().iloc[0] if len(member_df) > 0 else "General"
        theme_name = infer_unique_topic_title(
            top_words=top_words,
            dominant_category=dom_cat,
            used_titles=used_titles,
            is_corpus_wide=is_corpus_wide,
        )
        kw_preview = ", ".join(top_words[:3]).title()
        topic_label = f"Topic {t_id + 1}: {theme_name} ({kw_preview})"

        exemplar_text = member_df.iloc[int(top_doc_order[0])]["text"][:160] + "..."

        topic_keywords_dict[topic_label] = list(zip(top_words, top_weights))
        topic_records.append({
            "Topic ID": f"Topic {t_id + 1}",
            "Discovered Theme": topic_label,
            "Documents": int(len(member_df)),
            "Top Keywords (c-TF-IDF)": ", ".join(top_words[:6]),
            "Semantic Coherence": round(coherence, 4),
            "Exemplar Document": exemplar_text,
        })

    topic_diversity = len(set(all_top_words)) / max(1, len(all_top_words))

    return {
        "topic_summary_df": pd.DataFrame(topic_records),
        "topic_keywords_dict": topic_keywords_dict,
        "topic_ids": topic_ids,
        "mean_coherence": round(float(np.mean(coherence_scores)), 4),
        "topic_diversity": round(float(topic_diversity), 4),
    }


def discover_category_subtopics(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    category: str = "Technology",
    n_subtopics: int = 4,
) -> Dict[str, object]:
    mask = (df["category"] == category).values
    sub_df = df[mask].reset_index(drop=True)
    sub_emb = embeddings[mask]
    return compute_ctfidf_topics(sub_df, sub_emb, n_topics=n_subtopics, top_n_words=6)


# =====================================================================
# PART B: DIMENSIONALITY REDUCTION (PCA & t-SNE)
# =====================================================================

def compute_or_load_2d_projections(
    embeddings: np.ndarray, force_recompute: bool = False
) -> Dict[str, np.ndarray]:
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(PROJECTIONS_CACHE_PATH) and not force_recompute:
        data = np.load(PROJECTIONS_CACHE_PATH)
        if len(data["pca_2d"]) == len(embeddings):
            return {
                "pca_2d": data["pca_2d"],
                "tsne_2d": data["tsne_2d"],
                "pca_variance_ratio": data["pca_variance_ratio"],
            }

    print("[Module 4] Computing 2D PCA and t-SNE semantic space projections...")
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    pca_2d = pca.fit_transform(embeddings).astype(np.float32)
    pca_var = pca.explained_variance_ratio_.astype(np.float32)

    pca_30 = PCA(n_components=min(30, embeddings.shape[1]), random_state=RANDOM_SEED).fit_transform(embeddings)
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=RANDOM_SEED,
    )
    tsne_2d = tsne.fit_transform(pca_30).astype(np.float32)

    np.savez_compressed(
        PROJECTIONS_CACHE_PATH,
        pca_2d=pca_2d,
        tsne_2d=tsne_2d,
        pca_variance_ratio=pca_var,
    )
    return {
        "pca_2d": pca_2d,
        "tsne_2d": tsne_2d,
        "pca_variance_ratio": pca_var,
    }


# =====================================================================
# PART C: PUBLICATION-QUALITY VISUALIZATION SUITE (7 FIGURES)
# =====================================================================

def generate_all_visualizations(
    df: pd.DataFrame,
    projections: Dict[str, np.ndarray],
    clf_eval_l6: Dict[str, object],
    clf_eval_tfidf: Dict[str, object],
    clustering_res: Dict[str, object],
    k_sweep_df: pd.DataFrame,
    sim_demo_df: pd.DataFrame,
    sim_matrix: np.ndarray,
    sim_sentences: List[str],
    model_comparison_df: pd.DataFrame,
    corpus_topics: Dict[str, object],
    tech_subtopics: Dict[str, object],
) -> Dict[str, str]:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.0)
    saved_paths: Dict[str, str] = {}

    # -----------------------------------------------------------------
    # Figure 1: Dataset Category Distribution & Document Length Profile
    # -----------------------------------------------------------------
    fig1_path = os.path.join(FIGURES_DIR, "fig1_dataset_distribution.png")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=220)

    cat_counts = df["category"].value_counts().reindex(CATEGORIES)
    bars = axes[0].bar(cat_counts.index, cat_counts.values, color=PALETTE[:5], edgecolor="black", alpha=0.9)
    axes[0].set_title("A. Document Count per Category (Balanced Corpus)", fontweight="bold", fontsize=11)
    axes[0].set_xlabel("Category")
    axes[0].set_ylabel("Number of Documents")
    axes[0].set_ylim(0, max(cat_counts.values) * 1.18)
    for bar in bars:
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 4,
            f"{int(bar.get_height())}",
            ha="center",
            fontweight="bold",
        )

    sns.boxplot(
        data=df,
        x="category",
        y="word_count",
        hue="category",
        order=CATEGORIES,
        palette=PALETTE[:5],
        legend=False,
        ax=axes[1],
        width=0.5,
    )
    axes[1].set_title("B. Document Length Distribution (Word Count)", fontweight="bold", fontsize=11)
    axes[1].set_xlabel("Category")
    axes[1].set_ylabel("Words per Document")

    plt.tight_layout(pad=1.2)
    plt.savefig(fig1_path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    saved_paths["fig1_dataset"] = fig1_path

    # -----------------------------------------------------------------
    # Figure 2: Classification Confusion Matrix (Pretrained LLM vs TF-IDF)
    # -----------------------------------------------------------------
    fig2_path = os.path.join(FIGURES_DIR, "fig2_confusion_matrix.png")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=220)

    sns.heatmap(
        clf_eval_l6["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CATEGORIES,
        yticklabels=CATEGORIES,
        cbar=False,
        ax=axes[0],
        annot_kws={"size": 11, "weight": "bold"},
    )
    axes[0].set_title(
        f"A. Pretrained LLM (all-MiniLM-L6-v2)\nAccuracy: {clf_eval_l6['accuracy']*100:.1f}% | Macro F1: {clf_eval_l6['f1_score']:.3f}",
        fontweight="bold",
        fontsize=11,
    )
    axes[0].set_xlabel("Predicted Category")
    axes[0].set_ylabel("True Category")

    sns.heatmap(
        clf_eval_tfidf["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=CATEGORIES,
        yticklabels=CATEGORIES,
        cbar=False,
        ax=axes[1],
        annot_kws={"size": 11, "weight": "bold"},
    )
    axes[1].set_title(
        f"B. Lexical Baseline (TF-IDF)\nAccuracy: {clf_eval_tfidf['accuracy']*100:.1f}% | Macro F1: {clf_eval_tfidf['f1_score']:.3f}",
        fontweight="bold",
        fontsize=11,
    )
    axes[1].set_xlabel("Predicted Category")
    axes[1].set_ylabel("True Category")

    plt.tight_layout(pad=1.2)
    plt.savefig(fig2_path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    saved_paths["fig2_confusion_matrix"] = fig2_path

    # -----------------------------------------------------------------
    # Figure 3: K-Means Cluster Selection (Elbow Method & Silhouette)
    # -----------------------------------------------------------------
    fig3_path = os.path.join(FIGURES_DIR, "fig3_clustering_elbow_silhouette.png")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=220)

    axes[0].plot(
        k_sweep_df["K"],
        k_sweep_df["Inertia (WCSS)"],
        marker="o",
        linewidth=2.5,
        color="#2563EB",
        markersize=8,
    )
    axes[0].axvline(5, color="#EF4444", linestyle="--", linewidth=1.8, label="Optimal K = 5")
    axes[0].set_title("A. Elbow Curve: Inertia (WCSS) vs. K", fontweight="bold", fontsize=11)
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Within-Cluster Inertia (WCSS)")
    axes[0].set_xticks(k_sweep_df["K"])
    axes[0].legend()

    axes[1].plot(
        k_sweep_df["K"],
        k_sweep_df["Silhouette Score"],
        marker="s",
        linewidth=2.5,
        color="#10B981",
        markersize=8,
        label="Silhouette Score (Internal)",
    )
    axes[1].plot(
        k_sweep_df["K"],
        k_sweep_df["Normalized Mutual Info (NMI)"],
        marker="^",
        linewidth=2.2,
        color="#8B5CF6",
        markersize=8,
        label="NMI Score (External Alignment)",
    )
    axes[1].axvline(5, color="#EF4444", linestyle="--", linewidth=1.8, label="Optimal K = 5")
    axes[1].set_title("B. Cluster Quality Scores Across K = 2..10", fontweight="bold", fontsize=11)
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Score")
    axes[1].set_xticks(k_sweep_df["K"])
    axes[1].legend()

    plt.tight_layout(pad=1.2)
    plt.savefig(fig3_path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    saved_paths["fig3_elbow_silhouette"] = fig3_path

    # -----------------------------------------------------------------
    # Figure 4: 2D Semantic Space Visualization (PCA & t-SNE Projections)
    # -----------------------------------------------------------------
    fig4_path = os.path.join(FIGURES_DIR, "fig4_semantic_space_pca_tsne.png")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), dpi=220)

    tsne_coords = projections["tsne_2d"]
    cluster_ids = clustering_res["cluster_ids"]
    cluster_map = clustering_res["cluster_to_theme"]

    for idx, cat in enumerate(CATEGORIES):
        mask = (df["category"] == cat).values
        axes[0].scatter(
            tsne_coords[mask, 0],
            tsne_coords[mask, 1],
            label=cat,
            color=PALETTE[idx],
            alpha=0.78,
            s=32,
            edgecolors="white",
            linewidth=0.4,
        )
    axes[0].set_title("A. t-SNE 2D Semantic Space (Ground-Truth Categories)", fontweight="bold", fontsize=11)
    axes[0].set_xlabel("t-SNE Dimension 1")
    axes[0].set_ylabel("t-SNE Dimension 2")
    axes[0].legend(title="True Category", loc="best", frameon=True, fontsize=9)

    for c_id in sorted(np.unique(cluster_ids)):
        mask = cluster_ids == c_id
        axes[1].scatter(
            tsne_coords[mask, 0],
            tsne_coords[mask, 1],
            label=cluster_map[int(c_id)],
            color=PALETTE[int(c_id) % len(PALETTE)],
            alpha=0.78,
            s=32,
            edgecolors="white",
            linewidth=0.4,
        )
        cx, cy = tsne_coords[mask, 0].mean(), tsne_coords[mask, 1].mean()
        axes[1].scatter(cx, cy, color="black", marker="X", s=150, zorder=10)

    axes[1].set_title(
        f"B. Unsupervised K-Means Clusters (ARI={clustering_res['ari_score']:.3f}, NMI={clustering_res['nmi_score']:.3f})",
        fontweight="bold",
        fontsize=11,
    )
    axes[1].set_xlabel("t-SNE Dimension 1")
    axes[1].set_ylabel("t-SNE Dimension 2")
    axes[1].legend(title="Discovered Clusters (X = Center)", loc="best", frameon=True, fontsize=8.5)

    plt.tight_layout(pad=1.2)
    plt.savefig(fig4_path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    saved_paths["fig4_semantic_space"] = fig4_path

    # -----------------------------------------------------------------
    # Figure 5: Semantic Similarity Comparison (LLM Embeddings vs TF-IDF)
    # -----------------------------------------------------------------
    fig5_path = os.path.join(FIGURES_DIR, "fig5_semantic_similarity_heatmap.png")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), dpi=220)

    short_labels = [
        "S1: AI Healthcare",
        "S2: ML Diagnosis",
        "S3: Cricket Win",
        "S4: Cricket Victory",
        "S5: Stock Market",
        "S6: Equities Surge",
    ]
    sns.heatmap(
        sim_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=short_labels,
        yticklabels=short_labels,
        vmin=0.0,
        vmax=1.0,
        ax=axes[0],
        annot_kws={"size": 9.5, "weight": "bold"},
    )
    axes[0].set_title("A. Pretrained LLM Cosine Similarity Matrix", fontweight="bold", fontsize=11)
    axes[0].tick_params(axis="x", rotation=28)

    pair_labels = [
        "S1 vs S2\n(Healthcare AI\nParaphrase)",
        "S1 vs S3\n(Healthcare vs\nCricket - Unrel.)",
        "S3 vs S4\n(Cricket Win\nParaphrase)",
        "S5 vs S6\n(Stock Market\nParaphrase)",
    ]
    selected_rows = sim_demo_df.iloc[[0, 1, 3, 4]]
    x = np.arange(len(pair_labels))
    width = 0.35

    b1 = axes[1].bar(
        x - width / 2,
        selected_rows["TF-IDF Cosine Sim"],
        width,
        label="Traditional Lexical (TF-IDF)",
        color="#F59E0B",
        edgecolor="black",
    )
    b2 = axes[1].bar(
        x + width / 2,
        selected_rows["LLM Embedding Sim"],
        width,
        label="Pretrained LLM (all-MiniLM-L6-v2)",
        color="#2563EB",
        edgecolor="black",
    )
    axes[1].set_title("B. Lexical (TF-IDF) vs. Pretrained LLM on Paraphrases", fontweight="bold", fontsize=11)
    axes[1].set_ylabel("Cosine Similarity")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pair_labels)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.02,
            f"{h:.2f}",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )

    plt.tight_layout(pad=1.2)
    plt.savefig(fig5_path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    saved_paths["fig5_similarity"] = fig5_path

    # -----------------------------------------------------------------
    # Figure 6: Experimental Model Comparison Bar Chart
    # -----------------------------------------------------------------
    fig6_path = os.path.join(FIGURES_DIR, "fig6_model_comparison_metrics.png")
    fig, ax = plt.subplots(figsize=(11, 5.0), dpi=220)

    metrics_to_plot = [
        "Test Macro F1",
        "Paraphrase Generalization Acc",
        "Clustering NMI",
        "Clustering ARI",
    ]
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    models = model_comparison_df["Representation Model"].tolist()
    colors = ["#F59E0B", "#10B981", "#2563EB"]

    for i, model_label in enumerate(models):
        row = model_comparison_df[model_comparison_df["Representation Model"] == model_label].iloc[0]
        vals = [row[m] for m in metrics_to_plot]
        bars = ax.bar(
            x + (i - 1) * width,
            vals,
            width,
            label=model_label,
            color=colors[i % len(colors)],
            edgecolor="black",
        )
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.015,
                f"{h:.2f}",
                ha="center",
                fontsize=8.5,
                fontweight="bold",
            )

    ax.set_title(
        "Experimental Benchmark: Lexical TF-IDF vs. 3-Layer & 6-Layer Pretrained LLM Embeddings",
        fontweight="bold",
        fontsize=11,
    )
    ax.set_ylabel("Evaluation Score (0.0 to 1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(title="Representation Model", loc="upper right")

    plt.tight_layout(pad=1.2)
    plt.savefig(fig6_path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    saved_paths["fig6_model_comparison"] = fig6_path

    # -----------------------------------------------------------------
    # Figure 7: Discovered Topic Keywords (c-TF-IDF Weights)
    # -----------------------------------------------------------------
    fig7_path = os.path.join(FIGURES_DIR, "fig7_topic_keywords.png")
    kw_dict = tech_subtopics["topic_keywords_dict"]
    n_plots = len(kw_dict)
    fig, axes = plt.subplots(1, n_plots, figsize=(14.0, 4.4), dpi=220)
    if n_plots == 1:
        axes = [axes]

    for idx, (topic_title, kw_list) in enumerate(kw_dict.items()):
        words = [w for w, _ in kw_list][::-1]
        weights = [wt for _, wt in kw_list][::-1]
        axes[idx].barh(words, weights, color=PALETTE[idx % len(PALETTE)], edgecolor="black", alpha=0.88)
        short_title = topic_title.split("(")[0].strip()
        axes[idx].set_title(short_title, fontweight="bold", fontsize=9.5)
        axes[idx].set_xlabel("c-TF-IDF Weight", fontsize=9)

    fig.suptitle(
        "Discovered Technology Sub-Topics via Cluster c-TF-IDF",
        fontweight="bold",
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout(pad=1.1)
    plt.savefig(fig7_path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    saved_paths["fig7_topics"] = fig7_path

    return saved_paths
