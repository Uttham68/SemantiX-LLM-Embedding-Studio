"""
Module 5: Interactive Streamlit User Interface & Live Demonstration Dashboard
Responsible: Member 5 (Application Integration, UI/UX & Live Demo)

Run with:
    streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import BASE_DIR, CATEGORIES, load_or_create_dataset
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
    PALETTE,
)
from run_experiments import PIPELINE_PATH, run_all_experiments

st.set_page_config(
    page_title="Semantic Text Analysis — Pretrained LLM Embeddings",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Domain Semantic Color Tokens (Pure Pitch-Black OLED & Crisp Emerald Palette)
DOMAIN_COLORS = {
    "Technology": "#10B981",  # Electric Emerald
    "Healthcare": "#14B8A6",  # Crisp Teal-Mint
    "Finance": "#EAB308",     # Solar Gold
    "Sports": "#F97316",      # Vibrant Flame Coral
    "Politics": "#F43F5E",    # Crisp Crimson Rose
}

EXECUTIVE_PALETTE = [
    "#10B981",  # Electric Emerald
    "#14B8A6",  # Crisp Teal-Mint
    "#EAB308",  # Solar Gold
    "#F97316",  # Vibrant Flame Coral
    "#F43F5E",  # Crisp Crimson Rose
    "#84CC16",  # Vivid Lime
    "#34D399",  # Luminous Mint
    "#FACC15",  # Bright Gold
    "#FB923C",  # Bright Coral
    "#E4E4E7",  # Crisp Platinum
]

PLOTLY_CONFIG = {"displayModeBar": False}


def inject_obsidian_glass_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        /* Root Canvas & Typography (Pure Pitch-Black OLED) */
        .stApp {
            background-color: #000000 !important;
            background-image:
                radial-gradient(circle at 12% 10%, rgba(16, 185, 129, 0.07), transparent 38%),
                radial-gradient(circle at 88% 18%, rgba(20, 184, 166, 0.06), transparent 40%),
                radial-gradient(circle at 50% 85%, rgba(255, 255, 255, 0.03), transparent 42%);
            color: #FFFFFF !important;
            font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
            color: #FFFFFF !important;
            letter-spacing: -0.02em;
        }

        code, pre, [data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Sidebar Surface (#09090B) */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #09090B 0%, #000000 100%) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.10) !important;
        }

        [data-testid="stSidebar"] [role="radiogroup"] > label {
            background: rgba(12, 12, 14, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 11px;
            padding: 10px 14px !important;
            margin-bottom: 8px;
            transition: all 0.22s cubic-bezier(0.22, 1, 0.36, 1);
        }

        [data-testid="stSidebar"] [role="radiogroup"] > label:hover {
            border-color: rgba(16, 185, 129, 0.50);
            transform: translateX(3px);
        }

        /* Glassmorphic Metric Cards */
        [data-testid="stMetric"] {
            background: rgba(10, 10, 12, 0.85) !important;
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border: 1px solid rgba(255, 255, 255, 0.10) !important;
            border-top: 2px solid #10B981 !important;
            border-radius: 16px !important;
            padding: 18px 22px !important;
            box-shadow: 0 16px 40px -10px rgba(0, 0, 0, 0.85);
            transition: transform 0.25s cubic-bezier(0.22, 1, 0.36, 1), border-color 0.25s ease;
        }

        [data-testid="stMetric"]:hover {
            transform: translateY(-3px);
            border-color: rgba(16, 185, 129, 0.50) !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 11.5px !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #8E8E93 !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 24px !important;
            font-weight: 800 !important;
            color: #FFFFFF !important;
        }

        /* Custom Glass Containers */
        .obsidian-glass-card {
            background: rgba(10, 10, 12, 0.85);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 16px;
            padding: 24px 26px;
            margin-bottom: 20px;
            box-shadow: 0 18px 44px -10px rgba(0, 0, 0, 0.85);
            transition: border-color 0.25s ease, transform 0.25s cubic-bezier(0.22, 1, 0.36, 1);
        }

        .obsidian-glass-card:hover {
            border-color: rgba(16, 185, 129, 0.50);
        }

        .bento-kpi-card {
            background: rgba(10, 10, 12, 0.85);
            backdrop-filter: blur(18px);
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 16px;
            padding: 20px 22px;
            transition: transform 0.25s cubic-bezier(0.22, 1, 0.36, 1), border-color 0.25s ease;
        }

        .bento-kpi-card:hover {
            transform: translateY(-3px);
            border-color: rgba(16, 185, 129, 0.50);
        }

        .role-badge {
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 10.5px;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 6px;
            margin-bottom: 8px;
        }

        .kw-pill {
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            font-weight: 600;
            padding: 4px 10px;
            margin: 3px 4px 3px 0;
            border-radius: 99px;
            background: rgba(16, 185, 129, 0.14);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.38);
        }

        /* Primary Buttons (Electric Emerald) */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
            color: #000000 !important;
            border: 1px solid rgba(52, 211, 153, 0.45) !important;
            border-radius: 12px !important;
            font-weight: 800 !important;
            padding: 0.65rem 1.5rem !important;
            box-shadow: 0 8px 24px -4px rgba(16, 185, 129, 0.40) !important;
            transition: transform 0.2s cubic-bezier(0.22, 1, 0.36, 1), box-shadow 0.2s ease !important;
        }

        .stButton > button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 14px 30px -4px rgba(16, 185, 129, 0.55) !important;
        }

        /* Inputs & Textareas */
        .stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] {
            background-color: #09090B !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            border-radius: 12px !important;
            color: #FFFFFF !important;
        }

        /* Dataframes & Tables */
        [data-testid="stDataFrame"] {
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 14px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_dark_plotly_theme(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Plus Jakarta Sans, sans-serif", color="#FFFFFF", size=12),
        margin=dict(l=20, r=20, t=42, b=20),
        hoverlabel=dict(
            bgcolor="#09090B",
            bordercolor="#10B981",
            font=dict(family="Inter, sans-serif", color="#FFFFFF", size=12),
        ),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
        legend=dict(
            bgcolor="rgba(9, 9, 11, 0.85)",
            bordercolor="rgba(255, 255, 255, 0.10)",
            borderwidth=1,
        ),
    )
    return fig


@st.cache_resource(show_spinner="Loading dataset, cached LLM embeddings, and trained models...")
def load_system_state():
    if not os.path.exists(PIPELINE_PATH):
        run_all_experiments(force_rebuild=False)
    bundle = joblib.load(PIPELINE_PATH)
    df = load_or_create_dataset(force_rebuild=False)
    emb_bundle = get_or_create_corpus_embeddings(df, force_recompute=False)
    projections = compute_or_load_2d_projections(emb_bundle[PRIMARY_MODEL_NAME], force_recompute=False)
    return bundle, df, emb_bundle, projections


def render_bento_kpi_strip(df, bundle):
    k1, k2, k3, k4, k5 = st.columns(5)
    cards = [
        (k1, "BENCHMARK CORPUS", f"{len(df):,} Docs", "5 Balanced Domains • 100% Unique", "#10B981"),
        (k2, "VECTOR GEOMETRY", "384D Sphere", "all-MiniLM-L6-v2 (‖e‖₂ = 1)", "#14B8A6"),
        (
            k3,
            "CLASSIFICATION ACC.",
            f"{bundle['clf_eval_l6']['accuracy']*100:.1f}%",
            f"Macro F1: {bundle['clf_eval_l6']['f1_score']:.4f}",
            "#EAB308",
        ),
        (
            k4,
            "CLUSTERING NMI",
            f"{bundle['clustering_res_k5']['nmi_score']:.4f}",
            f"ARI: {bundle['clustering_res_k5']['ari_score']:.4f} (K=5)",
            "#F97316",
        ),
        (
            k5,
            "TOPIC DIVERSITY",
            f"{bundle['tech_subtopics']['topic_diversity']*100:.1f}%",
            "Cluster c-TF-IDF",
            "#F43F5E",
        ),
    ]
    for col, label, val, sub, accent in cards:
        with col:
            st.markdown(
                f"""
                <div class="bento-kpi-card" style="border-top: 2px solid {accent};">
                    <div style="font-size:11px; font-weight:700; color:#8E8E93; letter-spacing:0.05em;">{label}</div>
                    <div style="font-family:'JetBrains Mono',monospace; font-size:23px; font-weight:800; color:#FFFFFF; margin:4px 0;">{val}</div>
                    <div style="font-size:12px; font-weight:600; color:{accent};">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main():
    inject_obsidian_glass_css()
    bundle, df, emb_bundle, projections = load_system_state()
    emb_l6 = emb_bundle[PRIMARY_MODEL_NAME]
    fig_paths = bundle["fig_paths"]

    # Sidebar Navigation
    with st.sidebar:
        st.markdown("## 🧠 SemantiX Studio")
        st.caption(
            "**Topic #10:** Text classification/clustering and topic modeling "
            "using embeddings from pretrained LLMs"
        )
        st.divider()
        page = st.radio(
            "Navigate Modules:",
            [
                "🏠 Home & Semantic Basics",
                "🎯 Module 1: Text Classification",
                "🧩 Module 2: Text Clustering",
                "🔍 Module 3: Topic Discovery",
                "📊 Evaluation & Experiments",
            ],
        )
        st.divider()
        st.markdown("### ⚙️ Active Stack")
        st.markdown(
            "- **Primary LLM:** `all-MiniLM-L6-v2` (`384D`)\n"
            "- **Comparison LLM:** `paraphrase-MiniLM-L3-v2`\n"
            "- **Lexical Baseline:** `TF-IDF` (`1500D`)\n"
            f"- **Corpus Size:** `{len(df):,}` Unique Docs\n"
            f"- **Categories:** `{', '.join(CATEGORIES)}`"
        )

    # =========================================================================
    # PAGE 1: HOME & SEMANTIC BASICS (WITH 3-COLUMN SIMILARITY LAB)
    # =========================================================================
    if page == "🏠 Home & Semantic Basics":
        st.title("Text Classification, Clustering & Topic Modeling Using Pretrained LLM Embeddings")
        st.markdown(
            "##### *Polished Title: Semantic Text Analysis and Topic Discovery Using Pretrained LLM Embeddings*"
        )

        render_bento_kpi_strip(df, bundle)
        st.write("")

        col_left, col_right = st.columns([1.05, 1.25])
        with col_left:
            st.markdown(
                """
                <div class="obsidian-glass-card">
                    <span class="role-badge" style="background:rgba(217,107,67,0.16); color:#E6825C; border:1px solid rgba(217,107,67,0.4);">ARCHITECTURE OVERVIEW</span>
                    <h3 style="margin-top:4px;">1. Project Concept & Transformer Bi-Encoder</h3>
                    <p style="color:#D6D0C6; font-size:13.8px;">
                        Traditional NLP methods (<b>Bag-of-Words, TF-IDF</b>) rely on exact keyword overlap and fail when two sentences express the same meaning with different words.
                    </p>
                    <p style="color:#D6D0C6; font-size:13.8px;">
                        <b>Pretrained Language Embedding Models</b> (<code>all-MiniLM-L6-v2</code>) pass text through <b>6 Multi-Head Self-Attention</b> layers and <b>Mean Pooling</b> to project each document into a dense <code>384D</code> unit vector where Cosine Similarity captures <b>semantic meaning</b>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.code(
                "Text Dataset (1,000 Unique Articles: Sports, Tech, Finance, Politics, Healthcare)\n"
                "   ↓  Text Preprocessing (Punctuation restoration, context preservation)\n"
                "Pretrained Transformer Bi-Encoder (all-MiniLM-L6-v2)\n"
                "   ↓  L2-Normalized Dense Embeddings (R^384, ||e||_2 = 1)\n"
                "   ├──► Module 1: Supervised Classification (Predict Category + Confidence)\n"
                "   ├──► Module 2: Unsupervised Clustering (K-Means + PCA/t-SNE 2D Space)\n"
                "   └──► Module 3: Automated Topic Discovery (Class-based TF-IDF Themes)",
                language="text",
            )

        with col_right:
            st.markdown(
                """
                <div class="obsidian-glass-card">
                    <span class="role-badge" style="background:rgba(122,139,116,0.18); color:#A3B59D; border:1px solid rgba(122,139,116,0.42);">MODULE 2 • SEMANTIC VS. LEXICAL LAB</span>
                    <h3 style="margin-top:4px;">2. Interactive 3-Column Semantic Similarity Tester</h3>
                    <p style="color:#9E978E; font-size:13px; margin-bottom:0;">
                        Enter any three sentences in any order. Compare sparse lexical TF-IDF against <code>384D</code> Pretrained LLM Cosine Similarity:
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.markdown('<span class="role-badge" style="background:rgba(217,107,67,0.18); color:#E6825C;">ANCHOR A</span>', unsafe_allow_html=True)
                t_a = st.text_input("Text A:", value="Artificial intelligence is transforming healthcare.", label_visibility="collapsed")
            with sc2:
                st.markdown('<span class="role-badge" style="background:rgba(122,139,116,0.18); color:#A3B59D;">PARAPHRASE B</span>', unsafe_allow_html=True)
                t_b = st.text_input("Text B:", value="Machine learning is improving medical diagnosis.", label_visibility="collapsed")
            with sc3:
                st.markdown('<span class="role-badge" style="background:rgba(138,79,83,0.20); color:#C98A8E;">UNRELATED C</span>', unsafe_allow_html=True)
                t_c = st.text_input("Text C:", value="India won the cricket championship.", label_visibility="collapsed")

            def classify_relationship(sim_score: float) -> str:
                if sim_score >= 0.40:
                    return "🟢 Related (High Semantic Similarity)"
                elif sim_score >= 0.22:
                    return "🟡 Partially Related (Moderate Similarity)"
                return "🔴 Unrelated (Different Semantic Topics)"

            def count_shared_keywords(s1: str, s2: str) -> int:
                stop = {"is", "are", "was", "were", "the", "a", "an", "in", "on", "of", "to", "and", "for", "has", "have"}
                w1 = {w.strip(".,!?;:\"'") for w in s1.lower().split()} - stop
                w2 = {w.strip(".,!?;:\"'") for w in s2.lower().split()} - stop
                w1.discard("")
                w2.discard("")
                return len(w1.intersection(w2))

            if st.button("🔬 Compute Semantic vs. Lexical Similarity", type="primary"):
                emb_abc, _ = encode_texts([t_a, t_b, t_c], model_name=PRIMARY_MODEL_NAME)
                llm_sim = cosine_similarity(emb_abc)
                vec = TfidfVectorizer(stop_words="english")
                tfidf_sim = cosine_similarity(vec.fit_transform([t_a, t_b, t_c]).toarray())

                pairs = [
                    ("Text A vs. Text B", 0, 1, t_a, t_b),
                    ("Text A vs. Text C", 0, 2, t_a, t_c),
                    ("Text B vs. Text C", 1, 2, t_b, t_c),
                ]
                rows = []
                for pair_label, i, j, s_1, s_2 in pairs:
                    sim_val = float(llm_sim[i, j])
                    tf_val = float(tfidf_sim[i, j])
                    rows.append({
                        "Sentence Pair": pair_label,
                        "Shared Keywords": count_shared_keywords(s_1, s_2),
                        "TF-IDF Similarity (Keyword)": round(tf_val, 4),
                        "Pretrained LLM Cosine Sim": round(sim_val, 4),
                        "Semantic Gain (Δ)": f"+{sim_val - tf_val:.4f}",
                        "LLM Verdict": classify_relationship(sim_val),
                    })

                res_df = pd.DataFrame(rows)
                best_row = res_df.loc[res_df["Pretrained LLM Cosine Sim"].idxmax()]
                st.success(
                    f"🏆 **Most Semantically Similar Pair:** `{best_row['Sentence Pair']}` "
                    f"(Cosine Similarity = **{best_row['Pretrained LLM Cosine Sim']:.4f}** | "
                    f"Semantic Gain = `{best_row['Semantic Gain (Δ)']}` → {best_row['LLM Verdict']})"
                )
                st.dataframe(res_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("3. Dataset Overview & Category Distribution")
        if os.path.exists(fig_paths["fig1_dataset"]):
            st.image(fig_paths["fig1_dataset"], use_container_width=True)

        with st.expander("📄 Inspect Sample Preprocessed Documents from Corpus"):
            selected_cat = st.selectbox("Filter by Category:", ["All"] + CATEGORIES)
            view_df = df if selected_cat == "All" else df[df["category"] == selected_cat]
            st.dataframe(
                view_df[["doc_id", "category", "subtopic", "word_count", "text"]].head(15),
                use_container_width=True,
                hide_index=True,
            )

    # =========================================================================
    # PAGE 2: MODULE 1 — TEXT CLASSIFICATION
    # =========================================================================
    elif page == "🎯 Module 1: Text Classification":
        st.title("🎯 Module 1: Supervised Text Classification Using Pretrained Embeddings")
        st.markdown(
            "Classifies arbitrary text into predefined categories (**Sports, Technology, Finance, Politics, Healthcare**) "
            "by encoding the input into a **384D Transformer vector** and passing it to a calibrated classifier."
        )

        preset_examples = {
            "Example 1 (Sports - Prompt Demo)": "India defeated Australia in the final match.",
            "Example 2 (Technology - Prompt Demo)": "New artificial intelligence technology was released.",
            "Example 3 (Finance - Prompt Demo)": "The stock market recorded a significant increase.",
            "Example 4 (Healthcare - Paraphrase)": "Machine learning is improving medical diagnosis and clinical screening.",
            "Example 5 (Politics - Governance)": "Parliamentary lawmakers debated constitutional legislation and national budget policy.",
            "Example 6 (Space / NASA - Frontier Tech)": "NASA launched a new space telescope to study distant exoplanets and galaxies.",
            "Custom Input": "",
        }

        col_cfg1, col_cfg2 = st.columns([2, 1])
        with col_cfg1:
            choice = st.selectbox("Select a Preset Demo Sentence or Custom Text:", list(preset_examples.keys()))
        with col_cfg2:
            clf_choice = st.selectbox(
                "Select Classifier Head (trained on all-MiniLM-L6-v2):",
                ["Logistic Regression", "Linear SVM", "Random Forest"],
            )

        default_text = preset_examples[choice] if choice != "Custom Input" else "Zero-trust cybersecurity shields cloud servers from ransomware attacks."
        user_text = st.text_area("Input Text to Classify:", value=default_text, height=90)

        if st.button("🔍 Run Semantic Classification", type="primary"):
            if not user_text.strip():
                st.warning("Please enter valid text to classify.")
            else:
                clf_model = bundle["trained_classifiers"][clf_choice]
                pred_res = predict_single_text(user_text, clf_model, model_name=PRIMARY_MODEL_NAME)

                m1, m2, m3 = st.columns(3)
                m1.metric("Predicted Category", f"🏆 {pred_res['predicted_category']}")
                m2.metric("Model Confidence", f"{pred_res['confidence_pct']:.1f}%")
                m3.metric("Embedding Latency", f"{pred_res['encoding_time_ms']:.1f} ms")

                c_left, c_right = st.columns([1.05, 1.25])
                with c_left:
                    st.markdown("#### Category Probability Breakdown")
                    prob_df = pred_res["probability_df"].iloc[::-1]
                    bar_colors = [DOMAIN_COLORS.get(cat, "#D96B43") for cat in prob_df["Category"]]
                    fig_prob = go.Figure(
                        go.Bar(
                            x=prob_df["Percentage"],
                            y=prob_df["Category"],
                            orientation="h",
                            marker=dict(color=bar_colors, line=dict(color="rgba(255,255,255,0.2)", width=1)),
                            text=[f"{v:.1f}%" for v in prob_df["Percentage"]],
                            textposition="outside",
                            hovertemplate="<b>%{y}</b><br>Confidence: %{x:.2f}%<extra></extra>",
                        )
                    )
                    apply_dark_plotly_theme(fig_prob, height=290)
                    fig_prob.update_xaxes(range=[0, 112], title_text="Calibrated Probability (%)")
                    st.plotly_chart(fig_prob, use_container_width=True, config=PLOTLY_CONFIG)

                with c_right:
                    st.markdown("#### Top-4 Semantically Similar Documents in Corpus")
                    nearest_df = semantic_search_corpus(user_text, df, emb_l6, top_k=4)
                    st.dataframe(nearest_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Per-Class Classification Performance on Held-Out Test Set (250 Docs)")
        st.dataframe(bundle["clf_eval_l6"]["per_class_df"], use_container_width=True, hide_index=True)

    # =========================================================================
    # PAGE 3: MODULE 2 — TEXT CLUSTERING & INTERACTIVE 2D GALAXY
    # =========================================================================
    elif page == "🧩 Module 2: Text Clustering":
        st.title("🧩 Module 2: Unsupervised Document Clustering in Semantic Space")
        st.markdown(
            "Automatically groups unlabeled documents based solely on the geometric proximity of their "
            "**384-dimensional pretrained LLM embeddings** using **K-Means**."
        )

        col_k, col_proj = st.columns([1, 1])
        with col_k:
            k_val = st.slider("Select Number of Clusters (K):", min_value=2, max_value=10, value=5, step=1)
        with col_proj:
            proj_method = st.radio(
                "Select 2D Visualization Projection:",
                ["t-SNE (Non-Linear Manifold)", "PCA (Linear Principal Components)"],
                horizontal=True,
            )

        if k_val == 5:
            clust_res = bundle["clustering_res_k5"]
        else:
            clust_res = run_kmeans_clustering(emb_l6, df, n_clusters=k_val)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Number of Clusters (K)", f"{k_val}")
        m2.metric("Silhouette Score (Cosine)", f"{clust_res['silhouette_score']:.4f}")
        m3.metric("Adjusted Rand Index (ARI)", f"{clust_res['ari_score']:.4f}")
        m4.metric("Normalized Mutual Info (NMI)", f"{clust_res['nmi_score']:.4f}")

        st.subheader("1. Interactive 2D Semantic Galaxy (Plotly Dark Manifold)")
        coords = projections["tsne_2d"] if "t-SNE" in proj_method else projections["pca_2d"]
        axis_name = "t-SNE" if "t-SNE" in proj_method else "PCA"

        c_ids = clust_res["cluster_ids"]
        c_map = clust_res["cluster_to_theme"]

        fig_galaxy = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"A. Unsupervised K-Means Clusters (K={k_val}) in {axis_name} Space",
                f"B. True Ground-Truth Categories in {axis_name} Space",
            ),
            horizontal_spacing=0.06,
        )

        for c_id in sorted(np.unique(c_ids)):
            mask = c_ids == c_id
            theme_label = c_map[int(c_id)]
            fig_galaxy.add_trace(
                go.Scatter(
                    x=coords[mask, 0],
                    y=coords[mask, 1],
                    mode="markers",
                    name=theme_label,
                    marker=dict(size=7, color=EXECUTIVE_PALETTE[int(c_id) % len(EXECUTIVE_PALETTE)], opacity=0.85),
                    text=df.loc[mask, "text"].str.slice(0, 110) + "...",
                    hovertemplate="<b>" + theme_label + "</b><br>%{text}<extra></extra>",
                    legendgroup="clusters",
                ),
                row=1,
                col=1,
            )

        for cat in CATEGORIES:
            mask = (df["category"] == cat).values
            fig_galaxy.add_trace(
                go.Scatter(
                    x=coords[mask, 0],
                    y=coords[mask, 1],
                    mode="markers",
                    name=cat,
                    marker=dict(size=7, color=DOMAIN_COLORS.get(cat, "#F59E0B"), opacity=0.85),
                    text=df.loc[mask, "text"].str.slice(0, 110) + "...",
                    hovertemplate="<b>True: " + cat + "</b><br>%{text}<extra></extra>",
                    legendgroup="true_cats",
                ),
                row=1,
                col=2,
            )

        apply_dark_plotly_theme(fig_galaxy, height=480)
        fig_galaxy.update_xaxes(gridcolor="rgba(255,255,255,0.06)", title_text=f"{axis_name} Dim 1")
        fig_galaxy.update_yaxes(gridcolor="rgba(255,255,255,0.06)", title_text=f"{axis_name} Dim 2")
        st.plotly_chart(fig_galaxy, use_container_width=True, config=PLOTLY_CONFIG)

        st.subheader("2. Emergent Cluster Themes & Centroid-Nearest Exemplar Documents")
        st.dataframe(clust_res["cluster_summary_df"], use_container_width=True, hide_index=True)

        st.subheader("3. Selecting Optimal K: Elbow Method & Silhouette Analysis (K = 2..10)")
        if os.path.exists(fig_paths["fig3_elbow_silhouette"]):
            st.image(fig_paths["fig3_elbow_silhouette"], use_container_width=True)

    # =========================================================================
    # PAGE 4: MODULE 3 — TOPIC DISCOVERY (3x2 GLASS GRID + PLOTLY)
    # =========================================================================
    elif page == "🔍 Module 3: Topic Discovery":
        st.title("🔍 Module 3: Automated Topic Discovery (Embeddings + c-TF-IDF)")
        st.markdown(
            """
            - **Classification** (*Supervised*): Assigns documents to predefined labels.
            - **Clustering** (*Unsupervised Partitioning*): Groups similar documents together (`Cluster 1..K`).
            - **Topic Discovery** (*Unsupervised Thematic Extraction*): Combines semantic embedding clusters with **Class-based TF-IDF (`c-TF-IDF`)** to extract discriminative topic keywords, thematic labels, and representative exemplar documents.
            """
        )

        mode = st.radio(
            "Select Topic Discovery Scope:",
            [
                "Category Sub-Topic Discovery (e.g., AI, Cybersecurity, Gaming, Telecom inside Technology)",
                "Full Corpus Topic Discovery (1,000 Documents across all domains)",
            ],
        )

        if "Category Sub-Topic" in mode:
            col_c, col_n = st.columns(2)
            with col_c:
                target_cat = st.selectbox("Select Domain Category to Analyze:", CATEGORIES, index=1)
            with col_n:
                n_sub = st.slider("Number of Sub-Topics to Discover:", min_value=2, max_value=6, value=6)

            topic_res = discover_category_subtopics(df, emb_l6, category=target_cat, n_subtopics=n_sub)
        else:
            n_top = st.slider("Number of Corpus-Wide Topics to Discover:", min_value=4, max_value=8, value=6)
            topic_res = compute_ctfidf_topics(df, emb_l6, n_topics=n_top, top_n_words=8)

        k1, k2, k3 = st.columns(3)
        k1.metric("Discovered Topics", f"{len(topic_res['topic_summary_df'])}")
        k2.metric("Topic Diversity Score", f"{topic_res['topic_diversity']*100:.1f}%")
        k3.metric("Intra-Topic Semantic Coherence", f"{topic_res['mean_coherence']:.4f}")

        st.subheader("1. Discovered c-TF-IDF Topic Cards (Responsive 3×2 Glass Grid)")
        summary_rows = topic_res["topic_summary_df"].to_dict(orient="records")
        kw_dict = topic_res["topic_keywords_dict"]
        kw_keys = list(kw_dict.keys())

        for row_start in range(0, len(summary_rows), 3):
            cols = st.columns(3)
            for offset in range(3):
                idx = row_start + offset
                if idx >= len(summary_rows):
                    break
                row_data = summary_rows[idx]
                kw_list = kw_dict[kw_keys[idx]] if idx < len(kw_keys) else []
                pills_html = "".join(
                    [f'<span class="kw-pill">{w} ({wt:.2f})</span>' for w, wt in kw_list[:6]]
                )
                with cols[offset]:
                    st.markdown(
                        f"""
                        <div class="obsidian-glass-card" style="border-top: 2px solid {EXECUTIVE_PALETTE[idx % len(EXECUTIVE_PALETTE)]};">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                                <strong style="font-size:15px; color:#F4F1EA;">{row_data['Discovered Theme']}</strong>
                                <span class="role-badge" style="background:rgba(214,208,198,0.06); color:#D6D0C6; margin:0;">{row_data['Documents']} Docs</span>
                            </div>
                            <div style="font-size:12px; color:#A3B59D; font-family:'JetBrains Mono',monospace; margin-bottom:8px;">
                                Coherence: {row_data['Semantic Coherence']:.4f}
                            </div>
                            <div style="margin-bottom:10px;">{pills_html}</div>
                            <div style="font-size:12px; color:#9E978E; border-left:2px solid #D96B43; padding-left:8px; font-style:italic;">
                                "{row_data['Exemplar Document']}"
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.subheader("2. Topic Summary Telemetry Table")
        st.dataframe(topic_res["topic_summary_df"], use_container_width=True, hide_index=True)

    # =========================================================================
    # PAGE 5: EVALUATION & EXPERIMENTS DASHBOARD
    # =========================================================================
    elif page == "📊 Evaluation & Experiments":
        st.title("📊 Experimental Evaluation & Comparative Benchmarks")
        st.markdown(
            "Real quantitative results generated by `run_experiments.py` comparing **Lexical TF-IDF** against "
            "**3-Layer (`paraphrase-MiniLM-L3-v2`)** and **6-Layer (`all-MiniLM-L6-v2`)** Pretrained LLM Embeddings."
        )

        st.subheader("Experiment A: Representation & Pretrained Embedding Model Comparison")
        st.dataframe(bundle["model_comparison_df"], use_container_width=True, hide_index=True)
        if os.path.exists(fig_paths["fig6_model_comparison"]):
            st.image(fig_paths["fig6_model_comparison"], use_container_width=True)

        st.divider()
        st.subheader("Experiment B: Supervised Classifier Comparison & Confusion Matrix")
        st.dataframe(bundle["classifier_comparison_df"], use_container_width=True, hide_index=True)
        if os.path.exists(fig_paths["fig2_confusion_matrix"]):
            st.image(fig_paths["fig2_confusion_matrix"], use_container_width=True)

        st.divider()
        st.subheader("Experiment C: Semantic Similarity Benchmark (Paraphrases vs. Unrelated Texts)")
        st.dataframe(bundle["sim_demo_df"], use_container_width=True, hide_index=True)
        if os.path.exists(fig_paths["fig5_similarity"]):
            st.image(fig_paths["fig5_similarity"], use_container_width=True)


if __name__ == "__main__":
    main()
