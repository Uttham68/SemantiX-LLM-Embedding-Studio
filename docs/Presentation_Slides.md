# Course Project Presentation Deck (12 Slides)

**Official Title:** *Text classification/clustering and topic modeling using embeddings from pretrained LLMs*  
**Polished Title:** *Semantic Text Analysis and Topic Discovery Using Pretrained LLM Embeddings*  
**Course:** Generative AI Course Project (Topic #10)

---

## Slide 1: Title & Team Overview
- **Project Title:** Text Classification/Clustering and Topic Modeling Using Embeddings from Pretrained LLMs
- **Subtitle:** Semantic Text Analysis and Topic Discovery Using Pretrained LLM Embeddings
- **Team Members & Responsibilities:**
  - **Member 1:** Dataset Curation & Context-Preserving Preprocessing (`src/data_loader.py`)
  - **Member 2:** Pretrained Transformer Embeddings & Semantic Similarity (`src/embeddings.py`)
  - **Member 3:** Supervised Classification & Unsupervised K-Means Clustering (`src/models.py`)
  - **Member 4:** c-TF-IDF Topic Discovery & 2D PCA/t-SNE Visualizations (`src/topic_visualizer.py`)
  - **Member 5:** Streamlit Dashboard Integration, Evaluation & Documentation (`app.py`, `run_experiments.py`)

---

## Slide 2: Problem Statement & Motivation
- **The Challenge:** Modern organizations generate massive volumes of unstructured text (news, tickets, clinical notes, financial reports) requiring automated categorization and thematic discovery.
- **Why Traditional NLP Fails:**
  - Bag-of-Words (BoW) and TF-IDF rely strictly on **exact keyword matching**.
  - Example:
    - *Text A:* `"Artificial intelligence is transforming healthcare."`
    - *Text B:* `"Machine learning is improving medical diagnosis."`
    - **Shared Words:** `0` $\rightarrow$ **TF-IDF Cosine Similarity = `0.0000`**!
- **Why Calling Giant Generative LLMs per Document Is Impractical:**
  - Prompting GPT-4/Gemini for every document in a 10,000-document clustering loop is slow, expensive, and non-deterministic.
- **Our Solution:** Leverage **dense semantic embeddings from pretrained Transformer bi-encoders** (`all-MiniLM-L6-v2`) that run locally on a standard CPU in milliseconds while capturing deep contextual meaning.

---

## Slide 3: Background & Generative AI Connection
- **From Word Counts to Transformer Self-Attention:**
  - Pretrained Language Models (BERT, RoBERTa, MiniLM) use **Multi-Head Self-Attention** to contextualize every token relative to the entire sentence.
- **Sentence-BERT (Siamese Bi-Encoder Architecture):**
  - Tokenizes text $\rightarrow$ Passes through 6 Transformer layers $\rightarrow$ Applies **Mean Pooling** $\rightarrow$ Applies **L2 Normalization** to yield a unit vector $\vec{v} \in \mathbb{R}^{384}$ ($\|\vec{v}\|_2 = 1$).
- **Connection to Generative AI:**
  1. Built on the exact **Transformer foundation** (Vaswani et al., 2017) powering modern LLMs.
  2. Serves as the **semantic representation backbone** for Retrieval-Augmented Generation (RAG), LLM vector memory, and semantic guardrails.

---

## Slide 4: Proposed System Architecture
```text
Raw Text Dataset (1,000 Articles across 5 Domains)
        ↓
Context-Preserving Preprocessing (Noise removal, Sentence retention)
        ↓
Pretrained Transformer Bi-Encoder (all-MiniLM-L6-v2 vs. MiniLM-L3 & TF-IDF)
        ↓
L2-Normalized Dense Semantic Vectors (R^384)
        ├──► Module 1: Supervised Classification (Logistic Regression / SVM / RF)
        ├──► Module 2: Unsupervised Clustering (K-Means + Elbow & Silhouette)
        └──► Module 3: Automated Topic Discovery (Class-based TF-IDF / BERTopic)
        ↓
2D Semantic Projection (PCA & t-SNE) + Interactive Streamlit Dashboard
```

---

## Slide 5: Dataset Selection & Preprocessing (Member 1)
- **Benchmark Corpus (`data/dataset.csv`):**
  - **1,000 Balanced Documents** ($200$ per category) across **5 Real-World Domains**:
    1. **Sports** (*Cricket, Football, Athletics, Championships*)
    2. **Technology** (*AI, Cybersecurity, Cloud Computing, Robotics*)
    3. **Finance** (*Stock Market, Banking, Monetary Policy, Macroeconomics*)
    4. **Politics** (*Legislation, Elections, Diplomacy, Public Policy*)
    5. **Healthcare** (*Medical Diagnostics, Clinical Trials, Public Health, Biomedical*)
  - Sourced from **BBC News** and **20 Newsgroups (`sci.med`)** benchmarks + contemporary domain anchors.
- **Why Preprocessing Differs for Pretrained LLMs:**
  - We strip HTML, URLs, and header noise, but **preserve sentence structure and grammar** so Transformer self-attention heads can resolve word relationships.
- *Visual Reference:* `outputs/figures/fig1_dataset_distribution.png`

---

## Slide 6: Pretrained Embeddings & Semantic Similarity (Member 2)
- **Mathematical Equivalence on the Unit Hypersphere:**
  - Because embeddings are L2-normalized ($\|\vec{u}\| = \|\vec{v}\| = 1$):
    $$\cos(\vec{u}, \vec{v}) = \vec{u} \cdot \vec{v} \quad \text{and} \quad \|\vec{u} - \vec{v}\|^2 = 2(1 - \cos(\vec{u}, \vec{v}))$$
- **Empirical Paraphrase Verification:**
  - *Text A ("AI transforming healthcare")* vs. *Text B ("ML improving medical diagnosis")*:
    - **TF-IDF Similarity:** `0.0000` (Complete failure)
    - **Pretrained LLM Similarity (`all-MiniLM-L6-v2`):** High positive similarity (`> 0.65`)
  - *Text A ("AI transforming healthcare")* vs. *Text C ("India won cricket championship")*:
    - **Pretrained LLM Similarity:** Near zero (`< 0.10`)
- *Visual Reference:* `outputs/figures/fig5_semantic_similarity_heatmap.png`

---

## Slide 7: Module 1 — Supervised Text Classification (Member 3)
- **Three Paradigms Compared:**
  1. *Traditional ML:* TF-IDF + Classifier (fast, brittle to synonyms).
  2. *Pretrained Embedding Classification (Ours):* Frozen 384D Transformer vectors + Logistic Regression / Calibrated Linear SVM (CPU-fast, strong synonym generalization).
  3. *End-to-End LLM Fine-Tuning:* Updating 110M+ weights (requires GPU, risks overfitting on small data).
- **Key Results:**
  - High accuracy & Macro F1 on the 250-document held-out test set.
  - Dramatically outperforms TF-IDF on the **25-sentence Zero-Overlap Paraphrase Generalization Benchmark**.
- *Visual Reference:* `outputs/figures/fig2_confusion_matrix.png`

---

## Slide 8: Module 2 — Unsupervised Text Clustering (Member 3)
- **Objective:** Group unlabeled documents into coherent categories purely from embedding geometry.
- **Algorithm:** K-Means (`k-means++` initialization) in 384D unit-normalized embedding space.
- **How $K$ Is Selected ($K = 2 \dots 10$ Sweep):**
  - **Elbow Method (WCSS / Inertia)** and **Cosine Silhouette Score** identify a clear optimum at **$K = 5$**.
- **Evaluation Metrics:**
  - Internal: **Silhouette Score**
  - External: **Adjusted Rand Index (ARI)** & **Normalized Mutual Information (NMI)**
- *Visual Reference:* `outputs/figures/fig3_clustering_elbow_silhouette.png` & `fig4_semantic_space_pca_tsne.png`

---

## Slide 9: Module 3 — Topic Discovery via c-TF-IDF (Member 4)
- **How Topic Discovery Differs from Clustering:**
  - Clustering outputs *anonymous group IDs* (`Cluster 1..K`); Topic Discovery extracts **human-readable themes, weighted keywords, and exemplar documents**.
- **Methodology (BERTopic Class-based TF-IDF):**
  - Concatenates all documents in each semantic cluster $c$ and computes:
    $$\text{c-TF-IDF}_{w, c} = \text{tf}_{w, c} \times \log\left(1 + \frac{A}{\text{tf}_w}\right)$$
- **Discovered Sub-Topics Inside `Technology` Corpus:**
  - *Topic 1:* Artificial Intelligence (`ai`, `neural`, `language`, `models`)
  - *Topic 2:* Cybersecurity (`security`, `ransomware`, `network`, `encryption`)
  - *Topic 3:* Cloud Computing (`cloud`, `serverless`, `containers`, `infrastructure`)
  - *Topic 4:* Robotics & Hardware (`robotics`, `sensors`, `autonomous`, `chip`)
- *Visual Reference:* `outputs/figures/fig7_topic_keywords.png`

---

## Slide 10: Experimental Comparison & Results Summary
- **4 Controlled Experiments Executed (`run_experiments.py`):**
  - **Exp A:** `TF-IDF` vs. `paraphrase-MiniLM-L3-v2` (3-layer LLM) vs. `all-MiniLM-L6-v2` (6-layer LLM).
  - **Exp B:** Logistic Regression vs. Linear SVM vs. Random Forest on 384D embeddings.
  - **Exp C:** Unsupervised K-Means across $K = 2\dots 10$.
  - **Exp D:** Topic Diversity & Intra-Topic Semantic Coherence.
- *Visual Reference:* `outputs/figures/fig6_model_comparison_metrics.png`

---

## Slide 11: Interactive Streamlit Dashboard & Applications (Member 5)
- **5-Tab Live Demonstration Interface (`app.py`):**
  1. **Home & Semantic Basics:** Live sentence similarity calculator (LLM vs. TF-IDF).
  2. **Text Classification:** Real-time category prediction + confidence chart + top-3 similar articles.
  3. **Text Clustering:** Interactive $K$ slider + 2D PCA/t-SNE semantic map.
  4. **Topic Discovery:** Dynamic c-TF-IDF sub-topic explorer.
  5. **Evaluation Dashboard:** Full metric tables & plots.
- **Real-World Applications:** Customer support ticket routing, clinical literature triage, financial news monitoring, and RAG indexing.

---

## Slide 12: Limitations & Conclusion
- **Limitations:**
  1. **Context Window Limit:** MiniLM truncates sequences beyond 256 WordPiece tokens (~180 words).
  2. **Pre-specified $K$ in K-Means:** Assumes spherical clusters and requires choosing $K$.
  3. **Frozen Domain Knowledge:** Highly specialized domain slang may require contrastive domain adaptation.
- **Conclusion:**
  - Pretrained Transformer embeddings bridge the gap between brittle keyword methods and expensive generative LLM calls—delivering superior classification, unsupervised clustering, and interpretable topic discovery on standard CPU hardware.

