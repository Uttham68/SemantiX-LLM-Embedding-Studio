# Viva Voce Preparation Guide & Live Demonstration Script

**Official Course Topic (#10):** *"Text classification/clustering and topic modeling using embeddings from pretrained LLMs"*  
**Polished Title:** *"Semantic Text Analysis and Topic Discovery Using Pretrained LLM Embeddings"*

---

## PART 1: Step-by-Step Live Classroom Demonstration Script (5–7 Minutes)

> **Pre-Demo Reliability Guarantee:**  
> Because `run_experiments.py` caches the 1,000-document corpus (`data/dataset.csv`), the 384D pretrained embeddings (`data/embeddings_cache.npz`), the 2D PCA/t-SNE coordinates (`data/projections_cache.npz`), and the trained classifiers (`outputs/results/trained_pipeline.joblib`), the live Streamlit dashboard runs **100% offline** in `< 50 ms` per query.

### Step 1 — Launch the Dashboard (30 sec)
1. Open terminal in `d:\Practice\GenAI ProjectAG` and run:
   ```powershell
   python -m streamlit run app.py
   ```
2. On **`🏠 Home & Semantic Basics`**, point out the **5 KPI metrics** at the top:
   - **1,000 Documents** across **5 balanced categories** (*Sports, Technology, Finance, Politics, Healthcare*)
   - **384-Dimensional** Pretrained Transformer Embedding Space (`all-MiniLM-L6-v2`)

### Step 2 — Demonstrate Semantic Similarity vs. Keyword Matching (1 min)
1. Stay on **`🏠 Home & Semantic Basics`** and look at the **Interactive Semantic Similarity Tester**.
2. Click **"Compute Semantic vs. Lexical Similarity"** on the default sentences:
   - **Text A:** *"Artificial intelligence is transforming healthcare."*
   - **Text B:** *"Machine learning is improving medical diagnosis."*
   - **Text C:** *"India won the cricket championship."*
3. **Explain to the Instructor:**
   - Notice that **TF-IDF Cosine Similarity** between Text A and Text B is **`0.0000`** because they share **zero words**.
   - In contrast, **Pretrained LLM Embedding Similarity (`all-MiniLM-L6-v2`)** is **`~0.68`** (high semantic similarity), while Text A vs. Text C is **`< 0.10`** (unrelated). This proves the model captures *meaning* rather than word overlap.

### Step 3 — Demonstrate Module 1: Supervised Text Classification (1.5 min)
1. Click **`🎯 Module 1: Text Classification`** in the sidebar.
2. Test the three preset examples from the assignment:
   - `"India defeated Australia in the final match."` → Predicts **`Sports`** with high confidence + retrieves top-3 cricket/match articles from the corpus.
   - `"New artificial intelligence technology was released."` → Predicts **`Technology`**.
   - `"The stock market recorded a significant increase."` → Predicts **`Finance`**.
3. Invite the instructor to give a custom sentence (e.g., *"Hospital doctors tested a new vaccine for patients"*) and run live classification.

### Step 4 — Demonstrate Module 2: Unsupervised Text Clustering (1.5 min)
1. Click **`🧩 Module 2: Text Clustering`** in the sidebar.
2. Show the **2D t-SNE Semantic Space scatter plot**:
   - Even when all labels are hidden (**unsupervised K-Means with $K=5$**), the documents self-organize into 5 distinct semantic islands corresponding to *Sports, Technology, Finance, Politics, and Healthcare*.
3. Move the slider from $K=5$ to $K=3$ or $K=7$, and point out the **Elbow Curve and Silhouette Score plot** showing why $K=5$ is optimal.

### Step 5 — Demonstrate Module 3: Topic Discovery (1 min)
1. Click **`🔍 Module 3: Topic Discovery`** in the sidebar.
2. Show how within the **`Technology`** category, **Class-based TF-IDF (c-TF-IDF)** automatically discovers the 4 sub-topics:
   - *Artificial Intelligence* (`ai`, `neural`, `language`, `learning`)
   - *Cybersecurity* (`security`, `ransomware`, `encryption`, `network`)
   - *Cloud Computing* (`cloud`, `serverless`, `containers`, `scalable`)
   - *Robotics & Hardware* (`robotics`, `sensors`, `chip`, `autonomous`)

### Step 6 — Show Quantitative Evaluation Results (1 min)
1. Click **`📊 Evaluation & Experiments`** and highlight:
   - **Experiment A:** Pretrained LLM (`all-MiniLM-L6-v2`) dramatically outperforms `TF-IDF` on **Paraphrase Generalization** and **Unsupervised Clustering NMI/ARI**.
   - **Experiment B:** Confusion matrix showing clean diagonal classification across all 5 categories.

---

## PART 2: Comprehensive Viva Questions & Answers (16 Core Questions)

### Q1. What are text embeddings?
**Answer:**  
Text embeddings are dense, fixed-length numerical vectors (e.g., $384$ real numbers: $\vec{v} \in \mathbb{R}^{384}$) that represent words, sentences, or documents in a continuous geometric space. In this space, the direction and distance between vectors reflect semantic meaning—texts with similar meanings are mapped to nearby points, while unrelated texts are placed far apart.

### Q2. What are pretrained language models (PLMs)?
**Answer:**  
Pretrained language models are deep neural networks—typically based on the **Transformer** architecture—that have already been trained on massive text corpora (billions of sentences from Wikipedia, books, and web articles) using self-supervised objectives (such as Masked Language Modeling or contrastive sentence pairing). Because they have already learned grammar, context, synonyms, and world concepts during pretraining, we can use their learned representations directly for downstream tasks without training a model from scratch.

### Q3. What is semantic similarity, and how does it differ from lexical similarity?
**Answer:**  
- **Lexical similarity** measures character or exact word overlap (e.g., TF-IDF, Jaccard similarity). For example, *"AI is transforming healthcare"* and *"Machine learning improves medical diagnosis"* have **zero lexical similarity** because they share no common words.
- **Semantic similarity** measures how close two texts are in *meaning* and *intent*. Using pretrained LLM embeddings, those same two sentences have a high semantic similarity ($\approx 0.68$) because the model understands that *AI/machine learning* and *healthcare/medical diagnosis* are closely related concepts.

### Q4. Why are pretrained embeddings useful compared to Bag-of-Words or TF-IDF?
**Answer:**  
1. **Captures Context & Synonyms:** Understands polysemy (e.g., *"river bank"* vs. *"investment bank"*) and synonyms.
2. **Low Dimensionality & Dense:** Compresses representation into $384$ dense numbers instead of $10,000+$ sparse zeros (avoiding the curse of dimensionality in K-Means clustering).
3. **Sample Efficiency:** Enables high classification accuracy even with small training sets and generalizes to unseen vocabulary.

### Q5. How does Sentence-BERT (`all-MiniLM-L6-v2`) work?
**Answer:**  
Standard BERT requires passing two sentences together (`[CLS] Sent A [SEP] Sent B`) to compare them, which takes quadratic time $O(N^2)$ and is too slow for clustering. **Sentence-BERT (SBERT)** uses a **Siamese / Bi-Encoder architecture**:
1. It passes each sentence independently through a 6-layer Transformer encoder (`MiniLM-L6`).
2. It applies **Mean Pooling** across all contextualized token vectors to produce a single $384$-dimensional vector.
3. During contrastive pretraining on over 1 billion sentence pairs, it adjusts weights so semantically similar sentences have high cosine similarity.

### Q6. What is Cosine Similarity, and why do we use it instead of Euclidean distance?
**Answer:**  
Cosine similarity measures the cosine of the angle $\theta$ between two vectors $\vec{u}$ and $\vec{v}$:
$$\cos(\theta) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$$
Its value ranges from $-1$ (opposite) to $+1$ (identical direction), with $0$ indicating orthogonality. We prefer it for text embeddings because it measures **directional alignment (semantic topic)** regardless of vector magnitude (document length). Furthermore, because we **L2-normalize** our embeddings ($\|\vec{u}\| = \|\vec{v}\| = 1$), squared Euclidean distance satisfies:
$$\|\vec{u} - \vec{v}\|^2 = 2(1 - \cos(\theta))$$
making K-Means Euclidean clustering mathematically equivalent to cosine similarity clustering.

### Q7. What is K-Means clustering, and how does it work in our project?
**Answer:**  
K-Means is an iterative unsupervised partitioning algorithm:
1. **Initialization (`k-means++`):** Picks $K$ spread-out initial cluster centroids $\vec{\mu}_1, \dots, \vec{\mu}_K$ in $\mathbb{R}^{384}$.
2. **Assignment Step:** Assigns every document embedding $\vec{x}_i$ to its closest centroid.
3. **Update Step:** Recomputes each centroid $\vec{\mu}_k$ as the mean of all document vectors assigned to cluster $k$.
4. **Convergence:** Repeats Assignment and Update until cluster assignments stabilize.

### Q8. How did we select the optimal number of clusters ($K$)?
**Answer:**  
We evaluated $K \in \{2, 3, \dots, 10\}$ using two complementary methods:
1. **The Elbow Method (Inertia / WCSS):** Plots Within-Cluster Sum of Squares against $K$; the rate of decrease bends sharply ("elbow") at $K = 5$.
2. **Silhouette Score Maximization:** Measures how similar a document is to its own cluster compared to the nearest neighboring cluster (range $[-1, +1]$). Both internal Silhouette score and external NMI confirm $K = 5$ aligns with the corpus structure.

### Q9. What is the difference between Text Classification, Text Clustering, and Topic Modeling?
**Answer:**  
| Dimension | Text Classification (Module 1) | Text Clustering (Module 2) | Topic Modeling / Discovery (Module 3) |
| :--- | :--- | :--- | :--- |
| **Supervision** | **Supervised** (requires labeled training data) | **Unsupervised** (no labels used) | **Unsupervised** (no labels used) |
| **Core Question** | *"Which predefined category does this text belong to?"* | *"Which documents are similar enough to be grouped together?"* | *"What themes and keywords describe each group of documents?"* |
| **Output** | Predicted Label + Confidence (e.g., `Sports: 96%`) | Cluster ID assignment (`Cluster 1`, `Cluster 2`, ...) | Ranked Topic Keywords (c-TF-IDF) + Thematic Labels + Exemplars |

### Q10. How does our Topic Discovery algorithm (Embeddings + c-TF-IDF) work?
**Answer:**  
Following the **BERTopic** architecture (Grootendorst, 2022):
1. Documents are first grouped semantically using their pretrained LLM embeddings.
2. All documents inside cluster $c$ are concatenated into a single macro-document.
3. We apply **Class-based TF-IDF (c-TF-IDF)**:
   $$\text{c-TF-IDF}_{w, c} = \text{tf}_{w, c} \times \log\left(1 + \frac{A}{\text{tf}_w}\right)$$
   where $\text{tf}_{w, c}$ is the relative frequency of word $w$ in cluster $c$, $\text{tf}_w$ is its total frequency across all clusters, and $A$ is the average word count per cluster. This highlights words that are uniquely characteristic of each semantic topic.

### Q11. Why did we choose this dataset?
**Answer:**  
We constructed a balanced 1,000-document benchmark across 5 real-world categories (*Sports, Technology, Finance, Politics, Healthcare*) sourced from the **BBC News** and **20 Newsgroups (`sci.med`)** benchmarks. It is:
- Large enough ($1,000$ docs; $750$ train / $250$ test) for statistically valid macro-F1, confusion matrices, and clustering metrics.
- Compact enough to run end-to-end on a standard Windows CPU laptop in seconds.
- Cached locally in `data/dataset.csv` for 100% offline reproducibility.

### Q12. What evaluation metrics did we use and why?
**Answer:**  
- **For Classification:** **Accuracy, Macro Precision, Macro Recall, Macro F1-Score, and Confusion Matrix** (plus **Paraphrase Generalization Accuracy** on sentences with zero keyword overlap).
- **For Clustering:** **Silhouette Score** (internal cohesion vs. separation without labels), **Adjusted Rand Index (ARI)**, and **Normalized Mutual Information (NMI)** (measuring alignment between unsupervised clusters and ground-truth classes).
- **For Topic Discovery:** **Topic Diversity** (uniqueness of top keywords across topics) and **Intra-Topic Semantic Coherence** (mean pairwise cosine similarity of core documents in each topic).

### Q13. Why do we use PCA and t-SNE in this project?
**Answer:**  
Human eyes cannot visualize $384$-dimensional space. We use dimensionality reduction ($\mathbb{R}^{384} \to \mathbb{R}^2$) to plot the embeddings in 2D:
- **PCA (Principal Component Analysis):** Linear projection that preserves global variance along orthogonal axes.
- **t-SNE (t-Distributed Stochastic Neighbor Embedding):** Non-linear probabilistic projection that preserves local neighborhoods, clearly revealing the 5 semantic clusters visually.

### Q14. What is the actual Generative AI component of this project, and why is it in a GenAI course?
**Answer:**  
1. **Shared Transformer Architecture:** Pretrained embedding models (`all-MiniLM-L6-v2`) are built on the exact **Transformer self-attention mechanism** (Vaswani et al., 2017) that underpins all modern Generative AI models (GPT, Gemini, LLaMA).
2. **Representation Layer of GenAI Pipelines:** In real-world Generative AI systems—especially **Retrieval-Augmented Generation (RAG)**, agentic memory, guardrails, and automated knowledge indexing—dense pretrained LLM embeddings are the primary engine that converts unstructured human language into machine-understandable semantic vectors.

### Q15. Why did we compare `all-MiniLM-L6-v2` against `paraphrase-MiniLM-L3-v2` and `TF-IDF`?
**Answer:**  
- **`TF-IDF`** serves as our classical lexical baseline to quantify how much semantic embeddings improve paraphrase generalization and unsupervised clustering.
- **`paraphrase-MiniLM-L3-v2`** (3 Transformer layers) vs. **`all-MiniLM-L6-v2`** (6 Transformer layers) allows us to analyze the **depth vs. speed trade-off**: the 3-layer model encodes ~1.7x faster on CPU, while the 6-layer model captures deeper semantic abstractions, achieving higher classification F1 and clustering NMI.

### Q16. What are the limitations of our system?
**Answer:**  
1. **Context Window Truncation:** `MiniLM` has a maximum sequence length of $256$ WordPiece tokens (~180–200 words); very long documents must be truncated or chunked.
2. **Fixed Cluster Count ($K$) in K-Means:** K-Means requires specifying $K$ in advance and assumes roughly spherical clusters in embedding space.
3. **Static Pretrained Domain Knowledge:** Frozen embeddings without domain fine-tuning may struggle on highly specialized jargon (e.g., rare legal statutes or chemical formulas).

