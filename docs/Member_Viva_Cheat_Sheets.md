# Member-by-Member Viva Cheat Sheets (Group of 5)

**Official Course Title:** *"Text classification/clustering and topic modeling using embeddings from pretrained LLMs"*  
**Polished Title:** *"Semantic Text Analysis and Topic Discovery Using Pretrained LLM Embeddings"*

---

## 👤 MEMBER 1: Dataset Selection & Context-Preserving Preprocessing
**Owned File:** [`src/data_loader.py`](file:///d:/Practice/GenAI%20ProjectAG/src/data_loader.py) | **Output File:** [`data/dataset.csv`](file:///d:/Practice/GenAI%20ProjectAG/data/dataset.csv)  
**Key Figure:** [`outputs/figures/fig1_dataset_distribution.png`](file:///d:/Practice/GenAI%20ProjectAG/outputs/figures/fig1_dataset_distribution.png)

### 1. Functions You Own & Should Point To
- `restore_clean_formatting(raw_text)` (Lines 40–90): Restores sentence-case capitalization, uppercase acronyms (`US`, `UK`, `AI`, `NASA`, `GDP`), and first-line headline periods on raw BBC News & 20 Newsgroups texts, and caps length at 180 words (~256 WordPiece tokens).
- `infer_document_subtopic(text, category)` (Lines 105–163): Assigns content-accurate subtopics to each article based on domain vocabulary.
- `load_or_create_dataset()` & `get_train_test_split()` (Lines 350–380): Builds the 1,000-document balanced corpus ($200$ per class across **Sports, Technology, Finance, Politics, Healthcare**) with **1,000/1,000 unique texts** and performs a stratified $75\%$ Train ($750$ docs) / $25\%$ Test ($250$ docs) split.

### 2. Top 4 Viva Questions for Member 1
- **Q1: Why didn't you apply stemming, lemmatization, or stopword removal before feeding text to the pretrained LLM?**
  - *Answer:* Pretrained Transformer models like `all-MiniLM-L6-v2` use **Multi-Head Self-Attention** trained on natural, grammatical sentences. Removing stopwords (like *"not"* or *"to"*) or stemming words (turning *"organization"* into *"organ"*) destroys sentence grammar and changes the contextual meaning attended to by the Transformer heads. We only strip HTML/URL/header noise for the LLM, and reserve stopword removal strictly for the lexical TF-IDF baseline and c-TF-IDF keyword extraction (`clean_text_for_lexical()`).
- **Q2: Why did you truncate documents at 180 words?**
  - *Answer:* `all-MiniLM-L6-v2` has a maximum positional embedding window of **256 WordPiece subword tokens** (which corresponds to roughly 170–190 English words). Truncating cleanly at 180 words prevents silent token drop and ensures rapid CPU encoding.
- **Q3: Why is a stratified train-test split necessary here?**
  - *Answer:* Stratified splitting (`stratify=df["category"]`) guarantees that both the 750-document training set ($150$/class) and the 250-document test set ($50$/class) have an exact $20\%$ balance across all 5 categories, preventing class-imbalance bias in Accuracy and Macro-F1.
- **Q4: Are there any duplicate sentences leaking between your training and test sets?**
  - *Answer:* Zero. All $1,000$ documents in `data/dataset.csv` have $100\%$ unique full texts and $100\%$ unique prefixes (`df['text'].nunique() == 1000`), verified by our automated test suite (`test_project.py`).

---

## 👤 MEMBER 2: Pretrained LLM Embeddings & Semantic Similarity
**Owned File:** [`src/embeddings.py`](file:///d:/Practice/GenAI%20ProjectAG/src/embeddings.py) | **Output File:** [`data/embeddings_cache.npz`](file:///d:/Practice/GenAI%20ProjectAG/data/embeddings_cache.npz)  
**Key Figure:** [`outputs/figures/fig5_semantic_similarity_heatmap.png`](file:///d:/Practice/GenAI%20ProjectAG/outputs/figures/fig5_semantic_similarity_heatmap.png)

### 1. Functions You Own & Should Point To
- `encode_texts(texts, model_name, normalize=True)` (Lines 43–74): Passes cleaned texts through the pretrained Transformer bi-encoder (`all-MiniLM-L6-v2` or `paraphrase-MiniLM-L3-v2`), applies **Mean Pooling**, and **L2-normalizes** every output vector to unit length ($\|\vec{v}\|_2 = 1$).
- `evaluate_semantic_similarity_demo()` (Lines 135–195): Computes pairwise Cosine Similarity matrices comparing Pretrained LLM Embeddings against TF-IDF on paraphrase vs. unrelated sentence pairs.
- `semantic_search_corpus()` (Lines 204–234): Retrieves the top-$K$ distinct cosine-nearest documents in the corpus in $<15\text{ ms}$.

### 2. Top 4 Viva Questions for Member 2
- **Q1: How does `all-MiniLM-L6-v2` convert a variable-length sentence into a fixed 384-dimensional vector?**
  - *Answer:* First, the WordPiece tokenizer splits the sentence into subword tokens (`[CLS], t_1, ..., t_L, [SEP]`). Second, the 6 Transformer layers compute contextualized token embeddings $(\vec{h}_1, \dots, \vec{h}_L) \in \mathbb{R}^{L \times 384}$ via Multi-Head Self-Attention. Third, **Mean Pooling** averages those token vectors across the sequence length $L$. Finally, **L2 Normalization** divides the vector by its Euclidean norm so $\|\vec{e}\|_2 = 1$.
- **Q2: Why does TF-IDF give `0.0000` similarity on *"Artificial intelligence is transforming healthcare"* vs. *"Machine learning is improving medical diagnosis"*, while your LLM gives `0.5829`?**
  - *Answer:* TF-IDF only counts exact word overlap—since those two sentences share zero non-stopword tokens, their TF-IDF dot product is zero. In contrast, `all-MiniLM-L6-v2` was contrastively pretrained on over 1 billion sentence pairs, so its self-attention layers map synonyms (*AI $\leftrightarrow$ machine learning*, *healthcare $\leftrightarrow$ medical diagnosis*) to the same region of the 384D hypersphere.
- **Q3: Why do you normalize embeddings to unit length ($\|\vec{e}\|_2 = 1$)?**
  - *Answer:* When two vectors have unit length, their Cosine Similarity is simply their dot product ($\cos(\vec{u}, \vec{v}) = \vec{u} \cdot \vec{v}$), which makes nearest-neighbor search extremely fast. Furthermore, squared Euclidean distance becomes $\|\vec{u} - \vec{v}\|_2^2 = 2(1 - \cos(\vec{u}, \vec{v}))$, making standard Euclidean K-Means clustering mathematically equivalent to cosine similarity clustering.
- **Q4: What is the difference between your two compared LLM models (`all-MiniLM-L6-v2` vs. `paraphrase-MiniLM-L3-v2`)?**
  - *Answer:* Both output 384D vectors, but `paraphrase-MiniLM-L3-v2` has only **3 Transformer layers** (encoding ~4.5x faster on CPU and specifically tuned on paraphrase corpora), whereas `all-MiniLM-L6-v2` has **6 Transformer layers**, capturing deeper semantic structure and achieving higher Test Accuracy (`97.6%`) and Clustering ARI (`0.91+`).

---

## 👤 MEMBER 3: Supervised Classification & Unsupervised Clustering
**Owned File:** [`src/models.py`](file:///d:/Practice/GenAI%20ProjectAG/src/models.py) | **Output Files:** `outputs/results/model_comparison.csv`, `k_selection_sweep.csv`  
**Key Figures:** [`fig2_confusion_matrix.png`](file:///d:/Practice/GenAI%20ProjectAG/outputs/figures/fig2_confusion_matrix.png) & [`fig3_clustering_elbow_silhouette.png`](file:///d:/Practice/GenAI%20ProjectAG/outputs/figures/fig3_clustering_elbow_silhouette.png)

### 1. Functions You Own & Should Point To
- `build_classifier()` & `train_and_evaluate_classifier()` (Lines 43–110): Trains Logistic Regression (`C=12.0`), Calibrated Linear SVM, and Random Forest on the 384D embeddings and computes Accuracy, Precision, Recall, Macro-F1, and the Confusion Matrix.
- `run_kmeans_clustering()` & `align_clusters_to_categories()` (Lines 195–290): Runs unsupervised `KMeans(n_clusters=k, init='k-means++')` on unlabeled embeddings and aligns discovered clusters to ground-truth categories using the Hungarian algorithm (`linear_sum_assignment`) to evaluate Purity, ARI, and NMI.
- `evaluate_k_selection_sweep()` (Lines 295–318): Sweeps $K \in \{2\dots 10\}$ computing Inertia (WCSS) and Cosine Silhouette Score.

### 2. Top 4 Viva Questions for Member 3
- **Q1: Why did you set `C=12.0` in Logistic Regression instead of the default `C=1.0`?**
  - *Answer:* Because our input embeddings are L2-normalized ($\|\vec{x}\|_2 = 1$), the magnitude of the dot product $\vec{w} \cdot \vec{x}$ is bounded by $\|\vec{w}\|_2$. Strong L2 regularization (`C=1.0`) shrinks $\|\vec{w}\|_2$, which squashes softmax probabilities toward a flat uniform distribution ($20\%$ per class). Setting `C=12.0` acts as an inverse temperature parameter ($1/\tau$), allowing the classifier to output sharp, well-calibrated probabilities ($80\%\text{–}98\%$) on clear in-domain inputs.
- **Q2: How do you prove that $K=5$ is the optimal number of clusters without looking at the labels?**
  - *Answer:* We use two unsupervised internal metrics across $K = 2\dots 10$: (1) **The Elbow Method** on Within-Cluster Sum of Squares (Inertia), which bends at $K=5$, and (2) **Cosine Silhouette Score**, which measures intra-cluster cohesion vs. nearest-cluster separation and reaches its global maximum (`0.122+`) at $K=5$.
- **Q3: What do ARI (Adjusted Rand Index) and NMI (Normalized Mutual Information) measure?**
  - *Answer:* Both are external cluster validation metrics comparing unsupervised cluster assignments against the true categories: **ARI (`0.91+`)** measures pairwise agreement corrected for chance (where $0$ is random and $1$ is identical), and **NMI (`0.88+`)** measures shared information-theoretic reduction in entropy between clusters and true classes.
- **Q4: Why did you use Agglomerative Clustering in the Public Batch Sandbox for small inputs ($N \le 15$)?**
  - *Answer:* On tiny sample sizes ($N=5$ sentences), randomized `k-means++` centroid seeding can occasionally group two distinct singletons together. **Agglomerative Clustering** with cosine distance and average linkage is 100% deterministic and bottom-up, guaranteeing that 5 sentences from 5 distinct domains separate into 5 clean groups every time.

---

## 👤 MEMBER 4: Topic Discovery (c-TF-IDF) & 2D Visualizations
**Owned File:** [`src/topic_visualizer.py`](file:///d:/Practice/GenAI%20ProjectAG/src/topic_visualizer.py) | **Output File:** [`data/projections_cache.npz`](file:///d:/Practice/GenAI%20ProjectAG/data/projections_cache.npz)  
**Key Figures:** [`fig4_semantic_space_pca_tsne.png`](file:///d:/Practice/GenAI%20ProjectAG/outputs/figures/fig4_semantic_space_pca_tsne.png) & [`fig7_topic_keywords.png`](file:///d:/Practice/GenAI%20ProjectAG/outputs/figures/fig7_topic_keywords.png)

### 1. Functions You Own & Should Point To
- `compute_ctfidf_topics()` & `infer_unique_topic_title()` (Lines 135–265): Groups embeddings into semantic clusters, concatenates all documents in each cluster $c$ into a single macro-document, applies **Class-based TF-IDF (c-TF-IDF)** to extract top keywords, and maps keywords to unique human-readable topic titles.
- `compute_or_load_2d_projections()` (Lines 280–320): Reduces 384D embeddings to 2D using **PCA** and **t-SNE** (`PCA(384 -> 30)` followed by `TSNE(30 -> 2, perplexity=30)`).
- `generate_all_visualizations()` (Lines 325–560): Generates all 7 publication figures at 220 DPI.

### 2. Top 4 Viva Questions for Member 4
- **Q1: How is Topic Discovery different from Clustering?**
  - *Answer:* Clustering only partitions documents into unlabeled groups (`Cluster 1`, `Cluster 2`, etc.) without telling us *what* each group is about. Topic Discovery adds **semantic interpretability** by extracting the discriminative keywords, human-readable theme labels (e.g., *Cybersecurity & Email Security*, *Gaming & Digital Entertainment*, *Telecom, Mobile & Broadband*), and centroid-nearest exemplar documents for each group.
- **Q2: Write or explain the Class-based TF-IDF (c-TF-IDF) formula used in your code.**
  - *Answer:* First, all documents in cluster $c$ are joined into a single macro-document. Then for word $w$ in cluster $c$:
    $$\text{c-TF-IDF}_{w, c} = \frac{\text{tf}_{w, c}}{\sum_{w'} \text{tf}_{w', c}} \times \log\left(1 + \frac{A}{\sum_{c'} \text{tf}_{w, c'}}\right)$$
    where $A$ is the average number of words per cluster. Unlike standard TF-IDF (which scores words per *individual document*), c-TF-IDF scores words per *entire semantic cluster*, highlighting terms that distinguish one topic from all other topics.
- **Q3: How do you measure the quality of discovered topics quantitatively?**
  - *Answer:* We compute two metrics: (1) **Topic Diversity (`95%–100%`)**, which is the fraction of unique words among the top keywords across all discovered topics (proving topics do not overlap), and (2) **Intra-Topic Semantic Coherence (`~0.48`)**, which is the mean pairwise cosine similarity of the top 15 centroid-nearest document embeddings inside each topic.
- **Q4: Why did you reduce 384D to 30D with PCA before running t-SNE?**
  - *Answer:* Running t-SNE directly on 384 dimensions is slower and sensitive to high-dimensional noise. Projecting $384\text{D} \rightarrow 30\text{D}$ with PCA first preserves $>85\%$ of global variance while suppressing noise and accelerating t-SNE manifold convergence.

---

## 👤 MEMBER 5: Full-Stack Integration, Interactive Dashboards & Benchmarking
**Owned Files:** [`app.py`](file:///d:/Practice/GenAI%20ProjectAG/app.py), [`web_app.py`](file:///d:/Practice/GenAI%20ProjectAG/web_app.py), [`run_experiments.py`](file:///d:/Practice/GenAI%20ProjectAG/run_experiments.py), [`web/`](file:///d:/Practice/GenAI%20ProjectAG/web/index.html)  
**Key Figure:** [`outputs/figures/fig6_model_comparison_metrics.png`](file:///d:/Practice/GenAI%20ProjectAG/outputs/figures/fig6_model_comparison_metrics.png)

### 1. Functions You Own & Should Point To
- `run_all_experiments()` in `run_experiments.py`: Orchestrates all 4 modules end-to-end, benchmarks `TF-IDF` vs. `paraphrase-MiniLM-L3-v2` vs. `all-MiniLM-L6-v2`, and saves the precomputed pipeline state (`outputs/results/trained_pipeline.joblib`).
- `app.py` (Faculty Evaluation Streamlit UI) & `web_app.py` + `web/script.js` (Public Interactive FastAPI Web Studio with live HTML5 Canvas visualizations, 2D Live Query Star projection, and dynamic similarity triangles).

### 2. Top 4 Viva Questions for Member 5
- **Q1: How does your web dashboard respond in `< 15 ms` offline without a GPU?**
  - *Answer:* Encoding 1,000 documents takes ~2–3 minutes once during offline build (`run_experiments.py`). We cache the L2-normalized 384D corpus vectors in `data/embeddings_cache.npz`, the 2D projections in `data/projections_cache.npz`, and the fitted classifiers in `trained_pipeline.joblib`. At runtime, the server only encodes the single user query sentence (taking ~10 ms on CPU) and performs a fast matrix dot product against the cached corpus vectors.
- **Q2: How do you plot a live user query (`⭐ YOUR QUERY`) onto the 2D t-SNE Semantic Galaxy canvas when t-SNE has no `.transform()` method?**
  - *Answer:* Because t-SNE is non-parametric, we compute the cosine similarity between the query's 384D embedding $\vec{q}$ and all 1,000 corpus embeddings, select the Top-5 nearest neighbors, and compute a **temperature-scaled softmax barycentric interpolation** of their 2D coordinates:
    $$(x_q, y_q) = \sum_{m=1}^5 \frac{\exp(s_m / 0.04)}{\sum_{j=1}^5 \exp(s_j / 0.04)} (x_{i_m}, y_{i_m})$$
    This places the live query star accurately inside its semantic neighborhood in both t-SNE and PCA views in $<1\text{ ms}$.
- **Q3: Why does `TF-IDF` get `95.6%` on the test split, and why do we still need Pretrained LLM Embeddings?**
  - *Answer:* Standard benchmark articles within the same news source repeat common domain keywords (like `"parliament"` or `"match"`), which helps TF-IDF on supervised classification. However, when tested on **unseen paraphrases with synonyms**, TF-IDF drops sharply while `all-MiniLM-L6-v2` achieves **`96.0%`**, and on **unsupervised clustering**, `all-MiniLM-L6-v2` improves **ARI from `0.80` (TF-IDF) to `0.91+`** and **Silhouette Score by nearly $4\times$ (`0.032` $\rightarrow$ `0.122`)**.
- **Q4: How does your Similarity Lab handle sentences entered in any arbitrary order (A, B, C)?**
  - *Answer:* It computes the full $3 \times 3$ pairwise cosine similarity matrix dynamically, evaluates each pair against calibrated semantic thresholds (`>= 0.40` Related, `0.22–0.39` Moderate, `< 0.22` Unrelated), identifies `argmax` across `(A,B)`, `(A,C)`, and `(B,C)`, and renders the 2D geometric triangle canvas dynamically.

