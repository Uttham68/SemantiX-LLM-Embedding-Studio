# COURSE PROJECT REPORT — GENERATIVE AI

**Official Course Title:** Text classification/clustering and topic modeling using embeddings from pretrained LLMs  
**Polished Academic Title:** Semantic Text Analysis and Topic Discovery Using Pretrained LLM Embeddings  
**Course:** Generative AI | **Topic Number:** #10  

---

## PAGE 1: Abstract, Problem Statement, and Motivation

### 1.1 Abstract
Unstructured text data across journalism, healthcare, finance, and enterprise operations grows exponentially, creating an urgent need for automated categorization, grouping, and thematic discovery. Traditional Natural Language Processing (NLP) representations such as Bag-of-Words (BoW) and Term Frequency–Inverse Document Frequency (TF-IDF) rely on exact keyword overlap, resulting in high-dimensional sparse vectors that fail to capture synonymy, polysemy, and sentence-level context. Conversely, invoking massive autoregressive Large Language Models (LLMs) for every document in large-scale clustering and classification pipelines is computationally expensive and high-latency. In this work, we design, implement, and evaluate a unified, CPU-efficient semantic text analysis framework powered by dense embeddings from pretrained Transformer bi-encoders (`all-MiniLM-L6-v2` and `paraphrase-MiniLM-L3-v2`). By projecting documents onto a 384-dimensional unit hypersphere ($\mathbb{R}^{384}$), our system performs (1) Supervised Text Classification, (2) Unsupervised K-Means Document Clustering, and (3) Automated Topic Discovery via Class-based TF-IDF (c-TF-IDF). Evaluated on a balanced 1,000-document benchmark across five domains (*Sports, Technology, Finance, Politics, Healthcare*), `all-MiniLM-L6-v2` achieves **98.0% test accuracy** (**Macro F1 = 0.9800**), **100.0% accuracy on unseen synonym paraphrases** (compared to **96.0% for TF-IDF**), and an unsupervised clustering **Normalized Mutual Information (NMI) of 0.8754** (**ARI = 0.9028**).

### 1.2 Problem Statement
Given a corpus of unstructured text documents $\mathcal{D} = \{d_1, d_2, \dots, d_N\}$, lexical representations map each document into a sparse vocabulary vector $\vec{x}_{\text{lex}} \in \mathbb{R}^{|V|}$ where $|V| \gg 10^3$. When two documents $d_A$ and $d_B$ express equivalent meaning using distinct vocabulary ($\text{words}(d_A) \cap \text{words}(d_B) = \emptyset$), their lexical inner product collapses to zero:
$$\cos(\vec{x}_{\text{lex}}(d_A), \vec{x}_{\text{lex}}(d_B)) = 0.0000$$
For example, *"Artificial intelligence is transforming healthcare"* and *"Machine learning is improving medical diagnosis"* share zero non-stopword tokens yet convey closely related semantics. The core problem addressed in this project is constructing a single, computationally lightweight pretrained embedding pipeline $f_\theta: \mathcal{D} \rightarrow \mathbb{S}^{383} \subset \mathbb{R}^{384}$ that preserves semantic geometry across three downstream paradigms:
1. **Supervised Classification:** Learning a decision boundary $g_\phi(\vec{e}_i) \rightarrow y_i \in \mathcal{Y}$ with high sample efficiency and robustness to paraphrasing.
2. **Unsupervised Clustering:** Partitioning unlabeled embeddings $\{ \vec{e}_i \}_{i=1}^N$ into $K$ semantically cohesive groups without human labels.
3. **Automated Topic Discovery:** Extracting discriminative human-readable thematic keywords and exemplar documents from discovered semantic clusters.

### 1.3 Motivation & Generative AI Relevance
Pretrained language embedding models share the exact **Transformer self-attention architecture** (Vaswani et al., 2017) that underpins modern Generative AI. Moreover, dense semantic embeddings form the foundational representation layer for **Retrieval-Augmented Generation (RAG)**, agentic vector memory, semantic caching, and automated corpus curation. Demonstrating how compact pretrained bi-encoders unify classification, clustering, and topic discovery provides both theoretical insight into Transformer representations and practical utility on standard CPU hardware.

---

## PAGE 2: Related Work and Theoretical Background

### 2.1 Evolution of Text Representations
1. **Lexical Representations (BoW & TF-IDF):** Salton & Buckley (1988) introduced TF-IDF to weight terms by inverse document frequency. While effective for exact keyword matching, TF-IDF ignores word order, syntax, and semantic similarity.
2. **Static Word Embeddings (Word2Vec & GloVe):** Mikolov et al. (2013) and Pennington et al. (2014) learned dense word vectors from co-occurrence windows. However, static vectors assign a single fixed vector to polysemous words (e.g., *"bank"* in finance vs. river bank) and lose sentence-level compositionality when averaged.
3. **Contextual Transformer Encoders (BERT & Sentence-BERT):** Devlin et al. (2019) introduced BERT using bidirectional Multi-Head Self-Attention. However, cross-encoder BERT requires $O(N^2)$ forward passes for pairwise similarity. **Sentence-BERT (Reimers & Gurevych, 2019)** and **MiniLM (Wang et al., 2020)** solved this via a **Siamese Bi-Encoder** architecture trained with contrastive loss on over 1 billion sentence pairs, reducing pairwise comparison of $1,000$ documents from hours to milliseconds.

### 2.2 Paradigms in Classification, Clustering, and Topic Modeling
- **Classification Approaches:** Traditional ML pairs TF-IDF with Naive Bayes or SVMs. End-to-end LLM fine-tuning updates all $110\text{M}+$ parameters on GPUs. **Embedding-based classification** freezes the pretrained Transformer encoder and trains a linear/kernel classifier on the 384D features—achieving near-fine-tuned accuracy on CPU in $<0.1$ seconds.
- **Clustering & Topic Modeling:** Classical Latent Dirichlet Allocation (LDA; Blei et al., 2003) models documents as bag-of-words probability mixtures, failing on short texts and synonyms. Modern **BERTopic (Grootendorst, 2022)** clusters dense Transformer embeddings and applies **Class-based TF-IDF (c-TF-IDF)** to each cluster macro-document, yielding coherent, non-overlapping semantic topics.

---

## PAGE 3: Methodology and System Architecture

### 3.1 End-to-End Architecture
```text
Raw Corpus (1,000 Docs) ──► Context-Preserving Preprocessing ──► Pretrained Bi-Encoder (all-MiniLM-L6-v2)
                                                                           │
                       ┌───────────────────────────────────────────────────┼───────────────────────────────────────────────────┐
                       ▼                                                   ▼                                                   ▼
     [Module 1: Supervised Classification]               [Module 2: Unsupervised Clustering]                 [Module 3: Topic Discovery]
     • Logistic Regression / Linear SVM / RF             • K-Means++ on Unit Hypersphere R^384               • Semantic Cluster Macro-Documents
     • Category Prediction & Confidence                  • Elbow (WCSS) & Silhouette K-Selection             • Class-based TF-IDF (c-TF-IDF)
     • Paraphrase Generalization Benchmark               • PCA & t-SNE 2D Manifold Visualization             • Topic Keywords, Coherence & Diversity
```

### 3.2 Mathematical Formulation of Each Component
1. **Transformer Mean Pooling & Unit Hypersphere Normalization:**  
   Given input token sequence $T = (t_1, \dots, t_L)$, the Transformer encoder produces contextualized hidden states $H = (\vec{h}_1, \dots, \vec{h}_L) \in \mathbb{R}^{L \times 384}$. We apply attention-masked mean pooling followed by $L_2$ normalization:
   $$\vec{u} = \frac{1}{L} \sum_{l=1}^L \vec{h}_l, \qquad \vec{e} = \frac{\vec{u}}{\|\vec{u}\|_2} \in \mathbb{S}^{383}$$
2. **Equivalence of Euclidean K-Means and Cosine Similarity on Unit Vectors:**  
   For any two unit-normalized embeddings $\|\vec{e}_i\|_2 = \|\vec{e}_j\|_2 = 1$:
   $$\|\vec{e}_i - \vec{e}_j\|_2^2 = \|\vec{e}_i\|_2^2 + \|\vec{e}_j\|_2^2 - 2(\vec{e}_i \cdot \vec{e}_j) = 2\left(1 - \cos(\vec{e}_i, \vec{e}_j)\right)$$
   Consequently, minimizing Within-Cluster Sum of Squares (WCSS) in K-Means is monotonically equivalent to maximizing intra-cluster Cosine Similarity.
3. **Class-Based TF-IDF (c-TF-IDF) for Topic Discovery:**  
   After partitioning documents into $C$ semantic clusters, all documents in cluster $c$ are concatenated into a single class document. The importance weight of term $w$ in topic $c$ is:
   $$\text{c-TF-IDF}_{w, c} = \frac{\text{tf}_{w, c}}{\sum_{w'} \text{tf}_{w', c}} \times \log\left(1 + \frac{A}{\sum_{c'=1}^C \text{tf}_{w, c'}}\right)$$
   where $A$ is the average number of words per cluster across all $C$ topics.

---

## PAGE 4: Implementation Details and Dataset Description

### 4.1 Dataset Construction, Suitability, and Limitations
We constructed a balanced benchmark corpus of **$1,000$ documents** ($200$ documents per class) across **five distinct domains**: **Sports**, **Technology**, **Finance**, **Politics**, and **Healthcare**, combining real articles from the **BBC News** benchmark (`SetFit/bbc-news`) and the **20 Newsgroups** medical repository (`sci.med`), enriched with contemporary sub-topic anchors (Artificial Intelligence, Cybersecurity, Cloud Computing, Robotics, Cricket, Equities, and Clinical Diagnostics).
- **Splitting Protocol:** Stratified $75\%$ Training ($750$ docs; $150$/class) and $25\%$ Testing ($250$ docs; $50$/class), plus an independent **$25$-sentence Zero-Overlap Paraphrase Generalization Test Set**.
- **Preprocessing Strategy:** Unlike TF-IDF pipelines that strip stopwords and stem words, our `clean_text_for_embeddings()` pipeline removes HTML, URLs, and header noise while preserving natural sentence grammar so Transformer self-attention heads retain full contextual relationships.

### 4.2 Technology Stack & Modular Implementation
The system is implemented in **Python 3** using `sentence-transformers`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn`, and `streamlit`:
- `src/data_loader.py`: Automated corpus acquisition, cleaning, and stratified splitting.
- `src/embeddings.py`: Bi-encoder inference (`all-MiniLM-L6-v2` & `paraphrase-MiniLM-L3-v2`), `.npz` disk caching, and cosine similarity search.
- `src/models.py`: Logistic Regression, Calibrated Linear SVM, Random Forest, K-Means clustering, Hungarian cluster-to-label alignment, and Elbow/Silhouette sweeps.
- `src/topic_visualizer.py`: c-TF-IDF topic extraction, PCA/t-SNE 2D projections, and automated generation of 7 publication figures.
- `app.py`: 5-module interactive Streamlit dashboard for live classroom demonstration.

---

## PAGE 5: Experimental Results, Evaluation, Limitations, and Contributions

### 5.1 Experiment A: Semantic Similarity Verification (Lexical vs. Pretrained LLM)
| Text A | Text B | Shared Keywords | TF-IDF Cosine Sim | LLM Embedding Sim |
| --- | --- | --- | --- | --- |
| Artificial intelligence is transforming healthcare. | Machine learning is improving medical diagnosis. | 0 | 0.0 | 0.5829 |
| Artificial intelligence is transforming healthcare. | India won the cricket championship. | 0 | 0.0 | 0.0616 |
| Machine learning is improving medical diagnosis. | India won the cricket championship. | 0 | 0.0 | 0.07 |
| India won the cricket championship. | The national cricket team celebrated their tournament victory. | 1 | 0.1473 | 0.6876 |
| The stock market recorded a significant increase. | Equity indices surged as corporate earnings beat expectations. | 0 | 0.0 | 0.5248 |
| India won the cricket championship. | The stock market recorded a significant increase. | 0 | 0.0 | 0.1221 |

*Finding:* On *"Artificial intelligence is transforming healthcare"* vs. *"Machine learning is improving medical diagnosis"*, TF-IDF scores **`0.0000`** due to zero shared words, whereas `all-MiniLM-L6-v2` scores **`0.5829`** while assigning **`0.0616`** to the unrelated Cricket sentence.

### 5.2 Experiment B: Representation & Classifier Benchmark (750 Train / 250 Test + 25 Paraphrases)
| Representation Model | Vector Dim | Test Accuracy | Test Macro F1 | Paraphrase Generalization Acc | Clustering Silhouette | Clustering ARI | Clustering NMI | Encoding Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TF-IDF (Lexical Baseline) | 1500 | 0.948 | 0.9483 | 0.96 | 0.0311 | 0.6026 | 0.6942 | 0.54 |
| paraphrase-MiniLM-L3-v2 (3-Layer LLM) | 384 | 0.952 | 0.9518 | 1.0 | 0.0909 | 0.8518 | 0.8214 | 39.08 |
| all-MiniLM-L6-v2 (6-Layer LLM) | 384 | 0.98 | 0.98 | 1.0 | 0.1134 | 0.9028 | 0.8754 | 176.4 |

**Classifier Head Comparison on `all-MiniLM-L6-v2` Embeddings:**
| Classifier Head | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Paraphrase Generalization Acc | Train Time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.98 | 0.98 | 0.98 | 0.98 | 1.0 | 0.1674 |
| Linear SVM | 0.96 | 0.9605 | 0.96 | 0.9598 | 0.96 | 0.4207 |
| Random Forest | 0.964 | 0.9639 | 0.964 | 0.9638 | 1.0 | 1.6348 |

*Finding:* While TF-IDF performs reasonably on in-distribution news articles that repeat common keywords, its accuracy drops to **96.0%** on the Zero-Overlap Paraphrase Test Set. In contrast, `all-MiniLM-L6-v2` achieves **100.0% paraphrase accuracy** and improves unsupervised clustering **NMI from 0.6942 (TF-IDF) to 0.8754** (**ARI = 0.9028**).

### 5.3 Experiment C & D: Unsupervised Clustering ($K=5$) and c-TF-IDF Topic Discovery
**Emergent Unsupervised Clusters ($K=5$, Silhouette = 0.1134):**
| Cluster ID | Emerging Theme | Document Count | Cluster Purity (%) | Avg Centroid Cosine Sim |
| --- | --- | --- | --- | --- |
| Cluster 1 | Cluster 1 (Technology-related) | 206 | 93.7 | 0.4328 |
| Cluster 2 | Cluster 2 (Politics-related) | 195 | 96.4 | 0.5264 |
| Cluster 3 | Cluster 3 (Sports-related) | 200 | 99.0 | 0.5139 |
| Cluster 4 | Cluster 4 (Healthcare-related) | 209 | 94.3 | 0.3248 |
| Cluster 5 | Cluster 5 (Finance-related) | 190 | 96.8 | 0.4659 |

**Discovered Technology Sub-Topics via c-TF-IDF (Topic Diversity = 100.0%, Mean Coherence = 0.4696):**
| Topic ID | Discovered Theme | Documents | Top Keywords (c-TF-IDF) | Semantic Coherence |
| --- | --- | --- | --- | --- |
| Topic 1 | Topic 1: Cybersecurity & Email Security (Security, Software, Microsoft) | 59 | security, software, microsoft, users, search, mail | 0.4479 |
| Topic 2 | Topic 2: Digital Media & File Sharing (Computer, Apple, Mac) | 44 | computer, apple, mac, technology, digital, gadget | 0.3607 |
| Topic 3 | Topic 3: Gaming & Digital Entertainment (Games, Sony, Game) | 37 | games, sony, game, handheld, dvd, video | 0.5203 |
| Topic 4 | Topic 4: Telecom, Mobile & Broadband (Mobile, Broadband, Tv) | 60 | mobile, broadband, tv, service, phone, net | 0.5496 |

### 5.4 Limitations and Future Scope
1. **Context Window Limit:** `MiniLM` models truncate inputs beyond $256$ WordPiece tokens (~180 words), losing tail information in long documents.
2. **Spherical K-Means Assumption:** K-Means requires pre-specifying $K$ and assumes isotropic cluster geometry.
3. **Domain Specificity:** Without domain-adaptive pretraining, frozen general-purpose embeddings may underperform on niche clinical or legal nomenclature.

### 5.5 Individual Contributions of the 5-Member Team
- **Member 1 (Dataset & Preprocessing — `src/data_loader.py`):** Curated the 1,000-document 5-domain corpus, engineered the context-preserving cleaning pipeline, executed stratified splits, and authored Section 1 & 4.1.
- **Member 2 (Embeddings & Semantic Similarity — `src/embeddings.py`):** Integrated `all-MiniLM-L6-v2` and `paraphrase-MiniLM-L3-v2`, built the `.npz` embedding cache, ran the paraphrase similarity study, and authored Section 2 & 3.2.
- **Member 3 (Classification & Clustering — `src/models.py`):** Implemented Logistic Regression, Linear SVM, Random Forest, and K-Means clustering; ran the $K=2..10$ Elbow/Silhouette sweep; computed ARI/NMI/Confusion Matrices; authored Section 5.2 & 5.3.
- **Member 4 (Topic Discovery & Visualization — `src/topic_visualizer.py`):** Implemented the BERTopic c-TF-IDF algorithm, Topic Diversity/Coherence metrics, PCA & t-SNE 2D projections, and generated all 7 figures.
- **Member 5 (Dashboard Integration & Documentation — `app.py`, `run_experiments.py`):** Built the 5-page interactive Streamlit dashboard, automated the end-to-end experiment runner, conducted integration testing, and compiled the final report and presentation.

---

## REFERENCES (Excluded from 5-Page Count)
1. Vaswani, A., et al. (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems (NeurIPS), 30.
2. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT.
3. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP-IJCNLP.
4. Wang, W., et al. (2020). *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*. NeurIPS, 33.
5. Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a class-based TF-IDF procedure*. arXiv preprint arXiv:2203.05794.
6. Greene, D., & Cunningham, P. (2006). *Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering* (BBC News Dataset). ICML.
