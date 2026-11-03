# Text Classification, Clustering, and Topic Modeling Using Embeddings from Pretrained LLMs
Live Link: https://semantix-llm.onrender.com/

**Polished Academic Title:** *Semantic Text Analysis and Topic Discovery Using Pretrained LLM Embeddings*  
**Course:** Generative AI Course Project (Topic #10)  

---

## 1. Project Overview

Traditional Natural Language Processing (NLP) representations such as Bag-of-Words (BoW) and TF-IDF represent documents as high-dimensional sparse word-count vectors. Because they rely on exact keyword overlap, they fail to recognize semantic similarity when two texts use different words (synonyms or paraphrases).

This project implements a unified, CPU-efficient **Semantic Text Analysis Framework** powered by **Pretrained Transformer Language Model Embeddings** (`all-MiniLM-L6-v2` and `paraphrase-MiniLM-L3-v2`). Every text document is mapped via Multi-Head Self-Attention and Mean Pooling into a dense, L2-normalized **384-dimensional unit vector** ($\vec{v} \in \mathbb{R}^{384}$) where geometric proximity (Cosine Similarity) captures contextual meaning.

Using these pretrained embeddings, the system performs four integrated tasks:
1. **Semantic Similarity Analysis:** Demonstrates how semantically equivalent sentences (e.g., *"Artificial intelligence is transforming healthcare"* and *"Machine learning is improving medical diagnosis"*) achieve high cosine similarity in embedding space despite having zero shared words.
2. **Module 1 — Supervised Text Classification:** Trains lightweight classifiers (Logistic Regression, Calibrated Linear SVM, Random Forest) on frozen LLM embeddings to categorize documents into **Sports, Technology, Finance, Politics, and Healthcare**.
3. **Module 2 — Unsupervised Text Clustering:** Automatically discovers natural document groupings without labels using **K-Means** in normalized embedding space, validated via the Elbow Method, Silhouette Score, Adjusted Rand Index (ARI), and Normalized Mutual Information (NMI).
4. **Module 3 — Automated Topic Discovery:** Extracts human-interpretable themes, weighted topic keywords, and centroid-nearest exemplar documents using the **BERTopic Class-based TF-IDF (c-TF-IDF)** formulation over semantic clusters, alongside **2D PCA & t-SNE visualizations**.

---

## 2. Project Directory Structure

```text
d:\Practice\GenAI ProjectAG\
│
├── data\
│   ├── dataset.csv                  # 1,000 preprocessed documents across 5 balanced categories
│   ├── embeddings_cache.npz         # Cached 384D embeddings (all-MiniLM-L6-v2 & MiniLM-L3)
│   └── projections_cache.npz        # Cached 2D PCA & t-SNE coordinates
│
├── src\
│   ├── __init__.py
│   ├── data_loader.py               # Member 1: Dataset curation, cleaning & train/test split
│   ├── embeddings.py                # Member 2: Pretrained LLM embedding generation & similarity
│   ├── models.py                    # Member 3: Classification, K-Means clustering & evaluation
│   └── topic_visualizer.py          # Member 4: c-TF-IDF topic discovery, PCA/t-SNE & 7 figures
│
├── outputs\
│   ├── figures\                     # 7 high-resolution (220 DPI) PNG charts for Report & Slides
│   └── results\                     # Quantitative CSV tables & saved pipeline bundle (.joblib)
│
├── docs\
│   ├── Project_Report_5Pages.md     # Complete 5-page academic report (Markdown + PDF generator)
│   ├── Presentation_Slides.md       # Slide-by-slide presentation deck content (+ PPTX file)
│   └── Viva_Guide_and_Demo.md       # 20 Viva Questions & Answers + Live Demo Walkthrough
│
├── run_experiments.py               # Master CLI script to run all modules, experiments & plots
├── app.py                           # Member 5: Interactive 5-page Streamlit Web Dashboard
├── requirements.txt                 # Python package dependencies
└── README.md                        # Setup and execution instructions
```

---

## 3. Setup and Installation Instructions (Windows)

### Step 1: Open PowerShell in the Project Folder
```powershell
cd "d:\Practice\GenAI ProjectAG"
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run the Master Experiment & Pipeline Script
This script builds/loads `data/dataset.csv`, computes and caches the 384D embeddings in `data/embeddings_cache.npz`, runs all comparative experiments, saves CSV tables to `outputs/results/`, and generates all 7 figures in `outputs/figures/`:
```powershell
python run_experiments.py
```

### Step 4: Launch the Interactive Streamlit Dashboard
```powershell
python -m streamlit run app.py
```
Once launched, open `http://localhost:8501` in your browser to explore:
- **🏠 Home & Semantic Basics:** Corpus overview & live sentence similarity comparison (LLM vs. TF-IDF)
- **🎯 Module 1: Text Classification:** Live text classification with confidence bars & top-3 nearest corpus articles
- **🧩 Module 2: Text Clustering:** Interactive K-Means ($K=2\dots 10$) + 2D PCA/t-SNE cluster plots
- **🔍 Module 3: Topic Discovery:** Automated c-TF-IDF topic & sub-topic keyword discovery
- **📊 Evaluation & Experiments:** Full quantitative tables, confusion matrices, and benchmark graphs

---

## 4. Sample Input and Output

### Sample 1 — Semantic Similarity (Zero Lexical Overlap)
- **Text A:** `"Artificial intelligence is transforming healthcare."`
- **Text B:** `"Machine learning is improving medical diagnosis."`
- **Text C:** `"India won the cricket championship."`
- **Output:**
  - `TF-IDF Cosine Similarity (Text A, Text B)` = `0.0000` *(fails due to zero shared words)*
  - `Pretrained LLM Cosine Similarity (Text A, Text B)` = `0.66+` *(captures semantic equivalence)*
  - `Pretrained LLM Cosine Similarity (Text A, Text C)` = `< 0.10` *(recognizes unrelated domains)*

### Sample 2 — Module 1: Text Classification
- **Input:** `"India defeated Australia in the final match."` → **Output:** `Sports` (Confidence: `>90%`)
- **Input:** `"New artificial intelligence technology was released."` → **Output:** `Technology` (Confidence: `>90%`)
- **Input:** `"The stock market recorded a significant increase."` → **Output:** `Finance` (Confidence: `>90%`)

---

## 5. Individual Contributions (Group of 5)

| Member | Assigned Module | Core Technical Responsibilities |
| :--- | :--- | :--- |
| **Member 1** | **Dataset & Preprocessing** (`src/data_loader.py`) | Benchmark dataset curation (1,000 docs across 5 categories), context-preserving preprocessing pipeline, stratified train/test splitting, corpus statistical profiling. |
| **Member 2** | **Pretrained Embeddings & Similarity** (`src/embeddings.py`) | Transformer bi-encoder (`all-MiniLM-L6-v2` & `paraphrase-MiniLM-L3-v2`) integration, Mean Pooling & L2 normalization, embedding caching, semantic vs. lexical similarity analysis. |
| **Member 3** | **Classification & Clustering** (`src/models.py`) | Supervised classification heads (Logistic Regression, Linear SVM, Random Forest), K-Means clustering, Elbow & Silhouette optimization, Confusion Matrix, ARI & NMI evaluation. |
| **Member 4** | **Topic Discovery & Visualization** (`src/topic_visualizer.py`) | Class-based TF-IDF (c-TF-IDF) topic & sub-topic modeling, Topic Diversity & Semantic Coherence metrics, 2D PCA & t-SNE dimensionality reduction, 7 publication charts. |
| **Member 5** | **Integration, UI & Documentation** (`app.py`, `run_experiments.py`, `docs/`) | Multi-page Streamlit dashboard, end-to-end automated experiment runner, system testing, 5-page report compilation, presentation deck, and viva preparation. |

