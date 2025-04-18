
# 🤖 AI Code Authorship Classification

## 🔍 Overview

This project investigates the detection of AI-generated code versus human-written code, addressing academic integrity concerns in educational settings where generative models (e.g., ChatGPT, Codex) are increasingly used.

Traditional plagiarism tools like MOSS and TF-IDF clustering fail to distinguish between AI vs. human authorship. This notebook builds and compares several classification pipelines that utilize both lexical (TF-IDF) and structural (AST) features, integrating a contrastive learning network with a neural classifier for enhanced discrimination.

---

## 📘 Notebook Objective

This notebook:
- Loads a labeled dataset of AI-generated vs. human-written code
- Extracts TF-IDF and AST-based vector features
- Trains machine learning classifiers (e.g., Random Forests)
- Implements a contrastive learning pipeline with triplet loss
- Evaluates models using standard classification metrics (Accuracy, F1, AUC)
- Visualizes learned embeddings using t-SNE and PCA

> ✅ **Simply run the notebook top to bottom. All experiments and visualizations are included.**

---

## 📂 Files Used

- `data.jsonl`: Raw dataset with labeled code snippets  
  ➤ [Download from GitHub](https://github.com/marcoedingen/chatgpt-code-detection)

---

## 📥 Data Preparation

After downloading `data.jsonl`, place it in the notebook directory and run:

```python
import json
import pandas as pd

with open('data.jsonl', 'r') as file:
    lines = file.readlines()

new_data = pd.DataFrame([json.loads(line) for line in lines])
display(new_data)
```

This will load the dataset into a DataFrame for further processing.

---

## 🛠️ Libraries Used

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import json
```

---

## ⚙️ How to Run

1. Download `data.jsonl` from the official [GitHub dataset repo](https://github.com/marcoedingen/chatgpt-code-detection).
2. Place it in your working directory.
3. Open `project_Cs6890_PLP.ipynb` and run all cells.
4. The notebook will:
   - Train and validate models using stratified 2-fold cross-validation
   - Display classification metrics (Accuracy, Precision, Recall, F1-score, AUC)
   - Visualize embeddings using t-SNE and PCA

---

## 📊 Evaluation

Models are evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **AUC** (Area Under the ROC Curve)

Contrastive learning models use a combination of triplet loss and binary cross-entropy to improve class separation.

---

## ✅ Final Output

Running the notebook produces:
- Cleaned and vectorized code features
- Evaluation metrics per fold
- Mean ROC curve across folds
- 2D embedding visualizations using t-SNE or PCA
- A modular pipeline for AI-code detection

---

## 🧠 Author & Notes

Developed for **CS6890 - Programming Language Principles (PLP)**  
Focus: Fair detection of AI code generation in student programming submissions
