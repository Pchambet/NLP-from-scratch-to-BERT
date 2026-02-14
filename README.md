# 🧠 NLP — Travaux Pratiques

> Série de 4 TPs couvrant les fondamentaux du **Natural Language Processing**, du prétraitement de texte brut jusqu'au fine-tuning de BERT.

---

## 📂 Structure du Projet

```
TP-Final/
├── Solutions/                          ← Les 4 notebooks (à exécuter)
│   ├── TP1_Pretraitement.ipynb
│   ├── TP2_BOW_TFIDF.ipynb
│   ├── TP3_Word2Vec_FastText.ipynb
│   └── TP4_BERT.ipynb
│
├── data/                               ← Datasets utilisés
│   ├── alice_wonderland.txt            (texte pour tokenisation/Word2Vec)
│   ├── spam.csv                        (emails spam/ham pour TP2)
│   ├── Comment Spam.xls               (variante spam)
│   ├── train_tweets.csv               (tweets pour analyse de sentiments)
│   └── test_tweets.csv
│
├── Chapter 1 - Pretreatment/           ← Sujets PDF (référence)
├── Chapter 2 - frequency/
├── Chapter 3 - Prediction/
├── Chapter 4 - DNN/
│
├── pyproject.toml                      ← Dépendances (géré par uv)
└── README.md
```

---

## 🚀 Installation & Exécution

### Prérequis

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** (gestionnaire de packages, recommandé)

### Lancer les notebooks

```bash
# 1. Cloner le projet
cd TP-Final

# 2. Lancer Jupyter (uv installe tout automatiquement)
uv run jupyter lab
```

Ouvrir les notebooks dans `Solutions/` et les exécuter dans l'ordre (TP1 → TP4).

### Alternative sans uv

```bash
pip install pandas numpy nltk scikit-learn gensim seaborn matplotlib transformers datasets torch accelerate jupyterlab
jupyter lab
```

---

## 📘 Contenu des TPs

### TP1 — Prétraitement de Texte

| Thème | Ce qu'on fait |
|---|---|
| **Tokenisation** | Découpage en phrases et en mots (NLTK) |
| **Comparaison de Tokenizers** | TreebankWordTokenizer vs WordPunctTokenizer — quelles différences ? |
| **RegexTokenizer** | Définir ses propres règles, gestion de l'apostrophe |
| **Stemming vs Lemmatisation** | PorterStemmer vs WordNetLemmatizer, impact du POS tag, erreurs de frappe |
| **Stopwords** | Filtrage FR/EN, impact sur la fréquence des mots |
| **N-grams** | Génération (1 à 6-grams), analyse de rareté |
| **Analyse fréquentielle** | Distribution des mots avec et sans stopwords (graphiques) |

**Corpus** : *Alice in Wonderland*, textes en français, corpus Gutenberg

---

### TP2 — Bag of Words & TF-IDF

| Thème | Ce qu'on fait |
|---|---|
| **BOW** | Vectorisation avec CountVectorizer, rôle de `max_features` |
| **Classification** | MultinomialNB sur spam/ham, comparaison avec SVM |
| **TF-IDF Pipeline** | CountVectorizer → TfidfTransformer → LogisticRegression |
| **Validation croisée** | Cross-validation + test sur phrases exemples |
| **TF-IDF Manuel** | Calcul étape par étape : DF → IDF → TF-IDF brut → normalisation L2 |
| **Similarité documentaire** | Heatmap de similarité cosinus entre documents |
| **Clustering** | Dendrogramme hiérarchique (documents et mots) |

**Corpus** : `spam.csv` (5 572 emails) + corpus thématique (weather/animals/food)

---

### TP3 — Word2Vec, FastText & Sentiment Analysis

| Thème | Ce qu'on fait |
|---|---|
| **Word2Vec** | Entraînement CBOW et Skip-gram sur *Alice in Wonderland* |
| **Impact de `vector_size`** | Comparaison avec vector_size = 2, 10, 500 |
| **Modèle pré-entraîné** | Google News 300 (3M mots), analogies et similarités |
| **FastText** | Entraînement sur corpus Brown, gestion des mots inconnus (OOV) |
| **Visualisation** | PCA et t-SNE (rôle de la perplexity) |
| **Doc2Vec** | Vecteur moyen d'un document, clustering K-Means |
| **Clustering documents** | Dendrogramme hiérarchique, Adjusted Rand Index |
| **Sentiment Analysis** | Pipeline complet : nettoyage → 4 embeddings × 4 modèles (voir ci-dessous) |

#### Pipeline Sentiment Analysis (16 combinaisons)

|  | BoW | TF-IDF | Word2Vec | Doc2Vec |
|---|:---:|:---:|:---:|:---:|
| **Logistic Regression** | ✅ | ✅ | ✅ | ✅ |
| **SVM** | ✅ | ✅ | ✅ | ✅ |
| **Random Forest** | ✅ | ✅ | ✅ | ✅ |
| **XGBoost** | ✅ | ✅ | ✅ | ✅ |

Résultats comparés via heatmaps (Accuracy + F1-Score).

---

### TP4 — BERT (Transformers)

| Thème | Ce qu'on fait |
|---|---|
| **Tokenisation WordPiece** | Découpage en sous-mots, tokens spéciaux [CLS]/[SEP] |
| **Fine-tuning** | bert-base-uncased sur IMDB (analyse de sentiment) |
| **Transfer Learning** | Pourquoi BERT marche avec peu de données |
| **Évaluation** | Accuracy sur le test set |
| **Inférence** | Test sur 4 phrases personnalisées avec score de confiance |

**Corpus** : IMDB (25K reviews, sous-échantillonné à 500 pour la démo CPU)

---

## ⚠️ Notes Importantes

- **TP3** : Le téléchargement du modèle Google News (~1.5 GB) peut prendre du temps à la première exécution.
- **TP4** : L'entraînement BERT sur CPU prend ~5 min avec 500 exemples / 1 époque. Pour de meilleurs résultats, augmenter `num_samples` et `num_train_epochs`.
- **Ordre d'exécution** : Les notebooks sont indépendants, mais il est recommandé de les suivre dans l'ordre (TP1 → TP4) pour la progression pédagogique.

---

## 📚 Ressources

Les sujets originaux (PDF) sont dans les dossiers `Chapter 1` à `Chapter 4`. Chaque notebook couvre **tous les PDFs** de son chapitre :

| Notebook | PDFs couverts |
|---|---|
| TP1 | `TP_NLP_1_pretraitrement.pdf` + `TP_NLP_1_pretraitrement 2.pdf` |
| TP2 | `TP_NLP_2_BOW.pdf` + `TP_NLP_2_TFIDF.pdf` + `TP_NLP_2_other_example.pdf` |
| TP3 | `TP_NLP_3_word2vec.pdf` + `TP_NLP_3_Sentiment_analysis.pdf` + `TP_NLP_3_w2v_FastText.pdf` |
| TP4 | `TP_NLP_4_BERT_sentiment_analysis.pdf` |

---

## 🛠️ Stack Technique

| Lib | Usage |
|---|---|
| `nltk` | Tokenisation, stemming, lemmatisation, stopwords, N-grams |
| `scikit-learn` | CountVectorizer, TF-IDF, classifieurs (NB, SVM, RF, LR), PCA, t-SNE |
| `gensim` | Word2Vec, FastText, Doc2Vec, modèles pré-entraînés |
| `transformers` | BERT (tokenizer + modèle), Trainer API |
| `datasets` | Chargement IMDB |
| `torch` | Backend pour BERT |
| `matplotlib` / `seaborn` | Visualisations |
| `pandas` / `numpy` | Manipulation de données |
