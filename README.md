# 🧠 NLP — Du Texte Brut à BERT

> Quatre notebooks Jupyter qui explorent le **Natural Language Processing** de A à Z :
> du découpage de phrases simples jusqu'au fine-tuning d'un Transformer pré-entraîné.

---

## �️ Vue d'Ensemble

```
TP1  Prétraitement       →  Comment un ordinateur "lit" du texte
TP2  BOW & TF-IDF        →  Comment transformer du texte en chiffres
TP3  Word2Vec & FastText  →  Comment donner du "sens" aux mots
TP4  BERT                 →  Comment utiliser un modèle de deep learning pré-entraîné
```

Chaque notebook est **autonome** : il contient le code, les explications, et les résultats d'exécution.

---

## 📂 Structure

```
├── Solutions/
│   ├── TP1_Pretraitement.ipynb       (38 cellules)
│   ├── TP2_BOW_TFIDF.ipynb           (36 cellules)
│   ├── TP3_Word2Vec_FastText.ipynb    (59 cellules)
│   └── TP4_BERT.ipynb                (18 cellules)
│
├── data/
│   ├── alice_wonderland.txt          Texte complet d'Alice au pays des merveilles
│   ├── spam.csv                      5 572 emails (spam / ham)
│   ├── train_tweets.csv              Tweets pour sentiment analysis
│   ├── test_tweets.csv               Tweets (test)
│   └── Comment Spam.xls              Commentaires spam YouTube
│
├── pyproject.toml                    Dépendances Python
└── README.md
```

---

## 🚀 Installation

```bash
# Avec uv (recommandé)
uv run jupyter lab

# Ou classiquement
pip install pandas numpy nltk scikit-learn gensim matplotlib seaborn transformers datasets torch accelerate
jupyter lab
```

Puis ouvrir les notebooks dans `Solutions/` dans l'ordre TP1 → TP4.

---

## 📘 TP1 — Prétraitement de Texte

**Objectif** : Apprendre à préparer du texte brut avant toute analyse NLP.

### Ce qu'on y fait

- **Tokenisation** : Découper un texte en phrases, puis en mots.
  On compare 3 tokenizers (TreebankWord, WordPunct, Regex) et on observe comment chacun gère
  l'apostrophe, la ponctuation et les contractions (`can't`, `Alice's`).

- **Stemming vs Lemmatisation** : Deux méthodes pour ramener un mot à sa racine.
  Le stemmer coupe mécaniquement (`running` → `run`, mais aussi `universities` → `univers`).
  Le lemmatiseur utilise un dictionnaire et comprend la grammaire (`better` → `good` si on lui dit que c'est un adjectif).

- **Stopwords** : Les mots vides (`the`, `is`, `a`) dominent les fréquences.
  On montre avec des graphiques que les supprimer révèle les vrais mots-clés d'un texte.

- **N-grams** : Au-delà du mot unique — les bigrammes (`New York`), trigrammes (`not very good`)
  capturent le contexte. On observe qu'au-delà de 4-grams, les séquences sont souvent uniques.

### Concepts clés

> Un token n'est pas forcément un mot. C'est l'unité minimale que le modèle voit.
> Le choix du tokenizer change complètement ce que le modèle comprend.

---

## 📘 TP2 — Bag of Words & TF-IDF

**Objectif** : Transformer du texte en vecteurs numériques pour pouvoir faire de la classification.

### Ce qu'on y fait

- **Bag of Words (BOW)** : Chaque document = un vecteur de fréquences de mots.
  Simple mais efficace. On explore l'impact de `max_features` (limiter le vocabulaire aux N mots
  les plus fréquents) sur la précision d'un classifieur.

- **Classification spam/ham** : On entraîne Naive Bayes et SVM sur 5 500 emails.
  SVM est généralement meilleur car il gère mieux les espaces de haute dimension.

- **TF-IDF** : Pondère les mots par leur rareté dans le corpus.
  Le mot `free` dans un spam a un score TF-IDF élevé car il est fréquent dans ce document
  mais rare dans le corpus global. On construit un pipeline complet :
  `CountVectorizer → TfidfTransformer → LogisticRegression`.

- **Calcul manuel de TF-IDF** : On recalcule tout étape par étape (Document Frequency, IDF,
  normalisation L2) pour comprendre ce que sklearn fait en coulisses.
  On vérifie que notre résultat est identique à `TfidfVectorizer`.

- **Similarité documentaire** : Heatmap de similarité cosinus entre 8 documents thématiques.
  Les documents du même thème (weather, animals, food) se ressemblent davantage.

- **Clustering hiérarchique** : Dendrogramme qui regroupe automatiquement les documents
  (et les mots) par proximité sémantique, sans supervision.

### Concepts clés

> TF-IDF = "ce mot est-il important **pour ce document** par rapport à l'ensemble ?"
> Un mot présent partout (comme `the`) a un IDF proche de 0.
> Un mot rare et spécifique a un IDF élevé.

---

## 📘 TP3 — Word Embeddings & Sentiment Analysis

**Objectif** : Passer des comptages de mots à des **représentations sémantiques** :
un mot = un vecteur dense qui capture son sens.

### Ce qu'on y fait

#### Word2Vec
- **CBOW** (Continuous Bag of Words) : Prédire un mot à partir de son contexte.
- **Skip-gram** : Prédire le contexte à partir d'un mot.
- On entraîne les deux sur *Alice au Pays des Merveilles* et on compare les résultats
  avec différentes dimensions de vecteurs (`vector_size = 2, 10, 500`).
- On charge aussi le modèle pré-entraîné **Google News** (3 millions de mots, 300 dimensions)
  pour tester des analogies (`king - man + woman ≈ queen`).

#### FastText
- Fonctionne comme Word2Vec mais au niveau **sous-mot** (n-grams de caractères).
  Avantage : il peut traiter des mots jamais vus (`unfriendliest` → `un` + `friend` + `li` + ...).

#### Doc2Vec
- Représenter un **document entier** par un seul vecteur (moyenne des vecteurs de ses mots).
- On utilise ces vecteurs pour clusteriser des documents par thème et on mesure la qualité
  avec l'Adjusted Rand Index.

#### Visualisation
- **PCA** : Projection linéaire rapide, préserve les grandes distances.
- **t-SNE** : Projection non-linéaire, révèle les clusters locaux.
  Le paramètre `perplexity` contrôle l'équilibre local/global.

#### Sentiment Analysis (pipeline complet)
On compare systématiquement **16 combinaisons** :

|  | BoW | TF-IDF | Word2Vec | Doc2Vec |
|---|:---:|:---:|:---:|:---:|
| **Logistic Regression** | ✅ | ✅ | ✅ | ✅ |
| **SVM** | ✅ | ✅ | ✅ | ✅ |
| **Random Forest** | ✅ | ✅ | ✅ | ✅ |
| **XGBoost/GB** | ✅ | ✅ | ✅ | ✅ |

Résultats visualisés via des heatmaps d'Accuracy et F1-Score.

### Concepts clés

> Word2Vec apprend que `roi` et `reine` sont proches car ils apparaissent
> dans des contextes similaires. Il ne sait pas ce que ces mots *signifient*,
> mais il capture leurs relations d'usage.

---

## 📘 TP4 — BERT (Transformers)

**Objectif** : Utiliser un modèle de deep learning pré-entraîné pour classifier des sentiments,
en exploitant le **transfer learning**.

### Ce qu'on y fait

- **Tokenisation WordPiece** : BERT découpe les mots rares en sous-unités
  (`unbelievable` → `un`, `##bel`, `##iev`, `##able`). Il ajoute aussi des tokens spéciaux
  `[CLS]` (début) et `[SEP]` (fin).

- **Transfer Learning** : BERT a été pré-entraîné sur Wikipedia + BookCorpus (3.3 milliards de mots).
  Il a déjà "lu" tellement de texte qu'il comprend la grammaire, le contexte et les nuances.
  On ne fait que le **fine-tuner** (ajuster la dernière couche) sur notre tâche spécifique.

- **Fine-tuning** : On prend `bert-base-uncased` (110M paramètres) et on l'adapte
  à la classification de sentiments sur le dataset IMDB (critiques de films).
  Même avec 500 exemples et 1 seule époque, il obtient des résultats raisonnables.

- **Inférence** : On teste le modèle sur 4 phrases personnalisées et on observe
  les prédictions avec leur score de confiance.

### Concepts clés

> BERT est **bidirectionnel** : pour comprendre le mot `bank`, il regarde à la fois
> les mots avant ET après. C'est ce qui le différencie des modèles précédents
> qui lisaient le texte de gauche à droite uniquement.
>
> Le transfer learning est la raison pour laquelle BERT fonctionne avec peu de données :
> il part avec 110M de paramètres déjà entraînés, pas de zéro.

---

## 📊 Progression des Concepts

```
Texte brut
    │
    ▼
[TP1] Tokenisation, Nettoyage
    │   "Le chat mange" → ["chat", "mange"]
    ▼
[TP2] Vectorisation (BOW / TF-IDF)
    │   ["chat", "mange"] → [0, 1, 0, 1, 0, ...]  (vecteur creux)
    ▼
[TP3] Embeddings (Word2Vec / FastText)
    │   "chat" → [0.23, -0.41, 0.87, ...]  (vecteur dense, sémantique)
    ▼
[TP4] Transformers (BERT)
        "chat" → [contextualisé selon la phrase entière]
```

---

## 🛠️ Stack Technique

| Lib | Usage |
|---|---|
| `nltk` | Tokenisation, stemming, lemmatisation, stopwords |
| `scikit-learn` | Vectorisation (BOW, TF-IDF), classification, PCA, t-SNE |
| `gensim` | Word2Vec, FastText, Doc2Vec |
| `transformers` | BERT (Hugging Face) |
| `torch` | Backend deep learning |
| `matplotlib` / `seaborn` | Visualisations |
| `pandas` / `numpy` | Manipulation de données |
