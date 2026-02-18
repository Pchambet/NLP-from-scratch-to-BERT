# 🧠 L'Odyssée du NLP : Du Chaos vers le Sens

> *"Le langage est la source des malentendus."* — Antoine de Saint-Exupéry.
>
> Mais pour une machine, le langage n'est même pas un malentendu. C'est juste du bruit. Une suite inintelligible d'octets. Ce projet raconte l'histoire de comment nous avons appris aux machines à voir à travers ce bruit, à découvrir des structures, et finalement, à comprendre le sens.

---

## 🗺️ La Carte du Voyage

Ce dépôt n'est pas juste une collection de scripts. C'est une progression logique, une ascension en quatre étapes vers l'intelligence artificielle moderner.

### 🌑 Chapitre 1 : L'Atome (TP1 — Prétraitement)
Avant de comprendre une phrase, il faut isoler ses composants. C'est l'étape de la **Tokenisation**.
Ici, nous faisons exploser le texte. Nous nettoyons le bruit (ponctuation, majuscules), nous jetons ce qui est inutile (Stopwords), et nous cherchons la racine de chaque mot (Stemming & Lemmatisation).
👉 *Objectif : Transformer un flux de caractères informe en une séquence d'unités logiques.*

### 📊 Chapitre 2 : La Matrice (TP2 — BOW & TF-IDF)
Maintenant que nous avons des mots, comment les faire comprendre à un ordinateur qui ne parle que des mathématiques ? Nous les comptons.
Avec le **Bag of Words**, nous transformons chaque texte en un vecteur immense. Avec le **TF-IDF**, nous donnons du poids à la rareté : un mot commun comme "le" s'efface, tandis qu'un mot unique comme "trous noir" brille de mille feux.
👉 *Objectif : Transformer la littérature en statistique pour classifier des emails (Spam vs Ham).*

### 🌌 Chapitre 3 : La Géométrie (TP3 — Word2Vec & FastText)
Les statistiques ne suffisent pas. Dans le Bag of Words, "Roi" et "Reine" sont aussi différents que "Roi" et "Chaise". Ils sont juste des colonnes différentes.
Ici, nous entrons dans l'ère des **Embeddings**. Nous projetons les mots dans un espace vectoriel dense. Dans cet espace, la distance a un sens. La magie opère : `Vecteur(Roi) - Vecteur(Homme) + Vecteur(Femme) ≈ Vecteur(Reine)`.
👉 *Objectif : Capturer la sémantique et les analogies grâce à la géométrie spatiale.*

### 🧠 Chapitre 4 : L'Esprit (TP4 — BERT)
Nous avons l'atome, la statistique et la géométrie. Mais il manquait le **contexte**.
Jusqu'à maintenant, le mot "banque" avait le même vecteur qu'il s'agisse d'une "banque finance" ou d'un "banc de poissons". Avec **BERT** (Bidirectional Encoder Representations from Transformers), le modèle lit toute la phrase d'un coup. Il comprend les nuances. Il a "lu" tout Wikipédia. Il sait.
👉 *Objectif : Utiliser le Transfer Learning pour atteindre des sommets de performance avec peu de données.*

---

## 🛠️ Le Laboratoire (Installation)

Pour reproduire ces expériences, vous avez besoin de votre propre laboratoire.

### 1. Préparation
Assurez-vous d'avoir Python installé. Clonez ce dépôt, puis installez les dépendances :

```bash
# La méthode moderne (avec uv)
uv run jupyter lab

# Ou la méthode classique
pip install pandas numpy nltk scikit-learn gensim matplotlib seaborn transformers datasets torch accelerate
jupyter lab
```

### 2. Vérification
Nous avons inclus un script pour valider que votre environnement est prêt :

```bash
python verify_env.py
```

### 3. Exploration
Ouvrez le dossier `Solutions/`. Les notebooks sont numérotés pour suivre l'histoire dans l'ordre.
Chaque notebook est **autonome** : les résultats sont déjà calculés et visibles, mais vous pouvez tout ré-exécuter.

---

## 📂 Organisation du Dépôt

```
.
├── Solutions/               # Le cœur du projet (Les 4 Chapitres)
│   ├── TP1_Pretraitement.ipynb
│   ├── TP2_BOW_TFIDF.ipynb
│   ├── TP3_Word2Vec_FastText.ipynb
│   └── TP4_BERT.ipynb
│
├── data/                    # La matière première
│   ├── alice_wonderland.txt
│   ├── spam.csv
│   └── (autres datasets...)
│
└── verify_env.py            # Le filet de sécurité
```

---

> *"Toute technologie suffisamment avancée est indiscernable de la magie."* — Arthur C. Clarke.
>
> Bienvenue dans la magie du NLP.
