# 🧠 The NLP Odyssey: From Chaos to Understanding

> *"Language is the source of misunderstandings."* — Antoine de Saint-Exupéry.
>
> Yet, for a machine, language is not even a misunderstanding. It is merely noise. An unintelligible stream of bytes. This project chronicles the story of how we taught machines to see through this noise, to discern structure, and ultimately, to grasp meaning.

---

## 🗺️ The Map of the Journey

This repository is not merely a collection of scripts. It is a logical progression, an ascent in four stages towards modern artificial intelligence.

### 🌑 Chapter 1: The Atom (TP1 — Preprocessing)
Before one can comprehend a sentence, one must isolate its components. This is the stage of **Tokenization**.
Here, we shatter the text. We cleanse the noise (punctuation, capitalization), we discard the unnecessary (Stopwords), and we seek the very root of each word (Stemming & Lemmatization).
👉 *Goal: To transform a shapeless stream of characters into a sequence of logical units.*

### 📊 Chapter 2: The Matrix (TP2 — BOW & TF-IDF)
Now that we possess the words, how do we make them understood by a computer that speaks only in mathematics? We count them.
With the **Bag of Words**, we transform each text into an immense vector. With **TF-IDF**, we grant weight to rarity: a common word like "the" fades into the background, while a unique term like "black hole" shines with significance.
👉 *Goal: To transmute literature into statistics for the classification of emails (Spam vs. Ham).*

### 🌌 Chapter 3: The Geometry (TP3 — Word2Vec & FastText)
Statistics alone are insufficient. In the Bag of Words, "King" and "Queen" are as distinct as "King" and "Chair". They are merely different columns.
Here, we enter the era of **Embeddings**. We project words into a dense vector space. In this space, distance holds meaning. The magic unfolds: `Vector(King) - Vector(Man) + Vector(Woman) ≈ Vector(Queen)`.
👉 *Goal: To capture semantics and analogies through the elegance of spatial geometry.*

### 🧠 Chapter 4: The Mind (TP4 — BERT)
We had the atom, the statistics, and the geometry. But we lacked the **context**.
Until now, the word "bank" held the same vector whether it referred to a "financial bank" or a "river bank". With **BERT** (Bidirectional Encoder Representations from Transformers), the model reads the entire sentence at once. It perceives the nuances. It has "read" all of Wikipedia. It knows.
👉 *Goal: To harness Transfer Learning to reach the pinnacles of performance with minimal data.*

---

## 🛠️ The Laboratory (Installation)

To reproduce these experiments, you shall require your own laboratory.

### 1. Preparation
Ensure you have Python installed. Clone this repository, then install the dependencies:

```bash
# The modern method (with uv)
uv run jupyter lab

# Or the classic method
pip install pandas numpy nltk scikit-learn gensim matplotlib seaborn transformers datasets torch accelerate
jupyter lab
```

### 2. Verification
We have included a script to validate that your environment is prepared:

```bash
python verify_env.py
```

### 3. Exploration
Open the `Solutions/` directory. The notebooks are numbered to follow the tale in its proper order.
Each notebook is **self-contained**: the results are already calculated and visible, yet you are free to re-execute everything.

---

## 📂 Repository Structure

```
.
├── Solutions/               # The Heart of the Project (The 4 Chapters)
│   ├── TP1_Pretraitement.ipynb
│   ├── TP2_BOW_TFIDF.ipynb
│   ├── TP3_Word2Vec_FastText.ipynb
│   └── TP4_BERT.ipynb
│
├── data/                    # The Raw Material
│   ├── alice_wonderland.txt
│   ├── spam.csv
│   └── (other datasets...)
│
└── verify_env.py            # The Safety Net
```

---

> *"Any sufficiently advanced technology is indistinguishable from magic."* — Arthur C. Clarke.
>
> Welcome to the magic of NLP.
