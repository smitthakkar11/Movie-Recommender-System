<h1 align="center">🎬 WatchWise — Movie Recommender System</h1>

<p align="center">
  <b>A content-based movie recommendation engine that suggests 5 similar films for any title you pick — with posters, in a clean Streamlit UI.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/TMDB-API-01B4E4?style=flat-square&logo=themoviedatabase&logoColor=white" alt="TMDB"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"/>
</p>

---

## 🔗 Live Demo

**[▶ watchwise-smit.streamlit.app](https://watchwise-smit.streamlit.app)**

Pick a film, hit **Recommend**, get five neighbours with posters. No setup, no API key needed to browse.

---

## 📖 Overview

Ever finished a film and thought *"give me five more like that"*? That's the whole idea here.

**WatchWise** is a **content-based recommender**: it doesn't need your watch history, your ratings, or any other users at all. It looks purely at *what a movie is made of* — its plot, genres, keywords, cast and director — turns that into a numerical fingerprint, and finds the five films whose fingerprints sit closest to the one you chose.

Pick a movie from the dropdown, hit **Recommend**, and five posters appear side by side, pulled live from the TMDB API.

---

## 🧠 How It Works

The whole system reduces to one idea: **turn every movie into a vector, then measure the angle between vectors.**

```
TMDB 5000 dataset
        │
        ├─ merge movies + credits on title
        │
        ▼
  Feature extraction
  overview · genres · keywords · top-3 cast · director
        │
        ▼
   Combined "tags" string per movie
        │
        ├─ lowercase, remove spaces in names ("Sam Worthington" → "SamWorthington")
        ├─ stemming (PorterStemmer: "loving"/"loved" → "love")
        │
        ▼
  CountVectorizer (max_features=5000, stop_words='english')
        │
        ▼
  4806-movie × 5000-feature sparse matrix
        │
        ├─ L2-normalise each row (so a dot product IS cosine similarity)
        │
        ▼
   Saved to model/vectors.npz (~300 KB)
        │
        ▼
  At request time: one vector × matrix → similarity for that film only
        │
        ▼
  argpartition → top 5 (excluding itself)
```

**Why cosine similarity and not Euclidean distance?**
In high-dimensional text space, raw distance is dominated by document length — a movie with a long overview looks "far" from everything. Cosine similarity measures the *angle* between vectors instead of their magnitude, so it captures "these two films are about the same things" regardless of how much text each one had. That's the right question to ask here.

**Why remove spaces inside names?**
`Sam Worthington` would tokenize into `sam` and `worthington` — and every other Sam in the dataset would suddenly look similar. Collapsing it to `SamWorthington` keeps each person a single, unambiguous token.

**Why stemming?**
`action`, `actions`, and `acting` are three separate columns to a vectorizer but one concept to a human. Stemming collapses them so the 5000-feature budget isn't wasted on inflections.

**Why store vectors instead of the similarity matrix?**
The full 4806×4806 matrix is ~180 MB — too big for git, and 23 million pairwise scores to answer a question that needs exactly one row of them. Storing the normalised vectors instead costs ~300 KB, and the one row you actually want is a single sparse matrix-vector product at request time. Same results, 600× smaller, and the repo deploys anywhere.

---

## ⚙️ Tech Stack

| Layer | Tool | Role |
|---|---|---|
| Data | **TMDB 5000 Movie Dataset** | ~5000 films with metadata, cast and crew |
| Processing | **Pandas · NumPy** | Merging, JSON column parsing, feature engineering |
| NLP | **NLTK (PorterStemmer)** | Token normalization |
| Vectorization | **scikit-learn — CountVectorizer** | Bag-of-words → numeric vectors |
| Similarity | **SciPy sparse dot product** | Cosine similarity, computed per request |
| Posters | **TMDB REST API** | Live poster artwork by movie ID |
| Frontend | **Streamlit** | Dropdown, button, 5-column poster grid |
| Persistence | **SciPy `.npz` + CSV** | ~400 KB of committed model artifacts |

---

## 📁 Project Structure

```
Movie-Recommender-System/
├── app.py                  # Streamlit app — UI, recommend(), poster fetching
├── build_model.py          # Rebuilds model/ from the raw TMDB CSVs
├── movie.ipynb.zip         # Original notebook — EDA, preprocessing, model build
├── model/
│   ├── movies.csv          # movie_id + title, 4806 rows      (committed, 104 KB)
│   └── vectors.npz         # L2-normalised count vectors      (committed, 296 KB)
├── requirements.txt        # App dependencies
├── requirements-build.txt  # Extra deps for build_model.py only
├── .streamlit/
│   ├── config.toml         # Dark theme
│   └── secrets.toml        # TMDB API key (git-ignored, never commit)
└── README.md
```

> **The model artifacts are committed.** Both files together are ~400 KB, so cloning the repo gives you a runnable app with no dataset download and no notebook run. Only rebuild them if you change the feature pipeline.

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/smitthakkar11/Movie-Recommender-System.git
cd Movie-Recommender-System
```

### 2. Set up the environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run it

```bash
streamlit run app.py
```

Open `http://localhost:8501`. That's it — the model artifacts are in the repo, so there is nothing to download or train.

### 4. Add a TMDB API key (optional, for posters)

The app recommends films with or without a key. A key only adds poster artwork.

Get a free one at [themoviedb.org](https://www.themoviedb.org/settings/api) (v3 auth), then create `.streamlit/secrets.toml`:

```toml
TMDB_API_KEY = "your_key_here"
```

`app.py` reads `st.secrets` first, then the `TMDB_API_KEY` environment variable — the key is never hardcoded. `.streamlit/secrets.toml` is git-ignored.

> **🔐 Heads up:** earlier commits had an API key written directly into `fetch_poster()`. It is still in this repo's git history, so treat it as compromised — revoke it on TMDB and issue a fresh one.

### 5. Rebuilding the model (only if you change the pipeline)

```bash
pip install -r requirements-build.txt
# place tmdb_5000_movies.csv + tmdb_5000_credits.csv in the project root
python build_model.py
```

Grab the CSVs from the [TMDB 5000 dataset on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).

---

## ☁️ Deploying

**Streamlit Community Cloud** is free and takes about two minutes:

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **Create app** → **Deploy a public app from GitHub**.
3. Repository `smitthakkar11/Movie-Recommender-System`, branch `main`, main file `app.py`.
4. Under **Advanced settings → Secrets**, paste:
   ```toml
   TMDB_API_KEY = "your_key_here"
   ```
   Skip this and the app still works — posters just render as placeholders.
5. **Deploy.**

Nothing else is needed: the model artifacts are committed and total ~400 KB, so the container has everything on first boot. There is no dataset to download and no matrix to rebuild — which is exactly why the vectors are stored instead of the 180 MB similarity matrix.

---

## 🎯 Limitations & Roadmap

**Current limitations, stated honestly:**
- **Content-based only** — it has no idea what *you* like, only what a movie *is*. Two films can share genre and cast and still appeal to completely different audiences.
- **Fixed catalogue** — ~5000 films, nothing released after the dataset snapshot.
- **Bag-of-words ignores order and meaning** — "not a horror film" and "a horror film" look nearly identical to a count vectorizer.
- **Poster fetches depend on TMDB being reachable** — they run concurrently behind a pooled, retrying session and fall back to a placeholder, but a hard outage means no artwork.

**Planned improvements:**
- [ ] Swap `CountVectorizer` for **TF-IDF** so common tags stop dominating
- [ ] Try **sentence embeddings** (`sentence-transformers`) on overviews for semantic rather than lexical similarity
- [ ] Add a **hybrid layer** — blend content similarity with popularity/rating so recommendations aren't obscure by accident
- [ ] Show *why* a film was recommended (shared genres, shared cast) — recommenders are far more convincing when they explain themselves
- [ ] Rank exact ties by popularity rather than dataset order

---

## 🙏 Acknowledgements

- **[TMDB](https://www.themoviedb.org/)** — for the dataset and the poster API. This product uses the TMDB API but is not endorsed or certified by TMDB.
- The [TMDB 5000 dataset on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).

---

## 👤 Author

**Smit Thakkar**
Motilal Nehru National Institute of Technology, Allahabad

[![GitHub](https://img.shields.io/badge/GitHub-smitthakkar11-181717?style=flat-square&logo=github)](https://github.com/smitthakkar11)

---

<p align="center">
  <i>If this helped you, a ⭐ on the repo goes a long way.</i>
</p>
