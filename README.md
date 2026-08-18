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

**[▶ Try it here](#)** — *(deploy on Streamlit Community Cloud and drop the link here)*

<p align="center">
  <img src="assets/demo.gif" alt="Demo of the movie recommender" width="800"/>
</p>

> Add a `assets/demo.gif` or `assets/screenshot.png` to the repo so visitors see the app before they click anything.

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
   5000-movie × 5000-feature sparse matrix
        │
        ▼
  cosine_similarity → 5000 × 5000 similarity matrix
        │
        ▼
  Sort row for selected movie → take top 5 (excluding itself)
```

**Why cosine similarity and not Euclidean distance?**
In high-dimensional text space, raw distance is dominated by document length — a movie with a long overview looks "far" from everything. Cosine similarity measures the *angle* between vectors instead of their magnitude, so it captures "these two films are about the same things" regardless of how much text each one had. That's the right question to ask here.

**Why remove spaces inside names?**
`Sam Worthington` would tokenize into `sam` and `worthington` — and every other Sam in the dataset would suddenly look similar. Collapsing it to `SamWorthington` keeps each person a single, unambiguous token.

**Why stemming?**
`action`, `actions`, and `acting` are three separate columns to a vectorizer but one concept to a human. Stemming collapses them so the 5000-feature budget isn't wasted on inflections.

---

## ⚙️ Tech Stack

| Layer | Tool | Role |
|---|---|---|
| Data | **TMDB 5000 Movie Dataset** | ~5000 films with metadata, cast and crew |
| Processing | **Pandas · NumPy** | Merging, JSON column parsing, feature engineering |
| NLP | **NLTK (PorterStemmer)** | Token normalization |
| Vectorization | **scikit-learn — CountVectorizer** | Bag-of-words → numeric vectors |
| Similarity | **scikit-learn — cosine_similarity** | Movie-to-movie distance matrix |
| Posters | **TMDB REST API** | Live poster artwork by movie ID |
| Frontend | **Streamlit** | Dropdown, button, 5-column poster grid |
| Persistence | **Pickle** | Serialized dataframe + similarity matrix |

---

## 📁 Project Structure

```
Movie-Recommender-System/
├── app.py                  # Streamlit app — UI, recommend(), poster fetching
├── movie.ipynb             # Notebook — EDA, preprocessing, model build, pickle export
├── requirements.txt        # Dependencies
├── .streamlit/
│   └── secrets.toml        # TMDB API key (git-ignored, never commit)
├── assets/
│   └── screenshot.png      # Demo image for this README
├── movies.pkl              # Generated — movie titles + IDs        (not in repo)
├── similarity.pkl          # Generated — 5000×5000 matrix, ~180 MB (not in repo)
└── README.md
```

> **⚠️ `movies.pkl` and `similarity.pkl` are not tracked in git.** The similarity matrix is far too large for a normal repo. Run the notebook once to generate both files locally before starting the app — see below.

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

### 3. Get a TMDB API key

Create a free account at [themoviedb.org](https://www.themoviedb.org/settings/api) and request an API key (v3 auth).

Then store it as a Streamlit secret — **never hardcode it in `app.py`**:

```bash
mkdir -p .streamlit
```

`.streamlit/secrets.toml`
```toml
TMDB_API_KEY = "your_key_here"
```

And read it in `app.py`:
```python
api_key = st.secrets["TMDB_API_KEY"]
url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
```

Add `.streamlit/secrets.toml` to your `.gitignore`.

> **🔐 Heads up:** the version of `app.py` currently in this repo has an API key written directly into `fetch_poster()`. Since the repo is public, that key is exposed to anyone who reads the file — revoke it on TMDB, issue a new one, and switch to the secrets approach above before deploying.

### 4. Generate the model artifacts

Download the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) from Kaggle, place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in the project root, then run every cell in `movie.ipynb`. It writes out `movies.pkl` and `similarity.pkl`.

### 5. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

---

## ☁️ Deploying

**Streamlit Community Cloud** is free and takes about two minutes:

1. Push the repo to GitHub (without the `.pkl` files).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo, point it at `app.py`.
3. Add `TMDB_API_KEY` under **App settings → Secrets**.
4. Because `similarity.pkl` can't be committed, either:
   - **Recompute at startup** — wrap the notebook's pipeline in a `@st.cache_resource` function so it builds the matrix once on boot, or
   - **Store the vectors, not the matrix** — pickle the 5000×5000 *count vectors* (a few MB) and compute similarity for just the selected row on demand. Much lighter, and identical results.

The second option is the better engineering answer: you never need all 25 million pairwise scores, only one row of them.

---

## 🎯 Limitations & Roadmap

**Current limitations, stated honestly:**
- **Content-based only** — it has no idea what *you* like, only what a movie *is*. Two films can share genre and cast and still appeal to completely different audiences.
- **Fixed catalogue** — ~5000 films, nothing released after the dataset snapshot.
- **Bag-of-words ignores order and meaning** — "not a horror film" and "a horror film" look nearly identical to a count vectorizer.
- **Cold poster fetches** — five sequential API calls per recommendation makes the UI feel slow.

**Planned improvements:**
- [ ] Swap `CountVectorizer` for **TF-IDF** so common tags stop dominating
- [ ] Try **sentence embeddings** (`sentence-transformers`) on overviews for semantic rather than lexical similarity
- [ ] **Cache poster URLs** with `@st.cache_data` and fetch concurrently
- [ ] Add a **hybrid layer** — blend content similarity with popularity/rating so recommendations aren't obscure by accident
- [ ] Show *why* a film was recommended (shared genres, shared cast) — recommenders are far more convincing when they explain themselves
- [ ] Search-as-you-type instead of a 5000-item dropdown

---

## 🙏 Acknowledgements

- **[TMDB](https://www.themoviedb.org/)** — for the dataset and the poster API. This product uses the TMDB API but is not endorsed or certified by TMDB.
- **[Tutorial walkthrough](https://www.youtube.com/watch?v=1xtrIEwY_zY)** — the video this project follows.
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
