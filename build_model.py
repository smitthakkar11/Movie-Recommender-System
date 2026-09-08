"""Rebuild the model artifacts from the raw TMDB 5000 dataset.

This is the notebook pipeline (movie.ipynb) as a runnable script. It writes:

    model/movies.csv   movie_id + title, one row per film
    model/vectors.npz  L2-normalised sparse count vectors

The app computes cosine similarity from those vectors on demand. We deliberately
do NOT persist the full 4806x4806 similarity matrix: it is ~180 MB, too large for
git, and only one row of it is ever needed per recommendation.

Usage:
    pip install nltk
    # place tmdb_5000_movies.csv and tmdb_5000_credits.csv in the project root
    python build_model.py
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "model"
ps = PorterStemmer()


def names(obj):
    return [i["name"] for i in ast.literal_eval(obj)]


def top3_names(obj):
    return [i["name"] for i in ast.literal_eval(obj)[:3]]


def director(obj):
    return [i["name"] for i in ast.literal_eval(obj) if i["job"] == "Director"][:1]


def stem(text):
    return " ".join(ps.stem(word) for word in text.split())


def main():
    movies = pd.read_csv(BASE_DIR / "tmdb_5000_movies.csv")
    credits = pd.read_csv(BASE_DIR / "tmdb_5000_credits.csv")
    movies = movies.merge(credits, on="title")

    movies = movies[
        ["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]
    ].dropna()

    movies["genres"] = movies["genres"].apply(names)
    movies["keywords"] = movies["keywords"].apply(names)
    movies["cast"] = movies["cast"].apply(top3_names)
    movies["crew"] = movies["crew"].apply(director)
    movies["overview"] = movies["overview"].apply(lambda x: x.split())

    # "Sam Worthington" -> "SamWorthington", so each person stays one token and
    # every other Sam in the dataset doesn't suddenly look similar.
    for col in ("genres", "keywords", "cast", "crew"):
        movies[col] = movies[col].apply(lambda xs: [x.replace(" ", "") for x in xs])

    tags = (
        movies["overview"]
        + movies["genres"]
        + movies["keywords"]
        + movies["cast"]
        + movies["crew"]
    )
    tags = tags.apply(lambda x: " ".join(x).lower()).apply(stem)

    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = normalize(cv.fit_transform(tags)).astype(np.float32)

    OUT_DIR.mkdir(exist_ok=True)
    movies[["movie_id", "title"]].to_csv(OUT_DIR / "movies.csv", index=False)
    sp.save_npz(OUT_DIR / "vectors.npz", sp.csr_matrix(vectors))

    print(f"wrote {len(movies)} films to {OUT_DIR}/")
    print(f"  vectors: {vectors.shape}, {vectors.nnz} non-zeros")


if __name__ == "__main__":
    main()
