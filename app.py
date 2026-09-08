"""WatchWise — content-based movie recommender.

Streamlit frontend. Loads the count vectors built by build_model.py and computes
cosine similarity for the selected film on demand, then shows the 5 closest
titles with posters from TMDB.

Storing the 4806x5000 vectors (~300 KB) instead of the 4806x4806 similarity
matrix (~180 MB) is what makes this deployable: only one row of the matrix is
ever needed, so there is no reason to precompute all 23 million pairwise scores.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_DIR = Path(__file__).resolve().parent
VECTORS = BASE_DIR / "model" / "vectors.npz"
MOVIES = BASE_DIR / "model" / "movies.csv"

POSTER_BASE = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER = "https://placehold.co/500x750/1a1a2e/eaeaea?text=No+Poster"

st.set_page_config(
    page_title="WatchWise — Movie Recommender",
    page_icon="🎬",
    layout="wide",
)


def get_api_key():
    """TMDB key from Streamlit secrets, then env, then None."""
    try:
        if "TMDB_API_KEY" in st.secrets:
            return st.secrets["TMDB_API_KEY"]
    except Exception:
        # No secrets.toml present at all — st.secrets raises on access.
        pass
    return os.environ.get("TMDB_API_KEY")


@st.cache_resource(show_spinner="Loading the model…")
def load_artifacts():
    missing = [p.name for p in (VECTORS, MOVIES) if not p.exists()]
    if missing:
        return None, None, missing
    return pd.read_csv(MOVIES), sp.load_npz(VECTORS).tocsr(), []


@st.cache_resource
def _session():
    """One pooled HTTPS session for all poster lookups.

    Each fresh connection costs a TLS handshake, and on a flaky link that
    handshake is what drops. Reusing connections (plus urllib3-level retries on
    resets) turns most of those failures into a non-event.
    """
    sess = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    sess.mount("https://", adapter)
    return sess


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def _poster_path(movie_id, api_key):
    """TMDB poster path, or "" if the film genuinely has no poster.

    Raises on network failure — deliberately, so st.cache_data doesn't memoise a
    dropped connection as "no poster" for the next 24 hours.
    """
    last_error = None
    for attempt in range(5):
        try:
            resp = _session().get(
                f"https://api.themoviedb.org/3/movie/{movie_id}",
                params={"api_key": api_key, "language": "en-US"},
                timeout=8,
            )
            resp.raise_for_status()
            return resp.json().get("poster_path") or ""
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(0.3 * 2**attempt)  # 0.3s, 0.6s, 1.2s, 2.4s
    raise last_error


def fetch_poster(movie_id, api_key):
    """Poster URL for a TMDB movie id. Never raises — falls back to a placeholder."""
    if not api_key:
        return PLACEHOLDER
    try:
        path = _poster_path(movie_id, api_key)
    except Exception:
        return PLACEHOLDER
    return POSTER_BASE + path if path else PLACEHOLDER


def recommend(movie, movies, vectors, api_key, k=5):
    """Top-k most similar films to `movie`, as (title, poster_url) pairs."""
    # Positional, not label-based: the similarity row for a film is found by its
    # position in the matrix, which is not the same as any dataframe index label.
    positions = np.flatnonzero((movies["title"] == movie).to_numpy())
    if positions.size == 0:
        return []
    pos = int(positions[0])

    # Vectors are L2-normalised, so a dot product IS cosine similarity.
    scores = (vectors @ vectors[pos].T).toarray().ravel()
    scores[pos] = -1.0  # never recommend the film back to itself
    top = np.argpartition(-scores, k)[:k]
    top = top[np.argsort(-scores[top])]

    rows = movies.iloc[top]
    titles = rows["title"].tolist()
    ids = [int(i) for i in rows["movie_id"]]

    # Five sequential API calls felt slow; fetch them together instead.
    with ThreadPoolExecutor(max_workers=k) as pool:
        posters = list(pool.map(lambda mid: fetch_poster(mid, api_key), ids))

    return list(zip(titles, posters))


st.markdown(
    """
    <style>
      .hero { text-align: center; padding: 0.5rem 0 1.5rem; }
      .hero h1 { font-size: 3rem; margin-bottom: 0.2rem; }
      .hero p  { color: #9aa0b4; font-size: 1.05rem; margin-top: 0; }
      .card-title {
          text-align: center; font-weight: 600; font-size: 0.95rem;
          margin-top: 0.6rem; min-height: 2.6rem; line-height: 1.3;
      }
      div.stButton > button { width: 100%; height: 3rem; font-weight: 600; }
    </style>
    <div class="hero">
      <h1>🎬 WatchWise</h1>
      <p>Pick a film you liked — get five more built from the same stuff.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

movies, vectors, missing = load_artifacts()

if missing:
    st.error(
        "**Missing model files:** `" + "`, `".join(missing) + "`\n\n"
        "Generate them by running `python build_model.py` with the two TMDB CSVs "
        "in the project root, then reload this page."
    )
    st.stop()

api_key = get_api_key()
if not api_key:
    st.warning(
        "No TMDB API key found — recommendations still work, but posters will "
        "show as placeholders. Add one to `.streamlit/secrets.toml` as "
        "`TMDB_API_KEY = \"your_key\"` or set the `TMDB_API_KEY` env var.",
        icon="🔑",
    )

# dict.fromkeys keeps the dropdown free of the duplicate titles in the dataset.
titles = sorted(dict.fromkeys(movies["title"].astype(str).tolist()))

left, right = st.columns([3, 1], vertical_alignment="bottom")
with left:
    selected = st.selectbox(
        "Select or type a movie",
        titles,
        index=titles.index("Batman Begins") if "Batman Begins" in titles else 0,
    )
with right:
    go = st.button("Recommend", type="primary")

st.caption(f"{len(titles):,} films in the catalogue.")

if go:
    with st.spinner("Finding similar films…"):
        results = recommend(selected, movies, vectors, api_key)

    if not results:
        st.error(f"Couldn't find “{selected}” in the dataset.")
    else:
        st.subheader(f"Because you picked *{selected}*")
        for col, (title, poster) in zip(st.columns(len(results)), results):
            with col:
                st.image(poster, use_container_width=True)
                st.markdown(
                    f"<div class='card-title'>{title}</div>",
                    unsafe_allow_html=True,
                )

st.divider()
st.caption(
    "Content-based recommendations from the TMDB 5000 dataset · "
    "posters via the TMDB API. This product uses the TMDB API but is not "
    "endorsed or certified by TMDB."
)
