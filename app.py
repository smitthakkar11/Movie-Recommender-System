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
from html import escape
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

    # Fetch the seed film's poster alongside the five results, in one batch.
    seed_id = int(movies.iloc[pos]["movie_id"])
    with ThreadPoolExecutor(max_workers=k + 1) as pool:
        posters = list(pool.map(lambda mid: fetch_poster(mid, api_key), [seed_id] + ids))

    seed_poster, result_posters = posters[0], posters[1:]
    results = [
        {"title": ti, "poster": po, "score": float(scores[i])}
        for ti, po, i in zip(titles, result_posters, top)
    ]
    return {"seed_poster": seed_poster, "results": results}


CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  .stApp {
      background:
        radial-gradient(1100px 620px at 12% -8%,  rgba(229, 9, 20, 0.16), transparent 60%),
        radial-gradient(950px 560px at 88% 4%,   rgba(94, 92, 230, 0.16), transparent 62%),
        #08090d;
  }
  html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
  #MainMenu, footer { visibility: hidden; }
  .block-container { padding-top: 2.4rem; max-width: 1220px; }

  /* ---------- hero ---------- */
  .hero { text-align: center; padding: 0.4rem 0 1.6rem; }
  .hero h1 {
      font-size: clamp(2.6rem, 6vw, 4rem); font-weight: 800; letter-spacing: -0.03em;
      margin: 0 0 0.5rem; line-height: 1.05;
      background: linear-gradient(100deg, #fff 8%, #ffb3b8 46%, #a5a3ff 92%);
      -webkit-background-clip: text; background-clip: text; color: transparent;
  }
  .hero p { color: #9599ad; font-size: 1.06rem; margin: 0; font-weight: 400; }
  .hero .pill {
      display: inline-block; margin-bottom: 1.05rem; padding: 0.32rem 0.9rem;
      border: 1px solid rgba(255,255,255,0.14); border-radius: 999px;
      background: rgba(255,255,255,0.05); color: #c9ccd8;
      font-size: 0.735rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
  }

  /* ---------- controls ---------- */
  div[data-testid="stSelectbox"] > div > div {
      background: rgba(255,255,255,0.05) !important;
      border: 1px solid rgba(255,255,255,0.13) !important;
      border-radius: 12px !important; min-height: 3.05rem;
  }
  div[data-testid="stSelectbox"] > div > div:focus-within {
      border-color: rgba(229,9,20,0.65) !important;
      box-shadow: 0 0 0 3px rgba(229,9,20,0.16) !important;
  }
  div.stButton > button {
      width: 100%; height: 3.05rem; border-radius: 12px; border: 0;
      font-weight: 700; font-size: 0.97rem; letter-spacing: 0.01em; color: #fff;
      background: linear-gradient(135deg, #e50914 0%, #ff4d57 100%);
      box-shadow: 0 8px 22px rgba(229,9,20,0.32); transition: all 0.18s ease;
  }
  div.stButton > button:hover {
      transform: translateY(-2px); box-shadow: 0 12px 30px rgba(229,9,20,0.45); color: #fff;
  }
  div.stButton > button:active { transform: translateY(0); }

  /* ---------- results ---------- */
  .seedbar {
      display: flex; align-items: center; gap: 0.95rem; margin: 2.4rem 0 1.5rem;
      padding: 0.85rem 1.1rem; border-radius: 14px;
      background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.09);
  }
  .seedbar img { width: 44px; height: 66px; object-fit: cover; border-radius: 7px; flex: none; }
  .seedbar .lbl {
      color: #8b8fa3; font-size: 0.7rem; font-weight: 700;
      letter-spacing: 0.13em; text-transform: uppercase; margin-bottom: 0.16rem;
  }
  .seedbar .ttl { color: #fff; font-size: 1.16rem; font-weight: 700; line-height: 1.2; }

  .grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 1.15rem; }
  @media (max-width: 1000px) { .grid { grid-template-columns: repeat(3, 1fr); } }
  @media (max-width: 620px)  { .grid { grid-template-columns: repeat(2, 1fr); } }

  .card {
      position: relative; border-radius: 15px; overflow: hidden;
      background: #14161d; border: 1px solid rgba(255,255,255,0.08);
      box-shadow: 0 8px 24px rgba(0,0,0,0.42);
      transition: transform 0.22s cubic-bezier(.2,.7,.3,1), box-shadow 0.22s ease, border-color 0.22s ease;
  }
  .card:hover {
      transform: translateY(-8px);
      border-color: rgba(229,9,20,0.5);
      box-shadow: 0 20px 42px rgba(0,0,0,0.6), 0 0 0 1px rgba(229,9,20,0.22);
  }
  .card .shot { position: relative; aspect-ratio: 2/3; background: #1b1e27; }
  .card .shot img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .card .shade {
      position: absolute; inset: 0;
      background: linear-gradient(to top, rgba(6,7,10,0.94) 0%, rgba(6,7,10,0.5) 26%, transparent 52%);
  }
  .rank {
      position: absolute; top: 0.6rem; left: 0.6rem; width: 1.65rem; height: 1.65rem;
      display: flex; align-items: center; justify-content: center;
      border-radius: 8px; background: rgba(8,9,13,0.8); backdrop-filter: blur(7px);
      border: 1px solid rgba(255,255,255,0.18);
      color: #fff; font-size: 0.76rem; font-weight: 800;
  }
  .match {
      position: absolute; top: 0.6rem; right: 0.6rem;
      padding: 0.2rem 0.5rem; border-radius: 999px;
      background: rgba(229,9,20,0.9); backdrop-filter: blur(7px);
      color: #fff; font-size: 0.68rem; font-weight: 800; letter-spacing: 0.02em;
  }
  .card .name {
      position: absolute; left: 0.72rem; right: 0.72rem; bottom: 0.68rem;
      color: #fff; font-size: 0.87rem; font-weight: 700; line-height: 1.28;
      text-shadow: 0 2px 10px rgba(0,0,0,0.9);
  }

  .foot {
      margin-top: 2.6rem; padding-top: 1.15rem;
      border-top: 1px solid rgba(255,255,255,0.08);
      color: #6f7386; font-size: 0.8rem; text-align: center; line-height: 1.65;
  }
  .foot a { color: #9599ad; text-decoration: none; border-bottom: 1px solid rgba(255,255,255,0.16); }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero">
      <div class="pill">Content-based · TMDB 5000</div>
      <h1>WatchWise</h1>
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
        payload = recommend(selected, movies, vectors, api_key)

    if not payload or not payload["results"]:
        st.error(f"Couldn't find \u201c{selected}\u201d in the dataset.")
    else:
        st.markdown(
            f'''<div class="seedbar">
                 <img src="{payload["seed_poster"]}" alt="">
                 <div>
                   <div class="lbl">Because you picked</div>
                   <div class="ttl">{escape(selected)}</div>
                 </div>
               </div>''',
            unsafe_allow_html=True,
        )

        cards = []
        for n, item in enumerate(payload["results"], start=1):
            cards.append(
                f'''<div class="card">
                     <div class="shot">
                       <img src="{item["poster"]}" alt="{escape(item["title"], quote=True)}" loading="lazy">
                       <div class="shade"></div>
                       <div class="rank">{n}</div>
                       <div class="match">{round(item["score"] * 100)}% match</div>
                       <div class="name">{escape(item["title"])}</div>
                     </div>
                   </div>'''
            )
        st.markdown(f'<div class="grid">{"".join(cards)}</div>', unsafe_allow_html=True)

st.markdown(
    '''<div class="foot">
         Content-based recommendations from the TMDB 5000 dataset · posters via the
         <a href="https://www.themoviedb.org/" target="_blank" rel="noopener">TMDB API</a>.<br>
         This product uses the TMDB API but is not endorsed or certified by TMDB.
       </div>''',
    unsafe_allow_html=True,
)
