
import os
import sys
import subprocess
from pathlib import Path

import boto3
import pandas as pd
import plotly.express as px
import requests
import re
import streamlit as st
from dotenv import load_dotenv

# --------------------------------------------------
# Configuration
# --------------------------------------------------
ROOT = Path.home() / "movie-recommendation"
DASHBOARD_DIR = ROOT / "dashboard"
RESULTS_FILE = ROOT / "results" / "recommendations.csv"
MOVIES_FILE = Path("/tmp/movies.dat")
SPARK_SCRIPT = ROOT / "recommend_movies_multi.py"
SPARK_SUBMIT = Path.home() / "spark-4.0.1-bin-hadoop3" / "bin" / "spark-submit"

load_dotenv(DASHBOARD_DIR / ".env")

@st.cache_data(ttl=86400, show_spinner=False)
def get_tmdb_movie(title):
    """Find a movie and its poster using TMDB."""
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        return None

    match = re.search(r"\((\d{4})\)\s*$", str(title))
    year = match.group(1) if match else None
    query = re.sub(r"\s*\(\d{4}\)\s*$", "", str(title)).strip()

    try:
        response = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={
                "api_key": api_key,
                "query": query,
                **({"year": year} if year else {}),
            },
            timeout=10,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            return None

        movie = results[0]
        poster = movie.get("poster_path")
        return {
            "poster": (
                f"https://image.tmdb.org/t/p/w342{poster}"
                if poster else None
            ),
            "overview": movie.get("overview", ""),
        }
    except (requests.RequestException, ValueError, KeyError):
        return None



S3_BUCKET = os.getenv("S3_BUCKET", "movie-recommendation-spark-jay-2026")
S3_KEY = os.getenv("S3_KEY", "")
CACHE_PREFIX = "cache/recommendations"
CACHE_VERSION = "v1"
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
)

# --------------------------------------------------
# Styling
# --------------------------------------------------
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 1.4rem;
        border-radius: 14px;
        background: linear-gradient(120deg, #172554, #1d4ed8);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 {
        color: white;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #e0e7ff;
        margin-bottom: 0;
    }
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 14px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load movie titles
# --------------------------------------------------
@st.cache_data
def load_movie_titles():
    if not MOVIES_FILE.exists():
        return []

    titles = []
    with open(MOVIES_FILE, "r", encoding="latin-1") as file:
        for line in file:
            parts = line.rstrip("\n").split("::")
            if len(parts) >= 3:
                titles.append(parts[1])

    return sorted(set(titles))


@st.cache_data
def load_movie_metadata():
    if not MOVIES_FILE.exists():
        return pd.DataFrame(columns=["movie_id", "title", "genres"])

    rows = []
    with open(MOVIES_FILE, "r", encoding="latin-1") as file:
        for line in file:
            parts = line.rstrip("\n").split("::")
            if len(parts) >= 3:
                rows.append({
                    "movie_id": parts[0],
                    "title": parts[1],
                    "genres": parts[2].replace("|", ", "),
                })

    return pd.DataFrame(rows)


@st.cache_data
def load_title_to_id():
    metadata = load_movie_metadata()
    if metadata.empty:
        return {}
    return dict(zip(metadata["title"], metadata["movie_id"].astype(str)))


title_to_id = load_title_to_id()


# --------------------------------------------------
# Load recommendation results
# --------------------------------------------------
def recommendation_cache_key(selected_movies):
    import hashlib
    import json

    # Sorting makes the key consistent regardless of selection order.
    movie_ids = sorted(
        title_to_id[title]
        for title in selected_movies
        if title in title_to_id
    )

    cache_input = {
        "version": CACHE_VERSION,
        "movie_ids": movie_ids,
    }

    digest = hashlib.sha256(
        json.dumps(cache_input, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return f"{CACHE_PREFIX}/{CACHE_VERSION}/{digest}.csv"


def load_cached_recommendations(cache_key):
    try:
        s3 = boto3.client("s3", region_name=AWS_REGION)
        response = s3.get_object(Bucket=S3_BUCKET, Key=cache_key)
        return pd.read_csv(response["Body"])
    except s3.exceptions.NoSuchKey:
        return None
    except Exception as exc:
        # Treat cache read errors as misses; Spark can still run.
        print(f"Recommendation cache read failed: {exc}")
        return None


def save_cached_recommendations(df, cache_key):
    try:
        s3 = boto3.client("s3", region_name=AWS_REGION)
        csv_data = df.to_csv(index=False).encode("utf-8")
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=cache_key,
            Body=csv_data,
            ContentType="text/csv",
        )
        return True
    except Exception as exc:
        print(f"Recommendation cache save failed: {exc}")
        return False

def load_s3_results():
    if not S3_KEY:
        return pd.DataFrame()

    try:
        s3 = boto3.client("s3", region_name=AWS_REGION)
        response = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
        return pd.read_csv(response["Body"])
    except Exception:
        return pd.DataFrame()


def load_initial_results():
    # Prefer the latest local Spark output.
    if RESULTS_FILE.exists():
        try:
            df = pd.read_csv(RESULTS_FILE)
            if not df.empty:
                return df
        except Exception:
            pass

    # Use S3 only if there is no usable local output.
    return load_s3_results()


def prepare_results(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Support common variations in column names.
    aliases = {
        "movieId": "movie_id",
        "Movie ID": "movie_id",
        "Movie title": "title",
        "Title": "title",
        "Genres": "genres",
        "similarity_score": "similarity",
        "pearson_correlation": "similarity",
        "common_users_per_match": "common_users",
        "selected_movies_matched": "matched_selected_movies",
        "total_rating_overlaps": "total_common_users",
    }

    df.rename(columns={
        old: new for old, new in aliases.items() if old in df.columns
    }, inplace=True)

    for column in [
        "movie_id", "similarity", "common_users",
        "matched_selected_movies", "total_common_users"
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "title" not in df.columns:
        df["title"] = "Unknown movie"

    if "genres" not in df.columns:
        df["genres"] = "Unknown"

    # Enrich missing titles/genres from MovieLens metadata.
    metadata = load_movie_metadata()
    if not metadata.empty and "movie_id" in df.columns:
        metadata["movie_id"] = pd.to_numeric(
            metadata["movie_id"], errors="coerce"
        )
        df = df.merge(
            metadata,
            on="movie_id",
            how="left",
            suffixes=("", "_metadata"),
        )

        if "title_metadata" in df.columns:
            df["title"] = df["title_metadata"].fillna(df["title"])
            df.drop(columns=["title_metadata"], inplace=True)

        if "genres_metadata" in df.columns:
            df["genres"] = df["genres_metadata"].fillna(df["genres"])
            df.drop(columns=["genres_metadata"], inplace=True)

    return df


# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🎬 Movie Recommendation System</h1>
    <p>Discover movies you may love using Apache Spark and collaborative filtering.</p>
</div>
""", unsafe_allow_html=True)

st.caption("Powered by Apache Spark · MovieLens data · AWS S3 · Streamlit")

# --------------------------------------------------
# Sidebar: select movies and run Spark
# --------------------------------------------------
st.subheader("🎯 Choose Your Movies")

movie_titles = load_movie_titles()

if not movie_titles:
    st.warning(
        "Movie titles were not found at /tmp/movies.dat. "
        "Download the MovieLens movies.dat file to enable selection."
    )

selected_movies = st.multiselect(
    "Select movies you like",
    options=movie_titles,
    default=["Toy Story (1995)"] if "Toy Story (1995)" in movie_titles else [],
    max_selections=10,
    help="Choose up to 10 movies to generate personalized recommendations.",
)

st.caption("Choose up to 10 movies to personalize your recommendations.")

run_recommendations = st.button(
    "✨ Get Recommendations",
    type="primary",
    width="stretch",
)

# --------------------------------------------------
# Run the recommendation workflow: S3 cache first, Spark on cache miss
if run_recommendations:
    if not selected_movies:
        st.error("Please select at least one movie.")
    elif not SPARK_SCRIPT.exists():
        st.error(f"Spark script not found: {SPARK_SCRIPT}")
    elif not SPARK_SUBMIT.exists():
        st.error(f"spark-submit not found: {SPARK_SUBMIT}")
    else:
        cache_key = recommendation_cache_key(selected_movies)

        with st.spinner("Checking saved recommendations..."):
            cached_df = load_cached_recommendations(cache_key)

        if cached_df is not None and not cached_df.empty:
            st.session_state["recommendations_df"] = cached_df
            st.session_state["selected_movies"] = selected_movies
            st.success("Loaded saved recommendations from S3 cache!")
        else:
            command = [
                str(SPARK_SUBMIT),
                "--packages",
                "org.apache.hadoop:hadoop-aws:3.4.1",
                str(SPARK_SCRIPT),
                *selected_movies,
            ]

            with st.spinner(
                "Cache miss — running Apache Spark to generate recommendations..."
            ):
                try:
                    result = subprocess.run(
                        command,
                        cwd=str(ROOT),
                        capture_output=True,
                        text=True,
                        timeout=900,
                    )

                    if result.returncode != 0:
                        st.error("Spark could not generate recommendations.")
                        with st.expander("View Spark error details"):
                            st.code(
                                result.stderr[-12000:]
                                or result.stdout[-12000:]
                            )
                    elif RESULTS_FILE.exists():
                        fresh_df = pd.read_csv(RESULTS_FILE)

                        if fresh_df.empty:
                            st.warning(
                                "Spark completed, but the results CSV is empty."
                            )
                        else:
                            st.session_state["recommendations_df"] = fresh_df
                            st.session_state["selected_movies"] = selected_movies

                            if save_cached_recommendations(fresh_df, cache_key):
                                st.success(
                                    f"Generated and cached {len(fresh_df)} recommendations!"
                                )
                            else:
                                st.warning(
                                    f"Generated {len(fresh_df)} recommendations, "
                                    "but S3 caching failed. Results are available "
                                    "for this session."
                                )
                    else:
                        st.error(
                            "Spark finished, but results/recommendations.csv "
                            "was not created."
                        )
                        with st.expander("View Spark output"):
                            st.code(result.stdout[-8000:])

                except subprocess.TimeoutExpired:
                    st.error(
                        "Spark took longer than 15 minutes and was stopped. "
                        "Check the EC2 terminal for details."
                    )
                except Exception as exc:
                    st.error(f"Error running recommendations: {exc}")

# Display latest results
# --------------------------------------------------
if "recommendations_df" in st.session_state:
    raw_df = st.session_state["recommendations_df"]
else:
    raw_df = load_initial_results()

df = prepare_results(raw_df)

if df.empty:
    st.info(
        "No recommendations are loaded yet. Select one or more movies "
        "above and click **Get Recommendations**."
    )
    st.stop()

# --------------------------------------------------
# Sidebar: filter results
# --------------------------------------------------
st.divider()
available_genres = []
if "genres" in df.columns:
    for value in df["genres"].dropna().astype(str):
        available_genres.extend(
            genre.strip()
            for genre in value.split(",")
            if genre.strip()
        )

available_genres = sorted(set(available_genres))

st.sidebar.subheader("🔎 Filter Recommendations")

min_similarity = st.sidebar.slider(
    "Minimum Pearson correlation",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
)

selected_genres = st.sidebar.multiselect(
    "Filter by genre",
    options=available_genres,
)

max_results = st.sidebar.slider(
    "Number of movies to display",
    min_value=5,
    max_value=10,
    value=10,
    step=1,
)

view = df.copy()

if "similarity" in view.columns:
    view = view[view["similarity"].fillna(0) >= min_similarity]

if selected_genres:
    genre_pattern = "|".join(
        __import__("re").escape(genre) for genre in selected_genres
    )
    view = view[
        view["genres"].fillna("").str.contains(
            genre_pattern, case=False, regex=True
        )
    ]

sort_columns = [
    column for column in [
        "matched_selected_movies",
        "similarity",
        "total_common_users",
        "common_users",
    ]
    if column in view.columns
]

if sort_columns:
    view = view.sort_values(
        sort_columns,
        ascending=[False] * len(sort_columns),
        na_position="last",
    )

view = view.head(min(max_results, 10))

# --------------------------------------------------
# KPI cards
# --------------------------------------------------
st.subheader("📊 Recommendation Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Movies displayed", f"{len(view):,}")
c2.metric("Movies selected", f"{len(st.session_state.get('selected_movies', [])):,}")

if "similarity" in view.columns and not view.empty:
    c3.metric(
        "Average correlation",
        f"{view['similarity'].mean():.3f}",
    )
    c4.metric(
        "Highest correlation",
        f"{view['similarity'].max():.3f}",
    )
else:
    c3.metric("Average correlation", "N/A")
    c4.metric("Highest correlation", "N/A")

st.divider()

# --------------------------------------------------
# Movie poster gallery
if not view.empty:
    st.subheader("🎞️ Your Movie Recommendations")
    st.caption("Discover your recommended movies visually.")

    # Keep poster cards in descending hybrid-score order by default.
    poster_sort_column = (
        "final_score" if "final_score" in view.columns
        else "similarity" if "similarity" in view.columns
        else None
    )

    poster_view = view.copy()
    if poster_sort_column:
        poster_view[poster_sort_column] = pd.to_numeric(
            poster_view[poster_sort_column], errors="coerce"
        )
        poster_view = poster_view.sort_values(
            poster_sort_column, ascending=False, na_position="last"
        )

    poster_rows = poster_view.head(10).to_dict("records")
    gallery_columns = st.columns(5)

    for index, movie in enumerate(poster_rows):
        with gallery_columns[index % 5]:
            title = str(movie.get("title", "Unknown movie"))
            details = get_tmdb_movie(title)

            if details and details.get("poster"):
                st.image(details["poster"], width="stretch")
            else:
                st.markdown("🎬 *Poster unavailable*")

            st.markdown(f"**{title}**")
            genres = movie.get("genres", "")
            st.caption(str(genres) if pd.notna(genres) else "Genres unavailable")

            score = movie.get("similarity")
            if score is not None and pd.notna(score):
                st.caption(f"Pearson correlation: {float(score):.3f}")

    st.caption(
        "Movie posters and metadata are provided by TMDB. "
        "This product uses the TMDB API but is not endorsed or certified by TMDB."
    )

st.divider()

# Recommendations table
# --------------------------------------------------
st.subheader("🍿 Recommended Movies")
st.caption(
    "Recommendations combine collaborative filtering with genre similarity. "
    "The final hybrid score weights confidence-adjusted Pearson correlation "
    "at 70% and genre similarity at 30%."
)

if view.empty:
    st.warning("No movies match the current filters. Try lowering the correlation threshold.")
else:
    display_columns = [
        "movie_id",
        "title",
        "genres",
        "final_score",
        "similarity",
        "confidence_score",
        "genre_similarity",
        "common_users",
        "matched_selected_movies",
        "total_common_users",
    ]

    available_columns = [
        column for column in display_columns if column in view.columns
    ]

    display_df = view[available_columns].rename(columns={
        "movie_id": "Movie ID",
        "title": "Movie title",
        "genres": "Genres",
        "final_score": "Final hybrid score",
        "similarity": "Pearson correlation",
        "confidence_score": "Confidence score",
        "genre_similarity": "Genre similarity",
        "common_users": "Common users per match",
        "matched_selected_movies": "Selected movies matched",
        "total_common_users": "Total rating overlaps",
    })

    number_formats = {
        "Movie ID": "{:.0f}",
        "Final hybrid score": "{:.4f}",
        "Pearson correlation": "{:.4f}",
        "Confidence score": "{:.4f}",
        "Genre similarity": "{:.4f}",
        "Common users per match": "{:.0f}",
        "Selected movies matched": "{:.0f}",
        "Total rating overlaps": "{:.0f}",
    }

    number_formats = {
        column: fmt for column, fmt in number_formats.items()
        if column in display_df.columns
    }

    # Interactive table sorting
    sort_options = [
        column for column in display_df.columns
        if column != "Genres"
    ]
    default_sort = (
        "Final hybrid score"
        if "Final hybrid score" in sort_options
        else sort_options[0]
    )

    sort_col1, sort_col2 = st.columns(2)
    with sort_col1:
        sort_by = st.selectbox(
            "Sort table by",
            options=sort_options,
            index=sort_options.index(default_sort),
            key="recommendations_sort_by",
        )
    with sort_col2:
        sort_order = st.selectbox(
            "Sort order",
            options=["Descending", "Ascending"],
            index=0,
            key="recommendations_sort_order",
        )

    sorted_df = display_df.copy()
    if sort_by in sorted_df.columns:
        sorted_df = sorted_df.sort_values(
            by=sort_by,
            ascending=(sort_order == "Ascending"),
            na_position="last",
            kind="stable",
        )

    st.dataframe(
        sorted_df.style.format(number_formats, na_rep="—"),
        use_container_width=True,
        hide_index=True,
        height=420,
    )

    csv_data = sorted_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download recommendations CSV",
        data=csv_data,
        file_name="movie_recommendations.csv",
        mime="text/csv",
    )

# --------------------------------------------------
# Charts
# --------------------------------------------------
if not view.empty and "similarity" in view.columns:
    st.divider()
    st.subheader("📊 Understand Your Recommendations")
    st.caption(
        "These charts explain how closely movies match your selected movies "
        "and how much shared rating data supports each result."
    )

    chart_df = view.copy()
    chart_df["similarity"] = pd.to_numeric(
        chart_df["similarity"], errors="coerce"
    )
    chart_df = chart_df.dropna(subset=["similarity"])

    if not chart_df.empty:
        # Chart 1: strongest recommendations
        st.markdown("### ⭐ Which movies match most closely?")
        st.write(
            "Longer bars and higher scores mean users' rating patterns "
            "for that movie are more positively correlated with your selections. "
            "A score is not a percentage chance that you will like the movie."
        )

        score_column = (
            "final_score" if "final_score" in chart_df.columns
            else "similarity"
        )
        score_label = (
            "Final hybrid score" if score_column == "final_score"
            else "Pearson correlation score"
        )

        chart_df[score_column] = pd.to_numeric(
            chart_df[score_column], errors="coerce"
        )
        top_df = chart_df.nlargest(10, score_column).sort_values(
            score_column, ascending=True
        )
        top_df = top_df.copy()
        top_df["score_label"] = top_df[score_column].map(
            lambda x: f"{x:.3f}"
        )

        fig_bar = px.bar(
            top_df,
            x=score_column,
            y="title",
            orientation="h",
            text="score_label",
            labels={
                score_column: score_label,
                "title": "Recommended movie",
            },
            hover_data=[
                c for c in ["genres", "common_users",
                            "matched_selected_movies", "total_common_users"]
                if c in top_df.columns
            ],
        )
        fig_bar.update_traces(
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                + score_label + ": %{x:.3f}<extra></extra>"
            ),
        )
        fig_bar.update_layout(
            height=max(350, 38 * len(top_df)),
            margin=dict(l=10, r=65, t=15, b=45),
            showlegend=False,
        )
        fig_bar.update_xaxes(range=[0, 1], tickformat=".1f")
        st.plotly_chart(fig_bar, width="stretch")

        # Chart 2: how much shared rating evidence exists
        overlap_column = (
            "total_common_users"
            if "total_common_users" in chart_df.columns
            else "common_users"
            if "common_users" in chart_df.columns
            else None
        )

        if overlap_column:
            st.markdown("### 👥 How much shared rating data supports each movie?")
            st.write(
                "This shows the number of shared rating overlaps recorded by "
                "the algorithm. Larger values mean more shared rating data; "
                "they do not automatically mean a movie is better."
            )

            evidence_df = chart_df.nlargest(10, overlap_column).copy()
            evidence_df[overlap_column] = pd.to_numeric(
                evidence_df[overlap_column], errors="coerce"
            )
            evidence_df = evidence_df.dropna(subset=[overlap_column])
            evidence_df = evidence_df.sort_values(overlap_column)

            fig_evidence = px.bar(
                evidence_df,
                x=overlap_column,
                y="title",
                orientation="h",
                labels={
                    overlap_column: "Shared rating overlaps",
                    "title": "Recommended movie",
                },
                hover_data=[
                    c for c in ["similarity", "genres",
                                "matched_selected_movies"]
                    if c in evidence_df.columns
                ],
            )
            fig_evidence.update_traces(
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Shared rating overlaps: %{x:,.0f}<extra></extra>"
                )
            )
            fig_evidence.update_layout(
                height=max(350, 38 * len(evidence_df)),
                margin=dict(l=10, r=20, t=15, b=45),
                showlegend=False,
            )
            st.plotly_chart(fig_evidence, width="stretch")

# Algorithm details
# --------------------------------------------------
with st.expander("ℹ️ How the recommendation algorithm works"):
    st.markdown("""
    1. **Input:** Choose one or more movies.
    2. **Big data processing:** Apache Spark processes movie and rating data
       stored in AWS S3.
    3. **Collaborative filtering:** Pearson correlation compares rating
       patterns among users who rated both movies.
    4. **Confidence adjustment:** Correlation is adjusted using the number
       of common users, reducing the influence of weakly supported matches.
    5. **Genre similarity:** Jaccard similarity measures the overlap between
       the selected movies' genres and each candidate movie's genres.
    6. **Hybrid score:** The final score combines 70% confidence-adjusted
       correlation and 30% genre similarity.
    7. **Ranking:** Movies are ranked using the number of selected movies
       matched, the final hybrid score, and rating overlap.
    8. **Output:** View recommendations and download them as a CSV file.

    **Important:** Pearson correlation ranges from -1 to +1. Genre similarity
    ranges from 0 to 1. The final hybrid score is a ranking measure, not a
    percentage probability that you will like a movie.
    """)
