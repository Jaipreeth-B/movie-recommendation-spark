# Movie Recommendation System using Apache Spark

A Big Data Analytics project that recommends movies using user rating patterns and genre similarity.

## Features
- Multi-movie selection for personalized recommendations.
- Apache Spark processing of MovieLens ratings.
- Hybrid recommendation scoring using Pearson correlation and genre similarity.
- Interactive Streamlit dashboard with charts, sorting, and TMDB movie posters.
- Amazon S3 storage for datasets and persistent recommendation caching.
- Cached results are reused to avoid repeating Spark computations.

## Technology Stack
- Python and PySpark
- Apache Spark
- Amazon S3 and EC2
- Streamlit, Pandas, and Plotly
- TMDB API for movie posters

## Project Structure
```text
movie-recommendation/
├── dashboard/
│   ├── app.py
│   └── requirements.txt
├── recommend_movies_multi.py
├── README.md
└── .gitignore
```

## Setup
1. Install Java 17, Apache Spark, and Python.
2. Install dashboard dependencies with `pip install -r dashboard/requirements.txt`.
3. Configure `dashboard/.env` with your S3 bucket, AWS region, and TMDB API key.
4. Configure AWS permissions securely, preferably with an EC2 IAM role.
5. Ensure the Spark script and dashboard paths match your environment.
6. Run the dashboard from its directory:
```bash
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## S3 Configuration
The Spark script currently reads MovieLens data from `s3a://movie-recommendation-spark-jay-2026/ml-10m/`.
Update the input path if you use a different bucket.
The EC2 IAM role needs read access to the input data and read/write access to the recommendation cache.

## Recommendation Workflow
1. Select movies in the dashboard.
2. Generate a versioned cache key from the selected movie IDs.
3. Load recommendations from S3 if the cache exists.
4. Otherwise, run Spark and save the resulting CSV to S3.

## Dataset and Disclaimer
The project uses MovieLens data. Follow the dataset provider terms of use.
Recommendation scores are heuristic ranking signals and are not guaranteed measures of user preference.
This project is intended for educational and demonstration purposes.
