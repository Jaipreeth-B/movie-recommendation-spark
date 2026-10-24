
import sys
from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

S3_INPUT = "s3a://movie-recommendation-spark-jay-2026/ml-10m/"
OUTPUT_FILE = Path.home() / "movie-recommendation/results/recommendations.csv"
MIN_COMMON_USERS = 20
TOP_N = 10

if len(sys.argv) < 2:
    print('Usage: spark-submit recommend_movies_multi.py "Toy Story (1995)" "Jumanji (1995)"')
    sys.exit(1)

search_titles = list(dict.fromkeys(
    arg.strip() for arg in sys.argv[1:] if arg.strip()
))

spark = (
    SparkSession.builder
    .appName("ImprovedMultiMovieRecommendation")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

try:
    movies = (
        spark.read.text(S3_INPUT + "movies.dat")
        .select(F.split("value", "::").alias("p"))
        .select(
            F.col("p").getItem(0).cast("int").alias("movie_id"),
            F.col("p").getItem(1).alias("title"),
            F.col("p").getItem(2).alias("genres"),
        )
    )

    ratings = (
        spark.read.text(S3_INPUT + "ratings.dat")
        .select(F.split("value", "::").alias("p"))
        .select(
            F.col("p").getItem(0).cast("int").alias("user_id"),
            F.col("p").getItem(1).cast("int").alias("movie_id"),
            F.col("p").getItem(2).cast("double").alias("rating"),
        )
    )

    # Resolve selected titles to unique movie IDs.
    selected = []
    for title in search_titles:
        matches = (
            movies.filter(F.lower("title") == title.lower())
            .select("movie_id", "title")
            .limit(2)
            .collect()
        )
        if len(matches) != 1:
            print(f"Expected one exact title match for: {title}")
            sys.exit(2)
        selected.append(matches[0])

    selected_ids = list(dict.fromkeys(m["movie_id"] for m in selected))
    selected_titles = [m["title"] for m in selected]

    print("Selected movies:")
    for title in selected_titles:
        print(f" - {title}")

    # Only consider ratings from users who rated a selected movie
    # and a candidate movie.
    selected_ratings = (
        ratings.filter(F.col("movie_id").isin(selected_ids))
        .select(
            "user_id",
            F.col("movie_id").alias("selected_id"),
            F.col("rating").alias("selected_rating"),
        )
    )

    candidate_ratings = ratings.select(
        "user_id",
        F.col("movie_id").alias("candidate_id"),
        F.col("rating").alias("candidate_rating"),
    ).filter(~F.col("candidate_id").isin(selected_ids))

    paired = (
        selected_ratings.join(candidate_ratings, "user_id")
        .groupBy("selected_id", "candidate_id")
        .agg(
            F.count("*").alias("common_users"),
            F.sum("selected_rating").alias("sum_x"),
            F.sum("candidate_rating").alias("sum_y"),
            F.sum(F.col("selected_rating") ** 2).alias("sum_x2"),
            F.sum(F.col("candidate_rating") ** 2).alias("sum_y2"),
            F.sum(
                F.col("selected_rating") * F.col("candidate_rating")
            ).alias("sum_xy"),
        )
        .filter(F.col("common_users") >= MIN_COMMON_USERS)
    )

    # Pearson correlation computed from aggregate sums.
    n = F.col("common_users").cast("double")
    numerator = n * F.col("sum_xy") - F.col("sum_x") * F.col("sum_y")
    denominator_x = n * F.col("sum_x2") - F.col("sum_x") ** 2
    denominator_y = n * F.col("sum_y2") - F.col("sum_y") ** 2

    paired = (
        paired
        .withColumn("numerator", numerator)
        .withColumn("denominator_x", denominator_x)
        .withColumn("denominator_y", denominator_y)
        .filter(
            (F.col("denominator_x") > 0) &
            (F.col("denominator_y") > 0)
        )
        .withColumn(
            "correlation",
            F.col("numerator") /
            F.sqrt(F.col("denominator_x") * F.col("denominator_y"))
        )
        .filter(F.col("correlation").isNotNull())
    )

    # Rank candidate movies within each selected movie, then aggregate.
    ranking = Window.partitionBy("selected_id").orderBy(
        F.desc("correlation"), F.desc("common_users")
    )
    ranked = paired.withColumn("rank", F.row_number().over(ranking))

    # Use each selected movie's strongest positive associations.
    positive = ranked.filter(
        (F.col("correlation") > 0) & (F.col("rank") <= 100)
    )

    # Combine collaborative filtering with confidence and genre similarity.
    candidate_genres = movies.select(
        F.col("movie_id").alias("candidate_id"),
        F.col("genres").alias("candidate_genres"),
        "title",
    )

    selected_genres = (
        movies.filter(F.col("movie_id").isin(selected_ids))
        .select(
            F.col("movie_id").alias("selected_id"),
            F.col("genres").alias("selected_genres"),
        )
    )

    # Calculate genre similarity for each selected/candidate pair.
    genre_pairs = (
        positive
        .join(selected_genres, "selected_id")
        .join(candidate_genres, "candidate_id")
        .withColumn(
            "selected_genre_array",
            F.split(F.col("selected_genres"), r"\|"),
        )
        .withColumn(
            "candidate_genre_array",
            F.split(F.col("candidate_genres"), r"\|"),
        )
        .withColumn(
            "shared_genre_count",
            F.size(F.array_intersect(
                F.col("selected_genre_array"),
                F.col("candidate_genre_array"),
            )),
        )
        .withColumn(
            "all_genre_count",
            F.size(F.array_union(
                F.col("selected_genre_array"),
                F.col("candidate_genre_array"),
            )),
        )
        .withColumn(
            "genre_score",
            F.when(
                F.col("all_genre_count") > 0,
                F.col("shared_genre_count") / F.col("all_genre_count"),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "confidence_score",
            F.col("correlation") * (
                F.col("common_users") /
                (F.col("common_users") + F.lit(50.0))
            ),
        )
        .withColumn(
            "pair_score",
            F.lit(0.7) * F.col("confidence_score")
            + F.lit(0.3) * F.col("genre_score"),
        )
    )

    combined = (
        genre_pairs.groupBy("candidate_id")
        .agg(
            F.avg("correlation").alias("similarity"),
            F.avg("confidence_score").alias("confidence_score"),
            F.avg("genre_score").alias("genre_similarity"),
            F.avg("pair_score").alias("final_score"),
            F.countDistinct("selected_id").alias("matched_selected_movies"),
            F.sum("common_users").alias("total_common_users"),
            F.max("common_users").alias("common_users"),
            F.first("title").alias("title"),
            F.first("candidate_genres").alias("genres"),
        )
        .select(
            F.col("candidate_id").alias("movie_id"),
            "title",
            "genres",
            "similarity",
            "confidence_score",
            "genre_similarity",
            "final_score",
            "common_users",
            "total_common_users",
            "matched_selected_movies",
        )
        .orderBy(
            F.desc("matched_selected_movies"),
            F.desc("final_score"),
            F.desc("total_common_users"),
        )
        .limit(TOP_N)
    )

    result = combined.toPandas()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print("\nTop recommendations:")
    print(result.to_string(index=False))
    print(f"\nSaved CSV to: {OUTPUT_FILE}")

finally:
    spark.stop()
