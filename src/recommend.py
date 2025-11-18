import pandas as pd

def get_unrated_movies(user_id, ratings_df: pd.DataFrame) -> list:
    """
        Return a list of movieIds that `user_id` has NOT rated yet.
        Assumes ratings_df is the dataframe of all movielens ratings and has columns ['userId', 'movieId'].
    """
    uid = ratings_df["userId"].dtype.type(user_id)
    rated_movies = set(ratings_df[ratings_df["userId"] == uid]["movieId"])
    all_movies = set(ratings_df["movieId"])
    unrated_movies = all_movies - rated_movies
    return list(unrated_movies)


def predict_rating(algo, user_id, movie_id) -> float:
    """
        Predict the rating a user would give to a particular movie using a
        fitted Surprise algorithm (e.g., SVD, KNNBaseline).
        Returns the estimated rating, or None if prediction fails.
    """
    try:
        pred = algo.predict(str(user_id), str(movie_id))
        return float(pred.est)
    except Exception as e:
        print(f"Exception while predicting for user {user_id}, movie {movie_id}: {e}")
        raise e


def top_n_recommendations(algo, user_id: int|str, n: int=10, *, ratings_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
        Return dataframe of top-n recommended (unrated) movies for a given user, with metadata (title, release_date, runtime).
    """
    unrated_movies = get_unrated_movies(user_id, ratings_df)
    predictions = []
    for movie in unrated_movies:
        prediction = predict_rating(algo, user_id, movie)
        predictions.append((movie,prediction))

    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    top_n_df = pd.DataFrame(top_n, columns=['movieId', 'rating_estimate'])
    top_n_with_metadata = pd.merge(metadata_df, top_n_df, on='movieId', how='inner')
    return top_n_with_metadata[['movieId', 'title', 'release_date', 'runtime','rating_estimate']]


def compare_model_recommendations(svd_model, knn_model, user_id, ratings_df, metadata_df, n=10):
    """
        Compare top-N SVD and KNNBaseline recommendations for a user, printing
        counts and titles unique to each model and those in the overlap.
    """
    svd_top = top_n_recommendations(algo=svd_model, user_id=user_id, n=n, ratings_df=ratings_df, metadata_df=metadata_df)
    knn_top = top_n_recommendations(algo=knn_model, user_id=user_id, n=n, ratings_df=ratings_df, metadata_df=metadata_df)

    svd_ids = set(svd_top["movieId"])
    knn_ids = set(knn_top["movieId"])
    overlap = svd_ids & knn_ids
    svd_only_titles = svd_top[svd_top["movieId"].isin(svd_ids - knn_ids)]["title"]
    knn_only_titles = knn_top[knn_top["movieId"].isin(knn_ids - svd_ids)]["title"]
    overlap_titles = svd_top[svd_top["movieId"].isin(overlap)]["title"]

    print(f"\nComparison for user {user_id} (top-{n})")
    print(f"- SVD only: Number - {len(svd_only_titles)} | Titles - {list(svd_only_titles)}")
    print(f"- KNN only: Number - {len(knn_only_titles)} | Titles - {list(knn_only_titles)}")
    print(f"- Overlap: Number - {len(overlap)} | Titles - {list(overlap_titles)}")
    return None
