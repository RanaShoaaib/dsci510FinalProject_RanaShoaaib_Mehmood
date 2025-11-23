import pandas as pd
import numpy as np
from surprise import SVD, KNNBaseline
from process import parse_genres, one_hot_encode_genres
from analyze import imdb_validation
from models import build_surprise_data, surprise_train_test_split, predict_for_test, predictions_to_dataframe
from recommend import get_unrated_movies, predict_rating, top_n_recommendations


# ---------------------------------------------------------------------
# Shared toy data for multiple tests
# ---------------------------------------------------------------------

def make_toy_ratings() -> pd.DataFrame:
    """
    Tiny ratings dataframe for testing Surprise-related functions.
    Two users, three movies, four ratings total.
    """
    return pd.DataFrame({
        "userId":  [1, 1, 2, 2],
        "movieId": [10, 20, 10, 30],
        "rating":  [4.0, 5.0, 3.0, 2.0],
    })


def make_toy_metadata() -> pd.DataFrame:
    """
    Simple metadata dataframe that matches the toy ratings movieIds.
    """
    return pd.DataFrame({
        "movieId": [10, 20, 30],
        "title": ["Movie A", "Movie B", "Movie C"],
        "release_date": ["2000-01-01", "2001-01-01", "2002-01-01"],
        "runtime": [100, 110, 120],
    })


# ---------------------------------------------------------------------
# process.py tests
# ---------------------------------------------------------------------

def test_parse_genres_basic():
    raw = "[{'id': 28, 'name': 'Action'}, {'id': 35, 'name': 'Comedy'}]"
    result = parse_genres(raw)
    expected = ["Action", "Comedy"]
    assert result == expected, f"Expected {expected}, got {result}"

    # Non-string / empty handling
    assert parse_genres(None) == [], "Expected [] for None input."
    assert parse_genres("") == [], "Expected [] for empty string input."

    print("test_parse_genres_basic passed.")


def test_one_hot_encode_genres_basic():
    df = pd.DataFrame({
        "movieId": [1, 2, 3],
        "genre_lst": [
            ["Action", "Comedy"],
            ["Drama"],
            []  # no genres
        ],
    })

    encoded = one_hot_encode_genres(df)

    # We expect 3 new genre columns: Action, Comedy, Drama
    for col in ["Action", "Comedy", "Drama"]:
        assert col in encoded.columns, f"Missing genre column: {col}"

    # Check specific rows
    row0 = encoded.loc[encoded["movieId"] == 1].iloc[0]
    assert row0["Action"] == 1 and row0["Comedy"] == 1, "Row 0 one-hot encoding incorrect."
    row1 = encoded.loc[encoded["movieId"] == 2].iloc[0]
    assert row1["Drama"] == 1, "Row 1 one-hot encoding incorrect."
    row2 = encoded.loc[encoded["movieId"] == 3].iloc[0]
    assert row2[["Action", "Comedy", "Drama"]].sum() == 0, "Row 2 should have no genres encoded."

    print("test_one_hot_encode_genres_basic passed.")


# ---------------------------------------------------------------------
# analyze.py tests
# ---------------------------------------------------------------------

def test_imdb_validation_basic():
    # Two overlapping rows with non-constant values
    pred_df = pd.DataFrame({
        "movieId": ["1", "2", "3"],
        "estimate": [3.0, 4.0, 5.0],
    })
    meta_df = pd.DataFrame({
        "movieId": ["1", "2", "3"],
        "imdb_averageRating": [3.0, 4.0, 5.0],
    })

    corr = imdb_validation(pred_df, meta_df, model_name="TEST_MODEL")

    # Correlation should be close to 1.0
    assert corr is not None, "Expected a correlation value, got None."
    assert isinstance(corr, float), f"Expected float for correlation, got {type(corr)}"
    assert np.isclose(corr, 1.0), f"Expected correlation ~ 1.0, got {corr}"

    print("test_imdb_validation_basic passed.")


# ---------------------------------------------------------------------
# models.py tests
# ---------------------------------------------------------------------

def test_build_surprise_data_and_split():
    raw = make_toy_ratings()

    data = build_surprise_data(raw)  # should not raise
    assert data is not None, "build_surprise_data returned None."

    # 50/50 split to make counts easy to reason about
    trainset, testset = surprise_train_test_split(data, test_size=0.5, random_state=42)

    # There are 4 ratings total; half should go to test
    assert len(testset) == 2, f"Expected 2 test ratings, got {len(testset)}."
    assert trainset.n_ratings == 2, f"Expected 2 train ratings, got {trainset.n_ratings}."

    print("test_build_surprise_data_and_split passed.")


def test_predict_for_test_and_predictions_to_dataframe():
    raw = make_toy_ratings()
    data = build_surprise_data(raw)
    trainset, testset = surprise_train_test_split(data, test_size=0.5, random_state=42)

    # Simple, fast SVD model (no grid search)
    svd = SVD(n_factors=5, n_epochs=5, random_state=0, verbose=False)
    svd.fit(trainset)
    svd_model, svd_rmse, svd_preds = predict_for_test(svd, testset)

    assert isinstance(svd_rmse, float), "SVD RMSE should be a float."
    assert len(svd_preds) == len(testset), "Number of predictions must match test set size."

    svd_df = predictions_to_dataframe(svd_preds)
    expected_cols = ["userId", "movieId", "true_rating", "estimate", "details"]
    assert list(svd_df.columns) == expected_cols, f"Unexpected columns: {svd_df.columns}"
    assert svd_df.shape[0] == len(testset), "Predictions DataFrame row count mismatch."

    # userId and movieId should be strings (object dtype), since build_surprise_data casts them to str
    assert svd_df["userId"].dtype == object, "userId should be object (string) dtype."
    assert svd_df["movieId"].dtype == object, "movieId should be object (string) dtype."

    # Tiny KNNBaseline model as well
    knn = KNNBaseline(
        k=2,
        sim_options={"name": "cosine", "user_based": False},
        verbose=False,
    )
    knn.fit(trainset)
    knn_model, knn_rmse, knn_preds = predict_for_test(knn, testset)

    assert isinstance(knn_rmse, float), "KNN RMSE should be a float."
    assert len(knn_preds) == len(testset), "KNN predictions length mismatch."

    knn_df = predictions_to_dataframe(knn_preds)
    assert knn_df.shape[0] == len(testset), "KNN Predictions DataFrame row count mismatch."

    print("test_predict_for_test_and_predictions_to_dataframe passed.")


# ---------------------------------------------------------------------
# recommend.py tests
# ---------------------------------------------------------------------

def test_get_unrated_movies_basic():
    ratings_df = make_toy_ratings()

    unrated_user1 = get_unrated_movies(1, ratings_df)
    # user 1 has rated movieId 10 and 20, so 30 should be the only unrated movie
    assert set(unrated_user1) == {30}, f"User 1 unrated movies incorrect: {unrated_user1}"

    unrated_user3 = get_unrated_movies(3, ratings_df)
    # user 3 doesn't exist in ratings, so all movies in the df should be unrated for them
    assert set(unrated_user3) == {10, 20, 30}, f"User 3 unrated movies incorrect: {unrated_user3}"

    print("test_get_unrated_movies_basic passed.")


def _train_small_svd_on_toy():
    """
    Helper: build a surprise dataset on the toy ratings and train a tiny SVD.
    Returns:
        svd_model, trainset, testset, raw_ratings_df
    """
    raw = make_toy_ratings()
    data = build_surprise_data(raw)
    trainset, testset = surprise_train_test_split(data, test_size=0.5, random_state=42)

    svd = SVD(n_factors=5, n_epochs=5, random_state=0, verbose=False)
    svd.fit(trainset)
    return svd, trainset, testset, raw


def test_predict_rating_basic():
    svd_model, trainset, testset, raw = _train_small_svd_on_toy()

    # Pick a user/movie pair that appears in the data
    est = predict_rating(svd_model, user_id=1, movie_id=10)
    assert isinstance(est, float), f"Expected a float estimate, got {type(est)}"
    # Rating scale should be within 0.5–5.0 (MovieLens scale)
    assert 0.5 <= est <= 5.0, f"Predicted rating {est} outside expected range."

    print("test_predict_rating_basic passed.")


def test_top_n_recommendations_basic():
    svd_model, trainset, testset, raw = _train_small_svd_on_toy()
    meta_df = make_toy_metadata()

    # For user 1, only movieId 30 is unrated, so for n=1 the recommender must return that
    recs = top_n_recommendations(
        algo=svd_model,
        user_id=1,
        n=1,
        ratings_df=raw,
        metadata_df=meta_df,
    )

    assert len(recs) == 1, f"Expected 1 recommendation, got {len(recs)}"
    assert recs.iloc[0]["movieId"] == 30, f"Expected movieId 30, got {recs.iloc[0]['movieId']}"

    # Check that required columns are present
    for col in ["movieId", "title", "release_date", "runtime", "rating_estimate"]:
        assert col in recs.columns, f"Missing column in recommendations: {col}"

    print("test_top_n_recommendations_basic passed.")


# ---------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------

def run_all_tests():
    print("Running tests...\n")

    # process.py
    test_parse_genres_basic()
    test_one_hot_encode_genres_basic()

    # analyze.py
    test_imdb_validation_basic()

    # models.py
    test_build_surprise_data_and_split()
    test_predict_for_test_and_predictions_to_dataframe()

    # recommend.py
    test_get_unrated_movies_basic()
    test_predict_rating_basic()
    test_top_n_recommendations_basic()

    print("\nAll tests passed!")


if __name__ == "__main__":
    run_all_tests()
