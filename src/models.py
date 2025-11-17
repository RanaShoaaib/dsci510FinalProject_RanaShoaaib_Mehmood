from surprise import Dataset, Reader, SVD, KNNBaseline, accuracy
from surprise.model_selection import train_test_split, GridSearchCV
import numpy as np
import pandas as pd

SVD_PARAM_GRID = {"n_factors": [20, 40, 60],
        "reg_all": [0.05, 0.075, 0.1, 0.25],
        "lr_all": [0.005, 0.01, 0.025, 0.05]}

KNN_PARAM_GRID = {"k": [10, 30, 50, 70],
        "sim_options": {"name": ["cosine", "pearson_baseline"], "user_based": [False]},
        "verbose": [False]}

def build_surprise_data(raw_ratings_df, rating_scale=(0.5, 5.0)):
    """
        Builds a surprise dataset from the raw ratings dataframe.
    """
    reader = Reader(rating_scale=rating_scale)
    data = Dataset.load_from_df(raw_ratings_df[["userId", "movieId", "rating"]].astype({"userId": str, "movieId": str, "rating": float}), reader)
    return data

def surprise_train_test_split(data:Dataset, test_size:float = 0.2, random_state:int = 42):
    """
        Performs train and test split on the input surprise dataset.
        Returns:    a surprise train set and
                    test set - list of (uid, iid, r_ui) triples.
    """
    if test_size < 0 or test_size > 1:
        raise ValueError("test_size must be between 0 and 1")
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)
    return train, test

def tune_SVD(data:Dataset, cv:int = 3):
    """
        Grid search for Matrix Factorization (SVD).
        Input:      the surprise ratings dataset.
                    cv (int) - the number of folds to use - Default: 3
        Returns:    the dictionary of best parameters and
                    best score (float) obtained from cross-validation.
    """
    gs = GridSearchCV(algo_class=SVD, param_grid=SVD_PARAM_GRID, measures=["rmse"], cv=cv, n_jobs=2)
    gs.fit(data)

    best_params = gs.best_params["rmse"]
    best_score = gs.best_score["rmse"]

    return best_params, best_score


def tune_knn_baseline(data:Dataset, cv:int=3):
    """
    Grid search for KNNBaseline (memory-based CF).
    Input:      the surprise ratings dataset.
                cv (int) - the number of folds to use - Default: 3
    Returns:    best parameters as a dictionary and
                best score obtained from cross-validation.
    """

    gs = GridSearchCV(algo_class=KNNBaseline, param_grid=KNN_PARAM_GRID, measures=["rmse"], cv=cv, n_jobs=2)
    gs.fit(data)

    best_params = gs.best_params["rmse"]
    best_score = gs.best_score["rmse"]

    return best_params, best_score


def predict_for_test(algo, test_data):
    """
    Predicts ratings for a test set (list of (uid, iid, r_ui) triples).
    Input:
            test_data: the surprise test set.
            algo: the trained model to use for prediction.
    Returns:
            the model; rmse ; and predicted ratings for test_data
    """
    preds = algo.test(test_data)
    rmse = accuracy.rmse(preds, verbose=False)
    return algo, rmse, preds

def predictions_to_dataframe(pred_list):
    """
    Convert Surprise Prediction objects list into a pandas DataFrame
    with columns: userId, movieId, true_rating, est, details
    """
    rows = []
    for p in pred_list:
        rows.append({"userId": p.uid, "movieId": p.iid, "true_rating": p.r_ui, "estimate": p.est, "details": p.details})
    return pd.DataFrame(rows)