import pandas as pd
from surprise import KNNBaseline, SVD
import load
import models
from config import RESULTS_DIR as outdir
from process import filter_transform_metadata, one_hot_encode_genres, merge_datasets
from analyze import generate_plots

def main():
    # Downloading data
    ml_path = load.download_movielens_data() # data downloaded and path to directory returned
    kg_path = load.download_kaggle_data() # data downloaded and path to directory returned
    imdb_path = load.download_imdb_ratings() # data downloaded and path to file returned

    # Loading data
    ml_ratings = load.load_movielens_ratings(ml_path/"ratings.csv")
    ml_ratings_raw = load.load_movielens_ratings_raw(ml_path/"ratings.csv")
    ml_links = load.load_movielens_links(ml_path/"links.csv")
    imdb_ratings = load.load_imdb_ratings(imdb_path)
    kg_movies_metadata = load.load_kaggle_metadata(kg_path/"movies_metadata.csv")

    # Processing movies metadata
    movie_ids = ml_links["tmdbId"].dropna().astype("Int64").unique()
    meta_clean = filter_transform_metadata(kg_movies_metadata, movie_ids)
    meta_with_genres = one_hot_encode_genres(meta_clean) # One-hot encode genres
    meta_merged = merge_datasets(ml_links, meta_with_genres, imdb_ratings)

    # EDA
    generate_plots(meta_merged, ml_ratings)

    # Data prep for the models
    surprise_data = models.build_surprise_data(ml_ratings_raw)
    train_data, test_data = models.surprise_train_test_split(surprise_data, test_size=0.2, random_state=42)

    # 1. SVD
    best_param, best_score = models.tune_SVD(surprise_data, cv=3)
    print("\nFor the Matrix Factorization:\n"+"-"*30)
    print(f"Best score: {best_score:.4f}")
    print(f"Best params: {best_param}")
    svd_estimator = SVD(**best_param)
    svd_estimator.fit(train_data)
    algo, rmse, preds = models.predict_for_test(svd_estimator, test_data)
    print(f"Test RMSE: {rmse:.4f}")
    svd_row = {"best_cv_rmse": best_score, "test_rmse": rmse, "n_factors": best_param.get("n_factors"), "reg_all": best_param.get("reg_all"), "lr_all": best_param.get("lr_all"),
               # placeholders for SVD-only params
               "k": None, "sim_name": None, "user_based": None}
    svd_results_df = pd.DataFrame([svd_row], index=["SVD"])
    svd_pred_df = models.predictions_to_dataframe(preds)
    svd_pred_df.to_csv(outdir/"svd_predictions.csv", index=False)

    # 2. KNN
    best_param, best_score = models.tune_knn_baseline(surprise_data, cv=3)
    print("\nFor KNNBaseline:\n"+"-"*30)
    print(f"Best score: {best_score:.4f}")
    print(f"Best params: {best_param}")
    knn_estimator = KNNBaseline(**best_param)
    knn_estimator.fit(train_data)
    algo, rmse, preds = models.predict_for_test(knn_estimator, test_data)
    print(f"Test RMSE: {rmse:.4f}")
    sim_opts = best_param.get("sim_options", {})
    sim_name = sim_opts.get("name")
    user_based = sim_opts.get("user_based")
    knn_row = {
        "best_cv_rmse": best_score, "test_rmse": rmse, "k": best_param.get("k"), "sim_name": sim_name, "user_based": user_based,
        # placeholders for SVD-only params
        "n_factors": None, "reg_all": None, "lr_all": None}
    knn_results_df = pd.DataFrame([knn_row], index=["KNNBaseline"])
    knn_pred_df = models.predictions_to_dataframe(preds)
    knn_pred_df.to_csv(outdir/"knn_predictions.csv", index=False)

    # combining results for both models in a pandas DataFrame and saving it in results directory as csv file
    results_df = pd.concat([svd_results_df, knn_results_df], axis=0)
    results_df.to_csv(outdir/"model_results.csv")



if __name__ == "__main__":
    main()