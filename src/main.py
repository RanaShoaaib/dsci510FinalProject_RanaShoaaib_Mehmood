import load
from process import filter_transform_metadata,one_hot_encode_genres,merge_datasets
from analyze import generate_plots

def main():
    # Downloading data
    ml_path = load.download_movielens_data() # data downloaded and path to directory returned
    kg_path = load.download_kaggle_data() # data downloaded and path to directory returned
    imdb_path = load.download_imdb_ratings() # data downloaded and path to file returned

    # Loading data
    ml_ratings = load.load_movielens_ratings(ml_path/"ratings.csv")
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

if __name__ == "__main__":
    main()