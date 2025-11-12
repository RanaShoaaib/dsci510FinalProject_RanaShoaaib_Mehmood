import ast
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def parse_genres(s:str) -> list:
    """
        Parse the 'genres' field string into a list of genre names.
        The input string typically looks like:
            "[{'id': 18, 'name': 'Drama'}, {'id': 80, 'name': 'Crime'}]"
        Returns:
            A list of genre names (e.g., ['Drama', 'Crime']).
            Returns an empty list if parsing fails or input is invalid.
    """
    if not isinstance(s, str) or not s:
        return []
    try:
        lst = ast.literal_eval(s)
        return [d.get("name", "") for d in lst if isinstance(d, dict) and d.get("name")]
    except:
        return []


def filter_transform_metadata(df: pd.DataFrame, movie_ids) -> pd.DataFrame:
    """
        Filter and normalize kaggle movies metadata for a given set of movie (tmdb) IDs.
        - Validates required columns.
        - Coerces 'id' → int and filters rows to movie_ids.
        - Parses 'release_date' → datetime and adds 'year'.
        - Coerces ['runtime','budget','revenue'] → numeric.
        - Drops unnecessary fields ['title','original_language'].
        - Parses 'genres' into 'genre_lst' (list of genre names) and drops 'genres'.
        Returns:
            A cleaned DataFrame sorted by 'id', with columns:
            ['id','imdb_id','title','original_language','release_date','year',
             'runtime','budget','revenue','genre_lst'].
        """
    required = ['id', 'imdb_id', 'title', 'original_language', 'release_date', 'runtime', 'budget', 'revenue','genres']
    missing = set(required) - set(df.columns)
    if missing:
        raise KeyError(f"Passed dataframe is missing columns: {sorted(missing)}")

    # Retain only required columns and coerce ids to numeric
    transformed = df[required].copy()
    transformed["id"] = pd.to_numeric(transformed["id"], errors="coerce").astype("Int64")

    # Filter by the provided movie_ids
    movie_ids_ser = pd.Series(movie_ids, dtype="Int64")
    transformed = transformed[transformed["id"].isin(movie_ids_ser)].reset_index(drop=True)

    # Parse types
    transformed["release_date"] = pd.to_datetime(transformed["release_date"], errors="coerce")
    transformed["year"] = transformed["release_date"].dt.year
    transformed['year'] = pd.to_numeric(transformed['year'], errors='coerce').astype('Int64')

    for c in ["runtime", "budget", "revenue"]:
        transformed[c] = pd.to_numeric(transformed[c], errors="coerce")

    for c in ["imdb_id", "title", "original_language"]:
        transformed[c] = transformed[c].str.strip()

    transformed["genre_lst"] = transformed["genres"].map(parse_genres)
    transformed.drop(columns=["genres"], inplace=True)

    return transformed.sort_values("id").reset_index(drop=True)

def one_hot_encode_genres(df: pd.DataFrame) -> pd.DataFrame:
    """
        Expects df['genre_lst'] as list[str].
        Returns df with genre list column dropped and one-hot genre columns appended.
    """
    mlb = MultiLabelBinarizer()
    G = mlb.fit_transform(df["genre_lst"])
    genre_cols = [g for g in mlb.classes_]

    Gdf = pd.DataFrame(G, columns=genre_cols, index=df.index).astype("int8")
    return pd.concat([df.drop(columns=["genre_lst"]), Gdf], axis=1)


def merge_datasets(ml_links:pd.DataFrame, meta_with_genres:pd.DataFrame, imdb_ratings:pd.DataFrame) -> pd.DataFrame:
    """
    Merge MovieLens links with Kaggle metadata (via TMDB id) and IMDb ratings (via tconst).
    Returns:
      DataFrame with one row per MovieLens movieId and columns:
        - movieId, tmdbId, imdbId, title, original_language, release_date, runtime, budget, revenue, genre one-hots, …,'imdb_averageRating', 'imdb_numVotes'.
    """
    # Merge movielens links and Kaggle metadata via tmdbId
    meta = meta_with_genres.rename(columns={'id': 'tmdbId'}).copy()
    meta = meta.dropna(subset=['tmdbId']).drop_duplicates(subset=['tmdbId'], keep='first')
    merged = pd.merge(left=ml_links, right=meta, on='tmdbId', how='left')

    # Merge IMDb ratings via imdb_id/tconst
    merged['tconst'] = merged['imdb_id'].str.strip()
    imdb_ratings['tconst'] = imdb_ratings['tconst'].str.strip().copy()
    merged = pd.merge(left=merged, right=imdb_ratings, on='tconst', how='left')

    merged = merged.drop(columns=['imdbId', 'tconst'])
    merged = merged.rename(columns={'imdb_id': 'imdbId', 'numVotes': 'imdb_numVotes', 'averageRating': 'imdb_averageRating'})
    return merged