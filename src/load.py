import io, os, zipfile, gzip, requests, pandas as pd
import config
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi


def download_movielens_data(force_download:bool = False) -> Path:
    """
    Download the MovieLens “ml-latest-small” ZIP and extract it under data/movielens/.
    Skips download if it already exists and force_download is False.
    Input: force_download - Boolean to indicate whether to force download even if files already exist.
    Returns the path to the extracted/existing folder
    """
    url = config.ML_URL
    out_dir = config.DATA_DIR / "movielens"
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted = out_dir / "ml-latest-small"
    if not extracted.exists() or force_download:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(out_dir)
        print("Downloaded movielens data")
    else:
        print("MovieLens data already present. Skipping download.")
    return extracted


def load_movielens_ratings(file_path:Path) -> pd.DataFrame:
    """
        Return a user×movie ratings matrix (NaN = unrated) for MovieLens small.
        Rows: userId, Columns: movieId, Values: rating
    """
    try:
        ratings = pd.read_csv(file_path)
    except:
        raise FileNotFoundError(f"Could not find {file_path}")
    return ratings[["userId", "movieId", "rating"]].pivot(index="userId", columns="movieId", values="rating").sort_index().sort_index(axis=1)


def load_movielens_links(file_path:Path) -> pd.DataFrame:
    """
        Return a dataframe of links.csv from the MovieLens dataset.
        Drops duplicates before returning.
    """
    try:
        links = pd.read_csv(file_path)
    except:
        raise FileNotFoundError(f"Could not find {file_path}")
    for col in links.columns:
        links[col] = links[col].astype("Int64")
    return links.drop_duplicates()


def download_kaggle_data(force_download:bool = False) -> Path:
    """
    Download and extract files from Kaggle dataset into data/kaggle/.
    After extraction keep only required files and delete all others.
    Skips download if it already exists and force_download is False.
    Input: force_download - Boolean to indicate whether to force download even if files already exist.
    Returns: the kaggle directory path.
    """
    out_dir = config.DATA_DIR / "kaggle"
    out_dir.mkdir(parents=True, exist_ok=True)

    keep_set = {"movies_metadata.csv", "links_small.csv", "ratings_small.csv"}
    keep_paths = [out_dir / k for k in keep_set]
    if not force_download and (keep_paths[0].exists() and keep_paths[0].stat().st_size > 0) and (keep_paths[1].exists() and keep_paths[1].stat().st_size > 0) and (keep_paths[2].exists() and keep_paths[2].stat().st_size > 0):
        print("Required Kaggle files already present. Skipping download.")
        return out_dir

    if not (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")):
        raise RuntimeError("Kaggle credentials not found.")

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(config.KAGGLE_DATASET_SLUG, path=out_dir, quiet=False)

    # Extract then delete the zip
    zip_path = list(out_dir.glob("*.zip"))[0]
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
        print("Downloaded & extracted kaggle data")
    zip_path.unlink()

    # data cleanup - delete extra files and retain only keep_files
    print("Deleting extra files.")
    for p in out_dir.iterdir():
        if p.is_file() and p.name not in keep_set:
            try:
                p.unlink()
                print(f"Deleted {p.name}")
            except Exception as e:
                print(f"Warning: could not delete {p.name}: {e}")
    return out_dir

def load_kaggle_metadata(file_path:Path) -> pd.DataFrame:
    """
        Return metadata for all movies in movielens dataset as a dataframe.
        The metadata is for the entire movielens dataset and not just
        the smaller movielens dataset which is required for analysis.
    """
    try:
        metadata = pd.read_csv(file_path)
    except:
        raise FileNotFoundError(f"Could not find {file_path}")
    return metadata

def download_imdb_ratings(force_download:bool = False) -> Path:
    """
    Download IMDb ratings (title.ratings.tsv.gz), decompress in-memory,
    write 'title.ratings.tsv' under data/imdb/.
    Skips work if TSV already exists and force_download is False.
    Input: force_download - Boolean to indicate whether to force download even if files already exist.
    Returns the path of TSV file.
    """
    out_dir = config.DATA_DIR / "imdb"
    out_dir.mkdir(parents=True, exist_ok=True)

    tsv_path = out_dir / "title.ratings.tsv"
    if tsv_path.exists() and tsv_path.stat().st_size > 0 and not force_download:
        print("The IMDB TSV already exists. Skipping download.")
        return tsv_path

    resp = requests.get(config.IMDB_RATINGS_URL, timeout=120)
    resp.raise_for_status()

    # Decompress GZIP in memory and write TSV to disk
    with gzip.GzipFile(fileobj=io.BytesIO(resp.content)) as gz:
        data = gz.read()  # bytes of the TSV
    with open(tsv_path, "wb") as f:
        f.write(data)

    print("Downloaded IMDB ratings")
    return tsv_path

def load_imdb_ratings(file_path: Path) -> pd.DataFrame:
    """
        Return imdb ratings as a dataframe.
    """
    try:
        ratings = pd.read_csv(file_path, sep="\t")
    except:
        raise FileNotFoundError(f"Could not find {file_path}")
    ratings['tconst'] = ratings['tconst'].str.strip()
    return ratings