from pathlib import Path
from dotenv import load_dotenv

# project configuration from .env (secret part)
env_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path=env_path)  # loads into os.environ

# project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR    = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# data sources configuration
ML_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
KAGGLE_DATASET_SLUG = "rounakbanik/the-movies-dataset"
IMDB_RATINGS_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"

# Ensure folders exist; harmless if already present
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)