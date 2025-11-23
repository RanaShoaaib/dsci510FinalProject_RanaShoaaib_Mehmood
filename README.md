# Overview
This project implements an end-to-end, modular recommender system pipeline including:
- automated data download (MovieLens, Kaggle, IMDb)
- metadata processing and genre encoding
- exploratory data analysis with visualizations
- collaborative filtering model training (SVD and KNNBaseline)
- top-N recommendation generation for any user
- external validation using IMDb audience ratings
- a fully narrated Jupyter Notebook for analysis


# Project Structure
The repository is organized as follows.  
Note: `data/` and `results/` are **not tracked in Git** — they are generated automatically when running the project.

project/
│
├── data/                # (auto-created) downloaded datasets
├── results/             # (auto-created) saved plots, predictions, outputs
│
├── src/
│   ├── analysis.ipynb   # Full narrative analysis (EDA, modeling, validation)
│   ├── config.py        # Paths, URLs, environment setup
│   ├── load.py          # Downloading and loading datasets
│   ├── process.py       # Cleaning + merging metadata and ratings
│   ├── models.py        # SVD & KNN training, tuning, prediction
│   ├── analyze.py       # EDA plots and IMDb validation
│   ├── recommend.py     # Top-N recommendation functions
│   ├── main.py          # Full pipeline execution
│   └── tests.py         # Lightweight functional tests
│
├── requirements.txt
└── README.md


# Data sources
This project integrates three primary datasets: MovieLens ratings, Kaggle Movies Metadata, and IMDb ratings. These datasets collectively provide user–item interactions, rich movie attributes, and an external benchmark for recommendation evaluation.
| Dataset                    | Purpose                                           | Source        |
| -------------------------- | ------------------------------------------------- | ------------- |
| **MovieLens (100k)**       | User–movie ratings (core training data)           | GroupLens     |
| **Kaggle Movies Metadata** | Genres, runtime, budget, revenue, release date    | Kaggle API    |
| **IMDb Ratings (TSV)**     | Average rating + vote count (external validation) | IMDb Datasets |


# Results 
**Model Performance (RMSE)**

| Model           | CV RMSE | Test RMSE |
| --------------- | ------- | --------- |
| **SVD**         | ≈ 0.86  | ≈ 0.858  |
| **KNNBaseline** | ≈ 0.864 | ≈ 0.86   |

SVD slightly outperforms KNN in predictive accuracy.

**External Validation (Correlation with IMDb Ratings)**

| Model           | IMDb Correlation |
| --------------- | ---------------- |
| **SVD**         | **≈ 0.61**       |
| **KNNBaseline** | ≈ 0.54           |

SVD generalizes better and aligns more closely with real-world audience sentiment.

**Recommendation Comparison**

For three sample users examined, SVD and KNN delivered non-overlapping Top-N lists, highlighting:
- SVD captures latent preference patterns
- KNN emphasizes local item similarities
- The two models provide complementary recommendation styles with little overlap in recommendation for the three sample users


# Installation
This project uses the **Kaggle API** to download the Movies Metadata dataset.
1. Create a Kaggle account and generate an API token (`kaggle.json`) from your Kaggle account settings.
2. Set the following environment variables in a `.env` file:
     ```
     KAGGLE_USERNAME=your_kaggle_username
     KAGGLE_KEY=your_kaggle_api_key
     ```
MovieLens and IMDb data are downloaded from public URLs and do not require API keys; they are handled automatically by the project’s data-loading functions.
This project is written in Python and relies on the following main packages:
- `pandas`, `numpy` – data loading, cleaning, and manipulation  
- `requests` – HTTP requests for downloading external datasets  
- `kaggle` – Kaggle API client for programmatic dataset download  
- `matplotlib`, `seaborn` – exploratory data analysis and visualization  
- `scikit-learn` – modeling and evaluation (e.g., KNN, metrics)
- `python-dotenv` – loading environment variables from `.env`
Install all required packages via: pip install -r requirements.txt


# Jupyter Notebook
The complete narrative analysis—including EDA, model training, recommendation visualization, and external validation—is available in:
`src/analysis.ipynb`
This notebook presents the project as a story with explanations, plots, and interpretations.


# Running analysis 
From `src/` directory run:
`python main.py`
Results will appear in `results/` folder. All data obtained will be stored in `data/`


# Running tests 
From `src/` directory run:
`python tests.py`
This runs lightweight functional tests