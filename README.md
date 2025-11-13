# Movie Recommender System
This project develops a movie recommender system using the MovieLens ratings dataset enriched with Kaggle movie metadata and IMDb ratings. It performs exploratory data analysis and builds collaborative filtering models—including Matrix Factorization and KNN—to generate personalized movie recommendations. The integrated IMDb ratings serve as an external benchmark to evaluate prediction quality.


# Data sources
This project integrates three primary datasets: MovieLens ratings, Kaggle Movies Metadata, and IMDb ratings. These datasets collectively provide user–item interactions, rich movie attributes, and an external benchmark for recommendation evaluation.


# Results 
_describe your findings_


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


# Running analysis 
From `src/` directory run:
`python main.py`
Results will appear in `results/` folder. All data obtained will be stored in `data/`