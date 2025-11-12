import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from config import RESULTS_DIR as outdir


def generate_plots(meta_merged: pd.DataFrame, ml_ratings: pd.DataFrame):
    """
    Generates analysis plots and saves them to the results directory.
    """
    # Plot 1): Movies rated per user
    # --------------------------------
    user_counts = ml_ratings.count(axis=1).values
    plt.figure(figsize=(10, 6), dpi=120)
    # statistical markers
    q25 = np.percentile(user_counts, 25)
    med = np.percentile(user_counts, 50)
    q75 = np.percentile(user_counts, 75)
    plt.hist(user_counts, bins=50)
    plt.title('Distribution of Movies Rated per User (MovieLens 100K)')
    plt.xlabel('Number of Movies Rated')
    plt.ylabel('Number of Users')
    # vertical lines for statistical markers
    plt.axvline(med, linestyle='-', color='red', linewidth=1.6, label=f'Median = {med:.0f}')
    plt.axvline(q25, linestyle=':', color='orange', linewidth=1.6, label=f'25th = {q25:.0f}')
    plt.axvline(q75, linestyle=':', color='orange', linewidth=1.6, label=f'75th = {q75:.0f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "user_activity_hist.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Plot 2): Movies count by release year
    # --------------------------------
    plt.figure(figsize=(10, 6), dpi=120)
    sns.countplot(meta_merged, x='year')
    plt.xticks(rotation=90, size=5)
    plt.title("Distribution of Movies by Release Year")
    plt.tight_layout()
    plt.savefig(outdir / "movies_by_year.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Plot 3): Distribution of Movies by Original Language
    # --------------------------------
    plt.figure(figsize=(10, 6), dpi=120)
    lang_order = (meta_merged['original_language'].value_counts().index)
    sns.countplot(meta_merged, x='original_language', order=lang_order)
    plt.xticks(rotation=90, size=10)
    plt.title("Distribution of Movies by Original Language")
    plt.tight_layout()
    plt.savefig(outdir / "movies_by_lang.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Plot 4): Movies count per genre
    # --------------------------------
    non_genre = {'movieId', 'tmdbId', 'imdbId', 'title', 'original_language', 'release_date', 'year', 'runtime',
                 'budget', 'revenue', 'imdb_averageRating', 'imdb_numVotes'}
    genre_cols = [c for c in meta_merged.columns if c not in non_genre and meta_merged[c].dropna().isin([0, 1]).all()]
    genre_counts = meta_merged[genre_cols].sum().sort_values(ascending=True)
    # horizontal barplot (sorted)
    plt.figure(figsize=(10, 6), dpi=120)
    sns.barplot(x=genre_counts.values, y=genre_counts.index, orient='h')
    plt.title('Movie Counts by Genre')
    plt.xlabel('Number of Movies')
    plt.ylabel('Genre')
    plt.tight_layout()
    plt.savefig(outdir / "movie_by_genre.png", dpi=120, bbox_inches="tight")
    plt.close()
