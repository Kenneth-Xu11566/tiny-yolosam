import pandas as pd
import numpy as np
from load_data import load_movielens_100k

def compute_popularity(ratings: pd.DataFrame) -> pd.Series:
    """# of ratings per movieId (descending)."""
    pop = ratings.groupby("movieId").size().sort_values(ascending=False)
    pop.name = "popularity"
    return pop

def _entropy_one(group: pd.Series, rating_values=None) -> float:
    """
    Shannon entropy (base 2) for the ratings of a single movie.
    """
    if rating_values is None:
        rating_values = np.arange(0.5, 5.5, 0.5)

    counts = np.array([(group == v).sum() for v in rating_values], dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]   
    return float(-(p * np.log2(p)).sum())

def compute_entropy(ratings: pd.DataFrame) -> pd.Series:
    """Entropy for each movieId."""
    ent = ratings.groupby("movieId")["rating"].apply(_entropy_one)
    ent.name = "entropy"
    return ent

if __name__ == "__main__":
    ratings, movies = load_movielens_100k("data/ml-latest-small")

    pop = compute_popularity(ratings)
    top5_ids = pop.head(5).index
    print("\nTop 5 titles:")
    print(movies[movies["movieId"].isin(top5_ids)][["movieId","title"]])

    ent = compute_entropy(ratings)
    print("\nEntropy min/max:")
    print(ent.min(), ent.max())

    print("\nToy example (Section 3.2):")

    # Case 1: 2000 ratings, evenly spread across 1–5
    ratings_2000 = pd.Series([1,2,3,4,5] * 400)  # 2000 total
    H_2000 = _entropy_one(ratings_2000, rating_values=[1,2,3,4,5])

    # Case 2: 5 ratings, one of each 1–5
    ratings_5 = pd.Series([1,2,3,4,5])
    H_5 = _entropy_one(ratings_5, rating_values=[1,2,3,4,5])

    print(f"Entropy with 2000 evenly spread ratings: {H_2000:.3f} ")
    print(f"Entropy with 5 evenly spread ratings   : {H_5:.3f} ")
