from __future__ import annotations
import pandas as pd
from pathlib import Path

def load_movielens_100k(data_dir: str | Path):
    """
    Load MovieLens ratings and movies CSVs from the given directory.
    """
    data_dir = Path(data_dir)
    ratings = pd.read_csv(data_dir / "ratings.csv")
    movies  = pd.read_csv(data_dir / "movies.csv")
    return ratings, movies

if __name__ == "__main__":
    ratings, movies = load_movielens_100k("data/ml-latest-small")
    print("ratings:", ratings.shape)
    print("movies :", movies.shape)
    print(ratings.head(2))
    print(movies.head(2))
