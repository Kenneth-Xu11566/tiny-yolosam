# src/user_effort.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from load_data import load_movielens_100k

def eligible_users(ratings: pd.DataFrame, min_ratings: int = 80) -> pd.Index:
    counts = ratings.groupby("userId").size()
    return counts[counts >= min_ratings].index

def user_effort_curve(ratings: pd.DataFrame, Ns=(15, 30, 45, 60, 75)) -> pd.DataFrame:
    # Global popularity ranking (desc)
    pop_order = ratings.groupby("movieId").size().sort_values(ascending=False).index.to_list()

    users = eligible_users(ratings, 80)
    # Precompute each user's rated set
    user2movies = ratings.groupby("userId")["movieId"].apply(set)

    rows = []
    for N in Ns:
        topN = set(pop_order[:N])
        seen_counts = [len(user2movies[u] & topN) for u in users]
        rows.append({
            "N": N,
            "mean_seen": float(np.mean(seen_counts)),
            "median_seen": float(np.median(seen_counts)),
            "q25": float(np.percentile(seen_counts, 25)),
            "q75": float(np.percentile(seen_counts, 75)),
            "num_users": int(len(users)),
        })
    return pd.DataFrame(rows)

def plot_user_effort(df: pd.DataFrame, outpath="figures/user_effort.png"):
    plt.figure(figsize=(6.5, 3.5))
    plt.plot(df["N"], df["mean_seen"], marker="o")
    #IQR error band
    plt.fill_between(df["N"], df["q25"], df["q75"], alpha=0.2)
    plt.xlabel("# of movies presented (N)")
    plt.ylabel("# of movies seen (mean across users)")
    plt.title("User effort curve (Popularity)")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    print(f"Saved {outpath}")

if __name__ == "__main__":
    ratings, _ = load_movielens_100k("data/ml-latest-small")
    users = eligible_users(ratings, 80)
    print("Eligible users (>=80 ratings):", len(users))
    df = user_effort_curve(ratings, Ns=(15, 30, 45, 60, 75))
    print(df)
    plot_user_effort(df)
