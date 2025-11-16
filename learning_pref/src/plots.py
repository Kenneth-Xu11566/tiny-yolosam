import matplotlib.pyplot as plt
from load_data import load_movielens_100k
from metrics import compute_popularity, compute_entropy

def plot_popularity_hist(popularity, outpath="figures/pop_hist.png"):
    plt.figure(figsize=(6.5, 3.5))
    popularity.plot(kind="hist", bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel("# ratings per movie")
    plt.ylabel("# movies")
    plt.title("(a) Popularity distribution")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    print(f"Saved {outpath}")

def plot_entropy_hist(entropy, outpath="figures/entropy_hist.png"):
    plt.figure(figsize=(6.5, 3.5))
    entropy.plot(kind="hist", bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel("Entropy (bits)")
    plt.ylabel("Frequency")
    plt.title("(b) Entropy distribution")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    print(f"Saved {outpath}")

def plot_entropy_vs_popularity(entropy, popularity, outpath="figures/entropy_vs_pop.png"):
    plt.figure(figsize=(6.5, 3.5))
    plt.scatter(entropy, popularity, s=8, alpha=0.5)
    plt.xlabel("Entropy (bits)")
    plt.ylabel("Popularity (# ratings)")
    plt.title("(c) Entropy vs Popularity")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    print(f"Saved {outpath}")

if __name__ == "__main__":
    # load data
    ratings, _ = load_movielens_100k("data/ml-latest-small")

    # metrics
    pop = compute_popularity(ratings)
    ent = compute_entropy(ratings)

    # plots
    plot_popularity_hist(pop)
    plot_entropy_hist(ent)
    plot_entropy_vs_popularity(ent, pop)

    print("Min/max entropy:", float(ent.min()), float(ent.max()))
