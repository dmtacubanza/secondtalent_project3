from __future__ import annotations
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
from .io_utils import conn

PRODUCT_OUT_TABLE = "silver_product_sentiment"
REVIEW_OUT_TABLE  = "silver_llm_outputs"

# --- utils ------------------------------------------------------------

def _df_or_empty(sql: str) -> pd.DataFrame:
    try:
        return conn().execute(sql).df()
    except Exception:
        return pd.DataFrame()

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# --- 1) Leaderboard: which products are liked the most? ---------------
def plot_product_leaderboard(
    out_dir: str = "reports/charts",
    top_n: int = 15,
):
    """
    Bar chart of top-N products by mean_pos_prob (0..1).
    Encodes model confidence as text labels.
    """
    _ensure_dir(out_dir)
    df = _df_or_empty(f"""
        SELECT product_id, mean_pos_prob, confidence, n_reviews
        FROM {PRODUCT_OUT_TABLE}
        WHERE n_reviews > 0
    """)
    if df.empty:
        print("[viz] No product rows to plot.")
        return

    df = df.sort_values("mean_pos_prob", ascending=False).head(top_n)
    x = range(len(df))

    plt.figure(figsize=(10, 6))
    plt.bar(x, df["mean_pos_prob"])
    plt.xticks(list(x), df["product_id"], rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.title(f"Top {min(top_n, len(df))} Products by Mean Positive Probability")
    plt.xlabel("Product")
    plt.ylabel("Mean Positive Probability")

    # annotate confidence & n
    for i, (mpp, conf, n) in enumerate(zip(df["mean_pos_prob"], df["confidence"], df["n_reviews"])):
        plt.text(i, mpp + 0.02, f"conf {conf:.2f} • n={int(n)}", ha="center", va="bottom")

    out_path = os.path.join(out_dir, "1_product_leaderboard.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[viz] saved {out_path}")

# --- 2) Evidence vs opinion: volume vs sentiment (bubble scatter) -----
def plot_volume_vs_sentiment(
    out_dir: str = "reports/charts",
    label_top_k: int = 10,
):
    """
    Scatter: x = mean_pos_prob, y = n_reviews, bubble size ~ confidence.
    Helps answer: do we like it (x), and do we have enough reviews (y)?
    """
    _ensure_dir(out_dir)
    df = _df_or_empty(f"""
        SELECT product_id, mean_pos_prob, confidence, n_reviews
        FROM {PRODUCT_OUT_TABLE}
        WHERE n_reviews >= 1
    """)
    if df.empty:
        print("[viz] No product rows to plot.")
        return

    sizes = (df["confidence"].clip(0, 1) * 800) + 50  # bubble area

    plt.figure(figsize=(10, 6))
    plt.scatter(df["mean_pos_prob"], df["n_reviews"], s=sizes, alpha=0.6)
    plt.xlabel("Mean Positive Probability (0..1)")
    plt.ylabel("Number of Reviews")
    plt.title("Review Volume vs. Sentiment (bubble size = confidence)")
    plt.xlim(0, 1)

    # annotate top-k by volume
    df_large = df.sort_values("n_reviews", ascending=False).head(label_top_k)
    for _, row in df_large.iterrows():
        plt.annotate(
            str(row["product_id"]),
            (row["mean_pos_prob"], row["n_reviews"]),
            xytext=(5, 5),
            textcoords="offset points"
        )

    # optional quadrant guide at 0.6 (pos_threshold)
    plt.axvline(0.60, linestyle="--", linewidth=1)
    plt.text(0.605, plt.ylim()[1]*0.95, "pos_threshold", va="top")

    out_path = os.path.join(out_dir, "2_volume_vs_sentiment.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[viz] saved {out_path}")

# --- 3) Composition: pos/neu/neg shares per product (stacked) ---------
def plot_sentiment_composition(
    out_dir: str = "reports/charts",
    top_n_by_reviews: int = 12,
):
    """
    Stacked bar of frac_pos_pred, frac_neu_pred, frac_neg_pred for top-N products by n_reviews.
    """
    _ensure_dir(out_dir)
    df = _df_or_empty(f"""
        SELECT product_id, n_reviews, frac_pos_pred, frac_neg_pred, mean_pos_prob
        FROM {PRODUCT_OUT_TABLE}
        WHERE n_reviews > 0
    """)
    if df.empty:
        print("[viz] No product rows to plot.")
        return

    # compute neutral as remainder to avoid rounding drift
    df["frac_neu_pred"] = (1.0 - df["frac_pos_pred"] - df["frac_neg_pred"]).clip(lower=0.0, upper=1.0)
    df = df.sort_values("n_reviews", ascending=False).head(top_n_by_reviews).reset_index(drop=True)

    x = range(len(df))
    pos = df["frac_pos_pred"]
    neu = df["frac_neu_pred"]
    neg = df["frac_neg_pred"]

    plt.figure(figsize=(10, 6))
    plt.bar(x, pos)
    plt.bar(x, neu, bottom=pos)
    plt.bar(x, neg, bottom=pos+neu)
    plt.xticks(list(x), df["product_id"], rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.xlabel("Product (Top by n_reviews)")
    plt.ylabel("Share of Predictions")
    plt.title("Sentiment Composition per Product (pos/neu/neg)")

    # annotate with n and mean_pos_prob on top
    for i, (n, mpp) in enumerate(zip(df["n_reviews"], df["mean_pos_prob"])):
        plt.text(i, 1.02, f"n={int(n)} • mpp={mpp:.2f}", ha="center", va="bottom")

    out_path = os.path.join(out_dir, "3_sentiment_composition.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[viz] saved {out_path}")

# --- entry ------------------------------------------------------------
def render_all(out_dir: str = "reports/charts"):
    plot_product_leaderboard(out_dir)
    plot_volume_vs_sentiment(out_dir)
    plot_sentiment_composition(out_dir)

if __name__ == "__main__":
    from .visualization import render_all
    render_all("reports/charts")