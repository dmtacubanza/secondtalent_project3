import pandas as pd
from pathlib import Path
from .io_utils import read_csv_to_parquet, write_df

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
BRONZE = "bronze_reviews"

EXPECTED_COLS = {"Id","ProductId","UserId","ProfileName","HelpfulnessNumerator","HelpfulnessDenominator","Score","Time","Summary","Text"}

def run(csv_filename: str):
    raw_path = RAW_DIR / csv_filename
    if not raw_path.exists():
        raise FileNotFoundError(f"Place {csv_filename} under data/raw/")

    parquet_path = str(PROCESSED_DIR / f"{raw_path.stem}.parquet")
    read_csv_to_parquet(str(raw_path), parquet_path)

    df = pd.read_parquet(parquet_path)
    missing = EXPECTED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.rename(columns={
        "Id":"review_id","ProductId":"product_id","UserId":"user_id","ProfileName":"profile_name",
        "HelpfulnessNumerator":"helpful_numer","HelpfulnessDenominator":"helpful_denom",
        "Score":"rating","Time":"review_time","Summary":"summary_title","Text":"review_text",
    })

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce"); df = df[df["rating"].notna()]
    df["review_text"] = df["review_text"].fillna("").astype(str); df = df[df["review_text"].str.strip() != ""]

    try:
        df["review_time_iso"] = pd.to_datetime(df["review_time"], unit="s", utc=True)
    except Exception:
        df["review_time_iso"] = pd.to_datetime(df["review_time"], unit="ms", utc=True)

    df["helpful_numer"] = pd.to_numeric(df["helpful_numer"], errors="coerce").fillna(0).astype(int)
    df["helpful_denom"] = pd.to_numeric(df["helpful_denom"], errors="coerce").fillna(0).astype(int)
    df["helpful_ratio"] = df.apply(lambda r: (r.helpful_numer/r.helpful_denom) if r.helpful_denom>0 else None, axis=1)

    cols = ["review_id","product_id","user_id","profile_name","review_text","rating","review_time_iso","summary_title","helpful_numer","helpful_denom","helpful_ratio"]
    df = df[cols]

    write_df(BRONZE, df, mode="replace")
    return len(df)