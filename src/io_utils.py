import duckdb
import pandas as pd
from pathlib import Path
from .config import CFG

# Create folders
for p in ["data/processed","data/curated","data/enriched"]:
    Path(p).mkdir(parents=True, exist_ok=True)

_conn = duckdb.connect(CFG.duckdb_path)

def conn():
    return _conn

def read_csv_to_parquet(csv_path: str, out_path: str):
    df = pd.read_csv(csv_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path

def write_df(table: str, df: pd.DataFrame, mode: str = "append", chunk_rows: int = 100_000):
    """
    mode='replace'  -> CREATE OR REPLACE TABLE ... AS SELECT * FROM _df
    mode='append'   -> chunked INSERTs to keep memory bounded
    """
    # Make sure table exists if we'll append later
    if mode == "append":
        cols = ",".join([f"{c} {duck_type(df[c])}" for c in df.columns])
        _conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({cols});")

    if mode == "replace":
        _conn.register("_df", df)
        _conn.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _df;")
        _conn.unregister("_df")
        return

    # Append in chunks
    start = 0
    n = len(df)
    while start < n:
        end = min(start + chunk_rows, n)
        _conn.register("_df_chunk", df.iloc[start:end])
        _conn.execute(f"INSERT INTO {table} SELECT * FROM _df_chunk;")
        _conn.unregister("_df_chunk")
        start = end

_TYPE_MAP = {"int64":"BIGINT","float64":"DOUBLE","object":"TEXT","datetime64[ns]":"TIMESTAMP","bool":"BOOLEAN"}

def duck_type(series):
    return _TYPE_MAP.get(str(series.dtype), "TEXT")