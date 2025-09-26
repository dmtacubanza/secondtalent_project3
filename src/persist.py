# optional: export gold to parquet/csv for BI
import pandas as pd
from .io_utils import conn

def export_gold():
    df = conn().execute("SELECT * FROM gold_product_summary").df()
    df.to_parquet("data/enriched/product_summary.parquet", index=False)
    df.to_csv("data/enriched/product_summary.csv", index=False)