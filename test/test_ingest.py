from src.ingest_kaggle import EXPECTED_COLS
import pandas as pd

def test_expected_columns():
    assert {"product_id", "review_text", "rating"}.issubset(EXPECTED_COLS)