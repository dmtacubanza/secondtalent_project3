"""
Preprocess reviews before sending them to the LLM.
- Deduplicate reviews per product/review_id/text (keep latest by review_time_iso)
- Optionally sample a fraction of rows to control API cost
- Output: silver_reviews
# """

from .io_utils import conn

SILVER_IN = "bronze_reviews"
SILVER_OUT = "silver_reviews"  # review-level input to LLM


def run(sample_fraction: float | None = None):
    q = f"""
    CREATE OR REPLACE TABLE {SILVER_OUT} AS
    SELECT *
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY product_id, review_id, review_text
                   ORDER BY review_time_iso DESC
               ) AS rn
        FROM {SILVER_IN}
    )
    WHERE rn = 1;
    """
    conn().execute(q)

    if sample_fraction:
        conn().execute(f"DELETE FROM {SILVER_OUT} WHERE random() > {sample_fraction}")