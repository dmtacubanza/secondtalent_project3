"""
Aggregate review-level data into product-level metrics and text blobs.
Creates intermediate temp tables:
- tmp_avg(product_id, avg_rating)
- tmp_sent(product_id, sentiment)
- tmp_concat(product_id, joined)
Final gold table is created here; narrative text is filled in 4.10.
"""

from .io_utils import conn

GOLD_TABLE = "gold_product_summary"


def run():
    # 1) Average rating per product from bronze_reviews
    conn().execute(
        """
        CREATE OR REPLACE TABLE tmp_avg AS
        SELECT product_id, AVG(rating) AS avg_rating
        FROM bronze_reviews
        GROUP BY 1;
        """
    )

    # 2) Majority-vote sentiment from per-review LLM outputs (portable ROW_NUMBER approach)
    conn().execute(
        """
        CREATE OR REPLACE TABLE tmp_sent AS
        WITH counts AS (
            SELECT product_id, sentiment, COUNT(*) AS cnt
            FROM silver_llm_outputs
            WHERE sentiment IS NOT NULL AND length(sentiment) > 0
            GROUP BY 1, 2
        ), ranked AS (
            SELECT product_id, sentiment, cnt,
                   ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY cnt DESC, sentiment) AS rnk
            FROM counts
        )
        SELECT product_id, sentiment
        FROM ranked
        WHERE rnk = 1;
        """
    )

    # 3) Concatenate up to 30 summaries per product for the product-level narrative LLM call
    conn().execute(
        """
        CREATE OR REPLACE TABLE tmp_concat AS
        WITH base AS (
            SELECT product_id, summary,
                   ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY random()) AS rn
            FROM silver_llm_outputs
            WHERE summary IS NOT NULL AND length(summary) > 0
        )
        SELECT product_id,
               string_agg(summary, '
') AS joined
        FROM base
        WHERE rn <= 30
        GROUP BY 1;
        """
    )

    # 4) Ensure final gold table exists
    conn().execute(
        f"CREATE TABLE IF NOT EXISTS {GOLD_TABLE} (product_id TEXT, avg_rating DOUBLE, sentiment TEXT, narrative_summary TEXT)"
    )

    return True