import pandas as pd
from .io_utils import conn, write_df
from .llm_client import LLMClient
from .prompts import PRODUCT_PROMPT

GOLD_TABLE = "gold_product_summary"


def build_product_paragraphs():
    df = conn().execute("SELECT a.product_id, a.avg_rating, s.sentiment, c.joined FROM tmp_avg a JOIN tmp_sent s USING (product_id) JOIN tmp_concat c USING (product_id)").df()

    client = LLMClient()
    rows = []
    for _, r in df.iterrows():
        prompt = PRODUCT_PROMPT.format(reviews=r["joined"][:6000])
        text = client.complete_text(prompt)
        rows.append({
            "product_id": r["product_id"],
            "avg_rating": float(r["avg_rating"]),
            "sentiment": r["sentiment"],
            "narrative_summary": text.strip(),
        })

    out = pd.DataFrame(rows)
    write_df(GOLD_TABLE, out, mode="replace")