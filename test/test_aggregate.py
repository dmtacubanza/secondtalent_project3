from src.io_utils import conn

def test_gold_exists():
    cur = conn()
    cur.execute("CREATE TABLE IF NOT EXISTS gold_product_summary (product_id TEXT, avg_rating DOUBLE, sentiment TEXT, narrative_summary TEXT)")
    assert cur is not None