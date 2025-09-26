import duckdb
con = duckdb.connect("data/warehouse.duckdb")

print(con.execute("SHOW TABLES;").fetchall())

#print(con.execute("SELECT * FROM silver_reviews;").fetchall())

#print(con.execute("SELECT * FROM silver_llm_outputs;").fetchall())

#print(con.execute("SELECT * FROM silver_product_sentiment;").fetchall())

#print(con.execute("SELECT * FROM tmp_sent;").fetchall())

#print(con.execute("SELECT * FROM tmp_avg;").fetchall())

#print(con.execute("SELECT * FROM tmp_concat;").fetchall())

#print(con.execute("SELECT a.product_id, a.avg_rating, s.sentiment, c.joined FROM tmp_avg a JOIN tmp_sent s USING (product_id) JOIN tmp_concat c USING (product_id)").fetchall())

print(con.execute("SELECT * FROM gold_product_summary;").fetchall())