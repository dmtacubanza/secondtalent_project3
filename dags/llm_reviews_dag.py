from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

from src import ingest_kaggle, preprocess, llm_enrich, aggregate, orchestration, persist

DEFAULT_ARGS = {"owner": "data-eng", "retries": 1}

dag = DAG(
    dag_id="llm_reviews",
    default_args=DEFAULT_ARGS,
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
)

ingest = PythonOperator(
    task_id="ingest",
    python_callable=ingest_kaggle.run,
    op_kwargs={"csv_filename": "amazon_product_reviews.csv"},
    dag=dag,
)

prep = PythonOperator(
    task_id="preprocess",
    python_callable=preprocess.run,
    op_kwargs={"sample_fraction": 0.3},  # adjust to control LLM cost
    dag=dag,
)

review_llm = PythonOperator(
    task_id="llm_per_review",
    python_callable=llm_enrich.run,
    op_kwargs={"limit": None},
    dag=dag,
)

agg = PythonOperator(
    task_id="aggregate",
    python_callable=aggregate.run,
    dag=dag,
)

product_paragraphs = PythonOperator(
    task_id="product_paragraphs",
    python_callable=orchestration.build_product_paragraphs,
    dag=dag,
)

export = PythonOperator(
    task_id="export_gold",
    python_callable=persist.export_gold,
    dag=dag,
)

ingest >> prep >> review_llm >> agg >> product_paragraphs >> export