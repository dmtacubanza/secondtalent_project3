# secondtalent_project3# Data Engineering Assignments

This assessment is designed to evaluate your proficiency in **data engineering** through three distinct, hands-on projects.  
You will have **5 official days** to complete this comprehensive assessment.

The core objective across all projects is to demonstrate your ability to **design, implement, and optimize data pipelines** for:

- **Data ingestion**
- **Data transformation**
- **Data analysis**

Ultimately, your solution should address **specific business questions**.

You have the freedom to choose your preferred **tools and technologies**, including (but not limited to):

- [Apache Airflow](https://airflow.apache.org/)
- Various **AWS services** (utilizing the free tier is acceptable)
- Any other **relevant data engineering platforms** or frameworks

---

## Requirements

- You are required to complete **at least one project** from the three provided.
- Deliverables should include the following:

### 1. Source Code
Provide all scripts, notebooks, or workflows you created to implement your solution.  
Organize the codebase clearly (e.g., by modules, pipelines, or DAGs).

### 2. Documentation
Include instructions for:

- How to run your code
- Project architecture overview
- Key design decisions
- Any setup or dependencies required

### 3. Presentation / Walkthrough
Be prepared to explain your solution and decisions during a **video interview**.

---

## Suggested Workflow

1. **Understand the business questions** each project aims to solve.  
2. **Design your data pipeline architecture** (ingestion → transformation → analysis).  
3. **Implement the solution** using your chosen tools and technologies.  
4. **Test and optimize** for performance and scalability.  
5. **Document** your work clearly and prepare to present it.

---

## Recommended Tools

- **Workflow Orchestration:** Apache Airflow, AWS Step Functions, Prefect  
- **Data Storage:** S3, Redshift, Snowflake, BigQuery  
- **Processing:** AWS Glue, Spark, dbt, Pandas  
- **Monitoring:** CloudWatch, Prometheus + Grafana  
- **Version Control:** GitHub / GitLab

---

## Notes

- Using **free-tier cloud services** is encouraged where possible.  
- Focus on **clarity**, **scalability**, and **best practices** in your implementation.  
- Prepare to **justify your architectural and tool choices** during the interview.

# Project 3 — Pipeline that Integrates LLM Calls (Product Review Summary)

## Goal

This advanced project requires you to create a **data pipeline** that integrates calls to a **Large Language Model (LLM)** to automate the **summarization of product reviews** and perform **sentiment analysis**.

---

## Data Ingestion

- Ingest the **Amazon Product Reviews** dataset from Kaggle:  
  [https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews)
- Your pipeline should **efficiently load** these reviews for downstream processing.

---

## LLM Integration

- Integrate an LLM into your pipeline to process product reviews.
- Possible options include:
  - [OpenAI GPT models](https://platform.openai.com/docs/)
  - [Google Gemini](https://ai.google.dev/)
  - [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index)
- Your integration should be:
  - **Robust:** Handle multiple reviews (batch or single).
  - **Fault-tolerant:** Retry or gracefully skip on API errors/timeouts.
  - **Efficient:** Use batching or concurrency where appropriate.

---

## Data Analysis via LLM Output

Use the LLM’s output to:

1. **Generate concise summaries** of product reviews.
2. **Perform sentiment analysis** for each product.

This requires parsing the LLM response to extract key elements (summary text, sentiment classification).

---

## Key Questions to Address

1. **Narrative Product Review Summary**  
   For each unique product, generate a **single paragraph** summarizing:
   - Key themes
   - Pros
   - Cons  
   *(Use the LLM to synthesize this narrative.)*

2. **Average Product Rating Score**  
   Calculate the **average numerical rating** for each product using the dataset’s star ratings.

3. **Product Review Sentiment Analysis**  
   For each unique product, determine the **overall sentiment** of its reviews:  
   - `Positive`  
   - `Negative`  
   - `Neutral`  
   *(Derived from the LLM’s analysis of the review text.)*

---

## Code Quality Guidelines

Your code must demonstrate:

- **Well-structured design:** Use modules, classes, and functions where appropriate.
- **Readable naming:** Clear, descriptive variable and function names.
- **Meaningful comments:** Explain complex logic and API integrations.
- **Error handling:** Gracefully manage issues during ingestion, transformation, or LLM calls.

---

## Documentation Expectations

Provide **clear, comprehensive documentation** including:

- **Design choices:** Architecture, tools, schema, and reasoning.
- **Setup instructions:** Environment creation, dependency installation, API key configuration.
- **Execution guide:** How to run the pipeline (CLI commands, environment variables).
- **Assumptions:** Data quality, API behavior, or other project-specific notes.

---

## Problem-Solving & Scalability

- **Show your reasoning:** Describe how you addressed issues like:
  - Data inconsistencies
  - API rate limits or failures
  - Performance bottlenecks
- **Scalability (optional but highly recommended):**  
  Explain how this pipeline could handle **larger datasets** or higher data volume in production.

---

## Deliverables

Your final submission must include:

- **Source code** — organized in a **Git repository** (clear folder structure).
- **Documentation** — as described above.
- *(Be prepared to walk through your design and choices in a **video interview**.)*

---

### Suggested Folder Structure

project-3-llm-pipeline/
│── src/
│ ├── ingestion/
│ ├── processing/
│ ├── llm_integration/
│ └── utils/
│── notebooks/
│── requirements.txt
│── README.md
└── docs/
├── architecture-diagram.png
└── setup_instructions.md

---

## Tips

- Use **environment variables** or `.env` files for API keys.
- Consider **caching** or **storing API responses** to avoid repeated calls during testing.
- Handle **rate limits** and **timeout errors** gracefully.
- Document **cost considerations** if using paid LLM APIs.

## Scripts

python -c "from src import ingest_kaggle; ingest_kaggle.run('amazon_product_reviews.csv')"
python -c "from src import preprocess; preprocess.run(0.2)"
python -c "from src import llm_enrich; llm_enrich.run(limit=50)"
python -c "from src import aggregate; aggregate.run()"
python -c "from src import orchestration; orchestration.build_product_paragraphs()"
python -c "from src import persist; persist.export_gold()"

# secondtalent_project3