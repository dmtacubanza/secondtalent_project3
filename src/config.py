from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai").lower()
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    hf_api_key: str | None = os.getenv("HF_API_KEY")
    hf_model: str = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")

    max_tokens: int = int(os.getenv("MAX_TOKENS", 512))
    temperature: float = float(os.getenv("TEMPERATURE", 0.2))
    batch_size: int = int(os.getenv("BATCH_SIZE", 16))
    qps: float = float(os.getenv("RATE_LIMIT_QPS", 2))

    cache_dir: str = os.getenv("CACHE_DIR", ".cache")
    duckdb_path: str = os.getenv("DUCKDB_PATH", "./data/warehouse.duckdb")

CFG = Settings()