"""App settings. Values come from environment variables or a .env file (see .env.example)."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    DENSE_VECTOR_NAME: str = "dense"
    SPARSE_VECTOR_NAME: str = "sparse"

    # Data locations
    corpus_dir: Path = Path("corpus")
    qdrant_path: Path = Path("data/qdrant")  # embedded Qdrant: just a folder
    qdrant_collection: str = "helix_support"
    ingestion_report_path: Path = Path("evals/results/ingestion_report.json")

    # Models
    ollama_base_url: str = "http://localhost:11434"
    dense_model: str = "qwen3-embedding:0.6b"  # runs in Ollama
    sparse_model: str = (
        "Qdrant/bm25"  # fastembed BM25, the keyword side of hybrid search
    )

    # Retrieval
    retrieval_k: int = 6  # chunks handed to the LLM

    # Generation
    generation_model: str = "qwen3:8b"  # runs in Ollama
    num_ctx: int = (
        8192  # context window: must fit the retrieved chunks, the prompt and the answer
    )
    max_answer_tokens: int = 1024


settings = Settings()
