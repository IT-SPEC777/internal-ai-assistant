import os

from dotenv import load_dotenv
from pydantic import BaseModel

# Без этого вызова .env из README не подхватился бы.
load_dotenv()


class Settings(BaseModel):
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "internal_docs")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    embed_model: str = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
    users_file: str = os.getenv("USERS_FILE", "data/users.json")
    top_k_retrieve: int = int(os.getenv("TOP_K_RETRIEVE", "15"))
    top_k_final: int = int(os.getenv("TOP_K_FINAL", "4"))
    min_dense_score: float = float(os.getenv("MIN_DENSE_SCORE", "0.82"))
    enable_reranker: bool = os.getenv("ENABLE_RERANKER", "false").lower() == "true"
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")


settings = Settings()