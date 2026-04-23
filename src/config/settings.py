from typing import Optional, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    base_dir: Path = Path(__file__).parent.parent.parent
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    raw_data_dir: Path = Field(default_factory=lambda: Path("data/raw"))
    processed_data_dir: Path = Field(default_factory=lambda: Path("data/processed"))

    document_name: str = "popatkus.docx"
    document_type: Literal["pdf", "docx", "txt"] = "docx"

    default_chunk_size: int = 768
    default_chunk_overlap: int = 64

    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_device: Literal["cpu", "cuda"] = "cpu"

    chroma_persist_dir: Path = Field(default_factory=lambda: Path("chroma_db"))
    testset_size: int = 50

    log_level: str = "INFO"

    # Telegram Bot Configuration
    bot_token: str
    collection_name: str = "popatkus_semantic"

    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "deepseek-r1:14b"
    ollama_temperature: float = 0.6
    ollama_context_size: int = 4096

    # Retrieval Configuration
    top_k_initial: int = 10
    top_k_final: int = 3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_dir.mkdir(exist_ok=True)
        self.raw_data_dir.mkdir(exist_ok=True, parents=True)
        self.processed_data_dir.mkdir(exist_ok=True, parents=True)
        self.chroma_persist_dir.mkdir(exist_ok=True, parents=True)


settings = Settings()
