from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import socket

load_dotenv()

def _parse_bool(value: str | None, default: bool = False) -> bool:
    """Helper DRY per il parsing uniforme dei booleani dalle variabili d'ambiente."""
    if not value:
        return default
    return str(value).strip().lower() in {"true", "1", "yes", "y", "on"}

def _parse_set(value: str | None, default: set[str]) -> set[str]:
    """Helper DRY per il parsing di liste comma-separated dalle variabili d'ambiente in set di stringhe."""
    if not value:
        return default
    return {v.strip() for v in value.split(",") if v.strip()}

@dataclass(frozen=True)
class Settings:
    # App Config
    http_timeout_seconds: int
    data_folder: Path
    log_level: str
    
    # RabbitMQ
    broker_host: str
    broker_port: int
    broker_username: str
    broker_password: str
    broker_max_attempts: int

    # DocWSRicerche
    docws_ricerca_endpoint: str
    docws_atti_endpoint: str
    docws_codice_amministrazione: str
    docws_codice_aoo: str
    ws_username: str
    ws_password: str
    ruolo_docws: str

    verify_tls: bool
    soap_version: str

    db_host: str
    db_port: int
    db_user: str
    db_pass: str
    db_name: str

    ocr_model_name: str

    api_llm_key: str
    api_secret_key: str

    hostname: str

    qdrant_host: str
    qdrant_port: int

    allow_subtopic_selection: bool

    allowed_origins: list[str]

    max_history_items: int

    redis_host: str
    redis_port: int

    allow_general_knowledge: bool

    rerank_size: int
    rerank_truncate: int
    rerank_batch_size: int
    rerank_max_length: int
    answer_max_tokens: int
    grader_max_tokens: int
    qdrant_syntactic_size: int
    qdrant_semantic_size: int
    qdrant_semantic_threshold: float
    qdrant_concept_threshold: float
    max_context_chars: int
    max_sub_queries: int
    parent_budget_ratio: float
    onnx_model_cache_path: str

    min_prob_threshold: float
    parents_per_query: int

    mmr_similarity_threshold: float

    query_rewriter_model_name: str
    answer_generator_model_name: str
    grader_model_name: str

    llm_provider: str

    protected_keys: set[str]

    max_reranker_thread: int
    reranker_pool_size: int
    max_allowed_pages: int
    max_model_retries: int
    qdrant_concept_max_hits: int

    id_tipo_iter_atteso: dict[str, set[str]]

    answer_thinking_level: str
    grader_thinking_level: str


def load_settings() -> Settings:
    # L'operatore "or" protegge dalle stringhe vuote. 
    # Es: se HTTP_TIMEOUT_SECONDS="", os.getenv() o 30 restituisce 30.

    allowed_origins_raw = os.environ.get("ALLOWED_ORIGINS")
    allowed_origins = allowed_origins_raw.split(",") if allowed_origins_raw else ["*"]
    
    return Settings(
        http_timeout_seconds=int(os.getenv("HTTP_TIMEOUT_SECONDS") or 300),
        data_folder=Path(os.getenv("DATA_FOLDER") or Path(__file__).parent.resolve()),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        
        broker_host=os.getenv("BROKER_HOST", "rabbitmq-service.rag.svc.cluster.local"),
        broker_port=int(os.getenv("BROKER_PORT") or 5672),
        broker_username=os.getenv("BROKER_USERNAME", "guest"),
        broker_password=os.getenv("BROKER_PASSWORD", "guest"),
        broker_max_attempts = int(os.environ.get("BROKER_MAX_ATTEMPTS") or 30),
        
        docws_ricerca_endpoint=os.environ["DOCWS_RICERCA_ENDPOINT"],
        docws_atti_endpoint=os.environ["DOCWS_ATTI_ENDPOINT"],
        ws_username=os.getenv("WS_USERNAME", "").strip() or "sicraweb",
        ws_password=os.getenv("WS_PASSWORD", "").strip() or "sicraweb",
        docws_codice_amministrazione=os.environ["DOCWS_CODICE_AMMINISTRAZIONE"],
        docws_codice_aoo=os.environ["DOCWS_CODICE_AOO"],
        ruolo_docws=os.environ["RUOLO_DOCWS"],

        # Valuta correttamente i booleani confrontando la stringa
        verify_tls=_parse_bool(os.getenv("VERIFY_TLS"), default=True),
        soap_version=os.getenv("SOAP_VERSION", "1.1"),

        db_host = os.environ.get("MYSQL_SERVICE_HOST", "localhost"),
        db_port = int(os.environ.get("MYSQL_SERVICE_PORT") or 3306),
        db_user = os.environ.get("MYSQL_USER", "raguser"),
        db_pass = os.environ.get("MYSQL_PASSWORD", ""),
        db_name = os.environ.get("MYSQL_DATABASE", "rag_db"),

        ocr_model_name = os.environ.get("OCR_MODEL_NAME", "gemini-3-flash-preview"),

        api_llm_key = os.environ.get("API_LLM_KEY", "default_key"),
        api_secret_key  = os.environ.get("API_SECRET_KEY", "default_key"),

        hostname = os.environ.get("HOSTNAME", socket.gethostname()),

        qdrant_host = os.environ.get("QDRANT_HOST", "localhost"),
        qdrant_port = int(os.environ.get("QDRANT_PORT") or 6333),

        allow_subtopic_selection=_parse_bool(os.environ.get("ALLOW_SUBTOPIC_SELECTION"), default=True),
        allowed_origins=allowed_origins,

        max_history_items = int(os.environ.get("MAX_HISTORY_ITEMS") or 20),

        redis_host = os.environ.get("REDIS_HOST", "localhost"),
        redis_port = int(os.environ.get("REDIS_PORT") or 6379),

        rerank_size = int(os.environ.get("RERANK_SIZE") or 25),
        rerank_truncate = int(os.environ.get("RERANK_TRUNCATE") or 1200),
        rerank_batch_size = int(os.environ.get("RERANK_BATCH_SIZE") or  4),
        rerank_max_length = int(os.environ.get("RERANK_MAX_LENGTH") or 512),
        qdrant_concept_threshold = float(os.environ.get("QDRANT_CONCEPT_THRESHOLD") or 0.80),
        answer_max_tokens = int(os.environ.get("ANSWER_MAX_TOKENS") or  4096),
        grader_max_tokens = int(os.environ.get("GRADER_MAX_TOKENS") or  32),

        allow_general_knowledge = _parse_bool(os.environ.get("ALLOW_GENERAL_KNOWLEDGE"), default=True),

        qdrant_syntactic_size = int(os.environ.get("QDRANT_SYNTACTIC_SIZE") or 20),
        qdrant_semantic_size = int(os.environ.get("QDRANT_SEMANTIC_SIZE") or 30),

        qdrant_semantic_threshold = float(os.environ.get("QDRANT_SEMANTIC_THRESHOLD") or 0.60),
        max_context_chars = int(os.environ.get("MAX_CONTEXT_CHARS") or 30000),
        max_sub_queries = int(os.environ.get("MAX_SUB_QUERIES") or 5),
        parent_budget_ratio =float(os.environ.get("PARENT_BUDGET_RATIO") or 0.80),

        min_prob_threshold = float(os.environ.get("MIN_PROB_THRESHOLD") or 0.02),
        parents_per_query = int(os.environ.get("PARENTS_PER_QUERY") or 4),

        mmr_similarity_threshold = float(os.environ.get("MMR_SIMILARITY_THRESHOLD") or 0.92),

        query_rewriter_model_name = os.environ.get("QUERY_REWRITER_MODEL_NAME", "gemini-3.1-flash-lite"),
        answer_generator_model_name = os.environ.get("ANSWER_GENERATOR_MODEL_NAME", "gemini-3.5-flash"),
        grader_model_name = os.environ.get("GRADER_MODEL_NAME", "gemini-3.1-flash-lite"),

        onnx_model_cache_path = os.environ.get("RERANKER_MODEL_PATH", "./model_cache/mmarco-mMiniLMv2-L12-H384-v1"),

        llm_provider = os.environ.get("LLM_PROVIDER", "gemini").lower(),

        protected_keys = {"topic_id", "sub_topic_id", "source", "parent_id", "content",
                  "parent_index", "child_index", "file_name", "_ingestion_error", "_ingestion_id", "content_hash"},

        max_reranker_thread = int(os.environ.get("MAX_RERANKER_THREAD") or 2),
        answer_thinking_level = os.environ.get("ANSWER_THINKING_LEVEL") or 'LOW',
        grader_thinking_level = os.environ.get("GRADER_THINKING_LEVEL") or 'MINIMAL',

        reranker_pool_size = int(os.environ.get("RERANKER_POOL_SIZE") or 1),

        id_tipo_iter_atteso = {
            "decreto_presidenziale": _parse_set(os.environ.get("ID_TIPO_ITER_DECRETO_PRESIDENZIALE"), default={"8"}),
            "decreto_deliberativo": _parse_set(os.environ.get("ID_TIPO_ITER_DECRETO_DELIBERATIVO"), default={"9", "19"}),
        },

        max_allowed_pages = int(os.environ.get("MAX_ALLOWED_PAGES") or 300),

        max_model_retries = int(os.environ.get("MAX_MODEL_RETRIES") or 3),

        qdrant_concept_max_hits = int(os.environ.get("QDRANT_CONCEPT_MAX_HITS") or 3),
    )

settings = load_settings()