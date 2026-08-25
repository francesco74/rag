from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import socket

load_dotenv()

# I guardrail sotto scattano a import-time, PRIMA che Celery configuri il
# logging. I record di livello WARNING senza handler finiscono comunque su
# stderr tramite logging.lastResort, quindi restano visibili nei log del pod:
# una configurazione incoerente non passa mai sotto silenzio.
_log = logging.getLogger("CONFIG")

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
    # None = nessun tetto esplicito: si usa il massimo del modello. È il default.
    # Un numero qui sarebbe una copia locale di un dato che appartiene al
    # vendor e che diverge in silenzio al primo cambio di
    # ANSWER_GENERATOR_MODEL_NAME. Impostalo solo se hai una ragione specifica
    # per limitare la lunghezza della risposta (non il costo: i token si pagano
    # a consumo, e un troncamento costa PIÙ di una risposta lunga, perché il
    # primo tentativo è inutilizzabile e va rigenerato tutto).
    answer_max_tokens: Optional[int]
    # Resta un intero: qui il tetto NON è una capability da inseguire ma un
    # circuit breaker. Il grader emette un JSON di quattro campi, il rewriter
    # una lista di sottoquery: se uno dei due entra in loop di ripetizione,
    # senza tetto genera fino al soffitto del modello e lo paghi tutto.
    grader_max_tokens: int
    qdrant_syntactic_size: int
    qdrant_semantic_size: int
    qdrant_semantic_threshold: float
    qdrant_concept_threshold: float
    max_context_chars: int
    max_sub_queries: int
    onnx_model_cache_path: str

    # --- Allocazione del contesto per sottoquery -------------------------------
    # Governano come MAX_CONTEXT_CHARS viene ripartito tra le sottoquery generate
    # dal rewriter (es. un comune per sottoquery). Tarando questi tre valori si
    # ottengono tre politiche diverse SENZA toccare il codice:
    #
    #   Politica          FLOOR   CAP     WEIGHT   Comportamento
    #   ----------------  ------  ------  -------  --------------------------------
    #   Rigida            1.00    1.00    none     Parti uguali. Nessuna sottoquery
    #                                              può essere sacrificata, nemmeno
    #                                              se il suo materiale è debole.
    #   Morbida (default) 0.60    0.40    best     60% diviso equamente (nessuno
    #                                              sotto una soglia minima), 40%
    #                                              pesato sulla rilevanza trovata.
    #   Anti-monopolio    0.00    0.35    best     Tutto sullo score, ma nessuna
    #                                              sottoquery oltre il 35% del
    #                                              contesto totale.
    #
    # ATTENZIONE sul peso: gli score del cross-encoder NON sono nativamente
    # confrontabili tra sottoquery diverse (il reranker valuta la coppia
    # query-documento, quindi due formulazioni diverse producono distribuzioni
    # su scale diverse). La pesatura va letta come euristica di priorità, non
    # come misura assoluta di importanza. Se in rag_metrics vedi che le
    # sottoquery in coda restano cronicamente sotto-servite, la mossa corretta
    # è ALZARE context_floor_ratio, non ritoccare il peso.
    context_floor_ratio: float
    context_cap_ratio: float
    context_score_weight: str

    # Sforamento tollerato (frazione di max_context_chars) per includere un
    # documento INTERO invece di finestrarlo per pochi caratteri. Non può mai
    # servire ad aggiungere un parent NUOVO: solo a completarne uno già
    # selezionato. Viene usato solo dopo aver esaurito l'avanzo del water-filling.
    context_overflow_ratio: float
    # Tetto invalicabile, espresso come FRAZIONE di max_context_chars anziché in
    # caratteri assoluti. Un tetto assoluto è una bomba a orologeria: basta che
    # qualcuno alzi MAX_CONTEXT_CHARS senza ricordarsi di alzare anche il tetto
    # e build_context comincia a scartare in silenzio la maggior parte dei
    # parent recuperati (con MAX_CONTEXT_CHARS=150000 e un tetto fermo a 45000
    # si perdevano 17 parent su 24, cioè quasi tutto il lavoro di retrieval e
    # reranking già pagato). Legandolo al valore base, l'incoerenza non è più
    # nemmeno rappresentabile.
    context_hard_cap_ratio: float
    # DERIVATO, non letto da env: int(max_context_chars * (1 + ratio)).
    # Resta un campo a sé perché è ciò che il worker legge davvero.
    context_hard_cap_chars: int
    # Dove arretrare il taglio per non spezzare il testo a metà parola/frase:
    # 'sentence' | 'paragraph' | 'none'. Costo zero, nessuna espansione.
    context_snap_boundary: str

    min_prob_threshold: float
    parents_per_query: int

    mmr_similarity_threshold: float
    boilerplate_similarity_threshold: float

    query_rewriter_model_name: str
    answer_generator_model_name: str
    grader_model_name: str

    llm_provider: str

    # Provider usato per gli embedding (vedi common/embedding.py). A differenza
    # di llm_provider NON è "cambiabile a runtime senza conseguenze": i vettori
    # già scritti in Qdrant restano legati al modello/provider con cui sono
    # stati creati. Cambiare questo valore senza rifare l'ingestion completa
    # (su una collection nuova, con dimensione coerente) produce similarità
    # sbagliate in modo silenzioso.
    embedding_provider: str

    # Usati solo quando embedding_provider="local" (vedi LocalEmbeddingProvider
    # in common/embedding.py). Cambiare local_embedding_model non richiede
    # toccare il codice: la dimension del nuovo modello viene letta a runtime
    # da model.get_sentence_embedding_dimension(), non è un valore da
    # sincronizzare qui a mano.
    local_embedding_model: str
    local_embedding_device: str | None  # "cpu" | "cuda" | "mps" | None (auto-detect)

    # Endpoint del server Ollama, usato solo quando LLM_PROVIDER="local" (vedi
    # LocalLLMProvider in llm_provider.py). A differenza di local_embedding_model,
    # NON serve un "local_llm_model" qui: query_rewriter_model_name/
    # answer_generator_model_name/grader_model_name (sotto) esistono già e
    # bastano — con provider="local" contengono un tag Ollama (es. "qwen3:14b")
    # invece di un nome modello Gemini.
    local_llm_host: str

    # Quale backend usare quando LLM_PROVIDER="local": "ollama" (default,
    # sviluppo/basso-concorrenza, sequenziale) o "vllm" (produzione
    # multi-utente, continuous batching — vedi VLLMProvider in llm_provider.py
    # per il perché della soglia). Non tocca LLM_PROVIDER: la scelta resta
    # "sono in locale?" (llm_provider) separata da "con quale motore?"
    # (questo campo), così passare da un backend locale all'altro non cambia
    # nient'altro a valle (query_rewriter_model_name ecc. restano quelli).
    local_llm_backend: str

    # Usati solo quando local_llm_backend="vllm". base_url punta al server
    # OpenAI-compatible esposto da vLLM (es. "http://localhost:8000/v1" —
    # nota il suffisso /v1, diverso dal formato di local_llm_host per Ollama).
    # api_key è quasi sempre superflua in locale (vLLM in genere non
    # autentica); VLLMProvider usa un placeholder se lasciata vuota, perché
    # il client OpenAI pretende comunque una stringa non vuota.
    local_vllm_base_url: str
    local_vllm_api_key: str | None

    # Usata da GeminiEmbeddingProvider (output_dimensionality) e
    # OpenAIEmbeddingProvider (dimensions): entrambi supportano la troncatura
    # del vettore nativo a questa dimensione, a differenza di Mistral (nativo
    # fisso a 1024) e del provider locale (dimension letta dal modello
    # caricato, non troncabile in modo affidabile). Un solo valore condiviso
    # tra i due invece di "768" scritto due volte in embedding.py: se un
    # domani serve una dimensione diversa, cambia qui, non nel codice.
    embedding_dimension: int

    protected_keys: set[str]

    max_reranker_thread: int
    reranker_pool_size: int
    cpu_limit: int
    celery_concurrency: int
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

    # =========================================================================
    # VALORI INTERDIPENDENTI
    # Calcolati qui, prima della costruzione della dataclass, perché il valore
    # di uno vincola quello di un altro: passarli inline nel costruttore
    # renderebbe impossibile validarli l'uno contro l'altro.
    # =========================================================================

    # --- Contesto: il tetto insegue il budget --------------------------------
    max_context_chars = int(os.environ.get("MAX_CONTEXT_CHARS") or 30000)
    context_overflow_ratio = float(os.environ.get("CONTEXT_OVERFLOW_RATIO") or 0.15)
    context_hard_cap_ratio = float(os.environ.get("CONTEXT_HARD_CAP_RATIO") or 0.20)

    # Il tetto deve stare SOPRA l'espansione consentita in fase di allocazione.
    # allocate_context_budget calcola il proprio soffitto come
    #     min(budget * (1 + overflow_ratio), hard_cap_chars)
    # quindi un hard cap più basso dell'overflow taglierebbe SOTTO ciò che è
    # già stato allocato: build_context si troverebbe a scartare parent a cui
    # era stato appena assegnato del budget — di nuovo il bug che questa
    # modifica elimina, solo per un'altra strada.
    if context_hard_cap_ratio < context_overflow_ratio:
        _log.warning(
            "CONTEXT_HARD_CAP_RATIO (%.2f) è inferiore a CONTEXT_OVERFLOW_RATIO (%.2f): "
            "il tetto taglierebbe sotto l'allocazione. Allineato a %.2f.",
            context_hard_cap_ratio, context_overflow_ratio, context_overflow_ratio,
        )
        context_hard_cap_ratio = context_overflow_ratio

    # round() e non int(): la troncatura darebbe 114999 per 100000 * 1.15, che
    # in un log è solo rumore ma fa perdere tempo a chi cerca di capire perché
    # il tetto non è il numero tondo che si aspettava.
    context_hard_cap_chars = int(round(max_context_chars * (1 + context_hard_cap_ratio)))

    # --- Token di risposta: nessun tetto, per scelta ------------------------
    # Non passando max_tokens al provider, il tetto è quello del modello per
    # definizione: non c'è nessuna costante da tenere allineata a mano e
    # cambiare modello non richiede di ricordarsi niente. Il vecchio
    # ANSWER_MAX_TOKENS_HARD_CAP è stato rimosso: era già inerte in produzione
    # (con ANSWER_MAX_TOKENS=16384 la guardia "max_tokens < hard_cap" era
    # sempre falsa, quindi il retry sul troncamento non poteva scattare).
    # Vuoto, "0" o "auto" -> None. Un intero resta rispettato, per chi ha una
    # ragione specifica per limitare la lunghezza della risposta.
    _answer_tokens_raw = (os.environ.get("ANSWER_MAX_TOKENS") or "").strip().lower()
    if _answer_tokens_raw in ("", "0", "auto", "none", "model"):
        answer_max_tokens = None
    else:
        try:
            answer_max_tokens = int(_answer_tokens_raw)
            if answer_max_tokens <= 0:
                answer_max_tokens = None
        except ValueError:
            _log.warning(
                "ANSWER_MAX_TOKENS=%r non è un intero valido: uso il massimo del modello.",
                _answer_tokens_raw,
            )
            answer_max_tokens = None

    # --- Reranker: il pool non può essere più piccolo dei thread -------------
    # process_single_rerank pesca l'istanza con round-robin su q_idx: se le
    # istanze sono meno dei thread concorrenti, due thread finiscono sulla
    # STESSA InferenceSession, che è ORT_SEQUENTIAL — si serializzano sullo
    # stesso budget di calcolo invece di raddoppiarlo, cioè esattamente il
    # problema per cui il pool era stato introdotto.
    max_reranker_thread = int(os.environ.get("MAX_RERANKER_THREAD") or 2)
    reranker_pool_size = int(os.environ.get("RERANKER_POOL_SIZE") or 1)

    if reranker_pool_size < max_reranker_thread:
        _log.warning(
            "RERANKER_POOL_SIZE (%d) è inferiore a MAX_RERANKER_THREAD (%d): i thread in "
            "eccesso condividerebbero una sessione ONNX sequenziale, serializzandosi. "
            "Pool allineato a %d istanze (verificare che CPU_LIMIT sia adeguato).",
            reranker_pool_size, max_reranker_thread, max_reranker_thread,
        )
        reranker_pool_size = max_reranker_thread

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
        answer_max_tokens = answer_max_tokens,
        # Il grader ora restituisce un JSON strutturato (verdict/reason/
        # missing_targets/note), non più "YES"/"NO": 32 token si troncavano
        # sistematicamente. Su modelli che contano il thinking sullo stesso
        # budget (gemini-3.1-flash-lite) serve margine abbondante.
        grader_max_tokens = int(os.environ.get("GRADER_MAX_TOKENS") or 256),

        allow_general_knowledge = _parse_bool(os.environ.get("ALLOW_GENERAL_KNOWLEDGE"), default=True),

        qdrant_syntactic_size = int(os.environ.get("QDRANT_SYNTACTIC_SIZE") or 20),
        qdrant_semantic_size = int(os.environ.get("QDRANT_SEMANTIC_SIZE") or 30),

        qdrant_semantic_threshold = float(os.environ.get("QDRANT_SEMANTIC_THRESHOLD") or 0.60),
        max_context_chars = max_context_chars,
        max_sub_queries = int(os.environ.get("MAX_SUB_QUERIES") or 5),

        # Vedi la tabella delle tre politiche nel commento sulla dataclass.
        context_floor_ratio = float(os.environ.get("CONTEXT_FLOOR_RATIO") or 0.60),
        context_cap_ratio = float(os.environ.get("CONTEXT_CAP_RATIO") or 0.40),
        context_score_weight = (os.environ.get("CONTEXT_SCORE_WEIGHT") or "best").lower(),

        context_overflow_ratio = context_overflow_ratio,
        context_hard_cap_ratio = context_hard_cap_ratio,
        context_hard_cap_chars = context_hard_cap_chars,
        context_snap_boundary = (os.environ.get("CONTEXT_SNAP_BOUNDARY") or "sentence").lower(),

        min_prob_threshold = float(os.environ.get("MIN_PROB_THRESHOLD") or 0.02),
        parents_per_query = int(os.environ.get("PARENTS_PER_QUERY") or 4),

        mmr_similarity_threshold = float(os.environ.get("MMR_SIMILARITY_THRESHOLD") or 0.92),
        boilerplate_similarity_threshold = float(os.environ.get("BOILERPLATE_SIMILARITY_THRESHOLD") or 0.55),

        query_rewriter_model_name = os.environ.get("QUERY_REWRITER_MODEL_NAME", "gemini-3.1-flash-lite"),
        answer_generator_model_name = os.environ.get("ANSWER_GENERATOR_MODEL_NAME", "gemini-3.5-flash"),
        grader_model_name = os.environ.get("GRADER_MODEL_NAME", "gemini-3.1-flash-lite"),

        onnx_model_cache_path = os.environ.get("RERANKER_MODEL_PATH", "./model_cache/mmarco-mMiniLMv2-L12-H384-v1"),

        llm_provider = os.environ.get("LLM_PROVIDER", "gemini").lower(),

        # Default "gemini" per compatibilità con le collection Qdrant già
        # esistenti: chi non imposta la variabile continua a produrre vettori
        # nello stesso spazio vettoriale di prima. Cambiarla richiede una
        # collection nuova e una ingestion completa (vedi embedding_provider
        # nella dataclass sopra).
        embedding_provider = os.environ.get("EMBEDDING_PROVIDER", "gemini").lower(),

        # Letti solo se embedding_provider="local". local_embedding_device
        # vuoto/non impostato -> None -> sentence-transformers auto-rileva
        # l'hardware disponibile (CPU/CUDA/MPS).
        local_embedding_model = os.environ.get("LOCAL_EMBEDDING_MODEL", "").strip() or "BAAI/bge-m3",
        local_embedding_device = os.environ.get("LOCAL_EMBEDDING_DEVICE", "").strip() or None,

        local_llm_host = os.environ.get("LOCAL_LLM_HOST", "").strip() or "http://localhost:11434",

        local_llm_backend = os.environ.get("LOCAL_LLM_BACKEND", "ollama").strip().lower() or "ollama",

        local_vllm_base_url = os.environ.get("LOCAL_VLLM_BASE_URL", "").strip() or "http://localhost:8000/v1",
        local_vllm_api_key = os.environ.get("LOCAL_VLLM_API_KEY", "").strip() or None,

        # Default 768: dimensione con cui sono state create le collection
        # Qdrant originali (vedi embedding_provider sopra). Cambiarla NON
        # rende automaticamente compatibili le collection esistenti: come per
        # embedding_provider, serve una collection nuova + ingestion completa.
        embedding_dimension = int(os.environ.get("EMBEDDING_DIMENSION", "768")),

        protected_keys = {"topic_id", "sub_topic_id", "source", "parent_id", "content",
                  "parent_index", "child_index", "file_name", "_ingestion_error", "_ingestion_id", "content_hash"},

        max_reranker_thread = max_reranker_thread,
        answer_thinking_level = os.environ.get("ANSWER_THINKING_LEVEL") or 'LOW',
        grader_thinking_level = os.environ.get("GRADER_THINKING_LEVEL") or 'MINIMAL',

        reranker_pool_size = reranker_pool_size,

        # Core assegnati al pod (resources.limits.cpu nel deployment K8s), NON
        # letti da os.cpu_count() come unica fonte: quest'ultimo, in un
        # container Kubernetes, riflette i core del nodo host, non il CPU
        # limit imposto via CFS quota/period. Va impostato esplicitamente in
        # CPU_LIMIT nel ConfigMap, allineato al valore numerico di
        # resources.limits.cpu del pod worker (es. "4" -> CPU_LIMIT=4).
        # Se non impostato, ripiega su os.cpu_count() (i core della macchina)
        # come approssimazione — corretto in locale, sovrastimato in K8s.
        cpu_limit = int(os.environ.get("CPU_LIMIT") or os.cpu_count() or 1),

        # Deve combaciare col valore passato a --concurrency nel comando Celery
        # del deployment. Serve a get_reranker_pool() per capire quanti processi
        # Celery condividono gli stessi core (cpu_limit): senza questo, ogni
        # processo dimensionerebbe il proprio pool di reranker assumendo di
        # avere tutti i core del pod per sé, sovrasottoscrivendo quando più
        # processi fanno reranking nello stesso momento.
        celery_concurrency = int(os.environ.get("CELERY_CONCURRENCY") or 1),

        id_tipo_iter_atteso = {
            "decreto_presidenziale": _parse_set(os.environ.get("ID_TIPO_ITER_DECRETO_PRESIDENZIALE"), default={"8"}),
            "decreto_deliberativo": _parse_set(os.environ.get("ID_TIPO_ITER_DECRETO_DELIBERATIVO"), default={"9", "19"}),
        },

        max_allowed_pages = int(os.environ.get("MAX_ALLOWED_PAGES") or 300),

        max_model_retries = int(os.environ.get("MAX_MODEL_RETRIES") or 3),

        qdrant_concept_max_hits = int(os.environ.get("QDRANT_CONCEPT_MAX_HITS") or 3),
    )

settings = load_settings()