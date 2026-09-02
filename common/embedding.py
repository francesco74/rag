"""
embedding.py — generazione degli embedding (provider-agnostic).

Un embedding va SEMPRE confrontato solo con vettori generati dallo stesso
provider, stesso modello e stesso task_type (quando il provider lo prevede).
A differenza di LLM_PROVIDER (llm_provider.py), che si può cambiare a runtime
senza conseguenze perché ogni chiamata è indipendente, EMBEDDING_PROVIDER è
una decisione "congelata" al momento dell'ingestion: i vettori già scritti in
Qdrant restano nel modello con cui sono stati creati, e cambiare provider senza
rifare l'ingestion produce similarità sbagliate in modo silenzioso (Qdrant non
si accorge del mismatch, calcola comunque una distanza). Per passare a un
provider diverso serve una nuova collection (dimensione dei vettori diversa)
e una ingestion completa da capo.

Supported providers (set via EMBEDDING_PROVIDER env var, default: gemini):
  gemini    — Google (gemini-embedding-001). Unico dei tre con task_type
              asimmetrico: RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT / SEMANTIC_SIMILARITY
              producono proiezioni diverse per lo stesso testo, e vanno sempre
              accoppiati correttamente (query vs documento indicizzato).
  openai    — OpenAI (text-embedding-3-small di default). Embedding SIMMETRICO:
              non esiste un task_type separato per query e documento, quindi
              embed_query/embed_for_semantic_query/il lato documento chiamano
              tutti lo stesso identico endpoint.
  mistral   — Mistral AI (mistral-embed). Simmetrico come OpenAI.

Ogni provider espone un attributo `dimension`: è la dimensionalità dei vettori
prodotti, da usare per configurare VectorParams(size=...) in Qdrant quando si
crea la collection per quel provider.

Il client di ciascun provider è inizializzato UNA volta per processo da
init_embedding(), da chiamare DOPO il fork (nel worker Celery) oppure
all'avvio degli script standalone che fanno embedding (es. detect_boilerplate.py).
Prima di quella chiamata get_embedding_provider() solleva RuntimeError.

worker.py e ingest.py continuano a chiamare embed_query/embed_queries_batch/
embed_for_semantic_query/embed_documents_batch come funzioni libere di modulo:
sono thin wrapper che delegano al provider attivo, così aggiungere un nuovo
provider qui non richiede toccare i chiamanti.
"""

import logging
import random
import time
import asyncio
from abc import ABC, abstractmethod

from common.config import settings

log = logging.getLogger("EMBEDDING")


# ==============================================================================
# RETRY HELPERS (stesso schema di llm_provider.py: backoff esponenziale con
# jitter, normalizzato su una singola eccezione indipendente dal provider)
# ==============================================================================
class EmbeddingRateLimitError(Exception):
    """Raised by any embedding provider when the API signals rate-limit or overload."""


def _retry_with_backoff(fn, max_attempts=6, base_wait=4, max_wait=60):
    attempt = 0
    while True:
        try:
            return fn()
        except EmbeddingRateLimitError as e:
            attempt += 1
            if attempt >= max_attempts:
                raise
            wait = min(base_wait * (2 ** (attempt - 1)), max_wait)
            wait = wait * (0.5 + random.random() * 0.5)  # jitter
            log.warning(f"Embedding rate limit hit ({attempt}/{max_attempts}). Retrying in {wait:.1f}s... [{e}]")
            time.sleep(wait)


async def _retry_with_backoff_async(fn, max_attempts=6, base_wait=4, max_wait=60):
    attempt = 0
    while True:
        try:
            return await fn()
        except EmbeddingRateLimitError as e:
            attempt += 1
            if attempt >= max_attempts:
                raise
            wait = min(base_wait * (2 ** (attempt - 1)), max_wait)
            wait = wait * (0.5 + random.random() * 0.5)  # jitter
            log.warning(f"Embedding rate limit hit ({attempt}/{max_attempts}). Retrying in {wait:.1f}s... [{e}]")
            await asyncio.sleep(wait)


# ==============================================================================
# BASE PROVIDER
# ==============================================================================
class EmbeddingProvider(ABC):
    """
    model_name e dimension sono attributi di classe: identificano univocamente
    lo "spazio vettoriale" prodotto da questo provider, quello che va abbinato
    a una singola collection Qdrant.
    """
    model_name: str
    dimension: int

    @abstractmethod
    def embed_query(self, query):
        """Embedding di una singola query per il retrieval verso documenti indicizzati."""

    @abstractmethod
    def embed_queries_batch(self, queries_list):
        """Embedding di più query in un'unica chiamata (stesso task_type di embed_query)."""

    @abstractmethod
    def embed_for_semantic_query(self, query):
        """
        Embedding per confronti frase-contro-frase (dizionario concettuale,
        boilerplate), NON per il retrieval verso un documento indicizzato:
        quel caso usa embed_query(). Sui provider simmetrici (OpenAI, Mistral)
        questa distinzione collassa sullo stesso embedding di embed_query;
        su Gemini no, perché il task_type cambia la proiezione.
        """

    @abstractmethod
    async def embed_documents_batch(self, texts):
        """Embedding ASINCRONO in batch di DOCUMENTI da indicizzare (lato ingestion)."""


# ==============================================================================
# GEMINI PROVIDER
# ==============================================================================
class GeminiEmbeddingProvider(EmbeddingProvider):
    model_name = "gemini-embedding-001"

    def __init__(self):
        try:
            from google import genai
            from google.genai.errors import APIError
        except ImportError:
            raise RuntimeError("google-genai package not installed.")
        # Letta da config.py (EMBEDDING_DIMENSION, default 768) invece che
        # hardcoded qui: usata sia per il parametro output_dimensionality
        # nelle chiamate sotto, sia come schema della collection Qdrant in
        # migrate_embeddings.py/ingest.py.
        self.dimension = settings.embedding_dimension
        self._client = genai.Client(api_key=settings.api_llm_key)
        self._APIError = APIError
        log.info(f"✓ GeminiEmbeddingProvider initialized (dim={self.dimension}).")

    def _attempt(self, contents, task_type):
        try:
            return self._client.models.embed_content(
                model=self.model_name,
                contents=contents,
                config=dict(task_type=task_type, output_dimensionality=self.dimension),
            )
        except self._APIError as e:
            # Coerente col comportamento originale: qualunque APIError su
            # questo endpoint viene trattato come transitorio e riprovato.
            raise EmbeddingRateLimitError(str(e)) from e

    def embed_query(self, query):
        log.debug(f"Embedding query: '{query[:50]}...'")
        result = _retry_with_backoff(lambda: self._attempt(query, "RETRIEVAL_QUERY"))
        return result.embeddings[0].values

    def embed_queries_batch(self, queries_list):
        result = _retry_with_backoff(lambda: self._attempt(queries_list, "RETRIEVAL_QUERY"))
        if isinstance(queries_list, str):
            return [result.embeddings[0].values]
        return [emb.values for emb in result.embeddings]

    def embed_for_semantic_query(self, query):
        result = _retry_with_backoff(lambda: self._attempt(query, "SEMANTIC_SIMILARITY"))
        return result.embeddings[0].values

    async def embed_documents_batch(self, texts):
        if not texts:
            return []
        from google.genai import types

        async def _attempt():
            try:
                return await self._client.aio.models.embed_content(
                    model=self.model_name,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=self.dimension,
                    ),
                )
            except self._APIError as e:
                raise EmbeddingRateLimitError(str(e)) from e

        response = await _retry_with_backoff_async(_attempt)
        return [emb.values for emb in response.embeddings]


# ==============================================================================
# OPENAI PROVIDER
# ==============================================================================
class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI è simmetrico: un solo endpoint per query e documenti. Il parametro
    `dimensions` (supportato dalla famiglia text-embedding-3-*) tronca il
    vettore nativo alla dimensione richiesta — letta da settings.embedding_dimension
    (config.py, default 768) invece che hardcoded, stesso valore condiviso
    con GeminiEmbeddingProvider così una collection Qdrant a 768 dim resta
    valida come SCHEMA anche con OpenAI attivo — restano comunque due spazi
    vettoriali incompatibili tra loro e servono due collection separate.
    """
    model_name = "text-embedding-3-small"

    def __init__(self):
        try:
            from openai import OpenAI, AsyncOpenAI, RateLimitError, APIStatusError
        except ImportError:
            raise RuntimeError("openai package not installed.")
        self.dimension = settings.embedding_dimension
        self._client = OpenAI(api_key=settings.api_llm_key)
        self._async_client = AsyncOpenAI(api_key=settings.api_llm_key)
        self._RateLimitError = RateLimitError
        self._APIStatusError = APIStatusError
        log.info(f"✓ OpenAIEmbeddingProvider initialized (dim={self.dimension}).")

    def _raise_if_rate_limited(self, e):
        if isinstance(e, self._RateLimitError):
            raise EmbeddingRateLimitError(str(e)) from e
        if isinstance(e, self._APIStatusError) and e.status_code in (429, 503):
            raise EmbeddingRateLimitError(str(e)) from e
        raise e

    def _embed_sync(self, texts):
        try:
            response = self._client.embeddings.create(
                model=self.model_name, input=texts, dimensions=self.dimension,
            )
            return [d.embedding for d in response.data]
        except (self._RateLimitError, self._APIStatusError) as e:
            self._raise_if_rate_limited(e)

    def embed_query(self, query):
        return _retry_with_backoff(lambda: self._embed_sync([query]))[0]

    def embed_queries_batch(self, queries_list):
        texts = [queries_list] if isinstance(queries_list, str) else queries_list
        return _retry_with_backoff(lambda: self._embed_sync(texts))

    def embed_for_semantic_query(self, query):
        return _retry_with_backoff(lambda: self._embed_sync([query]))[0]

    async def embed_documents_batch(self, texts):
        if not texts:
            return []

        async def _attempt():
            try:
                response = await self._async_client.embeddings.create(
                    model=self.model_name, input=texts, dimensions=self.dimension,
                )
                return [d.embedding for d in response.data]
            except (self._RateLimitError, self._APIStatusError) as e:
                self._raise_if_rate_limited(e)

        return await _retry_with_backoff_async(_attempt)


# ==============================================================================
# MISTRAL PROVIDER
# ==============================================================================
class MistralEmbeddingProvider(EmbeddingProvider):
    """
    Simmetrico come OpenAI. mistral-embed produce nativamente vettori a 1024
    dimensioni: a differenza di OpenAI non tutti i modelli Mistral supportano
    una troncatura via `output_dimension` (dipende dal modello), quindi qui si
    tiene la dimensione nativa invece di forzarla a 768.
    """
    model_name = "mistral-embed"
    dimension = 1024

    def __init__(self):
        try:
            from mistralai import Mistral
            from mistralai.models import SDKError
        except ImportError:
            raise RuntimeError("mistralai package not installed.")
        self._client = Mistral(api_key=settings.api_llm_key)
        self._SDKError = SDKError
        log.info("✓ MistralEmbeddingProvider initialized.")

    def _raise_if_rate_limited(self, e):
        status_code = getattr(e, "status_code", None) or getattr(e, "status", None)
        if status_code in (429, 503):
            raise EmbeddingRateLimitError(str(e)) from e
        raise e

    def _embed_sync(self, texts):
        try:
            response = self._client.embeddings.create(model=self.model_name, inputs=texts)
            return [d.embedding for d in response.data]
        except self._SDKError as e:
            self._raise_if_rate_limited(e)

    def embed_query(self, query):
        return _retry_with_backoff(lambda: self._embed_sync([query]))[0]

    def embed_queries_batch(self, queries_list):
        texts = [queries_list] if isinstance(queries_list, str) else queries_list
        return _retry_with_backoff(lambda: self._embed_sync(texts))

    def embed_for_semantic_query(self, query):
        return _retry_with_backoff(lambda: self._embed_sync([query]))[0]

    async def embed_documents_batch(self, texts):
        if not texts:
            return []

        async def _attempt():
            try:
                response = await self._client.embeddings.create_async(
                    model=self.model_name, inputs=texts,
                )
                return [d.embedding for d in response.data]
            except self._SDKError as e:
                self._raise_if_rate_limited(e)

        return await _retry_with_backoff_async(_attempt)


# ==============================================================================
# LOCAL PROVIDER (self-hosted, nessuna chiamata di rete)
# ==============================================================================
class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Modello locale via sentence-transformers, eseguito in-process. Nessuna
    chiamata API, nessun rate limit, nessun bisogno di retry/backoff.

    Modello e device letti da config.py (settings.local_embedding_model,
    settings.local_embedding_device), NON hardcoded: cambiare modello locale
    (es. da BGE-M3 a Nomic) richiede solo una variabile d'ambiente, non un
    redeploy del codice. Default: BAAI/bge-m3 — multilingue (100+ lingue,
    incluso l'italiano e l'inglese), licenza MIT.

    Simmetrico come OpenAI/Mistral: non esiste un task_type separato per
    query e documento, quindi embed_query/embed_for_semantic_query/il lato
    documento producono lo stesso identico embedding per lo stesso testo.

    Il modello viene caricato UNA volta in __init__ (operazione pesante:
    qualche secondo e ~2GB di RAM/VRAM per bge-m3) e tenuto in memoria per
    tutta la vita del processo — coerente con init_embedding() chiamata una
    volta per processo, come da docstring in testa al modulo.

    NOTA su `dimension`: a differenza degli altri provider non è un valore
    fisso in classe, ma viene letto DAL MODELLO dopo il caricamento
    (model.get_sentence_embedding_dimension()). Cambiare
    local_embedding_model in config.py aggiorna automaticamente anche la
    dimensione usata da migrate_embeddings.py per creare la collection
    Qdrant — niente valori disallineati da tenere sincronizzati a mano.
    """

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError("sentence-transformers package not installed.")

        # Campi definiti in config.py: LOCAL_EMBEDDING_MODEL (default "BAAI/bge-m3")
        # e LOCAL_EMBEDDING_DEVICE ("cpu"|"cuda"|"mps", default None = auto-rilevato).
        self.model_name = settings.local_embedding_model
        device = settings.local_embedding_device

        self._model = SentenceTransformer(self.model_name, device=device)
        self.dimension = self._model.get_sentence_embedding_dimension()
        log.info(
            f"✓ LocalEmbeddingProvider initialized "
            f"(model={self.model_name}, dim={self.dimension}, device={device or 'auto'})."
        )

    def _encode(self, texts):
        # normalize_embeddings=True: vettori già L2-normalizzati, coerente con
        # Distance.COSINE usata per la collection Qdrant.
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, query):
        return self._encode([query])[0]

    def embed_queries_batch(self, queries_list):
        texts = [queries_list] if isinstance(queries_list, str) else queries_list
        return self._encode(texts)

    def embed_for_semantic_query(self, query):
        return self._encode([query])[0]

    async def embed_documents_batch(self, texts):
        if not texts:
            return []
        # SentenceTransformer.encode è sincrono e CPU/GPU-bound: lo giriamo in
        # un thread per non bloccare l'event loop asyncio. Niente retry qui:
        # a differenza delle API remote, un encode locale non può andare in
        # rate-limit — se fallisce è un errore reale (OOM, input malformato)
        # che deve propagarsi, non essere ritentato alla cieca.
        return await asyncio.to_thread(self._encode, texts)


# ==============================================================================
# PUBLIC API
# ==============================================================================
_EMBEDDING_PROVIDERS = {
    "gemini":  GeminiEmbeddingProvider,
    "openai":  OpenAIEmbeddingProvider,
    "mistral": MistralEmbeddingProvider,
    "local":   LocalEmbeddingProvider,
}

_PROVIDER_INSTANCE = None  # set once in init_embedding()


def init_embedding():
    """
    Inizializza il provider di embedding selezionato da settings.embedding_provider
    (default "gemini", per compatibilità con le collection già esistenti se la
    variabile non è ancora configurata).

    Va chiamata una volta per processo, dopo il fork nel worker Celery
    (init_worker_process) o all'avvio di uno script standalone che fa embedding.

    ATTENZIONE: questa funzione seleziona SOLO il client con cui generare nuovi
    vettori. Non converte né rende compatibili i vettori già presenti in
    Qdrant: cambiare provider richiede una collection nuova (dimensione e
    spazio vettoriale diversi) e una ingestion completa da capo.
    """
    global _PROVIDER_INSTANCE
    provider_name = settings.embedding_provider

    cls = _EMBEDDING_PROVIDERS.get(provider_name)
    if cls is None:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER='{provider_name}'. "
            f"Valid options: {list(_EMBEDDING_PROVIDERS.keys())}"
        )

    _PROVIDER_INSTANCE = cls()
    log.info(
        f"Embedding provider set to: {provider_name} "
        f"(model={_PROVIDER_INSTANCE.model_name}, dim={_PROVIDER_INSTANCE.dimension})"
    )
    return _PROVIDER_INSTANCE


def get_embedding_provider() -> EmbeddingProvider:
    """Return the already-initialized embedding provider. Raises if init_embedding() was not called."""
    if _PROVIDER_INSTANCE is None:
        raise RuntimeError("Embedding provider not initialized. Call init_embedding() first.")
    return _PROVIDER_INSTANCE


# ------------------------------------------------------------------------------
# Wrapper di modulo (backward-compatible): worker.py e ingest.py importano e
# chiamano queste funzioni direttamente, non passano da un'istanza provider.
# Restano thin delegate verso il provider attivo così i chiamanti non cambiano
# mai, indipendentemente da quanti provider vengono aggiunti qui sopra.
# ------------------------------------------------------------------------------
def embed_query(query):
    """Generate embedding for a single query (provider attivo)."""
    return get_embedding_provider().embed_query(query)


def embed_queries_batch(queries_list):
    """Generate embeddings for multiple queries in a single API call (provider attivo)."""
    return get_embedding_provider().embed_queries_batch(queries_list)


def embed_for_semantic_query(query):
    """Embedding per confronti frase-contro-frase (provider attivo)."""
    return get_embedding_provider().embed_for_semantic_query(query)


async def embed_documents_batch(texts):
    """Embedding asincrono in batch di documenti da indicizzare (provider attivo)."""
    return await get_embedding_provider().embed_documents_batch(texts)