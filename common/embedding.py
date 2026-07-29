"""
embedding.py — generazione degli embedding (Gemini only).

L'embedding resta su Google anche quando il resto del sistema usa un provider
LLM diverso (vedi llm_provider.py): i vettori in Qdrant sono calcolati con
gemini-embedding-001 e vanno confrontati solo con vettori dello stesso modello
e dello stesso task_type.

Il client è un globale di modulo assegnato da init_embedding(), che va chiamata
UNA volta per processo DOPO il fork (nel worker Celery) oppure all'avvio degli
script standalone che fanno embedding (es. detect_boilerplate.py). Prima di
quella chiamata embedding_client è None e le funzioni sollevano RuntimeError.
"""

import logging

from google import genai
from google.genai.errors import APIError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)

from common.config import settings

log = logging.getLogger("EMBEDDING")

EMBEDDING_MODEL = "gemini-embedding-001"

# Client globale assegnato dopo il fork (worker Celery) o all'avvio dello
# script standalone tramite init_embedding(). Resta None finché non inizializzato.
embedding_client = None

# Embedding retry (Gemini only — embedding stays on Google)
GEMINI_EMBEDDING_RETRY = retry(
    retry=retry_if_exception_type((APIError,)),
    wait=wait_random_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(6),
    before_sleep=lambda retry_state: log.warning(
        f"Embedding rate limit hit. Retrying in {retry_state.next_action.sleep}s..."
    )
)


def init_embedding():
    """Inizializza il client di embedding Gemini.

    Va chiamata una volta per processo, dopo il fork nel worker Celery
    (init_worker_process) o all'avvio di uno script standalone che fa embedding.
    """
    global embedding_client
    embedding_client = genai.Client(api_key=settings.api_llm_key)
    return embedding_client


@GEMINI_EMBEDDING_RETRY
def embed_query(query):
    """Generate embedding for a single query (Gemini only)."""
    log.debug(f"Embedding query: '{query[:50]}...'")
    result = embedding_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=dict(task_type="RETRIEVAL_QUERY", output_dimensionality=768)
    )
    return result.embeddings[0].values


@GEMINI_EMBEDDING_RETRY
def embed_queries_batch(queries_list):
    """Generate embeddings for multiple queries in a single API call (Gemini only)."""
    result = embedding_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=queries_list,
        config=dict(task_type="RETRIEVAL_QUERY", output_dimensionality=768)
    )
    if isinstance(queries_list, str):
        return [result.embeddings[0].values]
    return [emb.values for emb in result.embeddings]


@GEMINI_EMBEDDING_RETRY
def embed_for_semantic_query(query):
    """Embedding con task_type SEMANTIC_SIMILARITY, per confronti frase-contro-frase.

    Va usata ogni volta che si misura la vicinanza fra la query e un altro
    TESTO BREVE (voce del dizionario concettuale, frase di boilerplate), non fra
    la query e un DOCUMENTO indicizzato: quel caso usa RETRIEVAL_QUERY tramite
    embed_query(), perché Gemini produce proiezioni asimmetriche e confrontare
    vettori di task_type diversi falsa silenziosamente la similarità.
    """
    result = embedding_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=dict(task_type="SEMANTIC_SIMILARITY", output_dimensionality=768)
    )
    return result.embeddings[0].values