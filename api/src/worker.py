import os
import logging
import time
import uuid
from celery import Celery
from celery.signals import setup_logging
from celery.signals import worker_process_init, worker_shutdown
from celery.schedules import crontab
from mysql.connector import pooling
from qdrant_client import QdrantClient, models
from google import genai  # embedding only
from google.genai.errors import APIError
import math
import json, re
import hashlib
import numpy as np
# Use the optimized reranker
from reranker import ONNXReranker, RerankResult

# Provider-agnostic LLM adapter
from llm_provider import init_llm_provider, get_llm_provider

from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)

from common.config import settings
from common.db_logger import MySQLLogHandler, get_db_connection, init_db_pool


# ==============================================================================
# 1. CONFIGURATION & LOGGING
# ==============================================================================

@setup_logging.connect
def configure_worker_logging(*args, **kwargs):
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format='%(asctime)s - [WORKER-%(process)d] - %(levelname)s - %(message)s',
        force=True  # Svuota e sovrascrive gli handler precedentemente configurati da Celery
    )

log = logging.getLogger("WORKER")

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), 'prompts')

MAX_AGE_SECONDS = 86400  # 24 hours
_RERANKER_POOL: list = []
_POOL_SIZE = settings.reranker_pool_size

# Embedding retry (Gemini only — embedding stays on Google)
GEMINI_EMBEDDING_RETRY = retry(
    retry=retry_if_exception_type((APIError,)),
    wait=wait_random_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(6),
    before_sleep=lambda retry_state: log.warning(
        f"Embedding rate limit hit. Retrying in {retry_state.next_action.sleep}s..."
    )
)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["ONNXRUNTIME_EXECUTION_MODE"] = "PARALLEL"

QDRANT_COLLECTION = "document_chunks"
CACHE_COLLECTION = "semantic_cache"
CONCEPT_COLLECTION = "conceptual_dictionary"

# ==============================================================================
# 2. CELERY INITIALIZATION
# =========================================", "rag_system")=====================================
redis_conn_string = f"redis://{settings.redis_host}:{settings.redis_port}/0"
celery_app = Celery('rag_queue', broker=redis_conn_string, backend=redis_conn_string)
celery_app.conf.update(
    result_expires=3600,
    worker_concurrency=1,
    worker_prefetch_multiplier=1,   
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks
    worker_max_memory_per_child=2000000, # 2GB limit
    worker_hijack_root_logger=False,
)

# ==============================================================================
# PROCESS-SAFE INITIALIZATION 
# ==============================================================================
# Globals assigned strictly AFTER the fork
db_pool = None
qdrant_client = None
_RERANKER_INSTANCE = None

EMBEDDING_MODEL = "gemini-embedding-001"

@worker_process_init.connect
def init_worker_process(**kwargs):
    global qdrant_client, embedding_client
    log.info("Initializing Worker Resources (Post-Fork)...")

    try:
        init_db_pool()  # Inizializza il pool globalmente

        qdrant_client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        embedding_client = genai.Client(api_key=settings.api_llm_key)
        init_llm_provider()

        log.info("✓ Resources successfully initialized for this process.")
    except Exception as e:
        log.critical(f"✗ Failed to initialize worker resources: {e}")
        raise


# ==============================================================================
# 4. HELPER FUNCTIONS
# ==============================================================================

def generate_filters_key(metadata_filters):
    """Genera una chiave univoca basata sui filtri applicati per non inquinare la cache."""
    if not metadata_filters: return "no_filters"
    # Ordina le chiavi per garantire che lo stesso set generi sempre lo stesso hash
    filter_str = json.dumps(metadata_filters, sort_keys=True)
    return hashlib.md5(filter_str.encode()).hexdigest()

def safe_json_parse(raw_text: str, task_id: str = "UNKNOWN") -> dict:
    """
    Estrae robustamente il JSON con log intermedi per capire ESATTAMENTE
    dove e perché il parser fallisce.
    """
    raw_text = raw_text.strip()
    # Logghiamo l'inizio del parsing (limitato ai primi 300 caratteri per non intasare i log)
    log.debug(f"[{task_id}] [JSON_PARSE] Inizio estrazione. Testo grezzo ricevuto: \n{raw_text[:300]}...")
    
    # 1. Tentativo standard
    try:
        parsed_data = json.loads(raw_text)
        log.debug(f"[{task_id}] [JSON_PARSE] ✓ Successo al Livello 1 (Standard Parse).")
        return parsed_data
    except json.JSONDecodeError as e:
        log.debug(f"[{task_id}] [JSON_PARSE] Livello 1 fallito: {e}. Passo al Livello 1b (Raw Decode).")
 
    # 1b. Tentativo con raw_decode: parsa il primo oggetto JSON valido e ignora
    # eventuale "extra data" successiva (es. una '}' spuria aggiunta dal modello
    # dopo la chiusura corretta dell'oggetto).
    try:
        decoder = json.JSONDecoder()
        parsed_data, _ = decoder.raw_decode(raw_text)
        log.debug(f"[{task_id}] [JSON_PARSE] ✓ Successo al Livello 1b (raw_decode, extra data ignorata).")
        return parsed_data
    except json.JSONDecodeError as e:
        log.debug(f"[{task_id}] [JSON_PARSE] Livello 1b fallito: {e}. Passo al Livello 2 (Markdown Strip).")
 
    # 2. Tentativo con pulizia Markdown esplicita
    clean_text = re.sub(r'^```json\s*|\s*```$', '', raw_text, flags=re.MULTILINE).strip()
    try:
        parsed_data = json.loads(clean_text)
        log.debug(f"[{task_id}] [JSON_PARSE] ✓ Successo al Livello 2 (Markdown rimosso).")
        return parsed_data
    except json.JSONDecodeError as e:
        log.debug(f"[{task_id}] [JSON_PARSE] Livello 2 fallito: {e}. Passo al Livello 3 (Brute Force Regex).")
 
    # 2b. Tentativo di riparazione stringa troncata
    try:
        repaired = clean_text.rstrip() + '"}'
        parsed_data = json.loads(repaired)
        log.debug(f"[{task_id}] [JSON_PARSE] ✓ Successo al Livello 2b (Stringa riparata).")
        return parsed_data
    except json.JSONDecodeError as e:
        log.debug(f"[{task_id}] [JSON_PARSE] Livello 2b fallito: {e}. Passo al Livello 3 (Brute Force Regex).")
 
    # 3. Tentativo "Forza Bruta": Cerca tutto ciò che è tra parentesi graffe
    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if match:
        try:
            parsed_data = json.loads(match.group(0))
            log.debug(f"[{task_id}] [JSON_PARSE] ✓ Successo al Livello 3 (Regex Regex Brute Force).")
            return parsed_data
        except json.JSONDecodeError as e:
            log.debug(f"[{task_id}] [JSON_PARSE] Livello 3 fallito: {e}.")
            
    # Fallimento totale
    log.error(f"[{task_id}] [JSON_PARSE] ✗ Fallimento totale. Impossibile estrarre JSON. Testo originale:\n{raw_text}")
    raise ValueError("Impossibile estrarre un JSON valido dal testo fornito.")

def safe_sigmoid(x):
    """
    Converte i logit in probabilità (0-1) prevenendo OverflowError in Python.
    """
    if x < -100:  # Qualsiasi logit sotto -100 è di fatto 0 probabilità
        return 0.0
    return 1 / (1 + math.exp(-x))


# ==============================================================================
# 4bis. MMR (Maximal Marginal Relevance) — diversificazione dei parent
# ==============================================================================
# Applicato SOLO in fase di redistribuzione degli slot liberi (vedi retrieve_chunks,
# step 5). La quota garantita per sottoquery non viene mai toccata da MMR: questo
# limita il rischio di escludere un parent rilevante ma "solo" per somiglianza
# vettoriale con un altro (vedi discussione: il vettore-proxy, un centroid dei
# chunk recuperati, non rappresenta l'intero parent).

MMR_LOG_TAG = "[MMR]"

def cosine(v1, v2):
    """Similarità coseno tra due vettori. Ritorna 0.0 se uno dei due è nullo."""
    v1, v2 = np.asarray(v1), np.asarray(v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / denom) if denom else 0.0


def select_with_mmr(candidates, selected_vectors, lam, sim_threshold, k, task_id="UNKNOWN"):
    """
    Seleziona fino a k parent tra 'candidates' massimizzando rilevanza e penalizzando
    la ridondanza rispetto ai parent già selezionati (already_selected, quota garantita).

    candidates: list di dict {"pid": str, "score": float, "vector": np.ndarray}
    selected_vectors: dict {pid: vector} dei parent GIÀ in quota (fissi, non rimovibili,
                      usati solo per calcolare la penalità di similarità)
    lam: peso rilevanza vs diversità. lam=1.0 -> comportamento identico al ranking puro.
    sim_threshold: la penalità scatta solo se la similarità supera questa soglia
                   (evita di penalizzare parent solo "un po'" simili ma complementari)
    k: numero di slot da riempire

    Ritorna: (picked_pids, debug_info) dove debug_info è una lista di dict utile al log
    (pid, score, max_sim_to_selected, penalizzato: bool)
    """
    debug_info = []

    if lam >= 1.0:
        # Nessuna penalità: puro ranking per score, identico al comportamento pre-MMR.
        ordered = sorted(candidates, key=lambda c: c["score"], reverse=True)
        picked = [c["pid"] for c in ordered[:k]]
        log.debug(f"[{task_id}] {MMR_LOG_TAG} lam=1.0 -> bypass, ranking puro su {len(candidates)} candidati.")
        return picked, debug_info

    picked = []
    pool = list(candidates)
    current_vectors = dict(selected_vectors)  # cresce ad ogni pick

    log.info(
        f"[{task_id}] {MMR_LOG_TAG} Avvio selezione: {len(candidates)} candidati, "
        f"{len(selected_vectors)} già selezionati (quota), lam={lam}, "
        f"sim_threshold={sim_threshold}, slot da riempire={k}."
    )

    while pool and len(picked) < k:
        best_item, best_mmr, best_max_sim, best_penalized = None, float("-inf"), 0.0, False

        for item in pool:
            valid_selected_vectors = [v for v in current_vectors.values() if v is not None]
            if valid_selected_vectors and item.get("vector") is not None:
                max_sim = max(cosine(item["vector"], v) for v in valid_selected_vectors)
            else:
                max_sim = 0.0  # nessun vettore disponibile: non penalizzabile per similarità
            penalized = max_sim >= sim_threshold
            penalty = max_sim if penalized else 0.0
            mmr_score = lam * item["score"] - (1 - lam) * penalty

            if mmr_score > best_mmr:
                best_mmr, best_item = mmr_score, item
                best_max_sim, best_penalized = max_sim, penalized

        picked.append(best_item["pid"])
        current_vectors[best_item["pid"]] = best_item["vector"]
        pool.remove(best_item)

        debug_info.append({
            "pid": best_item["pid"],
            "score": round(best_item["score"], 4),
            "max_sim_to_selected": round(best_max_sim, 4),
            "penalized": best_penalized,
            "mmr_score": round(best_mmr, 4),
        })

        log.debug(
            f"[{task_id}] {MMR_LOG_TAG} Scelto parent_id={best_item['pid']} "
            f"(score={best_item['score']:.4f}, max_sim_to_selected={best_max_sim:.4f}, "
            f"penalizzato={'sì' if best_penalized else 'no'}, mmr_score={best_mmr:.4f})."
        )

    n_penalized = sum(1 for d in debug_info if d["penalized"])
    log.info(
        f"[{task_id}] {MMR_LOG_TAG} Selezione completata: {len(picked)}/{k} slot riempiti, "
        f"{n_penalized} scelti nonostante penalità di ridondanza (sopra soglia {sim_threshold})."
    )

    return picked, debug_info


# ==============================================================================
# 4ter. PROFILI DI RETRY — escalation progressiva
# ==============================================================================
# Tentativo 0: comportamento attuale (recall di base, MMR di fatto disattivato).
# Tentativo 1: allarga la RECALL (più candidati, soglia più permissiva), MMR ancora spento
#              -> "forse non ho pescato abbastanza materiale, allargo la rete".
# Tentativo 2: recall ancora più larga + MMR attivo (lam<1)
#              -> "ho materiale ma è probabilmente un cluster semantico ripetitivo,
#                  forzo diversità per esplorare angolazioni diverse".
RETRY_PROFILES = {
    0: {"size_mult": 1.0, "threshold_mult": 1.0, "mmr_lambda": 1.0},
    1: {"size_mult": 1.5, "threshold_mult": 0.8, "mmr_lambda": 1.0},
    2: {"size_mult": 2.0, "threshold_mult": 0.7, "mmr_lambda": 0.8},
}


def profile_kwargs(attempt, task_id="UNKNOWN"):
    """Traduce il numero di tentativo in parametri concreti per retrieve_chunks."""
    profile = RETRY_PROFILES.get(attempt, RETRY_PROFILES[max(RETRY_PROFILES)])
    kwargs = {
        "semantic_size": int(settings.qdrant_semantic_size * profile["size_mult"]),
        "semantic_threshold": settings.qdrant_semantic_threshold * profile["threshold_mult"],
        "mmr_lambda": profile["mmr_lambda"],
    }
    log.info(
        f"[{task_id}] [RETRY_PROFILE] Tentativo {attempt} -> "
        f"semantic_size={kwargs['semantic_size']} "
        f"(x{profile['size_mult']}), semantic_threshold={kwargs['semantic_threshold']:.3f} "
        f"(x{profile['threshold_mult']}), mmr_lambda={kwargs['mmr_lambda']}."
    )
    return kwargs


def load_prompt_template(filename):
    """Load a prompt template from the prompts directory."""
    try:
        file_path = os.path.join(PROMPTS_DIR, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        log.error(f"Prompt template '{filename}' not found. Using fallback.")
        return "{context_str}\n\nQuestion: {query}\n\nAnswer:"
    except Exception as e:
        log.error(f"Error loading prompt {filename}: {e}")
        return "{context_str}\n\nQuestion: {query}\n\nAnswer:"


# ==============================================================================
# 5. LLM API FUNCTIONS
# ==============================================================================

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
def embed_for_concept_lookup(query):
    """Embedding per confronto con il dizionario concettuale (task simmetrico)."""
    result = embedding_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=dict(task_type="SEMANTIC_SIMILARITY", output_dimensionality=768)
    )
    return result.embeddings[0].values

def format_verified_concepts(concepts):
    """
    Formatta i concetti trovati nel dizionario per l'iniezione nel prompt
    del rewriter, nel formato "Concept: alias1, alias2, ...", una riga per
    concetto. Stringa vuota se non ci sono concetti (il prompt gestisce il
    caso esplicitamente).
    """
    if not concepts:
        return ""
    return "\n".join(
        f"{c['concept']}: {', '.join(c['aliases'])}"
        for c in concepts if c.get("concept") and c.get("aliases")
    )


def collect_concept_aliases(concepts):
    """Unione (deduplicata) di tutti gli alias dei concetti trovati, per le keyword testuali."""
    seen = set()
    out = []
    for c in concepts:
        for a in c.get("aliases", []):
            if a not in seen:
                seen.add(a)
                out.append(a)
    return out


def transform_query(history, query, task_id="UNKNOWN", verified_concepts=None):
    """
    Transform conversational query into standalone query + search facets + keywords.

    verified_concepts: lista di concetti dal dizionario Qdrant (output di
    get_concepts_by_similarity). Se presenti, vengono iniettati nel prompt
    come vocabolario autoritativo: il rewriter li usa direttamente per la
    decomposizione geografica (comuni verificati invece che indovinati) e
    per l'espansione tematica, in un'unica chiamata — eliminando la vecchia
    catena rewrite -> lookup -> seconda chiamata di espansione -> 
    riconciliazione in Python.
    """
    log.info(f"[{task_id}] [REWRITER] Avvio analisi query: '{query}'")

    if not history:
        log.debug(f"[{task_id}] [REWRITER] Nessuna history fornita.")
        history_str = ""
    else:
        history_str = "\n".join([
            f"{msg.get('role', 'user')}: {msg.get('text', '')}"
            for msg in history
        ])
        log.debug(f"[{task_id}] [REWRITER] History iniettata (elementi: {len(history)})")

    verified_concepts_str = format_verified_concepts(verified_concepts or [])
    if verified_concepts_str:
        log.debug(f"[{task_id}] [REWRITER] Concetti verificati iniettati:\n{verified_concepts_str}")

    prompt = load_prompt_template("query_rewriter").format(
        history_str=history_str,
        query=query,
        max_sub_queries=settings.max_sub_queries,
        verified_concepts_str=verified_concepts_str
    )

    try:
        log.debug(f"[{task_id}] [REWRITER] Chiamata al provider LLM in corso...")
        raw = get_llm_provider().generate_json(
            settings.query_rewriter_model_name, prompt, temperature=0.1
        )
        log.debug(f"[{task_id}] [REWRITER] Risposta grezza:\n{raw}")

        data = safe_json_parse(raw, task_id)
        standalone = data.get("standalone_query", query)
        searches   = data.get("search_queries", [query])
        keywords   = data.get("keywords", [])

        log.info(
            f"[{task_id}] [REWRITER] ✓ Query processata. "
            f"Standalone: '{standalone}' | Facets: {len(searches)} | Keywords: {len(keywords)}"
        )

        log.debug(f"Standalone query: {standalone}")
        log.debug(f"Query individuate: {searches}")
        log.debug(f"Keywords: {keywords}")
        return {"standalone_query": standalone, "search_queries": searches, "keywords": keywords}

    except Exception as e:
        log.warning(
            f"[{task_id}] [REWRITER] ✗ Fallimento critico: {e}. Attivazione fallback (query raw).",
            exc_info=True
        )
        return {"standalone_query": query, "search_queries": [query], "keywords": []}


def resolve_citations(answer_html: str, index_to_item: dict) -> str:
    """
    Sostituisce i riferimenti numerici scritti dal modello (es. <i>[3]</i> o
    <i>[2, 5]</i>) con il nome file CANONICO, preso direttamente dai metadati
    di retrieval (index_to_item), mai dal testo generato dal modello.
 
    Questo elimina alla radice il disallineamento tra la citazione mostrata
    nella risposta e il file_name reale usato per costruire i link di
    download (vedi CITATION RULE nel prompt: il modello ora cita SOLO indici
    numerici, mai filename).
    """
    def replace_match(m):
        raw_numbers = m.group(1)
        numbers = [n.strip() for n in raw_numbers.split(',')]
        file_names = []
        for n in numbers:
            try:
                idx = int(n)
            except ValueError:
                continue
            item = index_to_item.get(idx)
            if not item:
                log.warning(f"[CITATION] Riferimento [{idx}] generato dal modello non risolvibile: nessun chunk con questo indice.")
                continue
            fname = item["file_name"]
            if fname not in file_names:
                file_names.append(fname)
        if not file_names:
            # Nessun riferimento valido: rimuoviamo la citazione invece di mostrare un numero grezzo/errato
            return ""
        return "[" + ", ".join(file_names) + "]"
 
    # Cattura i pattern tipo [3] o [2, 5] ovunque compaiano nella risposta
    pattern = r'\[([\d,\s]+)\]'
    return re.sub(pattern, replace_match, answer_html)


# ==============================================================================
# 5bis. WINDOWED TRUNCATION — ritaglio del parent centrato sui child che hanno
# fatto match, invece di un taglio cieco dall'inizio del documento.
# ==============================================================================
# Quanti child (per parent) teniamo come "ancore" per il windowing. Tenerne
# più di uno permette di coprire casi in cui più frammenti dello stesso
# documento sono stati rilevanti per query diverse.
SNIPPETS_PER_PARENT = 3
# Lunghezza dell'ancora usata per localizzare lo snippet dentro il parent:
# corta a sufficienza da tollerare piccole differenze di whitespace/normalizzazione
# tra come il child è stato salvato e come appare dentro il testo del parent.
SNIPPET_ANCHOR_LEN = 80


def extract_relevant_window(content: str, matched_snippets: list, budget: int, task_id: str = "UNKNOWN") -> str:
    """
    Se il parent intero sta nel budget, nessun problema. Altrimenti, invece di
    tagliare ciecamente i primi `budget` caratteri (rischiando di perdere
    proprio il passaggio che ha fatto vincere questo parent al retrieval),
    proviamo a localizzare i child/snippet che hanno fatto match e ritagliamo
    una finestra centrata su di loro.

    Fallback in cascata se la localizzazione fallisce (es. child normalizzato
    diversamente dal parent in fase di ingestion): torna al comportamento
    precedente (troncamento dall'inizio), mai un'eccezione.
    """
    if len(content) <= budget:
        return content

    if not matched_snippets:
        log.debug(f"[{task_id}] [WINDOW] Nessuno snippet disponibile: fallback a troncamento dall'inizio.")
        return content[:budget]

    spans = []
    for snippet in matched_snippets:
        if not snippet:
            continue
        anchor = " ".join(snippet.split())[:SNIPPET_ANCHOR_LEN]
        if not anchor:
            continue
        idx = content.find(anchor)
        if idx == -1:
            # Tentativo più permissivo: normalizza anche il documento (whitespace)
            # prima di cercare. Approssimato (gli indici non combaciano più 1:1
            # coi caratteri originali), ma preferibile a non trovare nulla.
            normalized = " ".join(content.split())
            idx_norm = normalized.find(anchor)
            if idx_norm != -1:
                idx = idx_norm
        if idx != -1:
            spans.append((idx, idx + len(snippet)))

    if not spans:
        log.debug(f"[{task_id}] [WINDOW] Nessuno snippet localizzato nel parent: fallback a troncamento dall'inizio.")
        return content[:budget]

    # Una finestra di margine per ogni ancora trovata, poi merge di quelle sovrapposte.
    spans.sort()
    margin = max(200, budget // (2 * len(spans)))
    windows = [(max(0, s - margin), min(len(content), e + margin)) for s, e in spans]

    merged = [windows[0]]
    for s, e in windows[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))

    pieces = [content[s:e] for s, e in merged]
    result = "\n[...]\n".join(pieces)

    if len(result) > budget:
        result = result[:budget]

    log.debug(
        f"[{task_id}] [WINDOW] Finestra costruita da {len(spans)} snippet localizzati, "
        f"{len(merged)} blocchi dopo merge, {len(result)}/{budget} chars usati "
        f"(vs {len(content)} chars totali nel parent)."
    )
    return result


def generate_answer(query, rich_context, topic_id):
    formatted_chunks = []
    curr_len = 0
    n_parents = len(rich_context)
    index_to_item = {}

    if n_parents == 0:
        context_str = ""
    else:
        theoretical_budget_per_parent = settings.max_context_chars // (settings.parents_per_query * settings.max_sub_queries)
        cap_per_parent = int(theoretical_budget_per_parent / settings.parent_budget_ratio)
        budget_per_parent = min(settings.max_context_chars // n_parents, cap_per_parent)

        log.debug(
            f"Budget per parent: {budget_per_parent} chars "
            f"(teorico={theoretical_budget_per_parent}, cap={cap_per_parent}, "
            f"parent effettivi={n_parents})."
        )

        for idx, item in enumerate(rich_context, start=1):
            content = item['content']
            if len(content) > budget_per_parent:
                original_len = len(content)
                content = extract_relevant_window(
                    content, item.get('matched_snippets', []), budget_per_parent
                )
                log.debug(
                    f"Parent '{item['source']}' (rif. [{idx}]) ridotto: "
                    f"{original_len} → {len(content)} chars (windowed su child match)."
                )

            # NOTA: l'header espone al modello SOLO un indice numerico, mai il
            # filename. Il modello non può più scrivere/storpiare un nome file:
            # può solo riferirsi a "idx", che viene poi risolto in modo
            # deterministico da resolve_citations() usando index_to_item.
            header = f"[{idx}"
            if item.get('date'):
                header += f" | Date: {item['date']}"
            else:
                header += f" | Date: unknown"
            header += "]"

            chunk = f"{header}\n{content}\n\n"

            if curr_len + len(chunk) > settings.max_context_chars:
                log.warning(
                    f"Budget complessivo raggiunto ({curr_len}/{settings.max_context_chars} chars). "
                    f"{n_parents - len(formatted_chunks)} parent scartati."
                )
                break

            formatted_chunks.append(chunk)
            curr_len += len(chunk)
            index_to_item[idx] = item

        context_str = "".join(formatted_chunks)

    log.info(
        f"Contesto finale: {len(formatted_chunks)}/{n_parents} parent, "
        f"{curr_len}/{settings.max_context_chars} chars utilizzati."
    )

    prompt_file  = get_topic_prompt(topic_id)
    prompt_tmpl  = load_prompt_template(prompt_file)
    prompt       = prompt_tmpl.format(context_str=context_str, query=query)

    log.info(f"Generating structured answer for topic '{topic_id}'")
    log.debug(f"Context size: {len(context_str)} chars")
    log.debug(f"Context  {context_str[:500]}...")

    raw = get_llm_provider().generate_json(settings.answer_generator_model_name, prompt, max_tokens=settings.answer_max_tokens, thinking_level=settings.answer_thinking_level )
    log.debug(f"Raw response:\n{raw[:500]}...")

    try:
        result_data = safe_json_parse(raw, task_id=topic_id)
        raw_answer = str(result_data.get("answer", ""))
        resolved_answer = resolve_citations(raw_answer, index_to_item)
        return {
            "is_found": bool(result_data.get("is_found", True)),
            "is_general_knowledge": bool(result_data.get("is_general_knowledge", False)),
            "answer": resolved_answer,
        }
    except json.JSONDecodeError as e:
        log.error(f"Generazione JSON fallita: {e}. Output grezzo: {raw}")
        return {"is_found": True, "is_general_knowledge": False, "answer": raw.strip()}
    

def grade_answer(query, context_snippet, answer):
    """Ask the grader model whether the answer is satisfactory. Returns True/False."""
    grader_tmpl   = load_prompt_template("grader")
    grader_prompt = grader_tmpl.format(
        query=query,
        context_snippet=context_snippet,
        answer=answer
    )
    raw = get_llm_provider().generate_text(
        settings.grader_model_name, grader_prompt, temperature=0.0, max_tokens=5
    )
    result = raw.strip().upper()
    log.info(f"Grader response: '{result}'")
    return result.startswith("YES")


# ==============================================================================
# 6. SEMANTIC CACHE FUNCTIONS
# ==============================================================================

def check_semantic_cache(query_vector, topic_id, sub_topics_key, filters_key):
    """Check if similar query exists in cache."""
    if not qdrant_client:
        return None
    
    try:
        hits = qdrant_client.query_points(
            collection_name=CACHE_COLLECTION,
            query=query_vector,
            limit=1,
            score_threshold=0.97,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(key="topic_id", match=models.MatchValue(value=topic_id)),
                    models.FieldCondition(key="sub_topics_key", match=models.MatchValue(value=sub_topics_key)),
                    models.FieldCondition(key="filters_key", match=models.MatchValue(value=filters_key)) 
                ]
            )
        ).points
        
        if hits:
            log.info(f"✓ Cache HIT (similarity: {hits[0].score:.3f}) for topic '{topic_id}' & sub_topics '{sub_topics_key}'")
            return hits[0].payload
        
        log.debug("Cache MISS")
        return None
        
    except Exception as e:
        log.error(f"Cache check failed: {e}")
        return None

def save_to_semantic_cache(query_vector, original_query, answer, sources, topic_id, sub_topics_key, filters_key):
    """Salva la risposta in cache includendo la chiave di combinazione dei sub-topic."""
    if not qdrant_client:
        return
    
    try:
        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector=query_vector,
            payload={
                "original_query": original_query,
                "answer": answer,
                "sources": sources,
                "topic_id": topic_id,
                "sub_topics_key": sub_topics_key, 
                "filters_key": filters_key,
                "timestamp": time.time()
            }
        )
        qdrant_client.upsert(collection_name=CACHE_COLLECTION, points=[point])
        log.debug("Cache entry saved.")
        
    except Exception as e:
        log.error(f"Cache save failed: {e}")

def get_topic_prompt(topic_id):
    """Recupera il template del prompt associato al topic. Ritorna 'general' come fallback."""
    conn = get_db_connection()
    if not conn:
        return 'general'
    
    try:
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute("SELECT prompt FROM topics WHERE topic_id = %s", (topic_id,))
            row = cursor.fetchone()
            # Se il campo prompt è NULL o vuoto, restituisce 'general'
            return row['prompt'] if row and row.get('prompt') else 'general'
    except Exception as e:
        log.error(f"Errore durante il recupero del prompt per il topic '{topic_id}': {e}")
        return 'general'
    finally:
        conn.close()

def generate_sub_topics_key(selected_sub_topics):
   return ",".join(sorted(selected_sub_topics))


@celery_app.task(name="prune_semantic_cache")
def prune_semantic_cache():
    """Delete cache entries older than MAX_AGE_SECONDS."""
    if not qdrant_client:
        log.warning("Qdrant client not available for cache pruning.")
        return
    
    try:
        cutoff_time = time.time() - MAX_AGE_SECONDS
        
        qdrant_client.delete(
            collection_name=CACHE_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="timestamp",
                            range=models.Range(lt=cutoff_time)
                        )
                    ]
                )
            )
        )
        
        log.info(f"✓ Cache pruned (entries older than {MAX_AGE_SECONDS}s removed).")
        
    except Exception as e:
        log.error(f"Cache pruning failed: {e}")


@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Setup scheduled tasks."""
    sender.add_periodic_task(
        crontab(minute=0),
        prune_semantic_cache.s(),
        name='hourly-cache-cleanup'
    )
    log.info("✓ Scheduled task: Cache cleanup every hour")


# ==============================================================================
# 7. DOCUMENT RETRIEVAL (With OPTIMIZED Reranking)
# ==============================================================================
def retrieve_chunks(search_queries, vectors_list, keywords, topic_id, selected_sub_topics,
                     metadata_filters, include_undated=True,
                     semantic_size=None, semantic_threshold=None, mmr_lambda=1.0,
                     task_id="UNKNOWN"):
    """
    Recupera, fonde e riordina i chunk dal Vector DB in modo sicuro.

    semantic_size / semantic_threshold: sovrascrivono i default di settings, usati
        dai profili di retry per allargare progressivamente la recall.
    mmr_lambda: peso rilevanza/diversità nella redistribuzione degli slot liberi.
        1.0 = comportamento identico a prima dell'introduzione di MMR.
    """
    if not qdrant_client:
        log.error("Qdrant client not available! Retrieval aborted.")
        return [], [], {}

    semantic_size = semantic_size or settings.qdrant_semantic_size
    semantic_threshold = semantic_threshold if semantic_threshold is not None else settings.qdrant_semantic_threshold

    log.info(f"=== Inizio Retrieval per topic '{topic_id}' ===")
    log.info(
        f"[{task_id}] [RETRIEVAL_PARAMS] semantic_size={semantic_size}, "
        f"semantic_threshold={semantic_threshold:.3f}, mmr_lambda={mmr_lambda}."
    )
    log.debug(f"Search Queries: {search_queries} | Keywords: {keywords}")
    
    try:
        must_conditions = [models.FieldCondition(key="topic_id", match=models.MatchValue(value=topic_id))]
        if selected_sub_topics:
            must_conditions.append(models.FieldCondition(key="sub_topic_id", match=models.MatchAny(any=selected_sub_topics)))

        # === INIEZIONE FILTRI METADATI ===
        if metadata_filters:
            for key, value in metadata_filters.items():
                
                # Ricerca full-text tokenizzata: {"oggetto__text": "disposizioni generali"}
                if key.endswith("__text"):
                    real_key = key[:-6]  # rimuove "__text"
                    must_conditions.append(models.FieldCondition(
                        key=real_key,
                        match=models.MatchText(text=str(value))
                    ))
                
                # Filtro range: {"data": {"gte": "2024-01-01", "lte": "2024-06-30"}}
                # oppure numerico: {"prezzo": {"gte": 10, "lte": 100}}
                # NB: i metadati sono dinamici (dipendono dalla fonte di ingestion),
                # quindi un documento può legittimamente non avere questo campo.
                # In tal caso lo includiamo comunque (non escludiamo per "assenza
                # di dato", solo per "dato fuori range"): range_condition è
                # soddisfatta OPPURE il campo non esiste affatto.
                elif isinstance(value, dict) and ("gte" in value or "lte" in value):
                    range_values = [v for v in (value.get("gte"), value.get("lte")) if v is not None]
                    is_date_range = any(isinstance(v, str) for v in range_values)

                    if is_date_range:
                        # models.Range accetta solo float: per stringhe (es. date
                        # ISO "YYYY-MM-DD") serve DatetimeRange, altrimenti Qdrant
                        # solleva un errore di validazione.
                        range_condition = models.FieldCondition(
                            key=key,
                            range=models.DatetimeRange(
                                gte=value.get("gte"),
                                lte=value.get("lte")
                            )
                        )
                    else:
                        range_condition = models.FieldCondition(
                            key=key,
                            range=models.Range(
                                gte=value.get("gte"),
                                lte=value.get("lte")
                            )
                        )

                    if include_undated:
                        # Il documento passa se rientra nel range OPPURE se il
                        # campo non esiste affatto (comportamento di default:
                        # non escludiamo per "assenza di dato").
                        must_conditions.append(models.Filter(
                            should=[
                                range_condition,
                                models.IsEmptyCondition(is_empty=models.PayloadField(key=key))
                            ]
                        ))
                    else:
                        # L'utente ha chiesto di escludere i documenti privi
                        # del campo: nessuna clausola di "should", il documento
                        # deve soddisfare strettamente il range.
                        must_conditions.append(range_condition)
                
                # Filtro multi-valore: {"stato": ["approvato", "pubblicato"]}
                elif isinstance(value, list):
                    must_conditions.append(models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value)
                    ))
                
                # Filtro esatto: {"numero_atto": "123"}
                else:
                    must_conditions.append(models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    ))

        # ==========================================================
        # 1. FUNZIONI DI RICERCA (Con Exception Handling Interno)
        # ==========================================================
        def safe_vector_search(vector, idx):
            try:
                res = qdrant_client.query_points_groups(
                    collection_name=QDRANT_COLLECTION,
                    query=vector,
                    group_by="content_hash",
                    group_size=1,
                    limit=semantic_size,
                    score_threshold=semantic_threshold,
                    query_filter=models.Filter(must=must_conditions),
                    with_vectors=True  # necessario per costruire i centroid-per-parent usati da MMR
                )
                points = [group.hits[0] for group in res.groups if group.hits]
                log.debug(f"Vector search #{idx} returned {len(points)} grouped hits.")
                return points
            except Exception as e:
                log.error(f"Vector search #{idx} failed: {e}")
                return []
            
        def safe_keyword_search():
            if not keywords: return []
            try:
                clean_kw = [w.lower().strip() for w in keywords if len(w.strip()) > 2]
                if not clean_kw: return []

                should_cond = [models.FieldCondition(key="content", match=models.MatchText(text=w)) for w in clean_kw]
                target_size = settings.qdrant_syntactic_size

                seen_hashes: set = set()
                unique_hits = []
                next_offset = None
                max_scroll_iterations = 5  # safety: evita loop infiniti se il corpus è quasi tutto duplicato
                iterations = 0

                while len(unique_hits) < target_size and iterations < max_scroll_iterations:
                    log.debug(f"Scroll {iterations} di {max_scroll_iterations}")

                    res, next_offset = qdrant_client.scroll(
                        collection_name=QDRANT_COLLECTION,
                        scroll_filter=models.Filter(must=must_conditions + [models.Filter(should=should_cond)]),
                        limit=target_size * 2,  # batch ampio per compensare i duplicati scartati
                        offset=next_offset,
                        with_payload=True,
                        with_vectors=True  # necessario per costruire i centroid-per-parent usati da MMR
                    )
                    iterations += 1

                    if not res:
                        break

                    for point in res:
                        h = point.payload.get("content_hash") if point.payload else None
                        if h and h not in seen_hashes:
                            seen_hashes.add(h)
                            unique_hits.append(point)
                            if len(unique_hits) >= target_size:
                                break

                    if next_offset is None:
                        break  # esauriti i punti nella collection per questo filtro

                log.debug(f"Keyword search returned {len(unique_hits)} unique hits (deduped, {iterations} scroll iterations).")
                return unique_hits
            except Exception as e:
                log.error(f"Keyword search failed: {e}")
                return []

        # ==========================================================
        # ESECUZIONE PARALLELA 
        # ==========================================================
        hits_per_query = [[] for _ in search_queries]
        
        with ThreadPoolExecutor(max_workers=min(10, len(vectors_list) + 1)) as executor:
            future_kw = executor.submit(safe_keyword_search)
            future_map = {
                executor.submit(safe_vector_search, vec, idx): idx
                for idx, vec in enumerate(vectors_list)
            }
            keyword_hits = future_kw.result()
            for future, idx in future_map.items():
                try:
                    hits_per_query[idx] = future.result()
                except Exception as e:
                    log.error(f"Error resolving vector search future #{idx}: {e}")
 
        # Keyword hits come candidati extra della query principale
        if keyword_hits:
            kw_ids = {h.id for h in hits_per_query[0]}
            hits_per_query[0] = hits_per_query[0] + [h for h in keyword_hits if h.id not in kw_ids]

        # ==========================================================
        # 3. RERANK PER-QUERY — salva top_chunks per riuso nella redistribuzione
        # ==========================================================
        reranker = get_reranker()
        n_queries = len(search_queries)
        top_chunks_per_query: list[list] = [[] for _ in range(n_queries)]

        def process_single_rerank(q_idx, query_str, candidates):
            if not candidates:
                log.debug(f"Sottoquery #{q_idx} '{query_str[:40]}': nessun candidato.")
                return []
            
            # De-duplica i chunk per id in modo più efficiente
            seen_chunk_ids = {c.id: c for c in candidates}
            unique_candidates = list(seen_chunk_ids.values())

            # De-duplica cross duplicati
            n_before_content_dedup = len(unique_candidates)
            seen_content_hashes = {}
            deduped_candidates = []
            for c in unique_candidates:
                h = c.payload.get("content_hash") if c.payload else None
                if not h:
                    content = (c.payload.get("content", "") if c.payload else "")
                    h = hashlib.md5(" ".join(content.lower().split()).encode()).hexdigest()
                if h not in seen_content_hashes:
                    seen_content_hashes[h] = True
                    deduped_candidates.append(c)
            unique_candidates = deduped_candidates

            if len(unique_candidates) < n_before_content_dedup:
                log.debug(
                    f"Sottoquery #{q_idx} '{query_str[:40]}': "
                    f"{n_before_content_dedup} → {len(unique_candidates)} candidati "
                    f"(rimossi {n_before_content_dedup - len(unique_candidates)} duplicati)."
                )

            # Rerank effettivo
            if reranker:
                docs_content = [
                    c.payload.get("content", "")[:settings.rerank_truncate]
                    for c in unique_candidates
                ]
                try:
                    reranked = reranker.rerank(query_str, docs_content)
                    reranked.sort(key=lambda x: x.score, reverse=True)
                    top_chunks = []
                    for res in reranked[:settings.rerank_size]:
                        prob = safe_sigmoid(res.score)
                        if prob >= settings.min_prob_threshold or len(top_chunks) < 2:
                            top_chunks.append(unique_candidates[res.index])
                        else:
                            break
                    return top_chunks
                except Exception as e:
                    log.error(f"Reranker failed for query #{q_idx}: {e}. Fallback.")
                    
            unique_candidates.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)
            return unique_candidates[:settings.rerank_size]
        
        max_workers = min(n_queries, settings.max_reranker_thread)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(process_single_rerank, q_idx, q_str, cands): q_idx
                for q_idx, (q_str, cands) in enumerate(zip(search_queries, hits_per_query))
            }
            
            for future in as_completed(future_to_idx):
                q_idx = future_to_idx[future]
                try:
                    top_chunks_per_query[q_idx] = future.result()
                    log.debug(f"Sottoquery #{q_idx}: rerank completato con {len(top_chunks_per_query[q_idx])} chunk finali.")
                except Exception as e:
                    log.error(f"Errore critico nel thread di rerank per la query #{q_idx}: {e}", exc_info=True)
                    top_chunks_per_query[q_idx] = []


        # ==========================================================
        # 3bis. CENTROID PER PARENT (vettore-proxy per MMR)
        #
        # Il parent non ha un proprio embedding: usiamo la media (centroid) dei
        # vettori di tutti i chunk recuperati per quel parent, tra tutte le
        # sottoquery. È più rappresentativo di un singolo "chunk vincitore"
        # perché riflette tutto ciò che il retrieval ha effettivamente trovato
        # di quel documento, non solo il suo frammento più rilevante.
        # ==========================================================
        parent_chunk_vectors: dict[str, list] = {}
        parent_best_score: dict[str, float] = {}
        # NEW: fino a SNIPPETS_PER_PARENT child (contenuto + score) per parent,
        # usati da generate_answer per il windowing invece del troncamento cieco.
        parent_top_snippets: dict[str, list] = {}

        for chunks in top_chunks_per_query:
            for chunk in chunks:
                pid = chunk.payload.get("parent_id") if chunk.payload else None
                if not pid:
                    continue
                score = getattr(chunk, "score", 0.0) or 0.0
                if pid not in parent_best_score or score > parent_best_score[pid]:
                    parent_best_score[pid] = score

                vec = getattr(chunk, "vector", None)
                if vec is not None:
                    parent_chunk_vectors.setdefault(pid, []).append(np.asarray(vec))

                chunk_content = chunk.payload.get("content", "") if chunk.payload else ""
                if chunk_content:
                    snippets = parent_top_snippets.setdefault(pid, [])
                    snippets.append({"content": chunk_content, "score": score})
                    snippets.sort(key=lambda s: s["score"], reverse=True)
                    del snippets[SNIPPETS_PER_PARENT:]

        parent_centroids = {
            pid: np.mean(vecs, axis=0) for pid, vecs in parent_chunk_vectors.items()
        }

        n_missing_vectors = sum(1 for pid in parent_best_score if pid not in parent_centroids)
        log.info(
            f"[{task_id}] {MMR_LOG_TAG} Centroid calcolati per {len(parent_centroids)} parent "
            f"(su {len(parent_best_score)} parent candidati; {n_missing_vectors} senza vettore disponibile)."
        )

        # ==========================================================
        # 4. QUOTA GARANTITA PER SOTTOQUERY
        #
        # Ogni sottoquery contribuisce al massimo PARENTS_PER_QUERY
        # parent distinti. Si scende la lista rerankkata e si prendono
        # i primi PARENTS_PER_QUERY parent_id unici incontrati.
        # I parent già selezionati da sottoquery precedenti non vengono
        # ricontati: ogni slot è un documento NUOVO nel pool finale.
        # ==========================================================
        quota_per_query: list[list[str]] = []
        already_selected: set[str] = set()
 
        for q_idx, (query_str, top_chunks) in enumerate(zip(search_queries, top_chunks_per_query)):
            selected_for_query: list[str] = []
            seen_parents_this_query: set[str] = set()
 
            for chunk in top_chunks:
                pid = chunk.payload.get("parent_id") if chunk.payload else None
                if not pid or pid in seen_parents_this_query:
                    continue
                seen_parents_this_query.add(pid)
                if pid not in already_selected:
                    already_selected.add(pid)
                    selected_for_query.append(pid)
                    if len(selected_for_query) >= settings.parents_per_query:
                        break
 
            quota_per_query.append(selected_for_query)
            log.info(
                f"Sottoquery #{q_idx} '{query_str[:40]}': "
                f"{len(top_chunks)} chunk → {len(selected_for_query)} parent in quota."
            )
 
        # ==========================================================
        # 5. REDISTRIBUZIONE SLOT LIBERI
        #
        # Se alcune sottoquery hanno trovato 0 risultati (o meno di
        # PARENTS_PER_QUERY), i loro slot vengono offerti alle
        # sottoquery con più materiale disponibile, in round-robin.
        # Questo evita di sprecare budget quando, ad esempio, 4 comuni
        # su 5 non hanno documenti e uno solo ha molti risultati.
        # ==========================================================
        total_budget = settings.parents_per_query * n_queries
        slots_free = total_budget - len(already_selected)

        # Inizializzati qui così restano definiti (a 0/vuoto) anche se non
        # scatta la redistribuzione — servono per le metriche finali.
        n_mmr_candidates = 0
        n_mmr_penalized = 0

        if slots_free > 0:
            log.info(f"Redistribuzione: {slots_free} slot liberi su {total_budget} totali.")

            # Riserve per ogni query: parent validi oltre la quota, nell'ordine del reranker.
            # pid_owner_query serve per sapere a quale sottoquery "restituire" il pid dopo
            # la selezione MMR, così quota_per_query resta coerente con la struttura precedente.
            reserve_per_query: list[list[str]] = []
            pid_owner_query: dict[str, int] = {}
            for q_idx, top_chunks in enumerate(top_chunks_per_query):
                reserve: list[str] = []
                seen_parents: set[str] = set()
                for chunk in top_chunks:
                    pid = chunk.payload.get("parent_id") if chunk.payload else None
                    if not pid or pid in seen_parents:
                        continue
                    seen_parents.add(pid)
                    if pid not in already_selected:
                        reserve.append(pid)
                        pid_owner_query.setdefault(pid, q_idx)
                reserve_per_query.append(reserve)

            # Pool unico di candidati (deduplicato) per la selezione MMR globale.
            # Solo i pid con centroid disponibile entrano nella valutazione MMR;
            # quelli senza vettore (raro: mismatch nel recupero with_vectors) vengono
            # solo loggati, non selezionabili da MMR (con lam=1.0 il problema non si pone,
            # perché MMR non serve nemmeno il vettore per ordinare per score).
            seen_reserve: set[str] = set()
            reserve_pool = []
            n_skipped_no_vector = 0
            for reserve in reserve_per_query:
                for pid in reserve:
                    if pid in seen_reserve:
                        continue
                    seen_reserve.add(pid)
                    if pid in parent_centroids:
                        reserve_pool.append({
                            "pid": pid,
                            "score": parent_best_score.get(pid, 0.0),
                            "vector": parent_centroids[pid],
                        })
                    else:
                        n_skipped_no_vector += 1
                        # Fallback: se manca il vettore includiamo comunque il pid con score,
                        # così anche con lam<1 non lo perdiamo (verrà solo trattato come non
                        # comparabile: nessuna penalità di similarità applicabile su di lui).
                        reserve_pool.append({
                            "pid": pid,
                            "score": parent_best_score.get(pid, 0.0),
                            "vector": None,
                        })

            if n_skipped_no_vector:
                log.warning(
                    f"[{task_id}] {MMR_LOG_TAG} {n_skipped_no_vector} parent candidati alla "
                    f"redistribuzione senza vettore disponibile (inclusi comunque, senza "
                    f"possibilità di essere penalizzati per similarità)."
                )

            already_selected_vectors = {
                pid: parent_centroids[pid] for pid in already_selected if pid in parent_centroids
            }

            picked_pids, mmr_debug = select_with_mmr(
                reserve_pool,
                already_selected_vectors,
                lam=mmr_lambda,
                sim_threshold=settings.mmr_similarity_threshold,
                k=slots_free,
                task_id=task_id,
            )

            redistributed = 0
            for pid in picked_pids:
                q_idx = pid_owner_query.get(pid)
                if q_idx is None:
                    log.warning(f"[{task_id}] {MMR_LOG_TAG} pid={pid} scelto ma senza sottoquery proprietaria: skip.")
                    continue
                quota_per_query[q_idx].append(pid)
                already_selected.add(pid)
                redistributed += 1

            log.info(
                f"[{task_id}] Redistribuzione completata: {redistributed} parent aggiunti "
                f"(mmr_lambda={mmr_lambda}, candidati valutati={len(reserve_pool)})."
            )
            if mmr_debug:
                log.debug(f"[{task_id}] {MMR_LOG_TAG} Dettaglio scelte: {mmr_debug}")

            n_mmr_candidates = len(reserve_pool)
            n_mmr_penalized = sum(1 for d in mmr_debug if d["penalized"])
 
        # ==========================================================
        # 5bis. STATISTICHE DI RETRIEVAL (per la tabella rag_metrics)
        # ==========================================================
        retrieval_stats = {
            "semantic_size": semantic_size,
            "semantic_threshold": round(semantic_threshold, 4),
            "mmr_lambda": mmr_lambda,
            "n_parent_candidates": len(parent_best_score),
            "n_mmr_candidates": n_mmr_candidates,
            "n_mmr_penalized": n_mmr_penalized,
            "n_parents_selected": len(already_selected),
        }

        # ==========================================================
        # 6. POOL FINALE — ordine FIFO per sottoquery
        # ==========================================================
        final_parent_ids_ordered: list[str] = []
        seen_final: set[str] = set()
 
        for selected_pids in quota_per_query:
            for pid in selected_pids:
                if pid not in seen_final:
                    final_parent_ids_ordered.append(pid)
                    seen_final.add(pid)
 
        if not final_parent_ids_ordered:
            log.warning("Nessun parent_id estratto da nessuna sottoquery.")
            return [], [], retrieval_stats
 
        log.info(
            f"Pool finale: {len(final_parent_ids_ordered)} parent distinti "
            f"(budget={settings.parents_per_query}×{n_queries}={total_budget})."
        )
 
 
        # ==========================================================
        # 7. RECUPERO PARENT DOCUMENTS (dal DB)
        # ==========================================================
        parent_records = []
        conn = get_db_connection()
        if not conn:
            return [], [], retrieval_stats
 
        try:
            log.debug(f"Parent ID da recuperare da DB: {final_parent_ids_ordered}")

            format_strings = ','.join(['%s'] * len(final_parent_ids_ordered))
            sql = f"""
                SELECT id, topic_id, sub_topic_id, source, file_name, content, metadata
                FROM parent_documents
                WHERE id IN ({format_strings})
            """
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute(sql, tuple(final_parent_ids_ordered))
                parent_records = cursor.fetchall()

            log.debug(f"Record restituiti dal DB: {len(parent_records)}")
            if not parent_records:
                log.error(
                    f"MySQL ha restituito 0 record per {len(final_parent_ids_ordered)} parent_id. "
                    f"Primo ID cercato: {final_parent_ids_ordered[0] if final_parent_ids_ordered else 'N/A'}. "
                    f"Verificare tipo colonna id e sincronizzazione Qdrant↔MySQL."
                )
        finally:
            conn.close()
 
        # Riordina i parent nell'ordine in cui sono stati aggiunti
        # (rispetta la priorità per sottoquery)
        pid_to_record = {r['id']: r for r in parent_records}
        parent_records = [pid_to_record[pid] for pid in final_parent_ids_ordered if pid in pid_to_record]
 
 
        # ==========================================================
        # 8. FORMATTAZIONE OUTPUT
        # ==========================================================
        rich_context = []
        unique_sources_map = {}
 
        for p_doc in parent_records:
            source = p_doc.get("source", "Fonte_Sconosciuta")
            file_name = p_doc.get("file_name", "File_Sconosciuto")
            sub_topic_id = p_doc.get("sub_topic_id", "")
            content = p_doc.get("content", "").strip()

            # Estrae la data rilevata dal documento (se presente) dal JSON di
            # metadata salvato in MySQL. Priorità: "data" > "data_pubblicazione"
            # > "data_esecutivita" (stessi campi indicizzati come DATETIME su
            # Qdrant, vedi setup_qdrant.py).
            doc_date = None
            raw_metadata = p_doc.get("metadata")
            if raw_metadata:
                try:
                    meta = raw_metadata if isinstance(raw_metadata, dict) else json.loads(raw_metadata)
                    doc_date = (
                        meta.get("data")
                        or meta.get("data_pubblicazione")
                        or meta.get("data_esecutivita")
                    )
                except (json.JSONDecodeError, TypeError, AttributeError) as e:
                    log.debug(f"Metadata non parsabile per parent_id={p_doc.get('id')}: {e}")

            if content:
                matched_snippets = [
                    s["content"] for s in parent_top_snippets.get(p_doc.get("id"), []) if s.get("content")
                ]
                rich_context.append({
                    "content": content,
                    "source": source,
                    "date": doc_date,
                    "file_name": file_name,
                    "sub_topic": sub_topic_id,
                    "matched_snippets": matched_snippets,
                })
                if source not in unique_sources_map:
                    unique_sources_map[source] = {"source": source, "sub_topic": sub_topic_id, "file_name": file_name, "date": doc_date}
 
        log.info(f"=== Retrieval completata. Parent al generatore: {len(rich_context)} ===")
        retrieval_stats["n_parents_final"] = len(rich_context)
        return rich_context, list(unique_sources_map.values()), retrieval_stats
 
    except Exception as e:
        log.critical(f"Errore critico nel Retrieval: {e}", exc_info=True)
        return [], [], {}

def get_reranker_pool():
    global _RERANKER_POOL
    if not _RERANKER_POOL:
        for i in range(_POOL_SIZE):
            log.info(f"Initializing ONNX Reranker pool instance {i+1}/{_POOL_SIZE}...")
            instance = ONNXReranker(
                model_folder=settings.onnx_model_cache_path,
                batch_size=settings.rerank_batch_size,
                max_length=settings.rerank_max_length,
                num_threads=2
            )
            _RERANKER_POOL.append(instance)
    return _RERANKER_POOL

def rerank_subquery(args):
    pool_idx, query_str, docs_content = args
    pool = get_reranker_pool()
    reranker_instance = pool[pool_idx % len(pool)]
    return reranker_instance.rerank(query_str, docs_content)


def get_reranker():
    """
    Lazy loader for the Reranker.
    Ensures initialization happens INSIDE the worker process, avoiding deadlocks.
    """
    global _RERANKER_INSTANCE
    if _RERANKER_INSTANCE is None:
        try:
            log.info("Initializing ONNX Reranker (Lazy Load)...")
            num_threads = min(os.cpu_count() or 2, settings.max_reranker_thread) 
            log.info(f"Using {num_threads} threads...")

            _RERANKER_INSTANCE = ONNXReranker(
                model_folder=settings.onnx_model_cache_path,
                batch_size=settings.rerank_batch_size,
                max_length=settings.rerank_max_length,
                num_threads=num_threads
            )
            log.info("✓ Reranker initialized successfully.")
        except Exception as e:
            log.error(f"✗ Failed to lazy load Reranker: {e}")
            _RERANKER_INSTANCE = False # Mark as failed so we don't retry every time
            
    return _RERANKER_INSTANCE if _RERANKER_INSTANCE is not False else None



def get_all_sub_topics(topic_id):
    """Recupera tutti i sub_topic_id associati a un topic dal database."""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute("SELECT sub_topic_id FROM sub_topics WHERE topic_id = %s", (topic_id,))
            rows = cursor.fetchall()
            return [row['sub_topic_id'] for row in rows]
    except Exception as e:
        log.error(f"Errore recupero tutti i sub-topics per '{topic_id}': {e}")
        return []
    finally:
        conn.close()

def get_concepts_by_similarity(query_vector, threshold=settings.qdrant_concept_threshold, task_id="UNKNOWN"):
    """
    Ricerca UNICA sul dizionario concettuale: ritorna TUTTI i concetti sopra
    soglia (fino a settings.qdrant_concept_max_hits), non solo il migliore.

    Sostituisce la vecchia get_aliases_by_concept (limit=1, poi limit=1 per
    categoria): il problema non era la mancanza di categorie ma il limite a
    un solo hit — una query che tocca più concetti insieme ("lavoro agile in
    Garfagnana") li recupera ora tutti in un colpo solo, quanti siano,
    generalizzando automaticamente a qualunque numero di concetti presenti
    nel dizionario, senza tassonomia da mantenere nel codice.

    Ritorna una lista di dict: [{"concept": ..., "aliases": [...],
    "category": ..., "score": ...}], ordinata per score decrescente (ordine
    nativo di Qdrant). Lista vuota su miss o errore, mai eccezioni.
    """
    if not qdrant_client:
        return []

    max_hits = getattr(settings, "qdrant_concept_max_hits", 5)

    try:
        hits = qdrant_client.query_points(
            collection_name=CONCEPT_COLLECTION,
            query=query_vector,
            limit=max_hits,
            score_threshold=threshold
        ).points

        concepts = []
        for hit in hits:
            payload = hit.payload or {}
            concepts.append({
                "concept": payload.get("concept", ""),
                "aliases": payload.get("aliases", []),
                "category": payload.get("category", ""),
                "score": hit.score,
            })

        if concepts:
            summary = ", ".join(f"'{c['concept']}' ({c['score']:.3f})" for c in concepts)
            log.info(f"[{task_id}] [CONCEPT HIT] {len(concepts)} concetti sopra soglia (max {max_hits}): {summary}")
        else:
            log.debug(f"[{task_id}] [CONCEPT MISS] Nessun concetto sopra soglia {threshold}.")
        return concepts
    except Exception as e:
        log.error(f"[{task_id}] Errore lookup concettuale su Qdrant: {e}")
        return []


def log_automatic_negative_feedback(query, answer, topic_id, history, task_id="UNKNOWN"):
    """
    Salva automaticamente nel database le query che non hanno trovato risposte
    valide o che sono state scartate dal Grader.
    """
    conn = get_db_connection()
    if not conn:
        log.error(f"[{task_id}] [AUTO_FEEDBACK] Impossibile salvare: DB pool non disponibile.")
        return
        
    try:
        # Serializziamo la cronologia come fatto sul gateway Flask
        history_json = json.dumps(history) if history else "[]"
        
        sql = """
            INSERT INTO chat_feedback 
            (user_query, ai_response, topic_id, rating, chat_history, comment) 
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        with conn.cursor() as cursor:
            cursor.execute(sql, (
                query,
                answer,
                topic_id,
                0,  # Rating 0 per identificare i fallimenti automatici del sistema
                history_json,
                "Auto-Log: Risposta non trovata o non soddisfacente (Grader/Self-Correction Fail)"
            ))
        conn.commit()
        log.info(f"[{task_id}] [AUTO_FEEDBACK] ✓ Query non risposta registrata con successo nel DB.")
        
    except Exception as e:
        log.error(f"[{task_id}] [AUTO_FEEDBACK] ✗ Errore durante il salvataggio a DB: {e}")
        conn.rollback()
    finally:
        conn.close()


def log_rag_metrics(task_id, topic_id, query, cache_hit, is_satisfactory,
                     total_attempts, duration_ms, attempts_metrics):
    """
    Salva su MySQL (tabella rag_metrics, vedi create_rag_metrics_table.sql) una
    riga di sintesi per ogni task RAG processato: quanti tentativi sono serviti,
    se è finito in cache, e il dettaglio per-tentativo (parametri di recall,
    lambda MMR, quanti candidati/penalizzati) come JSON in attempts_detail.

    Non blocca mai il flusso principale: un fallimento qui viene solo loggato,
    esattamente come già fatto per la cache semantica e il feedback automatico.
    """
    conn = get_db_connection()
    if not conn:
        log.error(f"[{task_id}] [RAG_METRICS] Impossibile salvare: DB pool non disponibile.")
        return

    try:
        attempts_json = json.dumps(attempts_metrics, default=str)

        sql = """
            INSERT INTO rag_metrics
            (task_id, topic_id, query_preview, cache_hit, is_satisfactory,
             total_attempts, duration_ms, attempts_detail)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """

        with conn.cursor() as cursor:
            cursor.execute(sql, (
                task_id,
                topic_id,
                query[:255] if query else None,
                cache_hit,
                is_satisfactory,
                total_attempts,
                duration_ms,
                attempts_json
            ))
        conn.commit()
        log.debug(f"[{task_id}] [RAG_METRICS] ✓ Metriche salvate (attempts={total_attempts}, cache_hit={cache_hit}).")

    except Exception as e:
        log.error(f"[{task_id}] [RAG_METRICS] ✗ Errore durante il salvataggio a DB: {e}")
        conn.rollback()
    finally:
        conn.close()



# ==============================================================================
# 8. MAIN CELERY TASK
# ==============================================================================

@celery_app.task(bind=True, name="rag_queue")
def process_rag_query(self, query, history, topic_id, selected_sub_topics=None, metadata_filters=None, include_undated=True):
    """Main RAG processing pipeline: JSON Mode, Multi-Query, and Defensive Error Handling."""
    start_time = time.time()
    task_id = self.request.id
    log.info(f"[{task_id}] Task started: '{query[:50]}...' su topic: {topic_id}")

    # Accumula, per ciascun tentativo di retrieval, i parametri usati e le stats
    # ritornate da retrieve_chunks. Alla fine viene salvato in rag_metrics
    # (vedi log_rag_metrics) come JSON, per analisi a posteriori sui profili.
    attempts_metrics: list[dict] = []

    # ==========================================================
    # SETUP & VALIDATION
    # ==========================================================
    try:
        if not selected_sub_topics:
            selected_sub_topics = get_all_sub_topics(topic_id)
            if not selected_sub_topics:
                log.error(f"[{task_id}] Config Error: Nessun sub-topic per il topic '{topic_id}'.")
                return {
                    "error": "Configuration Error",
                    "message": "Il topic selezionato non è configurato correttamente.",
                    "status": "failed"
                }
        st_key = generate_sub_topics_key(selected_sub_topics)

    except Exception as e:
        log.error(f"[{task_id}] Initialization DB failed: {e}", exc_info=True)
        return {"error": "Internal Error", "message": "Errore di inizializzazione.", "status": "failed"}

    # ==========================================================
    # QUERY TRANSFORMATION & MULTI-QUERY EXPANSION
    # (concept-first: il dizionario Qdrant viene interrogato PRIMA del
    # rewriter, e i concetti trovati vengono passati come vocabolario
    # autoritativo dentro l'unica chiamata LLM che decide tutto)
    # ==========================================================
    try:
        # 1. Lookup concettuale sul testo grezzo (ultimi turni + query corrente):
        #    nomi di zone e concetti tecnici sono quasi sempre espliciti, quindi
        #    non serve la risoluzione dei pronomi per trovarli. Nota: euristica —
        #    su follow-up molto impliciti ("e lì invece?") il lookup può mancare
        #    il concetto; in quel caso il rewriter ripiega sulle sue regole
        #    generali, come da prompt.
        lookup_text = query
        if history:
            recent_turns = [msg.get("text", "") for msg in history[-2:]]
            lookup_text = " ".join(recent_turns + [query])

        verified_concepts = []
        try:
            concept_vector = embed_for_concept_lookup(lookup_text)
            verified_concepts = get_concepts_by_similarity(concept_vector, task_id=task_id)
        except Exception as ce:
            log.warning(f"[{task_id}] Lookup concettuale fallito ({ce}): il rewriter procede senza concetti verificati.")

        # 2. Unica chiamata: rewrite + decomposizione + espansione insieme,
        #    con i concetti verificati come dato di fatto nel prompt.
        rewritten_data = transform_query(history, query, task_id, verified_concepts=verified_concepts)

        standalone_query = rewritten_data.get("standalone_query", query)
        primary_vector = embed_query(standalone_query)

        extracted_keywords = rewritten_data.get("keywords", [])
        # Fallback testuale di sicurezza
        if not extracted_keywords:
            log.debug(f"[{task_id}] Nessuna keyword restituita, attivo fallback testuale in Python.")
            extracted_keywords = [w.strip("?.,!'\"") for w in standalone_query.split() if len(w) > 3]

        search_queries = rewritten_data.get("search_queries", [standalone_query])
        db_aliases = collect_concept_aliases(verified_concepts)

        if standalone_query not in search_queries:
                    search_queries.insert(0, standalone_query)
        vectors_list = embed_queries_batch(search_queries) 
        all_keywords = list(set(extracted_keywords + db_aliases))
            
    except Exception as e:
        log.warning(f"[{task_id}] [MAIN_TASK] Pipeline di trasformazione caduta: {e}. Uso raw query.")
        standalone_query, search_queries, all_keywords = query, [query], []
        lookup_text = query  # garantito anche nel fallback: usato dal lookup concettuale in retry
        primary_vector = embed_query(query)  # fallback sul raw query
        vectors_list = [primary_vector]

    # ==========================================================
    # SEMANTIC CACHE CHECK
    # ==========================================================
    try:
        filters_key = generate_filters_key({**(metadata_filters or {}), "_include_undated": include_undated})

        cached = check_semantic_cache(primary_vector, topic_id, st_key, filters_key)
        if cached:
            log.info(f"[{task_id}] Cache HIT. Returning cached response.")
            log_rag_metrics(
                task_id=task_id, topic_id=topic_id, query=query,
                cache_hit=True, is_satisfactory=True,
                total_attempts=0, duration_ms=int((time.time() - start_time) * 1000),
                attempts_metrics=[]
            )
            return {
                "answer": cached['answer'],
                "sources": cached['sources'],
                "topic": cached['topic_id'],
                "cached": True,
                "status": "success"
            }
    except Exception as e:
        log.warning(f"[{task_id}] Cache check failed (non-blocking): {e}")

    # ==========================================================
    # RETRIEVAL (Parallel Vector + Keyword)
    # ==========================================================
    try:
        context, sources, retrieval_stats = retrieve_chunks(
            search_queries, 
            vectors_list, 
            all_keywords, 
            topic_id, 
            selected_sub_topics,
            metadata_filters,
            include_undated,
            task_id=task_id,
            **profile_kwargs(0, task_id=task_id)
        )
        attempts_metrics.append({"attempt": 0, "standalone_query": standalone_query, **retrieval_stats})
    except Exception as e:
        log.error(f"[{task_id}] Vector DB Retrieval failed: {e}", exc_info=True)
        return {"error": "Database Error", "message": "Errore durante il recupero dei documenti.", "status": "failed"}

    # ==========================================================
    # SELF-CORRECTION LOOP (Generation & Grading)
    # ==========================================================
    attempt = 0
    answer = None
    is_satisfactory = False

    while attempt < settings.max_model_retries and not is_satisfactory:
        if not context:
            answer = "<p>Non sono riuscito a trovare la risposta nei documenti che ho analizzato.</p>"
            log.debug(f"[{task_id}] Context empty. Breaking loop.")
            break
        else:
            log.debug(f"Contesto da passare alla generazione: {context[:1500]}")

        # A. Generazione (JSON Mode)
        try:
            gen_result = generate_answer(standalone_query, context, topic_id)
            is_found = gen_result.get("is_found", False)
            answer = gen_result.get("answer", "")

            log.debug(f"Risposta generata: {answer[:500]}... | is_found: {is_found}")
        except Exception as e:
            log.error(f"[{task_id}] Answer generation API failed: {e}", exc_info=True)
            return {"error": "Generation Failed", "message": "Impossibile elaborare la risposta.", "status": "failed"}

        # B. Logica Fail-Fast
        if not is_found:
            log.info(f"[{task_id}] Model explicitly flagged is_found=False. Skipping Grader.")
            log.debug(f"[{task_id}] Model answer: {answer}")
            is_satisfactory = False

        elif gen_result.get("is_general_knowledge"):
            if settings.allow_general_knowledge:
                log.info(f"[{task_id}] Answer based on general knowledge (allowed). Skipping Grader.")
                is_satisfactory = True
            else:
                # Il modello ha sforato i limiti: forza il retry senza sprecare una chiamata al grader
                log.warning(f"[{task_id}] Answer based on general knowledge but ALLOW_GENERAL_KNOWLEDGE=false. Forcing retry.")
                is_satisfactory = False

        else:
            # Risposta ancorata al contesto: valutazione normale
            try:
                if isinstance(context, list):
                    context_snippet = "\n\n".join(
                        f"[{item.get('source', '')}]\n{item.get('content', '')}"
                        for item in context
                    )
                else:
                    context_snippet = str(context)
                is_satisfactory = grade_answer(standalone_query, context_snippet, answer)
            except Exception as e:
                log.error(f"[{task_id}] Grader LLM failed (non-blocking): {e}. Defaulting to YES.")
                is_satisfactory = True

        # D. Gestione Retry Fallimento
        if not is_satisfactory:
            attempt += 1
            if attempt < settings.max_model_retries:
                log.warning(f"[{task_id}] Answer unsatisfactory. Retrying ({attempt}/{settings.max_model_retries}) with full pipeline...")
                try:
                    # 1. Iniettiamo un "falso" messaggio di sistema nella history per forzare l'LLM a cambiare approccio
                    retry_history = history.copy() if history else []
                    retry_history.append({
                        "role": "user", 
                        "text": f"La ricerca precedente per '{standalone_query}' non ha prodotto documenti validi. Riformula completamente la query usando sinonimi o concetti più ampi per esplorare un'angolazione semantica diversa."
                    })
                    
                    # 2. Lookup concettuale anche in retry, ma sul lookup_text
                    #    GREZZO del primo giro (history + query originale
                    #    dell'utente), NON sulla standalone_query del tentativo
                    #    fallito: se il fallimento era dovuto proprio a una
                    #    riformulazione distorta, ripartire da quella
                    #    propagherebbe l'errore anche nel lookup. Il testo
                    #    originale dell'utente è immune da errori di rewrite.
                    retry_concepts = []
                    try:
                        retry_concept_vector = embed_for_concept_lookup(lookup_text)
                        retry_concepts = get_concepts_by_similarity(retry_concept_vector, task_id=f"{task_id}-RETRY")
                    except Exception as ce:
                        log.warning(f"[{task_id}-RETRY] Lookup concettuale fallito ({ce}): retry senza concetti verificati.")

                    retry_data = transform_query(retry_history, query, f"{task_id}-RETRY", verified_concepts=retry_concepts)
                    
                    standalone_query = retry_data.get("standalone_query", query)
                    extracted_keywords = retry_data.get("keywords", [])
                    
                    if not extracted_keywords:
                        extracted_keywords = [w.strip("?.,!'\"") for w in standalone_query.split() if len(w) > 3]

                    search_queries = retry_data.get("search_queries", [standalone_query])
                    extracted_keywords = list(set(extracted_keywords + collect_concept_aliases(retry_concepts)))

                    if standalone_query not in search_queries:
                        search_queries.insert(0, standalone_query)
                    
                    # 3. Rieseguiamo il Batch Embedding e il Retrieval Multi-Query
                    vectors_list = embed_queries_batch(search_queries)
                    context, sources, retrieval_stats = retrieve_chunks(
                        search_queries, 
                        vectors_list, 
                        extracted_keywords, 
                        topic_id, 
                        selected_sub_topics,
                        metadata_filters,
                        include_undated,
                        task_id=task_id,
                        **profile_kwargs(attempt, task_id=task_id)
                    )
                    attempts_metrics.append({"attempt": attempt, "standalone_query": standalone_query, **retrieval_stats})
                except Exception as e:
                    log.error(f"[{task_id}] Retry infrastructure failed: {e}", exc_info=True)
                    break  # Usciamo usando l'ultima answer generata
            else:
                log.warning(f"[{task_id}] Max retries exhausted. Returning best effort answer.")

    # ==========================================================
    # FEEDBACK NEGATIVO AUTOMATICO
    # ==========================================================
    if not is_satisfactory:
        log.info(f"[{task_id}] [MAIN_TASK] Rilevato fallimento retrieval/grading. Avvio auto-logging.")
        # Usiamo la query originale dell'utente passata al task
        log_automatic_negative_feedback(
            query=query, 
            answer=answer, 
            topic_id=topic_id, 
            history=history, 
            task_id=task_id
        )

    # ==========================================================
    # CACHE SAVING & RETURN
    # ==========================================================
    try:
        # Evitiamo di mettere in cache risposte troppo brevi o palesemente vuote
        if answer and is_satisfactory and len(answer) > 20:
            save_to_semantic_cache(
                primary_vector, 
                standalone_query, 
                answer, 
                sources, 
                topic_id,
                st_key,
                filters_key
            )
    except Exception as e:
        log.warning(f"[{task_id}] Failed to save successful response to cache (non-blocking): {e}")

    duration = time.time() - start_time
    log.info(f"[{task_id}] ✓ Completed gracefully in {duration:.2f}s")

    log_rag_metrics(
        task_id=task_id, topic_id=topic_id, query=query,
        cache_hit=False, is_satisfactory=is_satisfactory,
        total_attempts=len(attempts_metrics), duration_ms=int(duration * 1000),
        attempts_metrics=attempts_metrics
    )

    return {
        "answer": answer,
        "sources": sources,
        "topic": topic_id,
        "cached": False,
        "status": "success"
    }
    
# ==============================================================================
# 9. WORKER LIFECYCLE
# ==============================================================================

@worker_shutdown.connect
def cleanup_worker(**kwargs):
    """Gracefully close connections when the worker shuts down."""
    if qdrant_client:
        qdrant_client.close()
        log.info("Qdrant client closed.")