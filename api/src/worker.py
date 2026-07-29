import os
import logging
import time
import uuid
from celery import Celery
from celery.signals import setup_logging
from celery.signals import worker_process_init, worker_shutdown
from celery.schedules import crontab
from common.utility import normalize_ws
from mysql.connector import pooling
from qdrant_client import QdrantClient, models
import math
import json, re
import hashlib
import numpy as np
# Use the optimized reranker
from reranker import ONNXReranker, RerankResult

# Provider-agnostic LLM adapter
from llm_provider import init_llm_provider, get_llm_provider

# Embedding (Gemini only) — estratto in embedding.py
from common.embedding import (
    init_embedding,
    embed_query,
    embed_queries_batch,
    embed_for_semantic_query,
)

from concurrent.futures import ThreadPoolExecutor, as_completed

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

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["ONNXRUNTIME_EXECUTION_MODE"] = "PARALLEL"

QDRANT_COLLECTION = "document_chunks"
CACHE_COLLECTION = "semantic_cache"
CONCEPT_COLLECTION = "conceptual_dictionary"
BOILERPLATE_COLLECTION = "boilerplate_phrases"

# Cache in-processo delle frasi di boilerplate attive, per topic. Evita una
# query MySQL a ogni richiesta: le frasi cambiano una volta a notte (script
# detect_boilerplate.py) e su approvazione manuale, quindi un TTL breve è
# ampiamente sufficiente. Struttura: {topic_id: (scadenza_epoch, [frasi...])}.
_BOILERPLATE_CACHE = {}
_BOILERPLATE_TTL_SECONDS = 300

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
    worker_max_memory_per_child=2400000, # 2GB limit
    worker_hijack_root_logger=False,
    # Secondi che Celery concede a un processo appena forkato per segnalare
    # "UP". Il default è 4.0 e NON basta più: init_worker_process carica il
    # pool ONNX (tokenizer + sessione per ciascuna istanza, ~6.9s misurati),
    # quindi ogni processo di rimpiazzo veniva ucciso con SIGKILL a 3.97s
    # esatti e il worker entrava in crash loop infinito — visibile nei log
    # come "Timed out waiting for UP message from <ForkProcess(...)>".
    # Va tenuto sopra il tempo di init peggiore, con margine: è un limite di
    # sicurezza contro i processi nati morti, non una manopola di performance.
    worker_proc_alive_timeout=60.0,
)

# ==============================================================================
# PROCESS-SAFE INITIALIZATION 
# ==============================================================================
# Globals assigned strictly AFTER the fork
db_pool = None
qdrant_client = None

@worker_process_init.connect
def init_worker_process(**kwargs):
    global qdrant_client
    log.info("Initializing Worker Resources (Post-Fork)...")

    try:
        init_db_pool()  # Inizializza il pool globalmente

        qdrant_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            # Il default della libreria è 5s. Le vector search con group_by
            # impiegavano 4,1-4,9s: si viaggiava sul filo, e la prima che
            # sforava perdeva l'INTERA sottoquery (safe_vector_search cattura
            # l'eccezione e restituisce una lista vuota, senza retry), con il
            # comune corrispondente assente dalla risposta finale.
            timeout=10,
        )
        init_embedding()
        init_llm_provider()

        # Il pool ONNX viene costruito QUI, non alla prima query. Era lazy
        # dentro retrieve_chunks: i ~6,7s di caricamento (tokenizer + sessione,
        # per ogni istanza, in serie) finivano dentro la latenza percepita
        # dall'utente che aveva la sfortuna di essere il primo a interrogare il
        # processo appena forkato. Post-fork è il punto giusto: prima del fork
        # la sessione ONNX non sarebbe ereditabile in sicurezza.
        get_reranker_pool()

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

def load_active_boilerplate(topic_id: str, sub_topic_ids):
    """Frasi di boilerplate ATTIVE per (topic, sub_topic), con cache a TTL.

    Ritorna {sub_topic_id: [(phrase, vector), ...]} oppure None se non ce ne
    sono. MySQL è la fonte di verità sull'active (rispetta la revisione umana);
    Qdrant fornisce i vettori, recuperati PER phrase_hash — un join esatto che
    non dipende dall'identità testuale delle stringhe fra i due sistemi.

    Una frase attiva in MySQL ma priva di vettore in Qdrant (es. sync notturna
    non ancora rigirata) non si perde: entra nella mappa con vettore None, e la
    guardia ricadrà su di lei col criterio lessicale.

    Fail-soft: qualunque errore -> None, e il reranker lavora sul testo grezzo.
    """
    if not sub_topic_ids:
        return None

    # Chiave di cache: topic + insieme ordinato dei sub_topic interrogati.
    cache_key = (topic_id, tuple(sorted(sub_topic_ids)))
    now = time.time()
    cached = _BOILERPLATE_CACHE.get(cache_key)
    if cached and cached[0] > now:
        return cached[1]

    # --- 1. MySQL: quali frasi (active=TRUE) e il loro hash ------------------
    rows = []
    try:
        conn = get_db_connection()
        if conn is None:
            raise RuntimeError("connessione MySQL non disponibile")
        try:
            with conn.cursor(dictionary=True) as cursor:
                placeholders = ",".join(["%s"] * len(sub_topic_ids))
                cursor.execute(
                    f"SELECT phrase, sub_topic_id, phrase_hash "
                    f"FROM boilerplate_phrases "
                    f"WHERE topic_id = %s AND active = TRUE "
                    f"AND sub_topic_id IN ({placeholders})",
                    (topic_id, *sub_topic_ids),
                )
                rows = [r for r in cursor.fetchall() if r.get("phrase")]

                log.debug(f"Caricate {len(rows)} frasi boilerplate attive da MySQL per topic '{topic_id}' ")
        finally:
            conn.close()
    except Exception as e:
        log.warning(f"Caricamento boilerplate MySQL per '{topic_id}' fallito ({e}): "
                    f"il reranker procede sul testo grezzo.")
        _BOILERPLATE_CACHE[cache_key] = (now + 30, None)
        return None

    if not rows:
        _BOILERPLATE_CACHE[cache_key] = (now + _BOILERPLATE_TTL_SECONDS, None)
        return None

    # --- 2. Qdrant: i vettori delle frasi attive, recuperati per hash --------
    wanted_hashes = [r["phrase_hash"] for r in rows]
    vec_by_hash = {}
    try:
        scrolled, _ = qdrant_client.scroll(
            collection_name=BOILERPLATE_COLLECTION,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="topic_id", match=models.MatchValue(value=topic_id)),
                models.FieldCondition(key="phrase_hash", match=models.MatchAny(any=wanted_hashes)),
            ]),
            with_vectors=True,
            with_payload=["phrase_hash"],
            limit=len(wanted_hashes) + 16,
        )
        for p in scrolled:
            h = (p.payload or {}).get("phrase_hash")
            if h and p.vector is not None:
                vec_by_hash[h] = np.asarray(p.vector, dtype=np.float32)
    except Exception as e:
        # Qdrant assente o collection non ancora popolata: si prosegue coi soli
        # vettori mancanti (tutti None) -> guardia lessicale. Non è un errore
        # fatale, è un degrado.
        log.warning(f"Recupero vettori boilerplate da Qdrant fallito ({e}): "
                    f"guardia lessicale per questo topic.")

    # --- 3. Raggruppa per sub_topic, agganciando il vettore quando c'è -------
    by_subtopic = {}
    for r in rows:
        st = r["sub_topic_id"]
        vec = vec_by_hash.get(r["phrase_hash"])   # None se mancante
        by_subtopic.setdefault(st, []).append((r["phrase"], vec))

    _BOILERPLATE_CACHE[cache_key] = (now + _BOILERPLATE_TTL_SECONDS, by_subtopic)
    return by_subtopic


def build_removal_by_subtopic(boilerplate_map, query_str, query_vector):
    """Decide UNA VOLTA per query quali frasi rimuovere, per ciascun sub_topic.

    Ritorna {sub_topic_id: compiled_regex} dove la regex contiene SOLO le frasi
    da rimuovere (quelle NON pertinenti alla query). Le frasi pertinenti non
    entrano nella regex, quindi restano nei chunk.

    Guardia PRIMARIA semantica (vettore query e vettore frase presenti): tiene
    la frase se la similarità coseno supera settings.boilerplate_similarity_
    threshold. FALLBACK lessicale (manca un vettore): tiene la frase se
    condivide un termine significativo con la query.

    Il confronto è per sub_topic perché il boilerplate di una serie non deve
    toccare i chunk di un'altra; ed è per query, non per chunk, perché la
    decisione dipende solo da query e frasi (i 200 chunk del gruppo ricevono
    tutti la stessa regex)."""
    if not boilerplate_map:
        return {}

    qv = None
    if query_vector is not None:
        qv = np.asarray(query_vector, dtype=np.float32)
        qv = qv / (np.linalg.norm(qv) + 1e-9)
    thr = settings.boilerplate_similarity_threshold
    q_terms = _significant_terms(query_str)

    removal = {}
    for st, phrase_vecs in boilerplate_map.items():
        to_remove = []
        kept = 0
        for phrase, pv in phrase_vecs:
            pertinent = False
            if qv is not None and pv is not None:
                sim = float(qv @ (pv / (np.linalg.norm(pv) + 1e-9)))
                pertinent = sim >= thr
            else:
                # Fallback lessicale: pertinente se condivide un termine.
                pertinent = bool(_significant_terms(phrase) & q_terms)
            if pertinent:
                kept += 1
            else:
                to_remove.append(phrase)

        if to_remove:
            ordered = sorted(set(to_remove), key=len, reverse=True)
            removal[st] = re.compile("|".join(re.escape(p) for p in ordered), re.IGNORECASE)
        log.debug(f"[BOILERPLATE] sub_topic '{st}': rimuovo {len(to_remove)}, "
                  f"tengo {kept} (pertinenti alla query).")

    return removal


_STOPWORDS = frozenset("""
il lo la i gli le un uno una di del dello della dei degli delle da dal dallo
dalla in nel nello nella con su sul sullo sulla per tra fra e ed o od a ad al
allo alla ai agli alle che chi cui non come più meno anche se ma però quindi
the a an of to in on for and or with by from at as is are be this that
""".split())


def _significant_terms(text: str) -> set:
    """Termini di un testo, minuscoli, esclusi stopword e token troppo corti.
    Un identificatore come '267/2000' o 'd.lgs' resta un termine significativo."""
    toks = re.findall(r"\w+(?:['\-./]\w+)*", (text or "").lower())
    return {t for t in toks if len(t) >= 3 and t not in _STOPWORDS}

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
        # Entità distinte che la risposta deve coprire (comuni, enti, oggetti).
        # Vuota quando la domanda non è aggregativa. Il .get con default
        # garantisce la retrocompatibilità con prompt di rewriter più vecchi.
        coverage   = data.get("coverage_targets", []) or []

        log.info(
            f"[{task_id}] [REWRITER] ✓ Query processata. "
            f"Standalone: '{standalone}' | Facets: {len(searches)} | Keywords: {len(keywords)} | "
            f"Coverage: {len(coverage)}"
        )

        log.debug(f"Standalone query: {standalone}")
        log.debug(f"Query individuate: {searches}")
        log.debug(f"Keywords: {keywords}")
        log.debug(f"Coverage targets: {coverage}")
        return {
            "standalone_query": standalone,
            "search_queries": searches,
            "keywords": keywords,
            "coverage_targets": coverage,
        }

    except Exception as e:
        log.warning(
            f"[{task_id}] [REWRITER] ✗ Fallimento critico: {e}. Attivazione fallback (query raw).",
            exc_info=True
        )
        return {"standalone_query": query, "search_queries": [query], "keywords": [], "coverage_targets": []}


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
# 5bis. COSTRUZIONE DEL CONTESTO
#
# Tre stadi, in quest'ordine:
#   1. compute_core_spans()        — dove stanno, dentro il parent, i child che
#                                    hanno fatto match. Sono INCOMPRIMIBILI.
#   2. allocate_context_budget()   — quanti caratteri spettano a ciascun parent,
#                                    con equità garantita tra le sottoquery.
#   3. render_parent_window()      — ritaglio effettivo: i core sopravvivono
#                                    sempre, a cedere sono i margini.
#
# Il principio è che il budget si spende sui contenuti che il retrieval ha
# giudicato rilevanti, non sul testo di contorno; e che nessuna sottoquery può
# monopolizzare il contesto solo perché il suo comune ha più documenti a
# catalogo.
# ==============================================================================

# Quanti child (per parent) teniamo come "ancore" per il windowing. Tenerne
# più di uno permette di coprire casi in cui più frammenti dello stesso
# documento sono stati rilevanti per query diverse.
SNIPPETS_PER_PARENT = 3
# Lunghezza dell'ancora usata per localizzare lo snippet dentro il parent:
# corta a sufficienza da tollerare piccole differenze di whitespace/normalizzazione
# tra come il child è stato salvato e come appare dentro il testo del parent.
SNIPPET_ANCHOR_LEN = 80

# Marcatore esplicito al posto del vecchio "[...]". NON è cosmetica: serve a
# impedire che il modello legga due passaggi distanti come contigui e ci
# costruisca sopra un nesso che nel documento non esiste. Su atti
# amministrativi, dove un comma può essere separato dalla sua deroga da venti
# pagine, il rischio è concreto.
OMISSION_MARKER = "\n[... porzione di documento omessa ...]\n"

# Se non riusciamo a localizzare nessuno snippet dentro il parent, questo è il
# minimo che gli garantiamo comunque (troncamento dall'inizio).
FALLBACK_MIN_CHARS = 600

# Terminatori di frase/paragrafo per lo "snap": arretriamo il taglio fino al
# confine più vicino DENTRO il budget, quindi a costo zero.
_SENTENCE_END_RE = re.compile(r"[.!?;:»)\]]\s")
_PARAGRAPH_END_RE = re.compile(r"\n\s*\n")


def _whitespace_tolerant_find(content: str, anchor: str):
    """
    Cerca `anchor` dentro `content` tollerando differenze di whitespace, e
    restituisce (start, end) come indici VALIDI SU `content`, oppure None.

    Sostituisce il vecchio fallback che cercava dentro
    `" ".join(content.split())` e poi usava l'indice trovato come se fosse un
    indice del testo originale: la normalizzazione collassa gli spazi, quindi
    lo scarto cresce con la quantità di whitespace e su un documento pieno di
    tabulazioni la finestra poteva slittare di centinaia di caratteri,
    centrandosi sul punto sbagliato.
    """
    tokens = anchor.split()
    if not tokens:
        return None
    pattern = r"\s+".join(re.escape(t) for t in tokens)
    try:
        m = re.search(pattern, content)
    except re.error:
        return None
    return (m.start(), m.end()) if m else None


def compute_core_spans(content: str, matched_snippets: list, task_id: str = "UNKNOWN") -> list:
    """
    Localizza dentro il parent i child che hanno fatto match e restituisce una
    lista ordinata e deduplicata di dict {start, end, score}: sono i "core",
    le porzioni di testo che NON devono mai essere sacrificate.

    Localizzazione a due livelli, in ordine di affidabilità:
      1. OFFSET dal payload (start_char/end_char, popolati dal backfill o
         dall'ingestion): esatti e gratuiti. Un sanity check (l'ancora deve
         comparire nel testo puntato dagli offset) protegge dal caso di parent
         modificato dopo il calcolo degli offset.
      2. Ricerca dell'ancora tollerante ai whitespace sul testo ORIGINALE.

    matched_snippets: lista di dict {content, score, start_char?, end_char?}.
    Per retrocompatibilità accetta anche semplici stringhe.
    """
    spans = []
    offset_hits = 0

    for snippet in matched_snippets or []:
        # Retrocompatibilità: snippet può essere una stringa o un dict
        if isinstance(snippet, str):
            snippet_text, s_char, e_char, score = snippet, None, None, 0.0
        else:
            snippet_text = snippet.get("content", "")
            s_char = snippet.get("start_char")
            e_char = snippet.get("end_char")
            score = snippet.get("score", 0.0) or 0.0

        if not snippet_text:
            continue
        anchor = " ".join(snippet_text.split())[:SNIPPET_ANCHOR_LEN]
        if not anchor:
            continue

        # Livello 1: offset dal payload, con sanity check
        if (
            s_char is not None and e_char is not None
            and 0 <= s_char < e_char <= len(content)
        ):
            window_text_norm = " ".join(content[s_char:e_char].split())
            if anchor[:40] in window_text_norm:
                spans.append({"start": s_char, "end": e_char, "score": score})
                offset_hits += 1
                continue
            log.debug(
                f"[{task_id}] [WINDOW] Offset ({s_char},{e_char}) non combaciano col contenuto "
                f"(parent modificato dopo il backfill?): fallback a ricerca testuale."
            )

        # Livello 2: ricerca tollerante ai whitespace, indici sempre coerenti
        found = _whitespace_tolerant_find(content, anchor)
        if found:
            start, _ = found
            # L'ancora è solo il PREFISSO dello snippet: il core si estende per
            # tutta la lunghezza dello snippet originale, non solo dell'ancora.
            end = min(len(content), start + len(snippet_text))
            spans.append({"start": start, "end": end, "score": score})

    if not spans:
        return []

    # Merge dei core sovrapposti: lo score del blocco unito è il massimo dei suoi.
    spans.sort(key=lambda s: s["start"])
    merged = [spans[0]]
    for s in spans[1:]:
        last = merged[-1]
        if s["start"] <= last["end"]:
            last["end"] = max(last["end"], s["end"])
            last["score"] = max(last["score"], s["score"])
        else:
            merged.append(s)

    log.debug(
        f"[{task_id}] [WINDOW] {len(merged)} core individuati "
        f"({offset_hits} via offset, {len(spans) - offset_hits} via ricerca testuale)."
    )
    return merged


def estimate_parent_needs(content: str, core_spans: list) -> tuple:
    """
    Restituisce (need_min, need_full) per un parent.

    need_min  = spazio sotto il quale si perde informazione che il retrieval
                aveva giudicato rilevante: la somma dei core più i marcatori
                di omissione che li separano.
    need_full = il documento intero.

    Sono i due numeri su cui lavora l'allocatore: garantisce need_min a tutti
    PRIMA di far crescere chiunque verso need_full.
    """
    need_full = len(content)
    if not core_spans:
        return min(need_full, FALLBACK_MIN_CHARS), need_full
    core_total = sum(s["end"] - s["start"] for s in core_spans)
    core_total += len(OMISSION_MARKER) * (len(core_spans) - 1)
    return min(core_total, need_full), need_full


def _water_fill(demands: list, budget: int) -> tuple:
    """
    Ripartizione max-min fair (water-filling) di `budget` tra richieste eterogenee.

    Tutti ricevono una quota uguale; chi chiede MENO della propria quota prende
    solo ciò che gli serve e RESTITUISCE l'avanzo, che viene ridistribuito a chi
    è ancora insoddisfatto. Si itera fino a convergenza.

    È il meccanismo che elimina lo spreco del vecchio calcolo a tetto fisso, in
    cui un parent lungo 900 caratteri con quota 1875 lasciava 975 caratteri che
    nessuno raccoglieva mentre altri parent venivano troncati.

    Restituisce (allocazioni, budget_residuo).
    """
    n = len(demands)
    alloc = [0] * n
    if n == 0 or budget <= 0:
        return alloc, max(0, budget)

    pending = set(range(n))
    remaining = budget

    while pending and remaining > 0:
        share = remaining // len(pending)
        if share <= 0:
            break
        satisfied = [i for i in pending if demands[i] <= share]
        if not satisfied:
            # Nessuno si accontenta della quota: la spartiamo e chiudiamo.
            for i in pending:
                alloc[i] += share
                remaining -= share
            break
        for i in satisfied:
            alloc[i] += demands[i]
            remaining -= demands[i]
            pending.discard(i)

    return alloc, max(0, remaining)


def _subquery_quotas(group_scores: list, total_budget: int, task_id: str = "UNKNOWN") -> list:
    """
    Quota di caratteri per ciascuna sottoquery, secondo la politica configurata
    (vedi la tabella delle tre politiche in common/config.py).

        quota = floor_equamente_diviso + resto_pesato_sulla_rilevanza
        quota = min(quota, cap)     # tetto anti-monopolio

    L'eccedenza tagliata dal cap viene ridistribuita alle sottoquery non ancora
    al tetto, così nessun carattere si perde per strada.
    """
    n = len(group_scores)
    if n == 0:
        return []

    floor_ratio = max(0.0, min(1.0, settings.context_floor_ratio))
    floor_pool = total_budget * floor_ratio
    weighted_pool = total_budget - floor_pool
    base = floor_pool / n

    if settings.context_score_weight == "none" or weighted_pool <= 0:
        weights = [1.0 / n] * n
    else:
        raw = [max(0.0, s) for s in group_scores]
        tot = sum(raw)
        weights = [r / tot for r in raw] if tot > 0 else [1.0 / n] * n

    quotas = [base + weighted_pool * w for w in weights]

    # Tetto anti-monopolio, con ridistribuzione iterativa dell'eccedenza.
    cap_ratio = settings.context_cap_ratio
    if 0 < cap_ratio < 1.0 and n > 1:
        cap = total_budget * cap_ratio
        for _ in range(4):
            excess = sum(q - cap for q in quotas if q > cap)
            if excess <= 1:
                break
            quotas = [min(q, cap) for q in quotas]
            free_idx = [i for i, q in enumerate(quotas) if q < cap - 1]
            if not free_idx:
                break
            free_weight = sum(weights[i] for i in free_idx) or len(free_idx)
            for i in free_idx:
                share = (weights[i] / free_weight) if free_weight else (1 / len(free_idx))
                quotas[i] = min(cap, quotas[i] + excess * share)

    quotas = [int(q) for q in quotas]
    log.debug(f"[{task_id}] [ALLOC] Quote per sottoquery: {quotas} (floor={floor_ratio}, cap={cap_ratio}).")
    return quotas


def allocate_context_budget(items: list, total_budget: int, task_id: str = "UNKNOWN") -> tuple:
    """
    Assegna a ciascun parent il proprio budget di caratteri, garantendo che ogni
    sottoquery contribuisca al contesto finale.

    Cascata, nell'ordine deciso:
      FASE A — ogni parent riceve almeno need_min (i suoi core), dentro la quota
               della propria sottoquery.
      FASE B — l'avanzo del water-filling (il budget che le sottoquery povere non
               riescono a consumare) va a chi è ancora in deficit.
      FASE C — se serve ancora, si sfora max_context_chars fino a
               context_overflow_ratio, comunque entro context_hard_cap_chars.
      FASE D — solo se anche questo non basta, il parent riceve meno di need_min
               e render_parent_window() scarterà il core con lo score più basso,
               loggando un WARNING: è il segnale che SNIPPETS_PER_PARENT o il
               budget vanno ritarati per quel topic.
      FASE E — con quello che resta, i parent crescono verso need_full (margini
               di contesto attorno ai core).
      FASE F — completamento: un documento che sfora di poco (<= overflow_ratio)
               viene preso INTERO invece che finestrato. L'espansione non può
               mai servire ad aggiungere un parent nuovo, solo a completarne uno
               già selezionato.

    Restituisce (allocazioni, stats).
    """
    n = len(items)
    if n == 0:
        return [], {}

    # --- raggruppamento per sottoquery, preservando l'ordine di apparizione ---
    groups = {}
    for i, it in enumerate(items):
        groups.setdefault(it.get("query_idx", 0), []).append(i)
    group_keys = list(groups.keys())

    if settings.context_score_weight == "mean":
        group_scores = [
            sum(items[i].get("score", 0.0) for i in groups[g]) / len(groups[g])
            for g in group_keys
        ]
    else:  # 'best' (default) — meno sensibile alla numerosità dei documenti
        group_scores = [max(items[i].get("score", 0.0) for i in groups[g]) for g in group_keys]

    quotas = _subquery_quotas(group_scores, total_budget, task_id=task_id)

    need_min = [it["need_min"] for it in items]
    need_full = [it["need_full"] for it in items]
    alloc = [0] * n

    # --- FASE A: need_min dentro la quota di gruppo ---------------------------
    leftover_pool = 0
    for gi, g in enumerate(group_keys):
        idxs = groups[g]
        sub_alloc, leftover = _water_fill([need_min[i] for i in idxs], quotas[gi])
        for k, i in enumerate(idxs):
            alloc[i] = sub_alloc[k]
        leftover_pool += leftover

    # --- FASE B: avanzo globale ai parent ancora sotto need_min ---------------
    deficit_idx = [i for i in range(n) if alloc[i] < need_min[i]]
    if deficit_idx and leftover_pool > 0:
        deficits = [need_min[i] - alloc[i] for i in deficit_idx]
        got, leftover_pool = _water_fill(deficits, leftover_pool)
        for k, i in enumerate(deficit_idx):
            alloc[i] += got[k]
        log.info(
            f"[{task_id}] [ALLOC] Avanzo redistribuito a {len(deficit_idx)} parent sotto il minimo vitale."
        )

    # --- FASE C: overflow controllato, entro l'hard cap -----------------------
    ceiling = min(
        int(total_budget * (1 + settings.context_overflow_ratio)),
        settings.context_hard_cap_chars,
    )
    n_expanded = 0
    deficit_idx = [i for i in range(n) if alloc[i] < need_min[i]]
    if deficit_idx:
        headroom = ceiling - sum(alloc)
        if headroom > 0:
            deficits = [need_min[i] - alloc[i] for i in deficit_idx]
            got, _ = _water_fill(deficits, headroom)
            for k, i in enumerate(deficit_idx):
                if got[k] > 0:
                    alloc[i] += got[k]
                    n_expanded += 1
            log.info(
                f"[{task_id}] [ALLOC] Espansione oltre max_context_chars per garantire i core: "
                f"{sum(alloc)}/{total_budget} chars (tetto {ceiling})."
            )

    # --- FASE D: deficit residuo → sacrificio di uno snippet (rumoroso) -------
    still_short = [i for i in range(n) if alloc[i] < need_min[i]]
    if still_short:
        log.warning(
            f"[{task_id}] [ALLOC] {len(still_short)} parent restano sotto il minimo vitale anche "
            f"dopo espansione: verrà scartato lo snippet meno rilevante. Valutare di ridurre "
            f"SNIPPETS_PER_PARENT o di alzare MAX_CONTEXT_CHARS per questo topic."
        )

    # --- FASE E: crescita verso need_full con il budget residuo ---------------
    used = sum(alloc)
    growth_pool = max(0, total_budget - used)
    if growth_pool > 0:
        # Prima dentro il gruppo (equità), poi il residuo globalmente.
        for gi, g in enumerate(group_keys):
            idxs = groups[g]
            group_used = sum(alloc[i] for i in idxs)
            group_pool = max(0, quotas[gi] - group_used)
            group_pool = min(group_pool, growth_pool)
            if group_pool <= 0:
                continue
            extra = [max(0, need_full[i] - alloc[i]) for i in idxs]
            got, back = _water_fill(extra, group_pool)
            for k, i in enumerate(idxs):
                alloc[i] += got[k]
            growth_pool -= (group_pool - back)

        # Residuo globale, in due giri. Il primo rispetta il tetto
        # anti-monopolio: se più sottoquery hanno ancora fame, nessuna può
        # sfondare context_cap_ratio approfittando dell'avanzo altrui.
        # Il secondo giro serve solo quando NESSUN altro è in grado di
        # assorbire quel budget (tipicamente perché tutti gli altri hanno già
        # ricevuto i loro documenti INTERI): a quel punto lasciarlo inutilizzato
        # sarebbe uno spreco senza vittime, non un monopolio.
        if growth_pool > 0:
            cap = int(total_budget * settings.context_cap_ratio) if 0 < settings.context_cap_ratio < 1 else None
            for capped_round in (True, False):
                if growth_pool <= 0:
                    break
                extra = []
                for i in range(n):
                    want = max(0, need_full[i] - alloc[i])
                    if capped_round and cap is not None:
                        g = items[i].get("query_idx", 0)
                        group_used = sum(alloc[j] for j in groups[g])
                        want = min(want, max(0, cap - group_used))
                    extra.append(want)
                if not any(extra):
                    continue
                got, growth_pool = _water_fill(extra, growth_pool)
                for i in range(n):
                    alloc[i] += got[i]

    # --- FASE F: completamento dei documenti che sforano di poco --------------
    n_completed = 0
    used = sum(alloc)
    gaps = sorted(
        (i for i in range(n) if alloc[i] < need_full[i]),
        key=lambda i: need_full[i] - alloc[i],
    )
    for i in gaps:
        gap = need_full[i] - alloc[i]
        tolerance = max(1, int(need_full[i] * settings.context_overflow_ratio))
        if gap <= tolerance and used + gap <= ceiling:
            alloc[i] += gap
            used += gap
            n_completed += 1

    if n_completed:
        log.info(
            f"[{task_id}] [ALLOC] {n_completed} parent inclusi INTERI grazie all'overflow "
            f"(sforamento <= {settings.context_overflow_ratio:.0%}); totale {used} chars."
        )

    stats = {
        "n_groups": len(group_keys),
        "quotas": quotas,
        "allocated_total": sum(alloc),
        "n_expanded": n_expanded,
        "n_completed": n_completed,
        "n_below_min": len(still_short),
        "ceiling": ceiling,
    }
    return alloc, stats


def _snap_start(content: str, pos: int, limit: int) -> int:
    """
    Sposta l'inizio della finestra IN AVANTI fino al confine di frase/paragrafo
    più vicino, senza mai superare `limit` (l'inizio del core, che è intoccabile).
    Costo zero: si restringe, non si espande.
    """
    mode = settings.context_snap_boundary
    if mode == "none" or pos <= 0 or pos >= limit:
        return pos
    window = content[pos:limit]
    rx = _PARAGRAPH_END_RE if mode == "paragraph" else _SENTENCE_END_RE
    m = rx.search(window)
    if m:
        return pos + m.end()
    return pos


def _snap_end(content: str, pos: int, limit: int) -> int:
    """
    Arretra la fine della finestra fino all'ultimo confine di frase/paragrafo,
    senza mai scendere sotto `limit` (la fine del core).
    """
    mode = settings.context_snap_boundary
    if mode == "none" or pos >= len(content) or pos <= limit:
        return pos
    window = content[limit:pos]
    rx = _PARAGRAPH_END_RE if mode == "paragraph" else _SENTENCE_END_RE
    last = None
    for m in rx.finditer(window):
        last = m
    if last:
        return limit + last.end()
    return pos


def render_parent_window(content: str, core_spans: list, budget: int, task_id: str = "UNKNOWN") -> str:
    """
    Costruisce il testo effettivo del parent entro `budget` caratteri.

    Regola fondamentale: i CORE (i child che hanno fatto match) sono
    incomprimibili, i MARGINI attorno a loro sono comprimibili. Se un parent ha
    un chunk rilevante all'inizio e uno alla fine, teniamo entrambi separandoli
    con OMISSION_MARKER, invece di sacrificarne uno: quel documento è
    probabilmente il più informativo del lotto, e mutilarlo sarebbe il peggior
    uso possibile del budget.

    Non tronca MAI a metà di un blocco: se lo spazio non basta, scarta per
    intero il core con lo score più basso (ultima risorsa, già segnalata a
    WARNING dall'allocatore).
    """
    if len(content) <= budget:
        return content

    if not core_spans:
        log.debug(f"[{task_id}] [WINDOW] Nessun core localizzato: troncamento dall'inizio con snap.")
        cut = _snap_end(content, budget, 0)
        return content[:cut]

    # Se i soli core non ci stanno, sacrifichiamo i meno rilevanti — mai un
    # taglio a metà blocco.
    cores = sorted(core_spans, key=lambda s: s["start"])
    while len(cores) > 1:
        core_total = sum(c["end"] - c["start"] for c in cores)
        core_total += len(OMISSION_MARKER) * (len(cores) - 1)
        if core_total <= budget:
            break
        weakest = min(cores, key=lambda c: c["score"])
        cores.remove(weakest)
        log.warning(
            f"[{task_id}] [WINDOW] Budget insufficiente per tutti i core: scartato uno snippet "
            f"(score={weakest['score']:.4f}). Restano {len(cores)} core."
        )

    core_total = sum(c["end"] - c["start"] for c in cores)
    separators = len(OMISSION_MARKER) * (len(cores) - 1)

    # Il margine è ciò che AVANZA dopo aver messo al sicuro i core, distribuito
    # equamente sui blocchi (metà prima, metà dopo). Sostituisce il vecchio
    # margin = max(200, budget // (2 * len(spans))), che veniva calcolato prima
    # di sapere se i core ci stavano.
    margin_pool = max(0, budget - core_total - separators)

    # Espansione iterativa del margine. Un core a inizio o fine documento non
    # può crescere da entrambi i lati, e nella prima versione quello spazio
    # andava semplicemente perso (su un parent con core agli estremi si
    # arrivava a usare meno della metà del budget). Qui ricalcoliamo cosa è
    # stato davvero consumato e rioffriamo il resto a chi può ancora crescere.
    def _expand(pool):
        per_side = pool // (2 * len(cores)) if cores else 0
        wins = []
        for c in cores:
            start = _snap_start(content, max(0, c["start"] - per_side), c["start"])
            end = _snap_end(content, min(len(content), c["end"] + per_side), c["end"])
            wins.append((start, end))
        return wins

    windows = _expand(margin_pool)
    for _ in range(3):
        used = sum(e - s for s, e in windows) + separators
        slack = budget - used
        # Ci fermiamo quando l'avanzo è irrilevante o quando l'ultima
        # iterazione non ha prodotto alcun guadagno (tutti i core già al
        # confine del documento).
        if slack < 200:
            break
        candidate = _expand(margin_pool + slack)
        if sum(e - s for s, e in candidate) <= sum(e - s for s, e in windows):
            break
        margin_pool += slack
        windows = candidate

    # Merge dei blocchi che dopo l'espansione dei margini si toccano.
    merged = [windows[0]]
    for s, e in windows[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))

    pieces = [content[s:e] for s, e in merged]
    result = OMISSION_MARKER.join(pieces)

    # Se lo snap ha allargato oltre il previsto, rimuoviamo margine invece di
    # tagliare il risultato a caratteri (che spezzerebbe l'ultimo blocco).
    if len(result) > budget and len(merged) == 1:
        s, e = merged[0]
        core_end = max(c["end"] for c in cores)
        result = content[s:max(core_end, e - (len(result) - budget))]

    log.debug(
        f"[{task_id}] [WINDOW] {len(cores)} core, {len(merged)} blocchi dopo merge, "
        f"{len(result)}/{budget} chars usati (parent originale: {len(content)} chars)."
    )
    return result


def build_context(rich_context, task_id="UNKNOWN"):
    """
    Assembla il contesto testuale per il generatore e restituisce
    (context_str, index_to_item, stats).

    L'ordine degli item è quello deciso da retrieve_chunks (round-robin tra
    sottoquery): se qualcosa deve cadere per esaurimento budget, cade la coda di
    TUTTE le sottoquery, non un comune intero.
    """
    n_parents = len(rich_context)
    if n_parents == 0:
        return "", {}, {}

    # --- 1. header: costruiti PRIMA, così il loro costo è sottratto dal budget
    # invece di sforare silenziosamente come accadeva prima (il vecchio codice
    # contava gli header solo nel break finale, non nel budget per parent).
    headers = []
    for idx, item in enumerate(rich_context, start=1):
        parts = [str(idx)]
        if item.get("sub_topic"):
            parts.append(f"Ambito: {item['sub_topic']}")
        # Provenienza della RICERCA, non del contenuto: il prompt istruisce il
        # modello a verificarla nel testo prima di attribuire il documento.
        if item.get("retrieved_by"):
            parts.append(f"Rif. ricerca: {'; '.join(item['retrieved_by'])}")
        parts.append(f"Date: {item.get('date') or 'unknown'}")
        headers.append("[" + " | ".join(parts) + "]\n")

    header_cost = sum(len(h) for h in headers) + 2 * n_parents  # "\n\n" tra i blocchi
    budget = max(1000, settings.max_context_chars - header_cost)

    # --- 2. core spans e fabbisogni
    for item in rich_context:
        content = item.get("content", "")
        spans = compute_core_spans(content, item.get("matched_snippets", []), task_id=task_id)
        item["_core_spans"] = spans
        need_min, need_full = estimate_parent_needs(content, spans)
        item["need_min"] = need_min
        item["need_full"] = need_full

    # --- 3. allocazione equa tra sottoquery
    alloc, alloc_stats = allocate_context_budget(rich_context, budget, task_id=task_id)

    # --- 4. rendering
    formatted_chunks = []
    index_to_item = {}
    curr_len = 0
    hard_cap = settings.context_hard_cap_chars

    for pos, (idx, item) in enumerate(enumerate(rich_context, start=1)):
        content = item.get("content", "")
        item_budget = alloc[pos]
        if item_budget <= 0:
            log.debug(f"[{task_id}] Parent [{idx}] senza budget assegnato: escluso dal contesto.")
            continue

        rendered = render_parent_window(content, item.get("_core_spans", []), item_budget, task_id=task_id)
        chunk = f"{headers[pos]}{rendered}\n\n"

        if curr_len + len(chunk) > hard_cap:
            log.warning(
                f"[{task_id}] Hard cap raggiunto ({curr_len}/{hard_cap} chars): "
                f"{n_parents - len(formatted_chunks)} parent scartati."
            )
            break

        formatted_chunks.append(chunk)
        curr_len += len(chunk)
        index_to_item[idx] = item
        # Caratteri REALMENTE finiti nel contesto per questo parent, header
        # incluso: è l'unico numero che serve per capire come il budget è stato
        # speso davvero. Va registrato qui perché dopo il windowing il dato non
        # è più ricostruibile dall'item.
        item["_rendered_chars"] = len(chunk)

    # --- 5. osservabilità: l'allocazione effettiva, per sottoquery
    # 'chars' contava need_full, cioè il fabbisogno TEORICO del parent, non i
    # caratteri effettivamente renderizzati: su un documento finestrato i due
    # numeri divergono di parecchio (9455 richiesti contro 7403 usati, nei log)
    # e la somma per sottoquery poteva superare il contesto reale, rendendo la
    # riga inservibile proprio quando serviva, cioè per capire chi si era preso
    # il budget.
    per_query = {}
    for idx, item in index_to_item.items():
        q = item.get("query_idx", 0)
        entry = per_query.setdefault(q, {"query": item.get("query_text", ""), "parents": 0, "chars": 0})
        entry["parents"] += 1
        entry["chars"] += item.get("_rendered_chars", 0)
    log.info(
        f"[{task_id}] [ALLOC] Contesto: {len(index_to_item)}/{n_parents} parent, "
        f"{curr_len} chars (budget {settings.max_context_chars}, tetto {alloc_stats.get('ceiling')}, "
        f"hard cap {hard_cap}). Ripartizione per sottoquery: "
        + " | ".join(
            f"#{q}({v['parents']} parent, {v['chars']} chars)"
            for q, v in sorted(per_query.items())
        )
    )

    stats = {**alloc_stats, "context_chars": curr_len, "n_parents_in_context": len(index_to_item)}
    return "".join(formatted_chunks), index_to_item, stats


def generate_answer(query, rich_context, topic_id, search_queries=None,
                    coverage_targets=None, task_id="UNKNOWN"):
    """
    Genera la risposta finale.

    search_queries / coverage_targets: le facce in cui il rewriter ha scomposto
    la domanda e le entità da coprire. PRIMA non venivano passate, e questo
    rendeva la RULE FOR PARTIAL CONTEXT del prompt di fatto inapplicabile: la
    standalone_query è deliberatamente generica ("Sono previsti lavori nei
    comuni della Garfagnana?"), quindi il modello non aveva alcuna lista di
    entità da confrontare col contesto e non poteva dichiarare le assenze.

    Restituisce anche used_sources / used_parent_ids: le fonti REALMENTE entrate
    nel prompt, non tutte quelle recuperate.
    """
    context_str, index_to_item, ctx_stats = build_context(rich_context, task_id=task_id)

    # Fonti effettive: derivate da index_to_item, mai dal pool di retrieval.
    # Deduplicate per `source`, ma i parent_id sono raccolti TUTTI (vedi nota su
    # used_parent_ids: servono interi al fallback per escludere davvero ciò che
    # il generatore ha già visto).
    used_sources = {}
    used_parent_ids = set()
    for item in index_to_item.values():
        src = item.get("source", "Fonte_Sconosciuta")
        if item.get("parent_id"):
            used_parent_ids.add(item["parent_id"])
        if src not in used_sources:
            used_sources[src] = {
                "source": src,
                "sub_topic": item.get("sub_topic", ""),
                "file_name": item.get("file_name", ""),
                "date": item.get("date"),
                "parent_id": item.get("parent_id"),
            }

    facets = [q for q in (search_queries or []) if q]
    targets = [t for t in (coverage_targets or []) if t]

    prompt_file = get_topic_prompt(topic_id)
    prompt_tmpl = load_prompt_template(prompt_file)
    fmt_kwargs = {
        "context_str": context_str,
        "query": query,
        "search_facets_str": "\n".join(f"- {f}" for f in facets),
        "coverage_targets_str": ", ".join(targets),
    }
    try:
        prompt = prompt_tmpl.format(**fmt_kwargs)
    except KeyError as e:
        # Retrocompatibilità: prompt di topic che non conoscono i nuovi
        # placeholder continuano a funzionare invariati.
        log.debug(f"[{task_id}] Prompt '{prompt_file}' senza placeholder {e}: uso il formato legacy.")
        prompt = prompt_tmpl.format(context_str=context_str, query=query)

    log.info(f"[{task_id}] Generating structured answer for topic '{topic_id}'")
    log.debug(f"[{task_id}] Context size: {len(context_str)} chars | facets={len(facets)} | targets={targets}")

    # Budget di OUTPUT (distinto da quello di input): con settings.answer_max_tokens
    # a None il parametro non viene passato affatto al provider, quindi il tetto è
    # quello del modello per definizione — nessuna costante da tenere allineata.
    #
    # Qui NON c'è retry, ed è deliberato. Il vecchio raddoppio era già codice
    # morto in produzione (con ANSWER_MAX_TOKENS pari all'hard cap la guardia era
    # sempre falsa) e al soffitto del modello non esiste comunque nessun numero
    # più grande da chiedere: ritentare significherebbe solo ripagare per intero
    # un contesto da decine di migliaia di token per ottenere lo stesso esito.
    # Il troncamento smette quindi di essere un caso da gestire e diventa un
    # ALLARME, con in log i dati che servono a capire su quale leva agire.
    max_tokens = settings.answer_max_tokens
    raw, meta = get_llm_provider().generate_json(
        settings.answer_generator_model_name, prompt,
        max_tokens=max_tokens,
        thinking_level=settings.answer_thinking_level,
        with_meta=True,
    )
    if meta.get("truncated"):
        thinking_tokens = meta.get("thinking_tokens")
        output_tokens = meta.get("output_tokens")
        # Le due cause richiedono interventi opposti: se a saturare il budget è
        # stato il thinking, la leva è ANSWER_THINKING_LEVEL; se invece l'output
        # visibile era davvero enorme, il problema sta a monte (troppi parent in
        # contesto, o un prompt che non impone sintesi) e abbassare il thinking
        # non servirebbe a niente.
        if thinking_tokens and output_tokens and thinking_tokens > output_tokens:
            causa = (
                f"il thinking ha prodotto più token dell'output visibile "
                f"({thinking_tokens} vs {output_tokens}): abbassare ANSWER_THINKING_LEVEL "
                f"(attuale: {settings.answer_thinking_level})"
            )
        else:
            causa = (
                "l'output visibile ha saturato il budget del modello: ridurre il "
                "materiale da sintetizzare (PARENTS_PER_QUERY, MAX_CONTEXT_CHARS) "
                "o rendere il prompt più stringente sulla sintesi"
            )
        log.error(
            f"[{task_id}] Risposta TRONCATA (finish_reason={meta.get('finish_reason')}, "
            f"budget={'modello' if max_tokens is None else max_tokens}, "
            f"thinking_tokens={thinking_tokens}, output_tokens={output_tokens}, "
            f"contesto={len(context_str)} chars su {len(index_to_item)} parent). "
            f"Probabile causa: {causa}."
        )

    log.debug(f"[{task_id}] Raw response:\n{raw[:500]}...")

    base_result = {
        "used_sources": list(used_sources.values()),
        "used_parent_ids": used_parent_ids,
        "context_str": context_str,
        "context_stats": ctx_stats,
    }

    try:
        result_data = safe_json_parse(raw, task_id=task_id)
        raw_answer = str(result_data.get("answer", ""))
        resolved_answer = resolve_citations(raw_answer, index_to_item)
        return {
            **base_result,
            "is_found": bool(result_data.get("is_found", True)),
            "is_general_knowledge": bool(result_data.get("is_general_knowledge", False)),
            "answer": resolved_answer,
            # Risposta PRIMA della risoluzione delle citazioni: conserva i
            # riferimenti numerici ([3]), gli stessi che numerano i blocchi del
            # contesto. È questa che va data al grader: la versione risolta
            # contiene filename, che nel contesto non compaiono da nessuna parte,
            # rendendo le citazioni non verificabili.
            "raw_answer": raw_answer,
        }
    except json.JSONDecodeError as e:
        log.error(f"[{task_id}] Generazione JSON fallita: {e}. Output grezzo: {raw}")
        return {**base_result, "is_found": True, "is_general_knowledge": False,
                "answer": raw.strip(), "raw_answer": raw.strip()}

def grade_answer(query, context_snippet, answer, coverage_targets=None,
                 is_general_knowledge=False, task_id="UNKNOWN"):
    """
    Valuta la risposta e restituisce un dict:
        {"ok": bool, "reason": str, "missing_targets": list, "note": str}

    Il grader boccia SOLO due casi: "grounding" (afferma come tratto dai
    documenti qualcosa che il contesto non sostiene) e "no_answer" (nessuna
    informazione utile). La copertura parziale NON è un fallimento: se per un
    comune non risultano documenti, dirlo è la risposta corretta, non un
    difetto. missing_targets viene raccolto per sola diagnostica e finisce in
    rag_metrics, dove sui volumi distingue un problema di ingestion (documenti
    mai indicizzati) da uno di soglie (presenti ma non agganciati).

    context_snippet DEVE essere il context_str realmente passato al generatore,
    non i content integrali: giudicare su un testo più ampio di quello che
    l'autore aveva davanti produce sia falsi negativi (retry inutili su
    informazioni che erano state finestrate) sia falsi positivi (il grader
    trova la prova che mancava al generatore e valida un'affermazione non
    ancorata — cioè è cieco proprio sul caso che dovrebbe intercettare).

    FAIL-OPEN: qualunque output non interpretabile vale PASS, coerentemente col
    trattamento delle eccezioni. Prima l'incoerenza era silenziosa e costosa:
    un'eccezione dava is_satisfactory=True, ma una stringa vuota (tipica del
    troncamento con max_tokens=32) falliva startswith("YES") e bruciava un
    tentativo di retry.
    """
    targets = [t for t in (coverage_targets or []) if t]
    grader_tmpl = load_prompt_template("grader")

    fmt_kwargs = {
        "query": query,
        "context_snippet": context_snippet,
        "answer": answer,
        "coverage_targets_str": ", ".join(targets),
        "is_general_knowledge": "true" if is_general_knowledge else "false",
    }
    try:
        grader_prompt = grader_tmpl.format(**fmt_kwargs)
    except KeyError as e:
        # Retrocompatibilità con il grader.txt legacy (solo query/context/answer).
        log.debug(f"[{task_id}] [GRADER] Prompt senza placeholder {e}: uso il formato legacy.")
        grader_prompt = grader_tmpl.format(
            query=query, context_snippet=context_snippet, answer=answer
        )

    raw = get_llm_provider().generate_json(
        settings.grader_model_name,
        grader_prompt,
        temperature=0.0,
        max_tokens=settings.grader_max_tokens,
        thinking_level=settings.grader_thinking_level,
    )

    result = {"ok": True, "reason": "ok", "missing_targets": [], "note": ""}
    text = (raw or "").strip()

    if not text:
        log.warning(f"[{task_id}] [GRADER] Output vuoto: fail-open, risposta accettata.")
        result["note"] = "grader output vuoto"
        return result

    try:
        data = safe_json_parse(text, task_id=task_id)
        verdict = str(data.get("verdict", "PASS")).strip().upper()
        result["ok"] = verdict != "FAIL"
        result["reason"] = str(data.get("reason", "ok")).strip().lower() or "ok"
        raw_missing = data.get("missing_targets") or []
        if isinstance(raw_missing, list):
            result["missing_targets"] = [str(m) for m in raw_missing if m]
        result["note"] = str(data.get("note", ""))[:300]
    except Exception:
        # Fallback sul contratto testuale precedente (YES/NO), così un
        # grader.txt non aggiornato continua a funzionare.
        upper = text.upper()
        if upper.startswith("NO"):
            result.update(ok=False, reason="grounding", note="verdetto legacy NO")
        elif upper.startswith("YES"):
            result["note"] = "verdetto legacy YES"
        else:
            log.warning(f"[{task_id}] [GRADER] Output non interpretabile ({text[:120]}): fail-open.")
            result["note"] = "output non interpretabile"

    # La copertura è diagnostica: non può mai ribaltare il verdetto.
    if result["reason"] == "coverage":
        log.debug(f"[{task_id}] [GRADER] reason='coverage' ignorata: la copertura parziale non è un fallimento.")
        result.update(ok=True, reason="ok")

    log.info(
        f"[{task_id}] [GRADER] verdict={'PASS' if result['ok'] else 'FAIL'} "
        f"reason={result['reason']} missing={result['missing_targets']} note='{result['note']}'"
    )
    return result


def evaluate_generation(gen_result, query, coverage_targets=None, task_id="UNKNOWN"):
    """
    Valutazione unificata di una risposta generata. Restituisce
    (is_satisfactory, reason, missing_targets). Usata sia dal loop principale
    sia dal fallback finale, che prima ne avevano due copie divergenti.

    reason ∈ {"ok", "not_found", "grounding", "no_answer", "general_knowledge"}
    e guida il messaggio iniettato nel retry: riformulare in modo ampio ha senso
    quando la ricerca ha fallito, non quando i documenti c'erano ma erano
    insufficienti su un aspetto specifico.

    NOTA sul flag is_general_knowledge: NON è più una scorciatoia che salta il
    grader. Il flag di `atti` si alza anche quando la risposta è ancorata al 95%
    ai documenti e contiene un solo esempio suggerito in fondo: far uscire
    l'INTERA risposta dal controllo di grounding lasciava senza verifica proprio
    le risposte miste — le più esposte al rischio di confondere ciò che il
    documento dice con ciò che il modello sa — e poi le metteva in cache
    semantica, dove venivano riproposte ad altri utenti senza essere mai state
    controllate. Ora il grader vede sempre la risposta, e sa distinguere un
    suggerimento dichiarato (legittimo) da un fatto attribuito ai documenti ma
    assente dal contesto (allucinazione).
    """
    if not gen_result.get("is_found", False):
        log.info(f"[{task_id}] Model explicitly flagged is_found=False.")
        return False, "not_found", []

    is_gk = bool(gen_result.get("is_general_knowledge"))

    try:
        verdict = grade_answer(
            query,
            gen_result.get("context_str", ""),
            # Risposta grezza: i riferimenti numerici sono verificabili contro
            # il contesto numerato; i filename della versione risolta no.
            gen_result.get("raw_answer") or gen_result.get("answer", ""),
            coverage_targets=coverage_targets,
            is_general_knowledge=is_gk,
            task_id=task_id,
        )
    except Exception as e:
        log.error(f"[{task_id}] Grader LLM failed (non-blocking): {e}. Fail-open.")
        verdict = {"ok": True, "reason": "ok", "missing_targets": [], "note": "grader exception"}

    missing = verdict.get("missing_targets", [])

    if not verdict["ok"]:
        return False, verdict["reason"], missing

    # Il grader ha promosso la risposta. Resta la policy sulla conoscenza
    # generale: se non è ammessa si torna nel loop per un altro tentativo di
    # retrieval, con i profili progressivamente più permissivi (semantic_size
    # più ampio, soglia più bassa). Se il modello ha dovuto ricorrere a
    # conoscenza esterna perché il contesto era povero, allargare la recall è
    # esattamente la mossa giusta.
    if is_gk and not settings.allow_general_knowledge:
        log.warning(
            f"[{task_id}] Conoscenza generale non ammessa (ALLOW_GENERAL_KNOWLEDGE=false): nuovo tentativo."
        )
        return False, "general_knowledge", missing

    return True, "ok", missing


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
                     exclude_parent_ids=None,
                     query_semantic_vector=None,
                     task_id="UNKNOWN"):
    """
    Recupera, fonde e riordina i chunk dal Vector DB in modo sicuro.

    semantic_size / semantic_threshold: sovrascrivono i default di settings, usati
        dai profili di retry per allargare progressivamente la recall.
    mmr_lambda: peso rilevanza/diversità nella redistribuzione degli slot liberi.
        1.0 = comportamento identico a prima dell'introduzione di MMR.
    exclude_parent_ids: set di parent_id da escludere del tutto dalla selezione
        (usato dall'ultimo tentativo di fallback, per non riproporre parent
        già passati al generatore in un tentativo precedente e già falliti).
        L'esclusione avviene PRIMA della quota/MMR, così il budget di parent
        non viene sprecato su candidati che verrebbero comunque scartati.
    """
    if not qdrant_client:
        log.error("Qdrant client not available! Retrieval aborted.")
        return [], [], {}

    # Boilerplate attivo per (topic, sub_topic), caricato una volta per
    # l'intero retrieval. La decisione "quali frasi rimuovere" dipende solo da
    # query e frasi, non dai chunk: si prende UNA volta qui, non per ogni chunk.
    # removal_by_subtopic: {sub_topic_id: regex delle frasi da togliere}.
    boilerplate_map = load_active_boilerplate(topic_id, selected_sub_topics)
    removal_by_subtopic = build_removal_by_subtopic(
        boilerplate_map, " ".join(search_queries), query_semantic_vector
    )

    semantic_size = semantic_size or settings.qdrant_semantic_size
    semantic_threshold = semantic_threshold if semantic_threshold is not None else settings.qdrant_semantic_threshold

    # I vettori dei chunk servono SOLO a costruire i centroid-per-parent usati
    # da MMR. Con mmr_lambda=1.0 (il profilo del tentativo 0, cioè il caso
    # normale) select_with_mmr va in bypass e i centroidi vengono buttati senza
    # essere mai letti: li si pagava due volte, in trasferimento di rete e in
    # calcolo. Ogni gruppo trasporta un embedding gemini da 3072 dimensioni
    # serializzato in JSON — su 30 gruppi × 6 sottoquery è il motivo per cui le
    # search sfioravano il timeout.
    needs_vectors = mmr_lambda < 1.0

    log.info(f"=== Inizio Retrieval per topic '{topic_id}' ===")
    log.info(
        f"[{task_id}] [RETRIEVAL_PARAMS] semantic_size={semantic_size}, "
        f"semantic_threshold={semantic_threshold:.3f}, mmr_lambda={mmr_lambda}, "
        f"with_vectors={needs_vectors}."
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
                    with_vectors=needs_vectors  # centroid-per-parent: solo se MMR è attivo
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
                        with_vectors=needs_vectors  # centroid-per-parent: solo se MMR è attivo
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
        #
        # Ogni sotto-query pesca dal pool di istanze ONNXReranker (get_reranker_pool),
        # una per thread concorrente invece di condividerne una sola. Prima, tutti i
        # thread del ThreadPoolExecutor sotto chiamavano .rerank() sulla STESSA
        # istanza globale (get_reranker()): la sessione ONNX è deliberatamente
        # ORT_SEQUENTIAL (per evitare deadlock in Celery), quindi 2 thread paralleli
        # si contendevano lo stesso budget di calcolo invece di raddoppiarlo.
        # ==========================================================
        pool = get_reranker_pool()
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

            # Rerank effettivo — istanza dedicata dal pool per questo q_idx,
            # non condivisa con gli altri thread concorrenti.
            reranker_instance = pool[q_idx % len(pool)] if pool else None
            if reranker_instance:
                # Preparazione del testo passato al reranker, in quest'ordine:
                #   1. normalizzazione whitespace/entità HTML (&nbsp;), PRIMA del
                #      taglio: un chunk pieno di indentazione sprecherebbe la
                #      finestra del cross-encoder in spazi invece che in testo;
                #   2. rimozione del boilerplate attivo, protetta dalla guardia
                #      lessicale (una frase presente nella query non si tocca);
                #   3. troncamento a rerank_truncate.
                # Tutto ciò riguarda SOLO il testo per il ranking: il contenuto
                # renderizzato per l'utente e per il generatore resta integro.
                docs_content = []
                for c in unique_candidates:
                    raw = normalize_ws(c.payload.get("content", "") if c.payload else "")
                    # Rimuove SOLO il boilerplate del sub_topic DI QUESTO chunk:
                    # la formula delle determine non tocca i chunk dei decreti.
                    # La decisione di pertinenza è già stata presa (regex pronta);
                    # qui è pura sostituzione, nessun calcolo per chunk.
                    st = c.payload.get("sub_topic_id") if c.payload else None
                    rx = removal_by_subtopic.get(st) if st else None
                    txt = rx.sub(" ", raw) if rx else raw
                    docs_content.append(txt[:settings.rerank_truncate])
                try:
                    reranked = reranker_instance.rerank(query_str, docs_content)
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
        # Tutte le sottoquery che hanno recuperato quel parent (non solo quella
        # che se lo aggiudica in quota): serve per l'header del contesto, dove
        # dichiariamo al modello la PROVENIENZA della ricerca.
        parent_retrieved_by: dict[str, set] = {}
        # NEW: fino a SNIPPETS_PER_PARENT child (contenuto + score) per parent,
        # usati da generate_answer per il windowing invece del troncamento cieco.
        parent_top_snippets: dict[str, list] = {}

        exclude_parent_ids = exclude_parent_ids or set()
        excluded_count = 0

        for q_idx, chunks in enumerate(top_chunks_per_query):
            for chunk in chunks:
                pid = chunk.payload.get("parent_id") if chunk.payload else None
                if not pid:
                    continue
                if pid in exclude_parent_ids:
                    excluded_count += 1
                    continue
                parent_retrieved_by.setdefault(pid, set()).add(q_idx)
                score = getattr(chunk, "score", 0.0) or 0.0
                if pid not in parent_best_score or score > parent_best_score[pid]:
                    parent_best_score[pid] = score

                if needs_vectors:
                    vec = getattr(chunk, "vector", None)
                    if vec is not None:
                        parent_chunk_vectors.setdefault(pid, []).append(np.asarray(vec))

                chunk_content = chunk.payload.get("content", "") if chunk.payload else ""
                if chunk_content:
                    snippets = parent_top_snippets.setdefault(pid, [])
                    snippets.append({
                        "content": chunk_content,
                        "score": score,
                        # Popolati dal backfill (backfill_chunk_offsets.py) o
                        # dall'ingestion futura; None se assenti — in quel caso
                        # compute_core_spans ripiega sulla ricerca testuale.
                        "start_char": chunk.payload.get("start_char"),
                        "end_char": chunk.payload.get("end_char"),
                    })
                    snippets.sort(key=lambda s: s["score"], reverse=True)
                    del snippets[SNIPPETS_PER_PARENT:]

        if exclude_parent_ids:
            log.info(f"[{task_id}] [EXCLUDE] {excluded_count} chunk scartati (parent già usati in tentativi precedenti: {len(exclude_parent_ids)}).")

        if needs_vectors:
            parent_centroids = {
                pid: np.mean(vecs, axis=0) for pid, vecs in parent_chunk_vectors.items()
            }
            n_missing_vectors = sum(1 for pid in parent_best_score if pid not in parent_centroids)
            log.info(
                f"[{task_id}] {MMR_LOG_TAG} Centroid calcolati per {len(parent_centroids)} parent "
                f"(su {len(parent_best_score)} parent candidati; {n_missing_vectors} senza vettore disponibile)."
            )
        else:
            # Dizionario vuoto e non None: la redistribuzione più sotto filtra
            # già i pid privi di centroid ("if pid in parent_centroids"), quindi
            # con MMR in bypass il ramo degrada da solo a ranking puro senza
            # bisogno di un secondo controllo su mmr_lambda.
            parent_centroids = {}
            log.debug(
                f"[{task_id}] {MMR_LOG_TAG} mmr_lambda={mmr_lambda}: centroidi non calcolati "
                f"(vettori non richiesti a Qdrant)."
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

            # Il WARNING ha senso solo se i vettori li AVEVAMO chiesti e non sono
            # arrivati (anomalia vera). Con MMR in bypass non li abbiamo chiesti
            # affatto: segnalarlo come mancanza produrrebbe un allarme su ogni
            # singolo candidato di ogni richiesta del caso normale, che è il modo
            # più rapido per insegnare a chi legge i log a ignorare i WARNING.
            if n_skipped_no_vector and needs_vectors:
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
        # 6. POOL FINALE — ordine ROUND-ROBIN tra sottoquery
        #
        # PRIMA: concatenazione a blocchi (tutti i parent della sottoquery 0,
        # poi quelli della 1, ...). Quando il budget del contesto si esauriva,
        # a cadere era sempre la coda — cioè l'ULTIMO comune per intero. E la
        # sottoquery 0 era privilegiata due volte, essendo la standalone_query
        # e ricevendo in appendice tutti i keyword_hits.
        #
        # ORA: rank 0 di ogni sottoquery, poi rank 1, ecc. Se qualcosa deve
        # cadere, cade la coda di TUTTE le sottoquery. Serve anche al modello,
        # che vede subito la diversità dei territori invece di una lunga
        # sequenza omogenea sul primo.
        # ==========================================================
        final_parent_ids_ordered: list[str] = []
        seen_final: set[str] = set()
        # Sottoquery "proprietaria" di ogni parent, per l'equità nell'allocatore.
        pid_to_query_idx: dict[str, int] = {}

        max_depth = max((len(q) for q in quota_per_query), default=0)
        for rank in range(max_depth):
            for q_idx, selected_pids in enumerate(quota_per_query):
                if rank >= len(selected_pids):
                    continue
                pid = selected_pids[rank]
                if pid not in seen_final:
                    final_parent_ids_ordered.append(pid)
                    seen_final.add(pid)
                    pid_to_query_idx[pid] = q_idx
 
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
                pid = p_doc.get("id")
                matched_snippets = [
                    s for s in parent_top_snippets.get(pid, []) if s.get("content")
                ]
                q_idx = pid_to_query_idx.get(pid, 0)
                # Testi delle sottoquery che hanno recuperato questo parent:
                # è PROVENIENZA della ricerca, non attribuzione del contenuto.
                retrieved_by = [
                    search_queries[i]
                    for i in sorted(parent_retrieved_by.get(pid, {q_idx}))
                    if i < len(search_queries)
                ]
                rich_context.append({
                    "parent_id": pid,
                    "content": content,
                    "source": source,
                    "date": doc_date,
                    "file_name": file_name,
                    "sub_topic": sub_topic_id,
                    "matched_snippets": matched_snippets,
                    # Chi ha "vinto" questo parent in quota: è l'asse su cui
                    # l'allocatore garantisce l'equità del contesto.
                    "query_idx": q_idx,
                    "query_text": search_queries[q_idx] if q_idx < len(search_queries) else "",
                    "retrieved_by": retrieved_by,
                    "score": parent_best_score.get(pid, 0.0),
                })
                if source not in unique_sources_map:
                    unique_sources_map[source] = {
                        "source": source, "sub_topic": sub_topic_id, "file_name": file_name,
                        "date": doc_date, "parent_id": p_doc.get("id"),
                    }
 
        log.info(f"=== Retrieval completata. Parent al generatore: {len(rich_context)} ===")
        retrieval_stats["n_parents_final"] = len(rich_context)
        return rich_context, list(unique_sources_map.values()), retrieval_stats
 
    except Exception as e:
        log.critical(f"Errore critico nel Retrieval: {e}", exc_info=True)
        return [], [], {}

def get_reranker_pool():
    """
    Pool di istanze ONNXReranker indipendenti, una per thread concorrente
    (settings.reranker_pool_size). Ogni sotto-query pesca la propria istanza
    via round-robin su q_idx (vedi process_single_rerank in retrieve_chunks),
    invece di contendersi un'unica sessione condivisa tra tutti i thread.

    num_threads per istanza è calcolato dividendo i core disponibili per la
    dimensione del pool. I core disponibili vengono letti da
    settings.cpu_limit (il resources.limits.cpu del pod, es. "2"), NON da
    os.cpu_count(): quest'ultimo, in un container Kubernetes, riflette i
    core del nodo host — i CPU limit di Kubernetes si applicano via CFS
    quota/period, non modificano l'affinity mask che os.cpu_count() legge.
    Su un pod limitato a 2 core ma su un nodo da 8, os.cpu_count() avrebbe
    fatto sovrastimare di 4 volte i thread assegnabili per istanza.
    Se settings.cpu_limit non è impostato esplicitamente (CPU_LIMIT), il
    fallback su os.cpu_count() avviene già in config.py — qui arriva sempre
    un intero valido. Da preferire comunque sempre CPU_LIMIT esplicito in
    un pod Kubernetes, dove os.cpu_count() riflette i core del nodo host.

    I core disponibili vengono inoltre divisi per settings.celery_concurrency
    (deve combaciare col --concurrency del comando Celery nel deployment):
    i core del pod si dividono tra TUTTI i processi Celery concorrenti, non
    solo tra le istanze del pool all'interno di un singolo processo. Senza
    questo, ogni processo dimensionerebbe il proprio pool assumendo di avere
    tutti i core del pod per sé, sovrasottoscrivendo quando più processi
    fanno reranking nello stesso momento.
    """
    global _RERANKER_POOL
    if not _RERANKER_POOL:
        total_cores = settings.cpu_limit
        cores_per_process = max(1, total_cores // max(1, settings.celery_concurrency))
        num_threads = max(1, cores_per_process // max(1, _POOL_SIZE))
        log.info(
            f"Core totali pod: {total_cores}, concorrenza Celery: {settings.celery_concurrency} "
            f"-> {cores_per_process} core per processo, {num_threads} thread per istanza del pool."
        )
        for i in range(_POOL_SIZE):
            log.info(f"Initializing ONNX Reranker pool instance {i+1}/{_POOL_SIZE} (num_threads={num_threads})...")
            instance = ONNXReranker(
                model_folder=settings.onnx_model_cache_path,
                batch_size=settings.rerank_batch_size,
                max_length=settings.rerank_max_length,
                num_threads=num_threads
            )
            _RERANKER_POOL.append(instance)
    return _RERANKER_POOL




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
        concept_vector = None 
        try:
            # Dizionario concettuale: usa lookup_text (query + ultimi turni),
            # perché il contesto conversazionale aiuta ad agganciare i concetti
            # su domande di follow-up brevi ("e quelle del 2024?").
            concept_vector = embed_for_semantic_query(lookup_text)
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
        coverage_targets = rewritten_data.get("coverage_targets", [])
        db_aliases = collect_concept_aliases(verified_concepts)

        if standalone_query not in search_queries:
                    search_queries.insert(0, standalone_query)
        vectors_list = embed_queries_batch(search_queries) 
        all_keywords = list(set(extracted_keywords + db_aliases))
            
    except Exception as e:
        log.warning(f"[{task_id}] [MAIN_TASK] Pipeline di trasformazione caduta: {e}. Uso raw query.")
        standalone_query, search_queries, all_keywords = query, [query], []
        coverage_targets = []
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
            query_semantic_vector=concept_vector,
            task_id=task_id,
            **profile_kwargs(0, task_id=task_id)
        )
        attempts_metrics.append({"attempt": 0, "standalone_query": standalone_query, **retrieval_stats})

        # Parent già MOSTRATI al generatore, accumulati tra i tentativi: usato
        # dall'ultimo fallback (dopo max_model_retries) per non riproporre
        # parent già passati e già giudicati insoddisfacenti.
        #
        # PRIMA veniva derivato da `sources`, che è deduplicata per `source` e
        # conserva un solo parent_id per documento: se un atto contribuiva con
        # 3 parent ne veniva registrato 1, e gli altri 2 rientravano indisturbati
        # nel contesto del fallback, vanificandone in buona parte lo scopo.
        # ORA viene popolato da generate_answer con i parent REALMENTE finiti nel
        # prompt: un parent recuperato ma mai mostrato non è stato giudicato da
        # nessuno e resta legittimamente candidabile.
        used_parent_ids = set()
    except Exception as e:
        log.error(f"[{task_id}] Vector DB Retrieval failed: {e}", exc_info=True)
        return {"error": "Database Error", "message": "Errore durante il recupero dei documenti.", "status": "failed"}

    # ==========================================================
    # SELF-CORRECTION LOOP (Generation & Grading)
    # ==========================================================
    attempt = 0
    answer = None
    is_satisfactory = False
    # Inizializzati qui: restano definiti anche se il loop esce subito per
    # contesto vuoto, e vengono letti dopo il loop per applicare la policy.
    fail_reason = "ok"
    missing_targets = []

    while attempt < settings.max_model_retries and not is_satisfactory:
        if not context:
            answer = "<p>Non sono riuscito a trovare la risposta nei documenti che ho analizzato.</p>"
            log.debug(f"[{task_id}] Context empty. Breaking loop.")
            break
        else:
            log.debug(f"Contesto da passare alla generazione: {str(context)[:1500]}")

        # A. Generazione (JSON Mode)
        try:
            gen_result = generate_answer(
                standalone_query, context, topic_id,
                search_queries=search_queries,
                coverage_targets=coverage_targets,
                task_id=task_id,
            )
            is_found = gen_result.get("is_found", False)
            answer = gen_result.get("answer", "")

            # Le fonti mostrate all'utente sono SOLO quelle realmente entrate nel
            # prompt: prima si elencavano tutti i parent recuperati, comprese le
            # fonti scartate per esaurimento budget e mai lette dal modello.
            sources = gen_result.get("used_sources", sources)
            used_parent_ids |= gen_result.get("used_parent_ids", set())

            log.debug(f"Risposta generata: {answer[:500]}... | is_found: {is_found}")
        except Exception as e:
            log.error(f"[{task_id}] Answer generation API failed: {e}", exc_info=True)
            return {"error": "Generation Failed", "message": "Impossibile elaborare la risposta.", "status": "failed"}

        # B. Valutazione unificata (fail-fast + grader + policy)
        is_satisfactory, fail_reason, missing_targets = evaluate_generation(
            gen_result, standalone_query, coverage_targets=coverage_targets, task_id=task_id
        )
        if attempts_metrics:
            attempts_metrics[-1]["grader_reason"] = fail_reason
            attempts_metrics[-1]["missing_targets"] = missing_targets

        # D. Gestione Retry Fallimento
        if not is_satisfactory:
            attempt += 1
            if attempt < settings.max_model_retries:
                log.warning(f"[{task_id}] Answer unsatisfactory. Retrying ({attempt}/{settings.max_model_retries}) with full pipeline...")
                try:
                    # 1. Iniettiamo un "falso" messaggio di sistema nella history per forzare l'LLM a cambiare approccio.
                    #    Il messaggio dipende dal MOTIVO del fallimento: dire al
                    #    rewriter che "la ricerca non ha prodotto documenti validi"
                    #    quando invece i documenti c'erano ma erano insufficienti su
                    #    un aspetto lo spinge ad allontanarsi dalla formulazione
                    #    originale, mentre servirebbe restare sul punto e scavare.
                    if fail_reason == "general_knowledge":
                        retry_instruction = (
                            f"La risposta precedente per '{standalone_query}' ha dovuto ricorrere a conoscenza "
                            f"esterna perché i documenti recuperati non coprivano tutti gli aspetti richiesti. "
                            f"Mantieni la stessa intenzione e lo stesso ambito, ma cerca documenti più specifici "
                            f"sugli aspetti rimasti scoperti."
                        )
                    elif fail_reason == "grounding":
                        retry_instruction = (
                            f"La risposta precedente per '{standalone_query}' conteneva affermazioni non "
                            f"sostenute dai documenti recuperati. Mantieni l'intenzione originale ma cerca "
                            f"documenti che trattino l'argomento in modo più diretto e verificabile."
                        )
                    else:
                        retry_instruction = (
                            f"La ricerca precedente per '{standalone_query}' non ha prodotto documenti validi. "
                            f"Riformula completamente la query usando sinonimi o concetti più ampi per esplorare "
                            f"un'angolazione semantica diversa."
                        )

                    retry_history = history.copy() if history else []
                    retry_history.append({
                        "role": "user",
                        "text": retry_instruction
                    })
                    
                    # 2. Lookup concettuale anche in retry, ma sul lookup_text
                    #    GREZZO del primo giro (history + query originale
                    #    dell'utente), NON sulla standalone_query del tentativo
                    #    fallito: se il fallimento era dovuto proprio a una
                    #    riformulazione distorta, ripartire da quella
                    #    propagherebbe l'errore anche nel lookup. Il testo
                    #    originale dell'utente è immune da errori di rewrite.
                    retry_concepts = []
                    retry_concept_vector = None
                    try:
                        retry_concept_vector = embed_for_semantic_query(lookup_text)
                        retry_concepts = get_concepts_by_similarity(retry_concept_vector, task_id=f"{task_id}-RETRY")
                    except Exception as ce:
                        log.warning(f"[{task_id}-RETRY] Lookup concettuale fallito ({ce}): retry senza concetti verificati.")

                    retry_data = transform_query(retry_history, query, f"{task_id}-RETRY", verified_concepts=retry_concepts)
                    
                    standalone_query = retry_data.get("standalone_query", query)
                    extracted_keywords = retry_data.get("keywords", [])
                    
                    if not extracted_keywords:
                        extracted_keywords = [w.strip("?.,!'\"") for w in standalone_query.split() if len(w) > 3]

                    search_queries = retry_data.get("search_queries", [standalone_query])
                    coverage_targets = retry_data.get("coverage_targets", coverage_targets)
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
                        query_semantic_vector=retry_concept_vector,
                        task_id=task_id,
                        **profile_kwargs(attempt, task_id=task_id)
                    )
                    attempts_metrics.append({"attempt": attempt, "standalone_query": standalone_query, **retrieval_stats})
                    # used_parent_ids NON viene più aggiornato qui: lo popola
                    # generate_answer col contenuto effettivo del prompt.
                except Exception as e:
                    log.error(f"[{task_id}] Retry infrastructure failed: {e}", exc_info=True)
                    break  # Usciamo usando l'ultima answer generata
            else:
                log.warning(f"[{task_id}] Max retries esauriti. Tento un ultimo fallback senza riformulazione, escludendo {len(used_parent_ids)} parent già usati.")
                try:
                    final_context, final_sources, final_stats = retrieve_chunks(
                        search_queries,
                        vectors_list,
                        extracted_keywords,
                        topic_id,
                        selected_sub_topics,
                        metadata_filters,
                        include_undated,
                        exclude_parent_ids=used_parent_ids,
                        query_semantic_vector=concept_vector,
                        task_id=f"{task_id}-FALLBACK",
                        **profile_kwargs(settings.max_model_retries - 1, task_id=task_id)
                    )
                    attempts_metrics.append({"attempt": "fallback", "standalone_query": standalone_query, **final_stats})

                    if not final_context:
                        log.info(f"[{task_id}] [FALLBACK] Nessun parent nuovo trovato dopo l'esclusione. Nessun ulteriore tentativo possibile.")
                    else:
                        context, sources = final_context, final_sources
                        gen_result = generate_answer(
                            standalone_query, context, topic_id,
                            search_queries=search_queries,
                            coverage_targets=coverage_targets,
                            task_id=f"{task_id}-FALLBACK",
                        )
                        is_found = gen_result.get("is_found", False)
                        answer = gen_result.get("answer", "")
                        sources = gen_result.get("used_sources", sources)
                        used_parent_ids |= gen_result.get("used_parent_ids", set())
                        log.debug(f"[{task_id}] [FALLBACK] Risposta generata: {answer[:500]}... | is_found: {is_found}")

                        is_satisfactory, fail_reason, missing_targets = evaluate_generation(
                            gen_result, standalone_query,
                            coverage_targets=coverage_targets,
                            task_id=f"{task_id}-FALLBACK",
                        )
                        if attempts_metrics:
                            attempts_metrics[-1]["grader_reason"] = fail_reason
                            attempts_metrics[-1]["missing_targets"] = missing_targets

                        log.info(f"[{task_id}] [FALLBACK] Esito ultimo tentativo: is_satisfactory={is_satisfactory} (reason={fail_reason}).")
                except Exception as fe:
                    log.error(f"[{task_id}] [FALLBACK] Ultimo tentativo fallito: {fe}", exc_info=True)

    # ==========================================================
    # APPLICAZIONE DELLA POLICY SULL'OUTPUT FINALE
    #
    # Con ALLOW_GENERAL_KNOWLEDGE=false, esaurire i tentativi lasciando in piedi
    # una risposta basata su conoscenza generale significherebbe applicare la
    # policy al giudizio ma non a ciò che l'utente legge: is_satisfactory=False
    # e, in mano al cittadino, esattamente la risposta che la policy vieta.
    # ==========================================================
    if not settings.allow_general_knowledge and fail_reason == "general_knowledge":
        log.warning(
            f"[{task_id}] Tentativi esauriti con risposta ancora basata su conoscenza generale: "
            f"sostituita con il messaggio di risposta non trovata (ALLOW_GENERAL_KNOWLEDGE=false)."
        )
        answer = "<p>Non sono riuscito a trovare la risposta nei documenti che ho analizzato.</p>"

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