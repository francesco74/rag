#!/usr/bin/env python3
"""
detect_boilerplate.py — analisi periodica delle frasi ricorrenti (Optimized for k8s memory limits).

# Nightly: rilevamento + import delle verificate, nello stesso run
python detect_boilerplate.py --import-file boilerplate.txt

# Solo aggiungere le verificate, senza toccare i vettori già presenti
python detect_boilerplate.py --import-file boilerplate.txt --import-only

# Anteprima senza scrivere
python detect_boilerplate.py --import-file boilerplate.txt --import-only --dry-run

#FORMATO boilerplate.txt (una tripletta per riga, separata da ';'):
# le righe con # e quelle vuote sono ignorate
TOPIC_A;SUB_1;il presente atto è pubblicato all'albo pretorio
TOPIC_B;SUB_2;visto l'art. 42; comma 3 del regolamento
"""

import argparse
import gc
import hashlib
import logging
import re
import uuid
from collections import defaultdict
from datetime import datetime

from common.config import settings
from common.db_logger import get_db_connection, init_db_pool
from common.utility import normalize_ws

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("detect_boilerplate")

BOILERPLATE_COLLECTION = "boilerplate_phrases"

NGRAM_MIN = 3
NGRAM_MAX = 8
DEFAULT_DF_THRESHOLD = 0.50
MIN_GROUP_DOCS = 5

# Valori sentinella per le frasi importate manualmente da file: non derivano da
# statistica sui documenti ma da curatela umana, quindi doc_count/group_docs non
# hanno un significato reale. doc_freq=1.0 = "sempre presente" (frase certa).
MANUAL_DOC_FREQ = 1.0
MANUAL_DOC_COUNT = 0
MANUAL_GROUP_DOCS = 0

_TOKEN_RE = re.compile(r"\w+(?:['\-/]\w+)*", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"[.;!?\n]+")


def phrase_hash(phrase: str) -> str:
    return hashlib.md5(normalize_ws(phrase).encode("utf-8")).hexdigest()


def ngrams_of(text: str, nmin: int, nmax: int) -> set:
    out = set()
    sentences = _SENTENCE_SPLIT_RE.split(text.lower())
    
    for sentence in sentences:
        tokens = _TOKEN_RE.findall(sentence)
        n_tokens = len(tokens)
        if n_tokens < nmin:
            continue

        for n in range(nmin, min(nmax, n_tokens) + 1):
            for i in range(n_tokens - n + 1):
                out.add(" ".join(tokens[i : i + n]))
    return out


def list_groups(conn):
    """Elenca i gruppi (topic_id, sub_topic_id). Il conteggio dei documenti
    ammessi è calcolato dentro analyze_group, dopo il filtro max_parts, perché
    il denominatore corretto è il numero di documenti-radice, non di parti."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT topic_id, sub_topic_id FROM parent_documents"
        )
        return cur.fetchall()


def analyze_group(conn, topic_id, sub_topic_id, nmin, nmax, threshold, max_parts):
    """Analizza un gruppo (topic, sub_topic) RICOMPONENDO i documenti dalle loro
    parti (parent_index dà l'ordine, source identifica il documento).

    Lettura 2 del filtro max_parts: analizza SOLO i documenti composti da al
    massimo `max_parts` parti, ricomposti PER INTERO; i documenti più lunghi
    (progetti, allegati tecnici da migliaia di parti) sono esclusi, perché il
    boilerplate cercato è quello degli ATTI, non di quei mega-allegati, e la
    loro ripetizione interna falserebbe la document-frequency.

    La document-frequency ha come denominatore il numero di DOCUMENTI ammessi
    (source distinti con ≤ max_parts parti), non il numero di parti — che era
    il bug per cui l'intestazione, presente solo nella prima parte, non
    raggiungeva mai la soglia.
    """
    # 1. Documenti ammessi: source con al massimo max_parts parti.
    with conn.cursor() as cur:
        cur.execute(
            "SELECT source FROM parent_documents "
            "WHERE topic_id = %s AND sub_topic_id = %s "
            "GROUP BY source HAVING COUNT(*) <= %s",
            (topic_id, sub_topic_id, max_parts),
        )
        admitted = {row[0] for row in cur}

    # Conteggio totale documenti del gruppo (per il log di esclusione).
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(DISTINCT source) FROM parent_documents "
            "WHERE topic_id = %s AND sub_topic_id = %s",
            (topic_id, sub_topic_id),
        )
        total_docs = cur.fetchone()[0]

    n_docs = len(admitted)
    excluded = total_docs - n_docs
    log.debug("Gruppo (%s, %s): %d documenti ammessi (<= %d parti), %d esclusi (piu' lunghi).",
              topic_id, sub_topic_id, n_docs, max_parts, excluded)
    if n_docs == 0:
        return [], n_docs, excluded

    # 2. Ricomposizione incrementale: una query ordinata per (source, index),
    #    si accumulano le parti finché il source non cambia, poi si analizza il
    #    documento intero e si libera. In RAM: un documento alla volta (<= N parti).
    counts = defaultdict(int)
    docs_processed = 0
    current_source = None
    parts = []

    def _flush(src, part_list):
        nonlocal docs_processed
        if src is None or src not in admitted:
            return
        text = normalize_ws(" ".join(part_list))
        if not text:
            return
        for g in ngrams_of(text, nmin, nmax):
            counts[g] += 1
        docs_processed += 1
        if docs_processed % 500 == 0:
            log.debug("  ... ricomposti %d/%d documenti, ngrammi unici: %d",
                      docs_processed, n_docs, len(counts))

    with conn.cursor() as cur:
        cur.execute(
            "SELECT source, content FROM parent_documents "
            "WHERE topic_id = %s AND sub_topic_id = %s "
            "ORDER BY source, parent_index",
            (topic_id, sub_topic_id),
        )
        for source, content in cur:
            if source != current_source:
                _flush(current_source, parts)   # chiudi il documento precedente
                current_source = source
                parts = []
            # Accumula solo se il documento è ammesso (evita di tenere in RAM
            # le 16.915 parti di un mega-documento escluso).
            if source in admitted:
                parts.append(content or "")
        _flush(current_source, parts)           # ultimo documento

    above = [(g, c) for g, c in counts.items() if c / n_docs >= threshold]
    log.debug("Gruppo (%s, %s): %d ngrammi sopra soglia (da %d unici totali, denom=%d doc).",
              topic_id, sub_topic_id, len(above), len(counts), n_docs)

    del counts
    gc.collect()

    if not above:
        return [], n_docs, excluded

    maximal_phrases = keep_maximal(above)
    log.debug("Gruppo (%s, %s): keep_maximal da %d a %d frasi.",
              topic_id, sub_topic_id, len(above), len(maximal_phrases))

    rows = []
    for phrase, count in maximal_phrases:
        rows.append({
            "topic_id": topic_id,
            "sub_topic_id": sub_topic_id,
            "phrase": phrase,
            "phrase_hash": phrase_hash(phrase),
            "doc_count": count,
            "group_docs": n_docs,
            "doc_freq": round(count / n_docs, 4),
        })
    return rows, n_docs, excluded


def keep_maximal(candidates):
    by_len = sorted(candidates, key=lambda c: len(c[0]), reverse=True)
    maximal = []
    for ph, cnt in by_len:
        if not any(ph != m and ph in m for m, _ in maximal):
            maximal.append((ph, cnt))

    def toks(s):
        return s.split()

    changed = True
    phrases = {p: c for p, c in maximal}
    merge_cycles = 0
    
    while changed:
        changed = False
        merge_cycles += 1
        items = list(phrases.items())
        for i in range(len(items)):
            a, ca = items[i]
            if a not in phrases:
                continue
            ta = toks(a)
            for j in range(len(items)):
                if i == j:
                    continue
                b, cb = items[j]
                if b not in phrases or a not in phrases:
                    continue
                tb = toks(b)
                merged = None
                maxk = min(len(ta), len(tb)) - 1
                for k in range(maxk, 0, -1):
                    if ta[-k:] == tb[:k]:
                        merged = " ".join(ta + tb[k:])
                        break
                if merged:
                    del phrases[a]
                    phrases.pop(b, None)
                    phrases[merged] = min(ca, cb)
                    changed = True
                    break
            if changed:
                break
                
    log.debug("  ... keep_maximal ha eseguito %d cicli di fusione.", merge_cycles)
    return list(phrases.items())


def load_manual_phrases(path):
    """Legge un file di frasi di boilerplate GIÀ VERIFICATE da inserire.

    Formato: una tripletta per riga, separata da ';':

        topic_id;sub_topic_id;frase

    La frase può contenere ';' (lo split è con maxsplit=2). Le righe vuote e
    quelle che iniziano con '#' sono ignorate. Le righe malformate (meno di 3
    colonne o con un campo vuoto) sono loggate e saltate, non fanno fallire il
    caricamento.

    Ritorna una lista di row dict nello stesso formato prodotto da
    analyze_group, così da poter riusare upsert_mysql e push_qdrant_batch. I
    conteggi sono valori sentinella (vedi MANUAL_*): queste frasi nascono da
    curatela, non da statistica.
    """
    rows = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split(";", 2)
            if len(parts) < 3:
                log.warning(
                    "Riga %d ignorata (attese 3 colonne 'topic_id;sub_topic_id;frase'): %r",
                    lineno, stripped,
                )
                continue

            topic_id, sub_topic_id, phrase = (p.strip() for p in parts)
            phrase = normalize_ws(phrase)
            if not topic_id or not sub_topic_id or not phrase:
                log.warning("Riga %d ignorata (campo vuoto): %r", lineno, stripped)
                continue

            h = phrase_hash(phrase)
            key = (topic_id, sub_topic_id, h)
            if key in seen:
                log.debug("Riga %d duplicata nel file, la salto: %r", lineno, phrase)
                continue
            seen.add(key)

            rows.append({
                "topic_id": topic_id,
                "sub_topic_id": sub_topic_id,
                "phrase": phrase,
                "phrase_hash": h,
                "doc_count": MANUAL_DOC_COUNT,
                "group_docs": MANUAL_GROUP_DOCS,
                "doc_freq": MANUAL_DOC_FREQ,
            })

    log.info("File '%s': %d frasi manuali valide caricate.", path, len(rows))
    return rows


def upsert_mysql(conn, rows, now, active=None):
    """UPSERT delle frasi su MySQL.

    active:
      - None  -> la colonna `active` NON viene toccata. È il comportamento
                 storico del rilevamento automatico: le frasi candidate restano
                 soggette alla revisione umana che decide quando attivarle.
      - True/False -> imposta esplicitamente `active` sia in INSERT sia in
                 UPDATE. Usato dall'import manuale per inserire frasi GIÀ
                 verificate (active=TRUE) senza passare dalla revisione.
    """
    if not rows:
        return 0

    active_col = active_val = active_update = ""
    if active is not None:
        lit = "TRUE" if active else "FALSE"          # bool interno: iniezione sicura
        active_col = ", active"
        active_val = f", {lit}"
        active_update = f",\n                active     = {lit}"

    sql = f"""
        INSERT INTO boilerplate_phrases
            (topic_id, sub_topic_id, phrase, phrase_hash, doc_freq,
             doc_count, group_docs, detected_at, last_seen{active_col})
        VALUES
            (%(topic_id)s, %(sub_topic_id)s, %(phrase)s, %(phrase_hash)s,
             %(doc_freq)s, %(doc_count)s, %(group_docs)s, %(now)s, %(now)s{active_val})
        ON DUPLICATE KEY UPDATE
            phrase     = VALUES(phrase),
            doc_freq   = VALUES(doc_freq),
            doc_count  = VALUES(doc_count),
            group_docs = VALUES(group_docs),
            last_seen  = VALUES(last_seen){active_update}
    """

    log.debug("Avvio UPSERT su MySQL per %d righe (active=%s).", len(rows), active)
    with conn.cursor() as cur:
        cur.executemany(sql, [{**r, "now": now} for r in rows])
    conn.commit()
    log.debug("UPSERT su MySQL completato.")
    return len(rows)


def _create_boilerplate_collection(qc):
    from qdrant_client import models
    from common.embedding import get_embedding_provider

    # Dimensione letta dal provider attivo, non più 768 hardcoded: chiamata
    # solo da init_qdrant_collection/ensure_qdrant_collection, entrambe
    # invocate in main() DOPO init_embedding() (vedi riga ~525), quindi
    # get_embedding_provider() qui trova sempre l'istanza già pronta. Con
    # 768 fisso, il rilevamento notturno con EMBEDDING_PROVIDER=local avrebbe
    # ricreato la collection a dimensione sbagliata a ogni run, cancellando
    # in silenzio l'eventuale migrazione fatta a mano su questa collection.
    dimension = get_embedding_provider().dimension
    qc.create_collection(
        collection_name=BOILERPLATE_COLLECTION,
        vectors_config=models.VectorParams(
            size=dimension, distance=models.Distance.COSINE
        ),
    )

    for field in ("topic_id", "sub_topic_id", "phrase_hash"):
        try:
            qc.create_payload_index(
                collection_name=BOILERPLATE_COLLECTION,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as e:
            log.warning("Indice payload '%s' non creato (%s).", field, e)


def init_qdrant_collection(qc):
    """DISTRUTTIVO: ricrea la collection da zero. Usato dal rilevamento
    completo, che ripopola interamente Qdrant a ogni run."""
    log.debug("Reinizializzazione (distruttiva) collection Qdrant '%s'.", BOILERPLATE_COLLECTION)

    # Nuovo pattern raccomandato al posto di recreate_collection
    if qc.collection_exists(collection_name=BOILERPLATE_COLLECTION):
        qc.delete_collection(collection_name=BOILERPLATE_COLLECTION)

    _create_boilerplate_collection(qc)


def ensure_qdrant_collection(qc):
    """NON distruttivo: crea la collection solo se manca. Usato dall'import-only,
    che deve aggiungere frasi senza cancellare i vettori già presenti."""
    if qc.collection_exists(collection_name=BOILERPLATE_COLLECTION):
        log.debug("Collection Qdrant '%s' già presente: la mantengo.", BOILERPLATE_COLLECTION)
        return
    log.debug("Collection Qdrant '%s' assente: la creo.", BOILERPLATE_COLLECTION)
    _create_boilerplate_collection(qc)

def push_qdrant_batch(qc, rows, embed_fn):
    if not rows:
        return
    from qdrant_client import models

    log.debug("Generazione vettori per %d frasi da inviare a Qdrant.", len(rows))
    points = []
    for r in rows:
        try:
            vec = embed_fn(r["phrase"])
        except Exception as e:
            log.warning("Embedding fallito per frase (%s): la salto.", e)
            continue

        point_id = str(uuid.UUID(r["phrase_hash"]))
        points.append(
            models.PointStruct(
                id=point_id,
                vector=vec,
                payload={
                    "topic_id": r["topic_id"],
                    "sub_topic_id": r["sub_topic_id"],
                    "phrase": r["phrase"],
                    "phrase_hash": r["phrase_hash"],
                    "doc_freq": r["doc_freq"],
                },
            )
        )
    if points:
        log.debug("Upsert batch di %d vettori su Qdrant.", len(points))
        qc.upsert(collection_name=BOILERPLATE_COLLECTION, points=points)


def main():
    ap = argparse.ArgumentParser(
        description="Rilevamento periodico del boilerplate."
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_DF_THRESHOLD,
        help="Document-frequency minima per considerare boilerplate una frase.",
    )
    ap.add_argument("--ngram-min", type=int, default=NGRAM_MIN)
    ap.add_argument("--ngram-max", type=int, default=NGRAM_MAX) 
    ap.add_argument("--min-group-docs", type=int, default=MIN_GROUP_DOCS)
    ap.add_argument(
        "--max-parts", type=int, default=10,
        help="Analizza solo i documenti composti da al massimo N parti "
             "(source con <= N parent_index). I documenti più lunghi "
             "(progetti, allegati tecnici) sono esclusi. Alzare se il "
             "boilerplate individuato è insufficiente.",
    )
    ap.add_argument(
        "--topic",
        default=None,
        help="Analizza solo questo topic_id (utile per test).",
    )
    ap.add_argument(
        "--no-vectors",
        action="store_true",
        help="Salta la sync dei vettori Qdrant (solo MySQL).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Analizza e stampa, senza scrivere nulla.",
    )
    ap.add_argument(
        "--import-file",
        default=None,
        help="File di frasi di boilerplate GIÀ VERIFICATE da inserire "
             "(active=TRUE). Una tripletta per riga: 'topic_id;sub_topic_id;frase' "
             "(la frase può contenere ';'). Righe vuote o che iniziano con '#' "
             "sono ignorate. Può essere combinato col rilevamento automatico.",
    )
    ap.add_argument(
        "--import-only",
        action="store_true",
        help="Importa SOLO le frasi di --import-file, senza eseguire il "
             "rilevamento automatico. Non ricrea la collection Qdrant: i "
             "vettori già presenti restano intatti.",
    )
    # Aggiunto flag per forzare il DEBUG level da linea di comando se necessario
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Attiva il logging a livello DEBUG.",
    )
    args = ap.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)
        log.debug("Modalità DEBUG attivata.")

    if args.import_only and not args.import_file:
        ap.error("--import-only richiede --import-file.")

    # Carica subito le frasi manuali (fail-fast su path/parsing errato).
    manual_rows = []
    if args.import_file:
        try:
            manual_rows = load_manual_phrases(args.import_file)
        except OSError as e:
            log.error("Impossibile leggere --import-file '%s': %s", args.import_file, e)
            return
        if not manual_rows:
            log.warning("Nessuna frase valida in '%s'.", args.import_file)
            if args.import_only:
                return

    init_db_pool()
    conn = get_db_connection()
    if conn is None:
        log.error("Connessione MySQL non disponibile: impossibile procedere.")
        return

    qc = None
    embed_fn = None

    if not args.no_vectors and not args.dry_run:
        try:
            log.debug("Caricamento QdrantClient e modello di embedding...")
            from qdrant_client import QdrantClient
            from common.embedding import init_embedding, embed_for_semantic_query

            qc = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                timeout=60,
            )
            # Lo script è standalone: il client di embedding va inizializzato
            # esplicitamente (nel worker Celery lo fa init_worker_process).
            init_embedding()
            embed_fn = embed_for_semantic_query
            if args.import_only:
                # Solo import: NON svuotare Qdrant, aggiungere ai vettori esistenti.
                ensure_qdrant_collection(qc)
            else:
                init_qdrant_collection(qc)
        except Exception as e:
            log.warning(
                "Inizializzazione Qdrant fallita (%s). Continuo solo su MySQL.",
                e,
            )
            qc = None

    try:
        now = datetime.utcnow()

        # ------------------------------------------------------------------
        # 1. Rilevamento automatico (saltato in --import-only)
        # ------------------------------------------------------------------
        if not args.import_only:
            groups = list_groups(conn)
            if args.topic:
                groups = [g for g in groups if g[0] == args.topic]
            log.info("Gruppi (topic, sub_topic) da analizzare: %d.", len(groups))

            total_phrases = 0
            skipped = 0

            for idx, (topic_id, sub_topic_id) in enumerate(groups, 1):
                log.debug("--- Avvio processing gruppo %d/%d ---", idx, len(groups))

                rows, n_docs, excluded = analyze_group(
                    conn,
                    topic_id,
                    sub_topic_id,
                    args.ngram_min,
                    args.ngram_max,
                    args.threshold,
                    args.max_parts,
                )
                # Il minimo si applica ai documenti AMMESSI (dopo il filtro parti):
                # un gruppo con pochi documenti brevi non dà statistica affidabile.
                if n_docs < args.min_group_docs:
                    log.debug("Gruppo saltato: %d documenti ammessi < %d (minimo).",
                              n_docs, args.min_group_docs)
                    skipped += 1
                    continue
                if not rows:
                    log.debug("Nessuna frase sopra soglia per questo gruppo "
                              "(%d documenti ammessi, %d esclusi).", n_docs, excluded)
                    continue

                total_phrases += len(rows)

                if args.dry_run:
                    for r in sorted(rows, key=lambda x: -x["doc_freq"]):
                        log.info(
                            "  [%s / %s] df=%.2f (%d/%d)  «%s»",
                            r["topic_id"],
                            r["sub_topic_id"],
                            r["doc_freq"],
                            r["doc_count"],
                            r["group_docs"],
                            r["phrase"],
                        )
                else:
                    upsert_mysql(conn, rows, now)
                    if qc and embed_fn:
                        push_qdrant_batch(qc, rows, embed_fn)

                gc.collect()
                log.debug("--- Fine processing gruppo %d/%d ---", idx, len(groups))

            log.info(
                "Frasi candidate sopra soglia %.2f: %d (gruppi saltati perché troppo piccoli: %d).",
                args.threshold,
                total_phrases,
                skipped,
            )

        # ------------------------------------------------------------------
        # 2. Import manuale di frasi GIÀ VERIFICATE (active=TRUE)
        # ------------------------------------------------------------------
        if manual_rows:
            log.info("Import manuale: %d frasi già verificate (active=TRUE).",
                     len(manual_rows))
            if args.dry_run:
                for r in manual_rows:
                    log.info("  [MANUAL] [%s / %s] «%s»",
                             r["topic_id"], r["sub_topic_id"], r["phrase"])
            else:
                upsert_mysql(conn, manual_rows, now, active=True)
                if qc and embed_fn:
                    push_qdrant_batch(qc, manual_rows, embed_fn)
                elif not args.no_vectors:
                    log.warning("Qdrant non disponibile: le frasi manuali sono "
                                "state inserite solo in MySQL (attive, ma senza "
                                "vettore finché non gira una sync Qdrant).")

    finally:
        conn.close()

    log.info("Completato.")

if __name__ == "__main__":
    main()