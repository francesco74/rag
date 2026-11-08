"""
migrate_embeddings.py — cambio del modello di embedding: ricalcolo dei vettori
di tutte le collection Qdrant a partire dal testo già salvato nel payload. Non
servono né la conversione né l'OCR. Procedura completa:
misc/migrations/MIGRAZIONE_EMBEDDING.md.

Collection coinvolte (tutte nello stesso spazio vettoriale):

    document_chunks         payload "content", lo stesso testo embeddato da ingest.py
    conceptual_dictionary   "concept: alias, alias" come in populate_concepts.py
    boilerplate_phrases     payload "phrase" come in detect_boilerplate.py
    semantic_cache          cache di risposte: non si ricalcola, si ricrea vuota

Comandi, da lanciare in ordine con la NUOVA configurazione di embedding
(EMBEDDING_PROVIDER, EMBEDDING_DIMENSION...) e i servizi fermi:

    python -m utils.migrate_embeddings stato
    python -m utils.migrate_embeddings ricalcola     # -> <nome>__nuovo, le originali non si toccano
    python -m utils.migrate_embeddings sostituisci   # originali -> <nome>__vecchio, __nuovo -> originali
    python -m utils.migrate_embeddings ripristina    # per tornare indietro: __vecchio -> originali
    python -m utils.migrate_embeddings pulisci --conferma   # a migrazione confermata: elimina le copie

Qdrant non sa rinominare una collection: "sostituisci" e "ripristina" copiano i
punti (vettori e payload, stessi id) senza ricalcolare nulla e ricreano gli
indici del payload uguali a quelli della collection di partenza. document_chunks
nasce con i parametri di utils/setup_qdrant.collection_params (vettori su disco e
copia int8 in RAM, se QDRANT_CHUNKS_QUANTIZATION è attiva).

Ogni comando si può rilanciare: "ricalcola" riparte da capo sulle collection
__nuovo, "sostituisci" salta quelle già sostituite, e nessuna collection è
cancellata prima che la sua copia sia completa e verificata.
"""
import argparse
import asyncio
import logging
import sys

from qdrant_client import QdrantClient, models

from common.config import settings
from common.embedding import embed_documents_batch, embed_for_semantic_query, get_embedding_provider, init_embedding
from utils.setup_qdrant import collection_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s - MIGRATE - %(levelname)s - %(message)s")
log = logging.getLogger("MigrateEmbeddings")

SCROLL_BATCH = 256       # punti letti per ogni giro di scroll
EMBED_SUBBATCH = 64      # testi per ogni chiamata di embedding
LOG_EVERY = 5000         # punti tra un messaggio di avanzamento e l'altro
NEW, OLD = "__nuovo", "__vecchio"
CACHE = "semantic_cache"


def _content(payload):
    return (payload.get("content") or "").strip() or None


def _concept(payload):
    concept = (payload.get("concept") or "").strip()
    aliases = payload.get("aliases") or []
    if not concept:
        return None
    return f"{concept}: {', '.join(aliases)}" if aliases else concept


def _phrase(payload):
    return (payload.get("phrase") or "").strip() or None


# Collection da ricalcolare: testo dal payload e tipo di embedding usato da chi la
# popola: "documenti" per i chunk (ingest.py), "frasi" per concetti e boilerplate
# (embed_for_semantic_query). Con Gemini sono proiezioni diverse; con i modelli
# simmetrici coincidono, salvo l'istruzione delle domande, che qui non entra mai.
SOURCES = {
    "document_chunks": (_content, "documenti"),
    "conceptual_dictionary": (_concept, "frasi"),
    "boilerplate_phrases": (_phrase, "frasi"),
}


def _exists(client, name):
    return client.collection_exists(name)


def _count(client, name):
    return client.count(name, exact=True).count


def _copy_indexes(client, schema, target):
    for field, index in (schema or {}).items():
        client.create_payload_index(target, field_name=field, field_schema=index.params or index.data_type, wait=True)


def _create_like(client, source, target, dimension=None):
    """Crea target con gli indici del payload di source e i suoi parametri, oppure,
    con dimension, i parametri di collection_params per la nuova dimensione."""
    info = client.get_collection(source)
    if dimension is None:
        vectors = info.config.params.vectors
        if not isinstance(vectors, models.VectorParams):
            sys.exit(f"'{source}' ha vettori con nome: non gestiti da questo script.")
        params = {"vectors_config": vectors, "quantization_config": info.config.quantization_config}
    else:
        params = collection_params(target.removesuffix(NEW), dimension)
    client.create_collection(collection_name=target, **params)
    _copy_indexes(client, info.payload_schema, target)


def _replace(client, source, target):
    """target = copia esatta di source (parametri, indici, punti), verificata."""
    if _exists(client, target):
        client.delete_collection(target)
    _create_like(client, source, target)
    offset = None
    while True:
        points, offset = client.scroll(source, limit=SCROLL_BATCH, offset=offset, with_payload=True, with_vectors=True)
        if points:
            client.upsert(target, wait=True,
                          points=[models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points])
        if offset is None:
            break
    expected, copied = _count(client, source), _count(client, target)
    if copied != expected:
        sys.exit(f"Copia '{source}' -> '{target}' incompleta: {copied} punti su {expected}. "
                 "Nulla è stato cancellato: rilanciare il comando.")
    log.info(f"'{source}' -> '{target}': {copied} punti copiati.")


async def _embed(texts, kind):
    if kind == "documenti":
        return await embed_documents_batch(texts)
    return [await asyncio.to_thread(embed_for_semantic_query, t) for t in texts]


async def _recompute(client, name, dimension):
    extract, kind = SOURCES[name]
    target = name + NEW
    if _exists(client, target):
        client.delete_collection(target)
    _create_like(client, name, target, dimension=dimension)
    offset, done, skipped, next_log = None, 0, 0, LOG_EVERY
    while True:
        points, offset = client.scroll(name, limit=SCROLL_BATCH, offset=offset, with_payload=True, with_vectors=False)
        texts = [(p, extract(p.payload or {})) for p in points]
        skipped += sum(1 for _, text in texts if text is None)
        texts = [(p, text) for p, text in texts if text is not None]
        for i in range(0, len(texts), EMBED_SUBBATCH):
            part = texts[i:i + EMBED_SUBBATCH]
            vectors = await _embed([text for _, text in part], kind)
            client.upsert(target, wait=True, points=[models.PointStruct(id=p.id, vector=v, payload=p.payload)
                                                     for (p, _), v in zip(part, vectors)])
            done += len(part)
        if done >= next_log:
            log.info(f"[{name}] {done} punti ricalcolati...")
            next_log = done + LOG_EVERY
        if offset is None:
            break
    total = _count(client, name)
    log.info(f"✓ '{name}' -> '{target}': {done} punti ricalcolati su {total}"
             f"{f', {skipped} senza testo (non copiati)' if skipped else ''}.")
    if total and not done:
        sys.exit(f"'{name}': nessun punto con testo nel payload, controllare il campo atteso.")


def _reset_cache(client, dimension):
    """La cache contiene vettori di domande del modello precedente: si ricrea vuota."""
    schema = client.get_collection(CACHE).payload_schema if _exists(client, CACHE) else {}
    if _exists(client, CACHE):
        client.delete_collection(CACHE)
    client.create_collection(collection_name=CACHE, **collection_params(CACHE, dimension))
    _copy_indexes(client, schema, CACHE)
    log.info(f"'{CACHE}' ricreata vuota (dimensione {dimension}).")


def cmd_stato(client, _args):
    for name in [*SOURCES, CACHE]:
        for full in (name, name + NEW, name + OLD):
            if not _exists(client, full):
                continue
            info = client.get_collection(full)
            vectors = info.config.params.vectors
            quant = "int8 in RAM" if info.config.quantization_config else "nessuna"
            print(f"{full:34} punti {_count(client, full):>9}   dimensione {vectors.size:>5}   "
                  f"vettori su disco {'sì' if vectors.on_disk else 'no'}   quantizzazione {quant}")


async def cmd_ricalcola(client, args):
    init_embedding()
    provider = get_embedding_provider()
    probe = await _embed(["prova di dimensione"], "documenti")   # il fornitore vllm verifica la lunghezza
    log.info(f"Modello {provider.model_name}: vettori di {len(probe[0])} dimensioni.")
    for name in args.collections:
        if _exists(client, name):
            await _recompute(client, name, provider.dimension)
        else:
            log.warning(f"'{name}' non esiste: saltata.")
    log.info("Ricalcolo completato. Prossimo passo: sostituisci.")


def cmd_sostituisci(client, args):
    for name in args.collections:
        if not _exists(client, name + NEW):
            log.info(f"'{name}': " + ("già sostituita." if _exists(client, name + OLD)
                                      else f"'{name + NEW}' non esiste, saltata (lanciare prima ricalcola)."))
            continue
        if not _exists(client, name + OLD):
            _replace(client, name, name + OLD)   # copia per tornare indietro, prima di toccare l'originale
        client.delete_collection(name)
        _replace(client, name + NEW, name)
        client.delete_collection(name + NEW)
        log.info(f"✓ '{name}' sostituita (la precedente è in '{name + OLD}').")
    _reset_cache(client, settings.embedding_dimension)
    log.info("Sostituzione completata. Dopo le verifiche: pulisci --conferma.")


def cmd_ripristina(client, args):
    dimension = None
    for name in args.collections:
        if not _exists(client, name + OLD):
            log.warning(f"'{name + OLD}' non esiste: '{name}' resta com'è.")
            continue
        if _exists(client, name):
            client.delete_collection(name)
        _replace(client, name + OLD, name)
        dimension = client.get_collection(name).config.params.vectors.size
        log.info(f"✓ '{name}' ripristinata da '{name + OLD}'.")
    if dimension:
        _reset_cache(client, dimension)
    log.info("Ripristino completato: rimettere la configurazione di embedding precedente.")


def cmd_pulisci(client, args):
    if not args.conferma:
        sys.exit("Elimina le copie __vecchio e le __nuovo rimaste: aggiungere --conferma.")
    for name in args.collections:
        for full in (name + OLD, name + NEW):
            if _exists(client, full):
                client.delete_collection(full)
                log.info(f"Eliminata '{full}'.")


def main():
    parser = argparse.ArgumentParser(description="Cambio del modello di embedding: ricalcolo delle collection Qdrant.")
    sub = parser.add_subparsers(dest="comando", required=True)
    for name in ("stato", "ricalcola", "sostituisci", "ripristina", "pulisci"):
        p = sub.add_parser(name)
        p.add_argument("--collections", default=",".join(SOURCES),
                       help=f"separate da virgola, tra {list(SOURCES)} (default: tutte)")
        if name == "pulisci":
            p.add_argument("--conferma", action="store_true")
    args = parser.parse_args()
    args.collections = [c.strip() for c in args.collections.split(",") if c.strip()]
    unknown = [c for c in args.collections if c not in SOURCES]
    if unknown:
        parser.error(f"Collection sconosciute: {unknown}. Valide: {list(SOURCES)}")

    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=300)
    if args.comando == "ricalcola":
        asyncio.run(cmd_ricalcola(client, args))
    else:
        {"stato": cmd_stato, "sostituisci": cmd_sostituisci, "ripristina": cmd_ripristina,
         "pulisci": cmd_pulisci}[args.comando](client, args)


if __name__ == "__main__":
    main()
