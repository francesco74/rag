"""
migrate_embeddings.py — Re-embedding di TUTTE le collection Qdrant legate a
EMBEDDING_PROVIDER, ripartendo dal testo già salvato nel payload dei punti.
Nessun bisogno di rifare OCR/conversion né richiamare converter.py.

Il worker (worker.py) tocca QUATTRO collection con vettori nello stesso
spazio embedding, non solo una:

    document_chunks         i chunk indicizzati (retrieval principale)
    conceptual_dictionary   dizionario concetti/alias (get_concepts_by_similarity)
    boilerplate_phrases     frasi ricorrenti da filtrare (load_active_boilerplate)
    semantic_cache          cache delle risposte già date

Le prime tre hanno contenuto "sorgente" (documenti/concetti/frasi) e vanno
RI-EMBEDDATE. semantic_cache è una cache derivata, non va migrata: si
ricrea VUOTA con la nuova dimensione, altrimenti il primo save_to_semantic_cache
dopo lo switch di provider fallisce con un dimension mismatch su una
collection che ha ancora vettori Gemini.

ATTENZIONE sui campi testo per conceptual_dictionary e boilerplate_phrases:
a differenza di document_chunks (payload["content"], confermato in ingest.py),
NON ho visto lo script che popola queste due collection. I campi qui sotto
(MIGRATIONS) sono la mia migliore ipotesi guardando come worker.py LEGGE quei
payload (get_concepts_by_similarity, load_active_boilerplate) — verifica
prima di lanciare in produzione: lo script si ferma da solo con un errore
esplicito se il campo atteso manca dal payload, non embedda mai stringhe vuote
in silenzio.

Uso:
    EMBEDDING_PROVIDER=local python migrate_embeddings.py
    EMBEDDING_PROVIDER=local python migrate_embeddings.py --collections document_chunks
    EMBEDDING_PROVIDER=local python migrate_embeddings.py --suffix _bge --reset-cache

Dopo aver verificato le nuove collection (query di prova, confronto risultati),
il modo più semplice per non toccare le costanti QDRANT_COLLECTION/
CACHE_COLLECTION/CONCEPT_COLLECTION/BOILERPLATE_COLLECTION in worker.py è:
droppare le vecchie collection e rinominare le nuove (rename_collection di
Qdrant) allo stesso nome originale. Fino a quel momento le vecchie restano
intatte come rollback immediato.
"""
import argparse
import asyncio
import logging

from qdrant_client import AsyncQdrantClient, models

from common.config import settings
from common.embedding import init_embedding, embed_documents_batch, get_embedding_provider

logging.basicConfig(level=logging.INFO, format="%(asctime)s - MIGRATE - %(levelname)s - %(message)s")
log = logging.getLogger("MigrateEmbeddings")

SCROLL_BATCH = 200       # punti letti per ogni giro di scroll da Qdrant
EMBED_SUBBATCH = 100     # testi per ogni chiamata a embed_documents_batch


# ==============================================================================
# REGISTRY: una entry per ogni collection "sorgente" da ri-embeddare.
# `extract_text(payload) -> str | None`: None => punto scartato (loggato).
# ==============================================================================
def _extract_content(payload: dict) -> str | None:
    text = (payload.get("content") or "").strip()
    return text or None


def _extract_concept(payload: dict) -> str | None:
    # ASSUNZIONE (vedi docstring): il testo embeddato è "concept" + eventuali
    # "aliases", concatenati — coerente con come get_concepts_by_similarity
    # interroga per similarità semantica su entrambi. Se il tuo script di
    # popolamento embeddava SOLO il concept, togli la riga aliases sotto.
    concept = (payload.get("concept") or "").strip()
    aliases = payload.get("aliases") or []
    if not concept:
        return None
    if aliases:
        return f"{concept}: {', '.join(aliases)}"
    return concept


def _extract_phrase(payload: dict) -> str | None:
    # ASSUNZIONE (vedi docstring): presume che il payload contenga il testo
    # della frase sotto "phrase". worker.py, in lettura, recupera solo
    # "phrase_hash" da Qdrant (il testo vero vive in MySQL) — se il tuo
    # script di popolamento NON scrive anche "phrase" nel payload Qdrant,
    # questa collection va ri-embeddata leggendo da MySQL
    # (tabella boilerplate_phrases, colonna phrase) invece che da qui.
    text = (payload.get("phrase") or "").strip()
    return text or None


MIGRATIONS = {
    "document_chunks": _extract_content,
    "conceptual_dictionary": _extract_concept,
    "boilerplate_phrases": _extract_phrase,
}

# Collection derivate: NON si ri-embeddano, si ricreano vuote con la nuova
# dimensione (vedi motivazione nella docstring del modulo).
CACHE_COLLECTIONS = ["semantic_cache"]


async def ensure_collection(client: AsyncQdrantClient, name: str, dimension: int, recreate: bool = False):
    exists = await client.collection_exists(name)
    if exists and recreate:
        await client.delete_collection(name)
        exists = False
        log.info(f"Collection '{name}' esistente droppata (--reset-cache).")
    if exists:
        log.info(f"Collection '{name}' già esistente, la riuso (verifica tu che la dimensione combaci).")
        return
    await client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
    )
    log.info(f"Collection '{name}' creata (dim={dimension}).")


async def migrate_one(client: AsyncQdrantClient, source: str, target: str, extract_text, dimension: int):
    log.info(f"--- Migrazione '{source}' -> '{target}' ---")
    await ensure_collection(client, target, dimension)

    next_offset = None
    total_migrated = 0
    total_skipped = 0

    while True:
        points, next_offset = await client.scroll(
            collection_name=source,
            limit=SCROLL_BATCH,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,   # non ci serve il vettore vecchio, solo il payload/testo
        )
        if not points:
            break

        texts, valid_points = [], []
        for p in points:
            text = extract_text(p.payload or {})
            if text is None:
                total_skipped += 1
                continue
            texts.append(text)
            valid_points.append(p)

        new_vectors = []
        for i in range(0, len(texts), EMBED_SUBBATCH):
            sub = texts[i:i + EMBED_SUBBATCH]
            new_vectors.extend(await embed_documents_batch(sub))

        new_points = [
            models.PointStruct(id=p.id, vector=vec, payload=p.payload)
            for p, vec in zip(valid_points, new_vectors)
        ]
        if new_points:
            await client.upsert(collection_name=target, points=new_points)

        total_migrated += len(new_points)
        log.info(f"[{source}] Migrati {total_migrated} punti finora (scartati: {total_skipped})...")

        if next_offset is None:
            break

    log.info(f"✓ [{source}] Completato: {total_migrated} punti in '{target}' (scartati: {total_skipped}).")
    if total_skipped and total_migrated == 0:
        log.error(
            f"[{source}] TUTTI i punti scartati: il campo testo atteso non è nel payload. "
            f"Controlla l'assunzione in MIGRATIONS per questa collection prima di considerarla migrata."
        )


async def run(collections: list[str], suffix: str, reset_cache: bool):
    init_embedding()  # legge EMBEDDING_PROVIDER dall'env corrente
    provider = get_embedding_provider()
    log.info(f"Provider attivo per il re-embedding: {provider.model_name} (dim={provider.dimension})")

    client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=120.0)

    for source in collections:
        extract_text = MIGRATIONS[source]
        target = f"{source}{suffix}"
        await migrate_one(client, source, target, extract_text, provider.dimension)

    if reset_cache:
        for source in CACHE_COLLECTIONS:
            target = f"{source}{suffix}"
            log.info(f"--- Reset (vuoto) '{target}' ---")
            await ensure_collection(client, target, provider.dimension, recreate=True)
            log.info(f"✓ '{target}' pronta, vuota — la cache si ripopola da sola con l'uso.")

    await client.close()
    log.info("✓ Migrazione completata per tutte le collection richieste.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ri-embedda le collection Qdrant sorgente con un nuovo provider e resetta la cache."
    )
    parser.add_argument(
        "--collections",
        default="all",
        help=f"Comma-separated tra {list(MIGRATIONS.keys())}, oppure 'all' (default).",
    )
    parser.add_argument(
        "--suffix",
        default="_local",
        help="Suffisso applicato al nome sorgente per ottenere il nome target (default: _local).",
    )
    parser.add_argument(
        "--reset-cache",
        action="store_true",
        help="Ricrea anche semantic_cache (vuota, nuova dimensione) col suffisso indicato.",
    )
    args = parser.parse_args()

    if args.collections == "all":
        selected = list(MIGRATIONS.keys())
    else:
        selected = [c.strip() for c in args.collections.split(",") if c.strip()]
        unknown = [c for c in selected if c not in MIGRATIONS]
        if unknown:
            parser.error(f"Collection sconosciute: {unknown}. Valide: {list(MIGRATIONS.keys())}")

    asyncio.run(run(selected, args.suffix, args.reset_cache))