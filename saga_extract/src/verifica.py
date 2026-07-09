import json
import os
import sys
import re
import logging
import argparse
import asyncio
import pathlib
from pathlib import Path
from typing import Optional, List

from common.config import settings
from common.db_logger import init_db_pool, get_db_connection
from common.risultati_ricerca_parser import run_search
from common.ricerca_filtri import RicercaFiltri, RicercaSemplice
from common.estrazione_documenti import build_client_from_env

from qdrant_client import AsyncQdrantClient
from qdrant_client import models

from common.utility import clean_iso_date

QDRANT_COLLECTION = "document_chunks"
ESTENSIONI_CONSENTITE = {".pdf", ".p7m"}

def build_fresh_metadata(dati: dict) -> dict:
    """
    Ricostruisce, a partire dai dati freschi restituiti da leggi_atto_plus(),
    SOLO le chiavi "di contenuto" del blocco metadati (non id_sicraweb né
    percorso_originale, che non cambiano nel tempo e non vengono toccate
    dall'update "solo metadati"). Usata sia per l'update MySQL (via
    JSON_SET, che tocca solo queste chiavi) sia per l'update Qdrant (via
    set_payload, che fa merge parziale by design).
    """
    return {
        "oggetto": dati.get("oggetto"),
        "classifica": dati.get("classifica"),
        "classifica_descrizione": dati.get("classifica_descrizione"),
        "trattamento_descrizione": dati.get("trattamento_descrizione"),
        "proponente_descrizione": dati.get("proponente_descrizione"),
        "dirigente_descrizione": dati.get("dirigente_descrizione"),
        "data": clean_iso_date(dati.get("data_atto")),
        "anno": dati.get("anno_atto"),
        "numero": dati.get("numero_atto"),
        "data_esecutivita": clean_iso_date(dati.get("data_esecutivita")),
        "data_pubblicazione": clean_iso_date(dati.get("data_pubblicazione")),
        "giorni_pubblicazione": dati.get("giorni_pubblicazione"),
        "id_tipo_iter": dati.get("id_tipo_iter"),
    }

log_level_str = settings.log_level
log_level = getattr(logging, log_level_str, logging.INFO)

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - VERIFICA - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
)
log = logging.getLogger("Verifier")

def parse_filters(json_filters_str: str) -> tuple[List[RicercaFiltri], str, Optional[str]]:
    """
    Restituisce (lista_filtri, tipo_atto, anno_atto_richiesto).
    Con decreto/delibera + range di più giorni, la lista contiene un filtro per
    ciascun giorno (vedi RicercaSemplice.to_filtri_list), dato che
    DeliberaFilter/DecretoFilter non hanno un campo range confermato nel WSDL.

    anno_atto_richiesto viene restituito a parte perché per il DECRETO il WSDL
    non ha nessun campo "Anno" nella richiesta di ricerca (vedi <Documento> nel
    template WSDL: solo Numero/Data/Tipo/Oggetto) — quindi l'anno, se
    richiesto, va applicato come FILTRO LATO CLIENT sui risultati già estratti
    (vedi verify_pipeline), confrontandolo con l'anno_atto restituito da
    LeggiAttoPlus per ciascun candidato.

    NOTA: il sotto-tipo di decreto/delibera (deliberativo vs presidenziale)
    NON viene più determinato qui né tramite il registro Verbale della
    risposta di ricerca (RicercaDocumentiString): quel registro può mancare
    del tutto per un atto reale (CONFERMATO: decreto presidenziale 1/2024,
    UID 2119188). Il discriminante ora è id_tipo_iter, ottenuto da
    LeggiAttoPlus per ogni candidato in verify_pipeline (vedi
    ID_TIPO_ITER_ATTESO), quindi semplice.registro_definitivo_atteso() non
    viene più chiamato/usato qui.

    Formato JSON atteso:
       {"tipo_atto": "determina", "numero_atto": "4", "anno_atto": "2025"}
       {"tipo_atto": "decreto_deliberativo", "oggetto": "interreg",
        "data_da": "01/06/2026", "data_a": "30/06/2026"}
       {"tipo_atto": "decreto_presidenziale", "numero_atto": "1", "anno_atto": "2026"}
       {"tipo_atto": "qualsiasi", "oggetto": "mozione nicodemo"}
    """
    raw_json = json.loads(json_filters_str)
    semplice = RicercaSemplice(**raw_json)
    return semplice.to_filtri_list(), semplice.tipo_atto, semplice.anno_atto

def get_expected_files_from_lista_allegati(uid: str, lista_allegati_raw: list) -> list:
    """
    Stessa logica di filtro/dedup di prima, ma ora prende in input la lista
    allegati GIA' MATERIALIZZATA (proveniente da get_dati_atto_da_leggi_atto_plus),
    invece di richiamare leggi_atto_plus() una seconda volta per lo stesso UID.
    """
    nomi_file_presenti = {f_name.lower() for _, f_name in lista_allegati_raw}
    expected_files = []

    for _, file_name in lista_allegati_raw:
        nome_lower = file_name.lower()
        estensione = Path(nome_lower).suffix

        if estensione not in ESTENSIONI_CONSENTITE:
            continue

        if nome_lower.endswith('.p7m'):
            nome_atteso_decifrato = nome_lower[:-4]
            if nome_atteso_decifrato in nomi_file_presenti:
                continue
            file_name = file_name[:-4]

        safe_name = f"doc_{uid}_{file_name}"
        expected_files.append(safe_name)

    return expected_files

# La mappa dei valori attesi di id_tipo_iter ora vive in settings
# (settings.id_tipo_iter_atteso, common/config.py), configurabile via env
# ID_TIPO_ITER_DECRETO_PRESIDENZIALE/ID_TIPO_ITER_DECRETO_DELIBERATIVO,
# così verifica.py ed estrattore.py usano sempre la stessa mappa.



def get_dati_atto_da_leggi_atto_plus(repwss_client, uid: str) -> Optional[dict]:
    """
    Chiama leggi_atto_plus(uid) UNA SOLA VOLTA per UID e restituisce sia gli
    allegati (già materializzati) sia TUTTI gli attributi disponibili
    (numero_atto, anno_atto, id_tipo_iter, oggetto, classifica,
    classifica_descrizione, data_atto, data_esecutivita,
    data_pubblicazione, giorni_pubblicazione, trattamento_descrizione,
    proponente_descrizione, dirigente_descrizione), così lo stesso risultato
    viene riusato più avanti sia per la verifica sotto-tipo/anno/definitività
    sia per il calcolo dei file attesi sia per il recovery (che ora porta con
    sé anche i metadati completi, invece di scartarli), evitando di
    richiamare leggi_atto_plus due volte per lo stesso UID.

    numero_atto: "0" (con data_atto sentinella "0001-01-01...") indica una
    mera proposta non ancora protocollata; qualunque altro valore indica un
    atto realmente protocollato (CONFERMATO su UID 2119188: numero_atto="1").

    Le date NON sono normalizzate qui: sono i valori grezzi restituiti da
    Sicr@Web. La normalizzazione avviene in un unico punto centrale,
    elabora_atto() in estrattore.py, tramite clean_iso_date() — mantenuta
    identica a quella usata lì per garantire lo stesso comportamento sia
    che l'atto arrivi dalla modalità ricerca sia dal recovery.

    Restituisce None in caso di errore nella chiamata (loggato).
    """
    try:
        lista_allegati_raw, attributi_plus = repwss_client.leggi_atto_plus(uid)
        # FIX: leggi_atto_plus() restituisce un generatore (yield in estrai_allegati()).
        # Va materializzato in lista PRIMA di essere iterato più volte.
        lista_allegati_raw = list(lista_allegati_raw)
    except Exception as e:
        log.error(f"Errore durante leggi_atto_plus per UID {uid}: {e}")
        return None

    attributi_plus = attributi_plus or {}

    def _get(nome):
        if isinstance(attributi_plus, dict):
            return attributi_plus.get(nome)
        return getattr(attributi_plus, nome, None)

    def _get_str(nome):
        val = _get(nome)
        return str(val).strip() if val is not None else None

    return {
        "allegati": lista_allegati_raw,
        "numero_atto": _get_str("numero_atto"),
        "anno_atto": _get_str("anno_atto"),
        "id_tipo_iter": _get_str("id_tipo_iter"),
        "oggetto": _get("oggetto"),
        "classifica": _get("classifica"),
        "classifica_descrizione": _get("classifica_descrizione"),
        "data_atto": _get("data_atto"),
        "data_esecutivita": _get("data_esecutivita"),
        "data_pubblicazione": _get("data_pubblicazione"),
        "giorni_pubblicazione": _get("giorni_pubblicazione"),
        "trattamento_descrizione": _get("trattamento_descrizione"),
        "proponente_descrizione": _get("proponente_descrizione"),
        "dirigente_descrizione": _get("dirigente_descrizione"),
    }


def check_mysql_file_presence(source: str, file_name: str) -> int:
    conn = get_db_connection()
    if not conn:
        raise ConnectionError("Database MySQL non raggiungibile.")
    try:
        with conn.cursor() as cursor:
            log.debug(f"Controllo in mysql source:{source} - filename:{file_name}")
            cursor.execute(
                "SELECT COUNT(*) FROM parent_documents WHERE source = %s AND file_name = %s",
                (source, file_name)
            )
            res = cursor.fetchone()
            return res[0] if res else 0
    finally:
        conn.close()


def update_mysql_metadata(source: str, fresh_metadata: dict) -> int:
    """
    Aggiorna SOLO le chiavi presenti in fresh_metadata dentro la colonna JSON
    'metadata' su parent_documents (via JSON_SET: le chiavi non elencate,
    es. id_sicraweb/percorso_originale, restano invariate). NON tocca
    'content' né alcun embedding/vettore — nessuna re-ingestion.

    Restituisce il numero di righe MySQL effettivamente aggiornate.
    """
    conn = get_db_connection()
    if not conn:
        raise ConnectionError("Database MySQL non raggiungibile.")
    try:
        set_clauses = ", ".join(f"'$.{k}', %s" for k in fresh_metadata)
        values = list(fresh_metadata.values()) + [source]
        with conn.cursor() as cursor:
            cursor.execute(
                f"UPDATE parent_documents SET metadata = JSON_SET(metadata, {set_clauses}) WHERE source = %s",
                values
            )
            conn.commit()
            return cursor.rowcount
    finally:
        conn.close()


async def _qdrant_call_with_retry(coro_factory, description: str, retries: int = 3, base_delay: float = 3.0):
    """
    Esegue una chiamata Qdrant con retry ed exponential backoff su timeout/errori
    di rete transitori — comuni quando il server è sotto carico (scritture
    massive di metadati, come in --update-metadata-only).
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return await coro_factory()
        except Exception as e:
            last_exc = e
            if attempt < retries:
                wait = base_delay * (2 ** (attempt - 1))
                log.warning(f"{description}: tentativo {attempt}/{retries} fallito ({e}); ritento tra {wait:.0f}s...")
                await asyncio.sleep(wait)
            else:
                log.error(f"{description}: falliti tutti i {retries} tentativi: {e}")
    raise last_exc


async def update_qdrant_metadata_batch(qdrant_client: AsyncQdrantClient, sources: list, fresh_metadata: dict) -> None:
    """
    Aggiorna SOLO il payload dei punti Qdrant il cui 'source' è in `sources`
    (set_payload fa merge parziale: le chiavi non elencate in fresh_metadata,
    es. topic_id/sub_topic_id/content_hash, restano invariate). NON tocca
    vettori né content — nessuna re-ingestion, nessun nuovo embedding.

    Un'UNICA chiamata per tutti gli allegati dello stesso atto (via MatchAny),
    invece di una per allegato: riduce il numero di round-trip verso Qdrant,
    che sotto carico è la causa più probabile dei timeout osservati.
    """
    if not sources:
        return
    await _qdrant_call_with_retry(
        lambda: qdrant_client.set_payload(
            collection_name=QDRANT_COLLECTION,
            payload=fresh_metadata,
            points=models.Filter(
                must=[models.FieldCondition(key="source", match=models.MatchAny(any=sources))]
            ),
        ),
        description=f"set_payload batch ({len(sources)} source)",
    )

# Aggiorna queste due funzioni dentro verifica.py:

async def retrigger_extraction(uid: str, tipo_atto: str, meta_atto: dict):
    """
    Richiama estrattore.py in MODALITA' DIRETTA: passa l'UID e i metadati che
    verify_pipeline ha già ottenuto e verificato (dopo i filtri registro/anno),
    così estrattore.py salta del tutto una nuova ricerca RicercaDocumentiString
    su Sicr@Web. Questo elimina sia la chiamata di rete ridondante sia il
    rischio di "infiltrati" (altri atti con lo stesso numero/anno che una
    ri-ricerca indipendente per solo numero+anno potrebbe ripescare, dato che
    <Documento><Numero> da solo può restituire atti di anni/sotto-tipi diversi).
    """
    if not uid:
        print(f"    [!] SALTO RECOVERY: UID mancante.")
        return

    payload = {
        "uid": str(uid),
        "tipo_atto": tipo_atto,
        "metadati": meta_atto,
    }

    cmd = [sys.executable, "estrattore.py", "--json-filters", json.dumps(payload)]
    print(f"    [>] Esecuzione recovery: python3 estrattore.py (UID: {uid})...")
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    
    if process.returncode == 0:
        print(f"    [✓] Atto re-inviato a RabbitMQ con successo.")
    else:
        err_msg = stderr.decode().strip() or stdout.decode().strip()
        print(f"    [✗] Fallimento recovery. Output estrattore: {err_msg[:200]}")


        

async def verify_pipeline(json_filters_str: str, auto_recover: bool, update_metadata_only: bool = False, sicraweb_delay: float = None):
    init_db_pool()
    filtri_list, tipo_atto, anno_atto_richiesto = parse_filters(json_filters_str)

    log.info(f"Esecuzione di {len(filtri_list)} query di ricerca su Sicr@Web...")

    ids_totali: list[str] = []
    documenti_metadata: dict = {}
    ultimo_errore = None

    for i, filtri in enumerate(filtri_list, start=1):
        if len(filtri_list) > 1:
            log.debug(f"Ricerca {i}/{len(filtri_list)}...")
        risultato = run_search(filtri, dry_run=False)

        if risultato.errore:
            ultimo_errore = risultato.errore
            log.warning(f"Ricerca {i}/{len(filtri_list)} fallita: {risultato.errore}")
            continue

        for uid in risultato.ids:
            uid_str = str(uid)
            if uid_str not in documenti_metadata:
                ids_totali.append(uid_str)
            # Se lo stesso UID compare in più giorni (non dovrebbe capitare, ma
            # per sicurezza teniamo l'ultima versione dei metadata trovata)
            documenti_metadata[uid_str] = risultato.documenti_metadata.get(uid_str, {})

    if not ids_totali:
        log.warning(f"Nessun atto restituito dalla ricerca. Ultimo errore: {ultimo_errore}")
        return

    try:
        repwss_client = build_client_from_env()
        if sicraweb_delay is not None:
            repwss_client.min_interval_seconds = sicraweb_delay
    except Exception as e:
        log.error(f"Impossibile istanziare il client Sicr@Web: {e}")
        return

    id_tipo_iter_attesi = settings.id_tipo_iter_atteso.get(tipo_atto)  # None per tipo_atto senza sotto-tipo (determina/qualsiasi/...)

    uids_prima = list(ids_totali)
    ids_totali = []
    dati_atti: dict = {}

    for uid_str in uids_prima:
        # Offload su thread: get_dati_atto_da_leggi_atto_plus è sincrona/bloccante
        # (requests + il nuovo time.sleep() di throttling). Chiamarla direttamente
        # dentro questa coroutine blocca l'intero event loop, e con il throttling
        # ora attivo il blocco può durare secondi consecutivi per centinaia di UID:
        # rischia di far scadere le connessioni HTTP mantenute aperte da
        # AsyncQdrantClient più avanti, causando errori di lettura risposta.
        dati = await asyncio.to_thread(get_dati_atto_da_leggi_atto_plus, repwss_client, uid_str)
        if dati is None:
            log.warning(f"UID {uid_str}: escluso, errore durante leggi_atto_plus.")
            continue

        numero_atto_reale = dati.get("numero_atto")
        if not numero_atto_reale or numero_atto_reale == "0":
            log.debug(f"UID {uid_str}: escluso, è una proposta non ancora protocollata (numero_atto={numero_atto_reale!r}).")
            continue

        if id_tipo_iter_attesi and dati.get("id_tipo_iter") not in id_tipo_iter_attesi:
            log.debug(f"UID {uid_str}: escluso, id_tipo_iter={dati.get('id_tipo_iter')!r} non compatibile con {tipo_atto} (attesi: {id_tipo_iter_attesi}).")
            continue

        if anno_atto_richiesto:
            anno_reale = dati.get("anno_atto")
            if str(anno_reale).strip() != str(anno_atto_richiesto).strip():
                log.debug(f"UID {uid_str}: escluso, anno {anno_reale} != {anno_atto_richiesto} richiesto.")
                continue

        dati_atti[uid_str] = dati
        ids_totali.append(uid_str)

    n_esclusi = len(uids_prima) - len(ids_totali)
    if n_esclusi:
        log.info(f"Verifica LeggiAttoPlus: esclusi {n_esclusi}/{len(uids_prima)} atti (sotto-tipo diverso, proposte non definitive, o anno diverso).")

    if not ids_totali:
        log.warning(f"Nessun atto valido tra i {len(uids_prima)} risultati trovati dopo la verifica via LeggiAttoPlus.")
        return

    uids = ids_totali
    log.info(f"Trovati {len(uids)} atti su Sicr@Web (dopo aver unito {len(filtri_list)} ricerche e verificato via LeggiAttoPlus). Inizio controlli puntuali allegati...")

    # Timeout più permissivo del default precedente (20s): sotto scritture
    # massive di metadati il server può rispondere più lentamente. Configurabile
    # via settings.qdrant_client_timeout_seconds se lo aggiungi al config.
    qdrant_timeout = getattr(settings, "qdrant_client_timeout_seconds", 60.0)
    qdrant_client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=qdrant_timeout)

    print("\n" + "="*128)
    if update_metadata_only:
        print(f"{'UID':<10} | {'NUMERO':<10} | {'FILE ALLEGATO ATTESO':<48} | {'MYSQL':<15} | {'QDRANT':<15} | {'STATO'}")
    else:
        print(f"{'UID':<10} | {'NUMERO':<10} | {'FILE ALLEGATO ATTESO':<48} | {'MYSQL (PARENTS)':<15} | {'QDRANT (CHUNKS)':<15} | {'STATO'}")
    print("="*128)

    for uid in uids:
        dati = dati_atti[uid]
        num_atto = dati.get("numero_atto", "N/A")
        anno_atto_val = dati.get("anno_atto", "")
        display_num = f"{num_atto}/{anno_atto_val}" if anno_atto_val else str(num_atto)

        expected_files = get_expected_files_from_lista_allegati(uid, dati["allegati"])

        if not expected_files:
            print(f"{uid:<10} | {display_num:<10} | {'NESSUN ALLEGATO VALIDO TROVATO (SKIP)':<48} | {'0':<15} | {'0':<15} | SKIPPED")
            continue

        needs_recovery = False
        fresh_metadata = build_fresh_metadata(dati) if update_metadata_only else None
        sources_da_aggiornare_qdrant = []  # popolato solo in update_metadata_only

        for safe_name in expected_files:
            target_source = f"sicraweb://{uid}::{safe_name}"
            
            mysql_count = check_mysql_file_presence(target_source, safe_name)
            
            try:
                qdrant_res = await _qdrant_call_with_retry(
                    lambda: qdrant_client.count(
                        collection_name=QDRANT_COLLECTION,
                        count_filter=models.Filter(
                            must=[models.FieldCondition(key="source", match=models.MatchValue(value=target_source))]
                        )
                    ),
                    description=f"count({target_source})",
                )
                qdrant_count = qdrant_res.count
            except Exception:
                qdrant_count = "ERR_QDRANT"

            if update_metadata_only:
                # Aggiorna SOLO i metadati dove il documento esiste già.
                # Non tocca content/vettori, non fa alcuna re-ingestion, non
                # crea nulla di nuovo dove manca (per quello serve --recover).
                # L'update Qdrant vero e proprio è differito a DOPO questo
                # loop (una sola chiamata batch per tutti gli allegati
                # dell'atto, invece di una per allegato — vedi più sotto).
                mysql_updated = 0
                if mysql_count > 0:
                    mysql_updated = update_mysql_metadata(target_source, fresh_metadata)
                if isinstance(qdrant_count, int) and qdrant_count > 0:
                    sources_da_aggiornare_qdrant.append(target_source)

                if mysql_updated or (isinstance(qdrant_count, int) and qdrant_count > 0):
                    stato = f"METADATA DA AGGIORNARE (mysql:{mysql_updated} righe, qdrant:{qdrant_count if isinstance(qdrant_count, int) else 0} chunk)"
                else:
                    stato = "SKIP (non presente né in MySQL né in Qdrant)"

                display_name = safe_name if len(safe_name) <= 46 else f"...{safe_name[-43:]}"
                print(f"{uid:<10} | {display_num:<10} | {display_name:<48} | {mysql_count:<15} | {qdrant_count:<15} | {stato}")
                continue

            if mysql_count == 0 and qdrant_count == 0:
                stato = "MISSING ENTIRELY"
                needs_recovery = True
            elif mysql_count == 0:
                stato = "MISSING IN MYSQL"
                needs_recovery = True
            elif qdrant_count == 0 or qdrant_count == "ERR_QDRANT":
                stato = "MISSING IN QDRANT"
                needs_recovery = True
            else:
                stato = "OK"

            display_name = safe_name if len(safe_name) <= 46 else f"...{safe_name[-43:]}"
            print(f"{uid:<10} | {display_num:<10} | {display_name:<48} | {mysql_count:<15} | {qdrant_count:<15} | {stato}")

        if update_metadata_only:
            # Un'unica chiamata Qdrant per TUTTI gli allegati di questo atto
            # (MatchAny sui source raccolti sopra), invece di una per allegato.
            if sources_da_aggiornare_qdrant:
                try:
                    await update_qdrant_metadata_batch(qdrant_client, sources_da_aggiornare_qdrant, fresh_metadata)
                    print(f"    [✓] Qdrant aggiornato per {len(sources_da_aggiornare_qdrant)} source (UID {uid}).")
                except Exception as e:
                    print(f"    [✗] Aggiornamento Qdrant fallito per UID {uid} dopo i retry: {e}")
            # Modalità "solo metadati": mai recovery/re-ingestion, qualunque
            # sia lo stato dei file.
            continue

        if needs_recovery:
            if auto_recover:
                meta_atto_recovery = {
                    "numero": dati.get("numero_atto"),
                    "anno": dati.get("anno_atto"),
                    "oggetto": dati.get("oggetto"),
                    "id_tipo_iter": dati.get("id_tipo_iter"),
                    "classifica": dati.get("classifica"),
                    "classifica_descrizione": dati.get("classifica_descrizione"),
                    "data": dati.get("data_atto"),
                    "data_esecutivita": dati.get("data_esecutivita"),
                    "data_pubblicazione": dati.get("data_pubblicazione"),
                    "giorni_pubblicazione": dati.get("giorni_pubblicazione"),
                    "trattamento_descrizione": dati.get("trattamento_descrizione"),
                    "proponente_descrizione": dati.get("proponente_descrizione"),
                    "dirigente_descrizione": dati.get("dirigente_descrizione"),
                }
                await retrigger_extraction(uid, tipo_atto, meta_atto_recovery)
            else:
                print(f"    [!] Recovery disabilitato. Utilizzare flag --recover per forzare l'inserimento.")
            
    print("="*128 + "\n")
    await qdrant_client.close()

def main():
    parser = argparse.ArgumentParser(description="Verificatore Pipeline Documentale con Auto-Recovery")
    parser.add_argument("--json-filters", type=str, required=True, help="Filtri JSON di ricerca.")
    parser.add_argument("--recover", action="store_true", help="Lancia estrattore.py per tentare il recupero dei file mancanti.")
    parser.add_argument(
        "--update-metadata-only",
        action="store_true",
        help=(
            "Aggiorna SOLO i metadati (MySQL 'metadata' + payload Qdrant) dei documenti "
            "già presenti, richiamando leggi_atto_plus() per dati freschi. Non tocca "
            "content/vettori, non fa alcuna re-ingestion. Ignora --recover."
        ),
    )
    parser.add_argument(
        "--sicraweb-delay",
        type=float,
        default=None,
        help="Pausa minima in secondi tra due chiamate LeggiAttoPlus consecutive (sovrascrive il default/config).",
    )
    args = parser.parse_args()

    if args.update_metadata_only and args.recover:
        log.warning("--recover ignorato: --update-metadata-only non fa mai recovery/re-ingestion.")

    asyncio.run(verify_pipeline(args.json_filters, args.recover, args.update_metadata_only, args.sicraweb_delay))

if __name__ == "__main__":
    main()