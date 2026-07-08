import json
import os
import sys
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

QDRANT_COLLECTION = "document_chunks"
ESTENSIONI_CONSENTITE = {".pdf", ".p7m"}

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

# Valori di id_tipo_iter CONFERMATI dal cliente (campo dentro
# Determina/Workflow/Attributi/Attributo con Nome="id_tipo_iter" nella
# risposta di LeggiAttoPlus — vedi estrazione_documenti.py). E' il segnale
# affidabile per il sotto-tipo del decreto: 8 = decreto del Presidente,
# 9/19 = decreto deliberativo. Sostituisce interamente il vecchio approccio
# basato sul parsing del registro Verbale nella risposta di ricerca
# (RicercaDocumentiString), che per alcuni atti reali può mancare del tutto
# (CONFERMATO: decreto presidenziale 1/2024, UID 2119188, aveva solo un
# registro "PR" generico nella ricerca pur essendo un decreto vero).
ID_TIPO_ITER_ATTESO = {
    "decreto_presidenziale": {"8"},
    "decreto_deliberativo": {"9", "19"},
}


def get_dati_atto_da_leggi_atto_plus(repwss_client, uid: str) -> Optional[dict]:
    """
    Chiama leggi_atto_plus(uid) UNA SOLA VOLTA per UID e restituisce sia gli
    allegati (già materializzati) sia gli attributi discriminanti
    (numero_atto, anno_atto, id_tipo_iter, oggetto), così lo stesso risultato
    viene riusato più avanti sia per la verifica sotto-tipo/anno/definitività
    sia per il calcolo dei file attesi, invece di richiamare leggi_atto_plus
    due volte per lo stesso UID (comportamento precedente).

    numero_atto: "0" (con data_atto sentinella "0001-01-01...") indica una
    mera proposta non ancora protocollata; qualunque altro valore indica un
    atto realmente protocollato (CONFERMATO su UID 2119188: numero_atto="1").

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
    if isinstance(attributi_plus, dict):
        numero_atto = attributi_plus.get("numero_atto")
        anno_atto = attributi_plus.get("anno_atto")
        id_tipo_iter = attributi_plus.get("id_tipo_iter")
        oggetto = attributi_plus.get("oggetto")
    else:
        numero_atto = getattr(attributi_plus, "numero_atto", None)
        anno_atto = getattr(attributi_plus, "anno_atto", None)
        id_tipo_iter = getattr(attributi_plus, "id_tipo_iter", None)
        oggetto = getattr(attributi_plus, "oggetto", None)

    return {
        "allegati": lista_allegati_raw,
        "numero_atto": str(numero_atto).strip() if numero_atto is not None else None,
        "anno_atto": str(anno_atto).strip() if anno_atto else None,
        "id_tipo_iter": str(id_tipo_iter).strip() if id_tipo_iter is not None else None,
        "oggetto": oggetto,
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


        

async def verify_pipeline(json_filters_str: str, auto_recover: bool):
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

    # VERIFICA SOTTO-TIPO/ANNO/DEFINITIVITA' VIA LeggiAttoPlus: sostituisce
    # interamente il vecchio doppio filtro (registro Verbale dalla risposta
    # di ricerca + anno dal registro definitivo), perché quel registro può
    # mancare del tutto per un atto reale (CONFERMATO: decreto presidenziale
    # 1/2024, UID 2119188, ha solo un registro "PR" generico nella ricerca
    # pur essendo un decreto vero). Il discriminante ora è LeggiAttoPlus:
    #   - numero_atto: "0"/assente = mera proposta non protocollata, esclusa;
    #   - id_tipo_iter: 8 = decreto presidenziale, 9/19 = decreto deliberativo
    #     (vedi ID_TIPO_ITER_ATTESO) — non si applica a tipo_atto senza
    #     sotto-tipo (es. "determina", "qualsiasi");
    #   - anno_atto: confrontato con anno_atto_richiesto, se presente.
    # Chiamiamo leggi_atto_plus() qui UNA SOLA VOLTA per candidato e ne
    # riusiamo il risultato più avanti anche per il calcolo dei file attesi,
    # invece di richiamarlo una seconda volta per lo stesso UID.
    try:
        repwss_client = build_client_from_env()
    except Exception as e:
        log.error(f"Impossibile istanziare il client Sicr@Web: {e}")
        return

    id_tipo_iter_attesi = ID_TIPO_ITER_ATTESO.get(tipo_atto)  # None per tipo_atto senza sotto-tipo (determina/qualsiasi/...)

    uids_prima = list(ids_totali)
    ids_totali = []
    dati_atti: dict = {}

    for uid_str in uids_prima:
        dati = get_dati_atto_da_leggi_atto_plus(repwss_client, uid_str)
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

    qdrant_client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=20.0)

    print("\n" + "="*128)
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

        for safe_name in expected_files:
            target_source = f"sicraweb://{uid}::{safe_name}"
            
            mysql_count = check_mysql_file_presence(target_source, safe_name)
            
            try:
                qdrant_res = await qdrant_client.count(
                    collection_name=QDRANT_COLLECTION,
                    count_filter=models.Filter(
                        must=[models.FieldCondition(key="source", match=models.MatchValue(value=target_source))]
                    )
                )
                qdrant_count = qdrant_res.count
            except Exception:
                qdrant_count = "ERR_QDRANT"

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

        if needs_recovery:
            if auto_recover:
                meta_atto_recovery = {
                    "numero": dati.get("numero_atto"),
                    "anno": dati.get("anno_atto"),
                    "oggetto": dati.get("oggetto"),
                    "id_tipo_iter": dati.get("id_tipo_iter"),
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
    args = parser.parse_args()
    
    asyncio.run(verify_pipeline(args.json_filters, args.recover))

if __name__ == "__main__":
    main()