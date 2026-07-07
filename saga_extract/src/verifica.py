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

def parse_filters(json_filters_str: str) -> tuple[List[RicercaFiltri], str, Optional[str], Optional[str]]:
    """
    Restituisce (lista_filtri, tipo_atto, anno_atto_richiesto, registro_atteso).
    Con decreto/delibera + range di più giorni, la lista contiene un filtro per
    ciascun giorno (vedi RicercaSemplice.to_filtri_list), dato che
    DeliberaFilter/DecretoFilter non hanno un campo range confermato nel WSDL.

    anno_atto_richiesto viene restituito a parte perché per il DECRETO il WSDL
    non ha nessun campo "Anno" (vedi <Documento> nel template WSDL: solo
    Numero/Data/Tipo/Oggetto) — quindi l'anno, se richiesto, va poi applicato
    come FILTRO LATO CLIENT sui risultati già estratti (vedi verify_pipeline),
    usando l'anno del registro definitivo che il parser estrae comunque.

    registro_atteso (es. "DEC_VBDD") serve a filtrare lato client anche il
    SOTTO-TIPO di decreto/delibera (deliberativo vs presidenziale, organo).
    CONFERMATO da log reale: <Tipo>DEC</Tipo> nella richiesta NON distingue
    deliberativo da presidenziale — una ricerca per decreti deliberativi ha
    restituito un decreto il cui unico registro è "Registro Verbale
    (DEC_VBMP)" (presidenziale). Il sotto-tipo va quindi verificato dopo la
    ricerca, confrontando il registro realmente trovato con quello atteso.
    Per questo non esiste più un parametro "tipo_decreto": il sotto-tipo è
    parte del tipo_atto stesso ("decreto_deliberativo"/"decreto_presidenziale").

    Formato JSON atteso:
       {"tipo_atto": "determina", "numero_atto": "4", "anno_atto": "2025"}
       {"tipo_atto": "decreto_deliberativo", "oggetto": "interreg",
        "data_da": "01/06/2026", "data_a": "30/06/2026"}
       {"tipo_atto": "decreto_presidenziale", "numero_atto": "1", "anno_atto": "2026"}
       {"tipo_atto": "qualsiasi", "oggetto": "mozione nicodemo"}
    """
    raw_json = json.loads(json_filters_str)
    semplice = RicercaSemplice(**raw_json)
    return semplice.to_filtri_list(), semplice.tipo_atto, semplice.anno_atto, semplice.registro_definitivo_atteso()

def get_expected_files_from_sicraweb(repwss_client, uid: str) -> list:
    try:
        lista_allegati_raw, _ = repwss_client.leggi_atto_plus(uid)
        # FIX: leggi_atto_plus() restituisce un generatore (yield in estrai_allegati()).
        # Va materializzato in lista PRIMA di essere iterato due volte, altrimenti dopo
        # la comprehension seguente risulta esaurito e il ciclo for successivo non
        # produce più alcun allegato (falso "NESSUN ALLEGATO VALIDO TROVATO").
        lista_allegati_raw = list(lista_allegati_raw)
    except Exception as e:
        log.error(f"Errore durante la leggi_atto_plus per UID {uid}: {e}")
        return None

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
    filtri_list, tipo_atto, anno_atto_richiesto, registro_atteso = parse_filters(json_filters_str)

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

    # FILTRO REGISTRO/SOTTOTIPO LATO CLIENT: necessario perché <Tipo> nella
    # richiesta SOAP filtra solo decreto/delibera/determina in generale, NON
    # il sotto-tipo. CONFERMATO da log reale: una ricerca decreto con
    # tipo_decreto="deliberativo" (quindi <Tipo>DEC</Tipo>) ha restituito un
    # decreto il cui unico registro definitivo è "Registro Verbale (DEC_VBMP)"
    # (presidenziale). Verifichiamo quindi qui che il registro EFFETTIVAMENTE
    # trovato per ogni documento corrisponda a quello atteso per il
    # tipo_atto/sotto-tipo richiesto (registro_atteso, es. "DEC_VBDD").
    # Non si applica a tipo_atto="qualsiasi" (registro_atteso è None lì).
    if registro_atteso:
        uids_prima = list(ids_totali)
        ids_totali = []
        for uid_str in uids_prima:
            codice_trovato = documenti_metadata.get(uid_str, {}).get("registro_definitivo_codice")
            if codice_trovato == registro_atteso:
                ids_totali.append(uid_str)
            else:
                log.debug(f"UID {uid_str}: escluso, registro trovato '{codice_trovato}' != atteso '{registro_atteso}' (sotto-tipo diverso).")

        n_esclusi = len(uids_prima) - len(ids_totali)
        if n_esclusi:
            log.info(f"Filtro registro={registro_atteso}: esclusi {n_esclusi}/{len(uids_prima)} atti di sotto-tipo/registro diverso.")

        if not ids_totali:
            log.warning(f"Nessun atto con registro definitivo {registro_atteso} tra i {len(uids_prima)} risultati trovati.")
            return

    # FILTRO ANNO LATO CLIENT: necessario perché il WSDL non espone un campo
    # "Anno" per il decreto (solo Numero/Data/Tipo/Oggetto in <Documento>).
    # Sfruttiamo l'anno che il parser estrae comunque dal registro definitivo
    # (vedi risultati_ricerca_parser.py) per scartare qui i falsi positivi.
    #
    # IMPORTANTE (scoperto empiricamente): <Documento><Numero> nel WSDL NON
    # cerca sul registro Verbale/definitivo, ma su un registro generico "PR"
    # presente storicamente su quasi ogni atto (di qualunque tipo e anno).
    # Una ricerca decreto per solo Numero=1 restituisce quindi documenti dal
    # 2010 al 2026, quasi tutti privi di un registro DEC_VBDD/DEC_VBMP vero e
    # proprio. Se un documento NON ha nessun registro definitivo del tipo
    # cercato, è quasi certamente un falso positivo su questo campo "PR"
    # legacy, non l'atto realmente cercato: per questo lo ESCLUDIAMO, invece
    # di tenerlo "per prudenza" (scelta precedente, che produceva decine di
    # falsi "MISSING ENTIRELY" fuorvianti).
    if anno_atto_richiesto:
        uids_prima = list(ids_totali)
        ids_totali = []
        for uid_str in uids_prima:
            anno_trovato = documenti_metadata.get(uid_str, {}).get("anno")
            if anno_trovato is None:
                log.debug(f"UID {uid_str}: escluso, nessun registro definitivo del tipo cercato (probabile falso positivo sul registro 'PR' generico).")
            elif str(anno_trovato).strip() == str(anno_atto_richiesto).strip():
                ids_totali.append(uid_str)
            else:
                log.debug(f"UID {uid_str}: escluso, anno {anno_trovato} != {anno_atto_richiesto} richiesto.")

        n_esclusi = len(uids_prima) - len(ids_totali)
        if n_esclusi:
            log.info(f"Filtro anno={anno_atto_richiesto}: esclusi {n_esclusi}/{len(uids_prima)} atti (anno diverso o nessun registro definitivo trovato).")

        if not ids_totali:
            log.warning(f"Nessun atto con anno {anno_atto_richiesto} tra i {len(uids_prima)} risultati trovati.")
            return


    uids = ids_totali
    log.info(f"Trovati {len(uids)} atti su Sicr@Web (dopo aver unito {len(filtri_list)} ricerche). Inizializzazione controlli puntuali allegati...")

    try:
        repwss_client = build_client_from_env()
    except Exception as e:
        log.error(f"Impossibile istanziare il client Sicr@Web: {e}")
        return

    qdrant_client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=20.0)

    print("\n" + "="*128)
    print(f"{'UID':<10} | {'NUMERO':<10} | {'FILE ALLEGATO ATTESO':<48} | {'MYSQL (PARENTS)':<15} | {'QDRANT (CHUNKS)':<15} | {'STATO'}")
    print("="*128)

    for uid in uids:
        meta_atto = documenti_metadata.get(uid, {})
        num_atto = meta_atto.get("numero", "N/A")
        anno_atto = meta_atto.get("anno", "")
        display_num = f"{num_atto}/{anno_atto}" if anno_atto else str(num_atto)

        expected_files = get_expected_files_from_sicraweb(repwss_client, uid)
        
        if expected_files is None:
            print(f"{uid:<10} | {display_num:<10} | {'RECOVERY ERROR (SICRAWEB FAILURE)':<48} | {'0':<15} | {'0':<15} | ERROR")
            continue
            
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
                await retrigger_extraction(uid, tipo_atto, meta_atto)
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