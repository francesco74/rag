import json
import sys
import logging
import argparse
import asyncio
import pathlib
from pathlib import Path

from config import settings
from db_logger import init_db_pool, get_db_connection
from risultati_ricerca_parser import run_search
from ricerca_filtri import RicercaFiltri, DeterminaFilter, DeliberaFilter, DecretoFilter
from estrazione_documenti import build_client_from_env

from qdrant_client import AsyncQdrantClient
from qdrant_client import models

QDRANT_COLLECTION = "document_chunks"
ESTENSIONI_CONSENTITE = {".pdf", ".p7m"}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - VERIFICATORE - %(levelname)s - %(message)s')
log = logging.getLogger("verifier")

def parse_filters(json_filters_str: str) -> tuple[RicercaFiltri, str]:
    raw_json = json.loads(json_filters_str)
    tipi_supportati = {"determina": "determine", "delibera": "delibere", "decreto": "decreti"}
    chiavi_atto = [k for k in raw_json.keys() if k in tipi_supportati]
    
    if len(chiavi_atto) != 1:
        raise ValueError(f"Il JSON deve contenere esattamente UN tipo di atto root tra: {list(tipi_supportati.keys())}")
        
    tipo_atto = chiavi_atto[0]
    filtri_kwargs = {"utente": "utente@wsprotocollo", "ruolo": "CED"}
    
    if tipo_atto == "determina":
        filtri_kwargs["determina"] = DeterminaFilter(**raw_json[tipo_atto])
    elif tipo_atto == "delibera":
        filtri_kwargs["delibera"] = DeliberaFilter(**raw_json[tipo_atto])
    elif tipo_atto == "decreto":
        filtri_kwargs["decreto"] = DecretoFilter(**raw_json[tipo_atto])
        
    return RicercaFiltri(**filtri_kwargs), tipo_atto

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

async def retrigger_extraction(tipo_atto: str, num_atto: str, anno_atto: str):
    if not num_atto or not anno_atto or num_atto == "N/A":
        print(f"    [!] SALTO RECOVERY: Numero o Anno mancanti per formare una query sicura.")
        return

    keys_map = {
        "determina": ("determina_numero_gen", "determina_anno_gen"),
        "delibera": ("delibera_numero_gen", "delibera_anno_gen"),
        "decreto": ("decreto_numero_gen", "decreto_anno_gen")
    }
    
    num_key, anno_key = keys_map[tipo_atto]
    payload = {
        tipo_atto: {
            num_key: str(num_atto),
            anno_key: str(anno_atto)
        }
    }
    
    cmd = [sys.executable, "estrattore.py", "--json-filters", json.dumps(payload)]
    
    print(f"    [>] Esecuzione recovery: python3 estrattore.py (Numero: {num_atto}, Anno: {anno_atto})...")
    
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
    filtri, tipo_atto = parse_filters(json_filters_str)
    
    log.info("Esecuzione query di ricerca su Sicr@Web...")
    risultato_ricerca = run_search(filtri, dry_run=False)
    
    if risultato_ricerca.errore or not risultato_ricerca.ids:
        log.warning(f"Nessun atto restituito dalla ricerca. Errore: {risultato_ricerca.errore}")
        return

    uids = [str(uid) for uid in risultato_ricerca.ids]
    log.info(f"Trovati {len(uids)} atti su Sicr@Web. Inizializzazione controlli puntuali allegati...")

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
        meta_atto = risultato_ricerca.documenti_metadata.get(uid, {})
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
                await retrigger_extraction(tipo_atto, num_atto, anno_atto)
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