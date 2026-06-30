import json
import logging
import argparse

from common.db_logger import get_db_connection, init_db_pool
from common.risultati_ricerca_parser import run_search
from common.ricerca_filtri import RicercaFiltri, DeterminaFilter, DeliberaFilter, DecretoFilter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - VERIFICA - %(levelname)s - %(message)s')
log = logging.getLogger("verifica")

def get_processed_uids(expected_uids: set) -> set:
    if not expected_uids:
        return set()

    processed = set()
    conn = get_db_connection()
    if not conn:
        log.error("Impossibile connettersi a MySQL.")
        return processed

    try:
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute("SELECT DISTINCT source FROM parent_documents WHERE source LIKE 'sicraweb://%'")
            rows = cursor.fetchall()
            
            for row in rows:
                source_str = row['source']
                try:
                    uid = source_str.split("://")[1].split("::")[0]
                    if uid in expected_uids:
                        processed.add(uid)
                except IndexError:
                    continue
    except Exception as e:
        log.error(f"Errore durante l'interrogazione del database: {e}")
    finally:
        conn.close()
        
    return processed

def main():
    parser = argparse.ArgumentParser(description="Verifica atti mancanti in MySQL")
    parser.add_argument("--json-filters", type=str, required=True, help="Filtri in JSON nel formato dell'estrattore.")
    args = parser.parseargs()

    init_db_pool()

    try:
        raw_json = json.loads(args.json_filters)
        
        tipi_supportati = {"determina", "delibera", "decreto"}
        chiavi_atto = [k for k in raw_json.keys() if k in tipi_supportati]
        
        if len(chiavi_atto) != 1:
            log.error(f"Il JSON deve contenere un singolo tipo di atto root tra: {list(tipi_supportati)}")
            return
            
        tipo_atto = chiavi_atto[0]
        filtri_kwargs = {"utente": "utente@wsprotocollo", "ruolo": "CED"}
        
        if tipo_atto == "determina":
            filtri_kwargs["determina"] = DeterminaFilter(**raw_json[tipo_atto])
        elif tipo_atto == "delibera":
            filtri_kwargs["delibera"] = DeliberaFilter(**raw_json[tipo_atto])
        elif tipo_atto == "decreto":
            filtri_kwargs["decreto"] = DecretoFilter(**raw_json[tipo_atto])
            
        filtri_dinamici = RicercaFiltri(**filtri_kwargs)
        
    except (json.JSONDecodeError, TypeError) as e:
        log.error(f"Errore nella decodifica dei filtri di input: {e}")
        return

    log.info("Interrogazione Sicr@Web in corso per ottenere l'elenco atteso...")
    risultato_ricerca = run_search(filtri_dinamici, dry_run=False)

    if risultato_ricerca.errore:
        log.error(f"Ricerca fallita: {risultato_ricerca.errore}")
        return
        
    expected_uids = {str(uid) for uid in risultato_ricerca.ids}
    total_expected = len(expected_uids)
    
    if total_expected == 0:
        log.info("Nessun documento trovato su Sicr@Web per i filtri specificati.")
        return

    log.info(f"Trovati {total_expected} documenti su Sicr@Web. Incrocio con il database...")
    
    processed_uids = get_processed_uids(expected_uids)
    missing_uids = expected_uids - processed_uids
    
    log.info("--- REPORT VERIFICA ---")
    log.info(f"Attesi (Sicr@Web): {total_expected}")
    log.info(f"Elaborati (MySQL): {len(processed_uids)}")
    log.info(f"Mancanti:          {len(missing_uids)}")
    
    if missing_uids:
        log.warning("Elenco documenti mancanti in archivio:")
        
        missing_details = []
        for uid in missing_uids:
            # Recupera i metadati associati a questo UID, se presenti
            meta = risultato_ricerca.documenti_metadata.get(uid, {})
            missing_details.append({
                "uid": uid,
                "oggetto": meta.get("oggetto", "Oggetto non disponibile")
            })
            
        print(json.dumps(missing_details, indent=2, ensure_ascii=False))
    else:
        log.info("✓ Tutti i documenti cercati risultano presenti in MySQL.")

if __name__ == "__main__":
    main()