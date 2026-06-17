import os
import json
import logging
import argparse
import pathlib
from pathlib import Path
from typing import Optional
import pika

from asn1crypto.cms import ContentInfo

from risultati_ricerca_parser import run_search
from estrazione_documenti import build_client_from_env

# Importiamo tutte le classi filtro
from ricerca_filtri import RicercaFiltri, DeterminaFilter, DeliberaFilter, DecretoFilter
from dotenv import load_dotenv

load_dotenv()

ESTENSIONI_CONSENTITE = {".pdf", ".p7m"}

log = logging.getLogger("main_extractor")
BASE_DIR = pathlib.Path(__file__).parent.resolve()
STAGING_ATTI_FOLDER = pathlib.Path(os.environ.get("DATA_FOLDER", str(BASE_DIR))) / "staging" / "provincia"

def clean_iso_date(date_raw: str) -> Optional[str]:
    """Uniforma le date al formato YYYY-MM-DD, rimuovendo le componenti temporali (T)."""
    if not date_raw: 
        return None
    return date_raw.split("T")[0] if "T" in date_raw else date_raw.strip()

def extract_file_from_p7m(p7m_bytes: bytes, filename: str) -> bytes:
    """Decodifica busta CAdES ed estrae il file originale in RAM."""
    try:
        content_info = ContentInfo.load(p7m_bytes)
        compressed_content = content_info['content']
        encap_content_info = compressed_content['encap_content_info']
        raw_content = encap_content_info['content'].native
        
        extracted_bytes = raw_content if isinstance(raw_content, bytes) else encap_content_info['content'].chosen.contents
        log.debug("Sbustamento completato per %s", filename)
        return extracted_bytes
    except Exception as e:
        log.error("Fallimento sbustamento P7M per %s: %s", filename, str(e))
        raise ValueError(f"Decodifica P7M fallita: {str(e)}")
    
def get_rabbitmq_channel():
    """Inizializza la connessione al broker per pubblicare gli eventi."""
    rabbitmq_host = os.environ.get("RABBITMQ_HOST", "rabbitmq-service.rag.svc.cluster.local")
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbitmq_host))
        channel = connection.channel()
        channel.queue_declare(queue='da-convertire', durable=True)
        return connection, channel
    except Exception as e:
        log.error(f"Errore critico di connessione a RabbitMQ su {rabbitmq_host}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Estrattore Massivo Sicr@Web - Generatore Manifest")
    parser.add_argument("--debug", action="store_true", help="Abilita log di livello DEBUG.")
    parser.add_argument("--dry", action="store_true", help="Simula la ricerca senza scaricare nulla.")
    parser.add_argument("--json-filters", type=str, required=True, help="Filtri di ricerca in JSON.")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - ESTRATTORE - %(levelname)s - %(message)s', force=True)

    try:
        raw_json = json.loads(args.json_filters)
        
        # 1. VALIDAZIONE STRICT DEL TIPO ATTO E ROUTING DIRECTORY
        tipi_supportati = {"determina": "determine", "delibera": "delibere", "decreto": "decreti"}
        chiavi_atto = [k for k in raw_json.keys() if k in tipi_supportati]
        
        if len(chiavi_atto) != 1:
            log.error("Il JSON deve contenere esattamente UN tipo di atto root tra: %s", list(tipi_supportati.keys()))
            return
            
        tipo_atto = chiavi_atto[0]
        tipo_cartella = tipi_supportati[tipo_atto]
        
        # ATTENZIONE: Variabile dinamica usata per TUTTO il ciclo di vita (creazione e scrittura)
        staging_json_dir = STAGING_ATTI_FOLDER / tipo_cartella
        staging_json_dir.mkdir(parents=True, exist_ok=True)

        # 2. CREAZIONE DINAMICA DEI FILTRI DI RICERCA
        filtri_kwargs = {"utente": "utente@wsprotocollo", "ruolo": "CED"}
        
        if tipo_atto == "determina":
            filtri_kwargs["determina"] = DeterminaFilter(**raw_json[tipo_atto])
        elif tipo_atto == "delibera":
            filtri_kwargs["delibera"] = DeliberaFilter(**raw_json[tipo_atto])
        elif tipo_atto == "decreto":
            filtri_kwargs["decreto"] = DecretoFilter(**raw_json[tipo_atto])
            
        filtri_dinamici = RicercaFiltri(**filtri_kwargs)
        
    except (json.JSONDecodeError, TypeError) as e:
        log.error("Errore nei filtri di input: %s", str(e))
        return

    log.info("Esecuzione ricerca documenti su Sicr@Web (Destinazione: %s)...", tipo_cartella)
    risultato_ricerca = run_search(filtri_dinamici, dry_run=args.dry)

    if args.dry: 
        log.info("Esecuzione DRY-RUN terminata.")
        return

    if risultato_ricerca.errore or not risultato_ricerca.ids:
        log.warning("Ricerca fallita o senza risultati. Errore: %s", risultato_ricerca.errore)
        return

    log.info("Trovati %d documenti. Inizio download dei pacchetti binari...", len(risultato_ricerca.ids))
    
    try:
        repwss_client = build_client_from_env()
        mq_conn, mq_channel = get_rabbitmq_channel()
    except Exception as e:
        log.error("Impossibile inizializzare il client WSAtti: %s", str(e))
        return

    success_count = 0

    # 3. ELABORAZIONE DIRETTA E UNPACKING MULTI-ALLEGATO
    for doc_uid in risultato_ricerca.ids:
        str_uid = str(doc_uid)
        json_filename = f"doc_{str_uid}.json"
        log.info("Elaborazione UID: %s", str_uid)

        tmp_files_paths = []
        file_scritti_nomi = []
        tmp_json_path = None

        try:
            lista_allegati_raw, attributi_plus = repwss_client.leggi_atto_plus(str_uid)
            meta_atto = risultato_ricerca.documenti_metadata.get(str_uid, {})
            
            nomi_file_presenti = {f_name.lower() for _, f_name in lista_allegati_raw}
            lista_allegati_filtrata = []
            
            for file_bytes, file_name in lista_allegati_raw:
                nome_lower = file_name.lower()
                
                # A. Whitelist Estensioni
                estensione = Path(nome_lower).suffix
                if estensione not in ESTENSIONI_CONSENTITE:
                    continue

                # B. Filtro deduplica
                if nome_lower.endswith('.p7m'):
                    nome_atteso_decifrato = nome_lower[:-4] # Rimuove '.p7m'
                    if nome_atteso_decifrato in nomi_file_presenti:
                        log.debug("Scartato '%s': file nativo già presente.", file_name)
                        continue
                    
                    # C. ESTRAZIONE P7M IN RAM
                    try:
                        log.info("Decodifica firma P7M per il file: %s", file_name)
                        file_bytes = extract_file_from_p7m(file_bytes, file_name)
                        # Rimuoviamo il ".p7m" dal nome del file per salvarlo col formato originale (es. .pdf)
                        file_name = file_name[:-4] 
                    except Exception as e:
                        log.warning("Impossibile decodificare %s, procedo con salvataggio originale. Errore: %s", file_name, e)

                lista_allegati_filtrata.append((file_bytes, file_name))
            
            if not lista_allegati_filtrata:
                log.warning("Nessun allegato valido rimasto per UID %s. Salto.", str_uid)
                continue

            for file_bytes, file_name in lista_allegati_filtrata:
                safe_name = f"doc_{str_uid}_{file_name}"
                
                tmp_path = staging_json_dir / f"{safe_name}.tmp"
                final_path = staging_json_dir / safe_name
                
                tmp_path.write_bytes(file_bytes)
                tmp_files_paths.append((tmp_path, final_path))
                file_scritti_nomi.append(safe_name)

            percorso_logico = f"{tipo_cartella}/atto_{str_uid}"

            sidecar_manifest = {
                "source": f"sicraweb://{str_uid}",
                "files": file_scritti_nomi,  
                "metadati": {
                    "id_sicraweb": str_uid,
                    "percorso_originale": percorso_logico,
                    "oggetto": meta_atto.get("oggetto"),
                    "trattamento_descrizione": meta_atto.get("trattamento_descrizione"),
                    "proponente_descrizione": meta_atto.get("proponente_descrizione"),
                    "dirigente_descrizione": meta_atto.get("dirigente_descrizione"),
                    "data": clean_iso_date(meta_atto.get("data")),
                    "anno": meta_atto.get("anno"),
                    "numero": meta_atto.get("numero"),
                    "data_esecutivita": clean_iso_date(meta_atto.get("data_esecutivita")),
                    "data_pubblicazione": clean_iso_date(meta_atto.get("data_pubblicazione")),
                    "giorni_pubblicazione": meta_atto.get("giorni_pubblicazione")
                }
            }

            tmp_json_path = staging_json_dir / f"{json_filename}.tmp"
            final_json_path = staging_json_dir / json_filename
            
            with open(tmp_json_path, "w", encoding="utf-8") as f:
                json.dump(sidecar_manifest, f, indent=2)

            for tmp_p, final_p in tmp_files_paths:
                tmp_p.rename(final_p)
            tmp_json_path.rename(final_json_path)

            success_count += 1
            log.info("✓ Atto %s elaborato (%d allegati).", str_uid, len(file_scritti_nomi))

            # --- NOTIFICA EVENT-DRIVEN ---
            rel_path_to_json = str((staging_json_dir / json_filename).relative_to(STAGING_ATTI_FOLDER.parent))
            
            payload = {
                "source_type": "json",
                "rel_path": rel_path_to_json
            }
            
            mq_channel.basic_publish(
                exchange='',
                routing_key='da-convertire',
                body=json.dumps(payload).encode(),
                properties=pika.BasicProperties(
                    delivery_mode=pika.DeliveryMode.Persistent # Il messaggio sopravvive al riavvio del broker
                )
            )
            log.debug(f"📨 Inviato evento a RabbitMQ per il file: {json_filename}")

        except Exception as e:
            log.error("✗ Errore elaborando UID %s: %s", str_uid, str(e), exc_info=True)
            for tmp_p, _ in tmp_files_paths:
                tmp_p.unlink(missing_ok=True)
            if tmp_json_path and tmp_json_path.exists():
                tmp_json_path.unlink(missing_ok=True)

    try:
        mq_conn.close()
    except: pass

    log.info("Completato (%d/%d estratti).", success_count, len(risultato_ricerca.ids))

if __name__ == "__main__":
    main()