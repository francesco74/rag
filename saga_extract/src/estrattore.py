import json
import logging
import argparse
import pathlib
import sys
from pathlib import Path
from typing import Optional
import pika
import os
from common.db_logger import MySQLLogHandler, init_db_pool
from dotenv import load_dotenv
from asn1crypto.cms import ContentInfo

load_dotenv()


from common.risultati_ricerca_parser import run_search
from common.estrazione_documenti import build_client_from_env
from common.ricerca_filtri import RicercaSemplice

# Importa l'unica fonte di verità
from common.config import settings

ESTENSIONI_CONSENTITE = {".pdf", ".p7m"}
STAGING_ATTI_FOLDER = pathlib.Path(settings.data_folder) / "staging" / "attiprovincia"

# Cartella STATICA per ciascun tipo_atto, decisa una volta sola dal tipo_atto
# richiesto. Il filtro registro/anno (in modalità ricerca) e la validazione
# già fatta da verifica.py (in modalità diretta) garantiscono a monte che i
# documenti arrivati qui siano già del sotto-tipo corretto, quindi non serve
# ridecidere la cartella documento per documento.
TIPI_SUPPORTATI = {
    "determina": "determine",
    "delibera": "delibere",
    "decreto_deliberativo": "decreti_deliberativi",
    "decreto_presidenziale": "decreti_presidenziali",
}

log = None
init_db_pool()

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
    try:
        credentials = pika.PlainCredentials(settings.broker_username, settings.broker_password)
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=settings.broker_host,
                port=settings.broker_port,
                credentials=credentials,
                heartbeat=60,
                blocked_connection_timeout=300
            )
        )
        channel = connection.channel()
        return connection, channel
    except Exception as e:
        log.error(f"Errore critico di connessione a RabbitMQ su {settings.broker_host}: {e}")
        raise


def elabora_atto(str_uid: str, meta_atto: dict, tipo_atto: str, tipo_cartella_statica: Optional[str], repwss_client) -> bool:
    """
    Estrae gli allegati di UN SINGOLO atto (UID) da Sicr@Web, li filtra,
    li scrive su disco, produce il sidecar JSON e pubblica l'evento su
    RabbitMQ. Restituisce True se completato con successo, False altrimenti
    (l'errore è già stato loggato).

    meta_atto è il dizionario di metadati (oggetto, anno, numero,
    registro_definitivo_codice, ...) — può provenire da una ricerca appena
    fatta (modalità ricerca) oppure essere passato già pronto da chi chiama
    questo script (modalità diretta, usata da verifica.py --recover: evita
    di rifare da capo la ricerca su Sicr@Web per un atto già identificato).
    """
    json_filename = f"doc_{str_uid}.json"
    log.info("Elaborazione UID: %s", str_uid)

    tmp_files_paths = []
    file_scritti_nomi = []
    tmp_json_path = None

    try:
        lista_allegati_raw, _attributi_plus = repwss_client.leggi_atto_plus(str_uid)
        # FIX: leggi_atto_plus() restituisce un generatore (estrai_allegati() usa yield).
        # Va materializzato in una lista PRIMA di essere consumato più volte,
        # altrimenti dopo il primo giro (nomi_file_presenti) risulta esaurito
        # e il ciclo successivo non produce più alcun allegato.
        lista_allegati_raw = list(lista_allegati_raw)

        log.debug(f"Metadati per UID {str_uid}: {meta_atto}")

        staging_json_dir = STAGING_ATTI_FOLDER / tipo_cartella_statica
        staging_json_dir.mkdir(parents=True, exist_ok=True)

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
                nome_atteso_decifrato = nome_lower[:-4]  # Rimuove '.p7m'
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
            return False

        nomi_visti = set()  # Traccia i nomi assegnati per questo documento

        for file_bytes, file_name in lista_allegati_filtrata:
            base_name = f"doc_{str_uid}_{file_name}"
            safe_name = base_name
            counter = 1

            while safe_name in nomi_visti:
                p = pathlib.Path(base_name)
                safe_name = f"{p.stem}_{counter}{p.suffix}"
                counter += 1

            nomi_visti.add(safe_name)

            tmp_path = staging_json_dir / f"{safe_name}.tmp"
            final_path = staging_json_dir / safe_name

            tmp_path.write_bytes(file_bytes)
            tmp_files_paths.append((tmp_path, final_path))
            file_scritti_nomi.append(safe_name)

        percorso_logico = f"{tipo_cartella_statica}/atto_{str_uid}"

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
                "giorni_pubblicazione": meta_atto.get("giorni_pubblicazione"),
                # Codice del registro effettivamente riconosciuto come "definitivo"
                # (es. "DEC_VBDD"/"DEC_VBMP"/"DLC_VB"/"DET_VB"): è l'unica traccia
                # persistente del sotto-tipo reale dell'atto (es. decreto
                # deliberativo vs presidenziale), dato che TipoDocumento nella
                # risposta SOAP è identico per entrambi i sotto-tipi.
                "registro_definitivo_codice": meta_atto.get("registro_definitivo_codice"),
                # True se è stato trovato un registro verbale/definitivo per
                # questo atto; False indica che numero/anno/data sopra sono
                # rimasti None (nessun registro di quel tipo nella risposta).
                "registro_definitivo_trovato": meta_atto.get("registro_definitivo_trovato"),
                # Nomi degli allegati come dichiarati dalla ricerca (prima di
                # whitelist estensioni/dedup .p7m/sbustamento): utile per audit,
                # da non confondere con "files" sopra (i nomi realmente scritti).
                "allegati_nomi_dichiarati": meta_atto.get("allegati_nomi"),
            }
        }

        tmp_json_path = staging_json_dir / f"{json_filename}.tmp"
        final_json_path = staging_json_dir / json_filename

        with open(tmp_json_path, "w", encoding="utf-8") as f:
            json.dump(sidecar_manifest, f, indent=2)

        for tmp_p, final_p in tmp_files_paths:
            tmp_p.rename(final_p)
        tmp_json_path.rename(final_json_path)

        log.info("✓ Atto %s elaborato (%d allegati).", str_uid, len(file_scritti_nomi))

        # --- NOTIFICA EVENT-DRIVEN ---
        rel_path_to_json = str((staging_json_dir / json_filename).relative_to(STAGING_ATTI_FOLDER.parent))

        payload = {
            "source_type": "json",
            "rel_path": rel_path_to_json
        }

        mq_conn, mq_channel = get_rabbitmq_channel()
        try:
            mq_channel.basic_publish(
                exchange='',
                routing_key='da-convertire',
                body=json.dumps(payload).encode(),
                properties=pika.BasicProperties(
                    delivery_mode=pika.DeliveryMode.Persistent
                )
            )
            log.debug(f"📨 Inviato evento a RabbitMQ per il file: {json_filename}")
        finally:
            # Garantisce la pulizia del socket in ogni caso
            mq_conn.close()

        return True

    except Exception as e:
        log.error("✗ Errore elaborando UID %s: %s", str_uid, str(e), exc_info=True)
        for tmp_p, _ in tmp_files_paths:
            tmp_p.unlink(missing_ok=True)
        if tmp_json_path and tmp_json_path.exists():
            tmp_json_path.unlink(missing_ok=True)
        return False


def main():
    global log

    parser = argparse.ArgumentParser(description="Estrattore Massivo")
    parser.add_argument("--debug", action="store_true", help="Abilita log di livello DEBUG.")
    parser.add_argument("--dry", action="store_true", help="Simula la ricerca.")
    parser.add_argument("--json-filters", type=str, required=True, help="Filtri in JSON.")
    args = parser.parse_args()

    init_db_pool()

    log_level = logging.DEBUG if args.debug else getattr(logging, settings.log_level, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - ESTRATTORE - %(levelname)s - %(message)s'
    )

    root_logger = logging.getLogger()
    db_handler = MySQLLogHandler()
    db_handler.setLevel(logging.WARNING)
    db_handler.setFormatter(logging.Formatter('%(asctime)s - ESTRATTORE - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'))
    root_logger.addHandler(db_handler)

    log = logging.getLogger("main_extractor")

    try:
        raw_json = json.loads(args.json_filters)
    except json.JSONDecodeError as e:
        log.error("JSON non valido: %s", str(e))
        sys.exit(1)

    # Non esiste più un tipo_atto="decreto" generico con un parametro
    # "tipo_decreto" separato: il sotto-tipo è parte del tipo_atto stesso
    # ("decreto_deliberativo"/"decreto_presidenziale"), perché <Tipo>DEC</Tipo>
    # nella richiesta SOAP non distingue affatto i due sotto-tipi (confermato
    # da log reale).
    tipo_atto = raw_json.get("tipo_atto")
    if tipo_atto not in TIPI_SUPPORTATI:
        log.error("tipo_atto mancante o non valido: deve essere uno tra %s (ricevuto: %r)", list(TIPI_SUPPORTATI.keys()), tipo_atto)
        sys.exit(1)

    tipo_cartella_statica = TIPI_SUPPORTATI[tipo_atto]
    (STAGING_ATTI_FOLDER / tipo_cartella_statica).mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------
    # MODALITA' DIRETTA: {"uid": "...", "tipo_atto": "...", "metadati": {...}}
    # Usata da verifica.py --recover, che ha GIA' identificato l'UID esatto
    # (dopo aver applicato i suoi stessi filtri registro/anno) e ne ha già
    # tutti i metadati: qui si salta del tutto la ricerca RicercaDocumentiString
    # e si va dritti a leggi_atto_plus(uid). Elimina sia la chiamata di rete
    # ridondante sia il rischio di "infiltrati" (altri atti con lo stesso
    # numero/anno ripescati da una ri-ricerca indipendente).
    # --------------------------------------------------------------------
    if "uid" in raw_json:
        str_uid = str(raw_json["uid"])
        meta_atto = raw_json.get("metadati") or {}

        if args.dry:
            log.info("DRY-RUN: elaborazione saltata per UID %s (modalità diretta).", str_uid)
            return

        try:
            repwss_client = build_client_from_env()
        except Exception as e:
            log.error("Impossibile inizializzare il client WSAtti: %s", str(e))
            sys.exit(1)

        log.info("Modalità diretta: elaborazione singolo UID %s (nessuna nuova ricerca su Sicr@Web).", str_uid)
        ok = elabora_atto(str_uid, meta_atto, tipo_atto, tipo_cartella_statica, repwss_client)
        log.info("Completato (%d/1 estratti).", 1 if ok else 0)
        sys.exit(0 if ok else 1)

    # --------------------------------------------------------------------
    # MODALITA' RICERCA (comportamento storico): {"tipo_atto": ..., "numero_atto": ...,
    # "anno_atto": ..., "oggetto": ..., "data_da": ..., "data_a": ...}
    # Utile per uso manuale / estrazioni massive quando non si ha già un UID.
    # --------------------------------------------------------------------
    try:
        semplice = RicercaSemplice(**{k: v for k, v in raw_json.items() if k != "uid"})
        filtri_dinamici = semplice.to_filtri_list()[0]
    except (TypeError, ValueError) as e:
        log.error("Errore nei filtri di input: %s", str(e))
        sys.exit(1)

    log.info("Esecuzione ricerca documenti su Sicr@Web (Destinazione: %s)...", tipo_cartella_statica)
    risultato_ricerca = run_search(filtri_dinamici, dry_run=args.dry)

    if args.dry:
        log.info("Esecuzione DRY-RUN terminata.")
        return

    if risultato_ricerca.errore or not risultato_ricerca.ids:
        log.warning("Ricerca fallita o senza risultati. Errore: %s", risultato_ricerca.errore)
        sys.exit(1)

    # FILTRO REGISTRO/ANNO LATO CLIENT (stessa logica di verify_pipeline in
    # verifica.py). NECESSARIO qui: <Documento><Numero> da solo può
    # restituire decine di atti di anni e sotto-tipi diversi (es. Numero=1
    # esistente sia come decreto deliberativo 2026 sia come presidenziale
    # 2025, oltre a vecchi atti pre-2024 senza nessun registro Verbale). Senza
    # questo filtro, si estrarrebbero e pubblicherebbero TUTTI questi
    # "infiltrati", non solo gli atti realmente richiesti.
    registro_atteso = semplice.registro_definitivo_atteso()
    anno_atto_richiesto = raw_json.get("anno_atto")
    ids_da_elaborare = list(risultato_ricerca.ids)

    if registro_atteso:
        prima = list(ids_da_elaborare)
        ids_da_elaborare = [
            u for u in prima
            if risultato_ricerca.documenti_metadata.get(str(u), {}).get("registro_definitivo_codice") == registro_atteso
        ]
        esclusi = len(prima) - len(ids_da_elaborare)
        if esclusi:
            log.info("Filtro registro=%s: esclusi %d/%d atti di sotto-tipo/registro diverso.", registro_atteso, esclusi, len(prima))

    if anno_atto_richiesto:
        prima = list(ids_da_elaborare)
        ids_da_elaborare = [
            u for u in prima
            if str(risultato_ricerca.documenti_metadata.get(str(u), {}).get("anno")).strip() == str(anno_atto_richiesto).strip()
        ]
        esclusi = len(prima) - len(ids_da_elaborare)
        if esclusi:
            log.info("Filtro anno=%s: esclusi %d/%d atti con anno diverso o senza registro definitivo.", anno_atto_richiesto, esclusi, len(prima))

    if not ids_da_elaborare:
        log.warning("Nessun atto corrispondente dopo il filtro registro/anno (candidati iniziali: %d). Recovery annullato.", len(risultato_ricerca.ids))
        sys.exit(1)

    log.info("Trovati %d documenti (dopo filtro registro/anno). Inizio download dei pacchetti binari...", len(ids_da_elaborare))

    try:
        repwss_client = build_client_from_env()
    except Exception as e:
        log.error("Impossibile inizializzare il client WSAtti: %s", str(e))
        sys.exit(1)

    success_count = 0
    for doc_uid in ids_da_elaborare:
        str_uid = str(doc_uid)
        meta_atto = risultato_ricerca.documenti_metadata.get(str_uid, {})
        if elabora_atto(str_uid, meta_atto, tipo_atto, tipo_cartella_statica, repwss_client):
            success_count += 1

    log.info("Completato (%d/%d estratti).", success_count, len(ids_da_elaborare))

    if success_count == 0:
        # Trovati atti dalla ricerca ma NESSUNO elaborato/pubblicato con successo:
        # non è un "successo" per chi ha lanciato questo script (es. il
        # recovery automatico di verifica.py), quindi segnaliamolo con un
        # exit code diverso da zero invece di uscire silenziosamente.
        sys.exit(1)

if __name__ == "__main__":
    main()