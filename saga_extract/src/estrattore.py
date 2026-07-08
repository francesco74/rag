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

# La mappa dei valori attesi di id_tipo_iter ora vive in settings
# (settings.id_tipo_iter_atteso, common/config.py), configurabile via env
# ID_TIPO_ITER_DECRETO_PRESIDENZIALE/ID_TIPO_ITER_DECRETO_DELIBERATIVO,
# così verifica.py ed estrattore.py usano sempre la stessa mappa.



def get_dati_atto_da_leggi_atto_plus(repwss_client, uid: str):
    """
    Chiama leggi_atto_plus(uid) UNA SOLA VOLTA e restituisce sia gli
    allegati (già materializzati) sia gli attributi discriminanti
    (numero_atto, anno_atto, id_tipo_iter, oggetto). Lo stesso risultato
    viene poi passato a elabora_atto (parametro lista_allegati_raw) per
    evitare una seconda chiamata di rete per lo stesso UID.

    Restituisce None in caso di errore nella chiamata (loggato).
    """
    try:
        lista_allegati_raw, attributi_plus = repwss_client.leggi_atto_plus(uid)
        lista_allegati_raw = list(lista_allegati_raw)
    except Exception as e:
        log.error("Errore durante leggi_atto_plus per UID %s: %s", uid, str(e))
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

def clean_iso_date(date_raw: str) -> Optional[str]:
    """Uniforma le date al formato YYYY-MM-DD, rimuovendo le componenti temporali (T)."""
    if not date_raw: 
        return None
    return date_raw.split("T")[0] if "T" in date_raw else date_raw.strip()

def try_extract_pdf_from_pkcs7(file_bytes: bytes, filename: str = "") -> Optional[bytes]:
    """
    Tenta di interpretare file_bytes come busta di firma PKCS7/CMS (CAdES) e,
    se lo è, ne estrae il contenuto incapsulato (il PDF originale).

    NON si basa sul nome del file (Sicr@Web può restituire allegati con
    estensione ".pdf" il cui contenuto è in realtà ancora una busta CAdES non
    sbustata — mislabeling a monte, confermato su file reali) né su euristiche
    sui byte grezzi: usiamo asn1crypto per un parsing ASN.1 vero, controllando
    semanticamente il campo content_type.

    Restituisce:
    - None se file_bytes NON è una struttura PKCS7 valida (caso normale: PDF
      nativo già in chiaro) — il chiamante deve procedere con file_bytes
      originali, nessun errore da segnalare.
    - i bytes del PDF estratto, se file_bytes è una busta PKCS7 di tipo
      signed_data e l'estrazione riesce.

    Solleva ValueError SOLO se la struttura è riconosciuta come PKCS7
    signed_data ma l'estrazione del contenuto incapsulato fallisce
    (caso anomalo/da indagare, distinto da "non era affatto una busta").
    """
    try:
        content_info = ContentInfo.load(file_bytes)
    except Exception:
        # Non è (o non è un valido) ASN.1: quasi certamente un PDF nativo normale.
        return None

    if content_info['content_type'].native != 'signed_data':
        # E' una struttura ASN.1 valida ma non una busta di firma: non il nostro caso.
        return None

    try:
        signed_data = content_info['content']
        encap_content_info = signed_data['encap_content_info']
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


def elabora_atto(str_uid: str, meta_atto: dict, tipo_atto: str, tipo_cartella_statica: Optional[str], repwss_client, lista_allegati_raw: Optional[list] = None) -> bool:
    """
    Estrae gli allegati di UN SINGOLO atto (UID) da Sicr@Web, li filtra,
    li scrive su disco, produce il sidecar JSON e pubblica l'evento su
    RabbitMQ. Restituisce True se completato con successo, False altrimenti
    (l'errore è già stato loggato).

    meta_atto è il dizionario di metadati (oggetto, anno, numero, ...) — può
    provenire da una ricerca appena fatta (modalità ricerca) oppure essere
    passato già pronto da chi chiama questo script (modalità diretta, usata
    da verifica.py --recover: evita di rifare da capo la ricerca su Sicr@Web
    per un atto già identificato).

    lista_allegati_raw: se già disponibile (perché il chiamante ha già
    interrogato leggi_atto_plus per verificare sotto-tipo/anno/definitività
    via get_dati_atto_da_leggi_atto_plus, vedi MODALITA' RICERCA in main()),
    viene riusata qui invece di richiamare leggi_atto_plus una seconda volta
    per lo stesso UID. Se None (comportamento storico, usato in MODALITA'
    DIRETTA dove non c'è nessuna verifica preventiva da fare), viene
    recuperata qui.
    """
    json_filename = f"doc_{str_uid}.json"
    log.info("Elaborazione UID: %s", str_uid)

    tmp_files_paths = []
    file_scritti_nomi = []
    tmp_json_path = None

    try:
        if lista_allegati_raw is None:
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

            # B. Filtro deduplica (si applica solo quando il file è DICHIARATO come
            # .p7m: qui il nome ci dice che dovrebbe esisterne una versione nativa
            # gemella già presente nell'elenco, da preferire).
            if nome_lower.endswith('.p7m'):
                nome_atteso_decifrato = nome_lower[:-4]  # Rimuove '.p7m'
                if nome_atteso_decifrato in nomi_file_presenti:
                    log.debug("Scartato '%s': file nativo già presente.", file_name)
                    continue

            # C. ESTRAZIONE P7M IN RAM — basata sul CONTENUTO (via asn1crypto),
            # non sul nome file dichiarato (vedi try_extract_pdf_from_pkcs7).
            try:
                extracted = try_extract_pdf_from_pkcs7(file_bytes, file_name)
            except ValueError as e:
                log.warning("Impossibile decodificare %s, procedo con salvataggio originale. Errore: %s", file_name, e)
                extracted = None

            if extracted is not None:
                file_bytes = extracted
                if nome_lower.endswith('.p7m'):
                    # Rimuoviamo il ".p7m" dal nome per salvarlo col formato originale (es. .pdf).
                    # Se invece il file era già dichiarato ".pdf" (mislabeling a monte), il nome
                    # resta invariato: era già corretto, mancava solo lo sbustamento del contenuto.
                    file_name = file_name[:-4]

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
                # id_tipo_iter (da LeggiAttoPlus): 8 = decreto del Presidente,
                # 9/19 = decreto deliberativo. E' l'unica traccia persistente
                # affidabile del sotto-tipo reale dell'atto — sostituisce il
                # vecchio registro_definitivo_codice, che poteva mancare del
                # tutto anche per atti reali (vedi decreto presidenziale
                # 1/2024, UID 2119188).
                "id_tipo_iter": meta_atto.get("id_tipo_iter"),
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
    
    anno_atto_richiesto = raw_json.get("anno_atto")
    id_tipo_iter_attesi = settings.id_tipo_iter_atteso.get(tipo_atto)  # None per tipo_atto senza sotto-tipo (determina/qualsiasi/...)

    try:
        repwss_client = build_client_from_env()
    except Exception as e:
        log.error("Impossibile inizializzare il client WSAtti: %s", str(e))
        sys.exit(1)

    candidati = [str(u) for u in risultato_ricerca.ids]
    dati_atti = {}
    ids_da_elaborare = []

    for uid_str in candidati:
        dati = get_dati_atto_da_leggi_atto_plus(repwss_client, uid_str)
        if dati is None:
            log.warning("UID %s: escluso, errore durante leggi_atto_plus.", uid_str)
            continue

        numero_atto_reale = dati.get("numero_atto")
        if not numero_atto_reale or numero_atto_reale == "0":
            log.debug("UID %s: escluso, è una proposta non ancora protocollata (numero_atto=%r).", uid_str, numero_atto_reale)
            continue

        if id_tipo_iter_attesi and dati.get("id_tipo_iter") not in id_tipo_iter_attesi:
            log.debug("UID %s: escluso, id_tipo_iter=%r non compatibile con %s (attesi: %s).", uid_str, dati.get("id_tipo_iter"), tipo_atto, id_tipo_iter_attesi)
            continue

        if anno_atto_richiesto:
            anno_reale = dati.get("anno_atto")
            if str(anno_reale).strip() != str(anno_atto_richiesto).strip():
                log.debug("UID %s: escluso, anno %s != %s richiesto.", uid_str, anno_reale, anno_atto_richiesto)
                continue

        dati_atti[uid_str] = dati
        ids_da_elaborare.append(uid_str)

    esclusi = len(candidati) - len(ids_da_elaborare)
    if esclusi:
        log.info("Verifica LeggiAttoPlus: esclusi %d/%d atti (sotto-tipo diverso, proposte non definitive, o anno diverso).", esclusi, len(candidati))

    if not ids_da_elaborare:
        log.warning("Nessun atto corrispondente dopo la verifica via LeggiAttoPlus (candidati iniziali: %d). Recovery annullato.", len(candidati))
        sys.exit(1)

    log.info("Trovati %d documenti (dopo verifica LeggiAttoPlus). Inizio scrittura pacchetti binari...", len(ids_da_elaborare))

    success_count = 0
    for str_uid in ids_da_elaborare:
        dati = dati_atti[str_uid]
        meta_atto = {
            "numero": dati.get("numero_atto"),
            "anno": dati.get("anno_atto"),
            "oggetto": dati.get("oggetto"),
            "id_tipo_iter": dati.get("id_tipo_iter"),
        }
        if elabora_atto(str_uid, meta_atto, tipo_atto, tipo_cartella_statica, repwss_client, lista_allegati_raw=dati["allegati"]):
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