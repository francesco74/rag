"""
Caricamento dei documenti da indicizzare, sotto /review/uploads.

Il file passa i controlli, finisce in staging con il suo manifest e parte subito
sulla coda del converter, la stessa di direct.py. Il campo facoltativo "folder"
conserva le sottocartelle di origine: come per direct.py entrano in file_name
("faldone1/fascicolo3/scan.pdf") e quindi nelle fonti della chat. La tabella document_uploads
tiene chi ha caricato cosa; lo stato lo aggiornano converter e ingestione.

    GET  /review/uploads              ultimi caricamenti degli archivi dell'utente
    POST /review/uploads              carica un file (multipart)
    POST /review/uploads/<id>/send    rimanda in coda un file rimasto da inviare
    POST /review/uploads/<id>/retry   riprova un file in errore dal punto in cui si è fermato
"""

import hashlib
import json
import logging
import pathlib
import uuid

import pika
from flask import Blueprint, g, jsonify, request

from common.config import settings
from common.file_intake import (INPUT_EXTENSIONS, detect_extension_from_content, try_extract_from_pkcs7,
                                unwrapped_name)
from common.review_permissions import PERM_UPLOAD
from common.upload_status import INDEXED, FAILED, NOT_SENT, PROCESSING, QUEUED
from review_routes import ApiError, DATA_FOLDER, Db, _check_topic, _iso, publish_to_queue, require_permission

log = logging.getLogger("UPLOADS")

bp = Blueprint("uploads", __name__, url_prefix="/uploads")

STAGING_FOLDER = DATA_FOLDER / "staging"
WATCH_FOLDER = DATA_FOLDER / "watch"
PROCESSED_FOLDER = DATA_FOLDER / "processed"
CONVERTER_ERRORS = DATA_FOLDER / "converter" / "error"
INGESTION_ERRORS = DATA_FOLDER / "ingestion" / "error"
# La coda dei manifest (come direct.py), non quella delle conversioni al volo di /external.
CONVERTER_QUEUE = "da-convertire"
INGESTION_QUEUE = "da-indicizzare"
LIST_LIMIT = 200
# Filtri dell'elenco: gli stati che richiedono un intervento stanno insieme.
LIST_FILTERS = {"errori": (FAILED, NOT_SENT), "in_lavorazione": (QUEUED, PROCESSING), "indicizzati": (INDEXED,)}
# I file indicizzati non richiedono più nulla: restano in elenco per pochi giorni.
INDEXED_DAYS = 3
MAX_FOLDER_NAME = 100
MAX_FILE_NAME = 255  # colonna file_name, percorso compreso
CHUNK_BYTES = 1024 * 1024


def _safe_name(name):
    name = (name or "").strip()
    if (not name or len(name) > 200 or name.startswith(".") or "/" in name or "\\" in name
            or any(ord(c) < 32 for c in name)):
        raise ApiError(422, f"Nome di file non valido: '{name}'")
    if pathlib.Path(name).suffix.lower() not in INPUT_EXTENSIONS:
        raise ApiError(422, f"Tipo di file non ammesso: '{name}' ({', '.join(sorted(INPUT_EXTENSIONS))})")
    return name


def _safe_folder(folder):
    """Sottocartelle sotto la sottocategoria, separate da "/". Vuoto: nessuna."""
    folder = (folder or "").strip().strip("/")
    if not folder:
        return ""
    for part in folder.split("/"):
        if (not part or part.startswith(".") or len(part) > MAX_FOLDER_NAME or "\\" in part
                or part != part.strip() or any(ord(c) < 32 for c in part)):
            raise ApiError(422, f"Nome di cartella non valido: '{part}'")
    return folder


def _location(row):
    """Cartella del file sotto le radici della pipeline, nome del file e nome del suo manifest."""
    path = pathlib.PurePosixPath(row["topic_id"], row["sub_topic_id"], row["file_name"])
    return pathlib.Path(path.parent), path.name, f"{path.stem}.json"


def _receive(upload, temp_path):
    """Salva il file su disco senza superare il limite. Restituisce l'impronta SHA-256."""
    max_bytes = settings.upload_max_mb * 1024 * 1024
    digest = hashlib.sha256()
    written = 0
    with open(temp_path, "wb") as out:
        while chunk := upload.stream.read(CHUNK_BYTES):
            written += len(chunk)
            if written > max_bytes:
                raise ApiError(413, f"File troppo grande (massimo {settings.upload_max_mb} MB)")
            digest.update(chunk)
            out.write(chunk)
    if not written:
        raise ApiError(422, "Il file è vuoto")
    return digest.hexdigest()


def _check_content(temp_path, name):
    """Il contenuto deve corrispondere all'estensione. Le buste .p7m si aprono: nome finale."""
    suffix = pathlib.Path(name).suffix.lower()
    if suffix == ".p7m":
        try:
            content = try_extract_from_pkcs7(temp_path.read_bytes(), name)
        except ValueError:
            content = None
        detected = detect_extension_from_content(content) if content else None
        if not detected:
            raise ApiError(422, f"'{name}': la busta firmata non è valida o non contiene un PDF o un'immagine")
        temp_path.write_bytes(content)
        return unwrapped_name(name, detected)
    with open(temp_path, "rb") as f:
        head = f.read(16)
    detected = detect_extension_from_content(head)
    if suffix in (".txt", ".md"):
        try:
            temp_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ApiError(422, f"'{name}': il testo deve essere in UTF-8")
        return name
    if detected != (".jpg" if suffix == ".jpeg" else suffix):
        raise ApiError(422, f"'{name}': il contenuto non corrisponde all'estensione")
    return name


def _check_duplicates(db, topic_id, sub_topic_id, source, sha256, replace):
    twin = db.one("SELECT file_name FROM document_uploads WHERE topic_id = %s AND sub_topic_id = %s "
                  "AND sha256 = %s AND status <> %s", (topic_id, sub_topic_id, sha256, FAILED))
    if twin:
        raise ApiError(409, f"Stesso contenuto già caricato come '{twin['file_name']}'")
    row = db.one("SELECT status FROM document_uploads WHERE source = %s", (source,))
    if row and row["status"] not in (INDEXED, FAILED):
        raise ApiError(409, "Un file con questo nome è ancora in lavorazione")
    exists = (row and row["status"] == INDEXED) or db.one(
        "SELECT 1 AS ok FROM parent_documents WHERE source = %s AND topic_id = %s AND sub_topic_id = %s LIMIT 1",
        (source, topic_id, sub_topic_id))
    if exists and not replace:
        raise ApiError(409, "Esiste già un documento con questo nome: scegli di sostituirlo per caricarlo")
    return row is not None


def _send(row):
    """Mette il file in coda. False se RabbitMQ non risponde: resta "da inviare" e si può ritentare."""
    folder, _, manifest = _location(row)
    rel_path = str(folder / manifest)
    try:
        publish_to_queue(CONVERTER_QUEUE, {"source_type": "json", "rel_path": rel_path})
        return True
    except Exception as e:
        log.error(f"Invio alla coda fallito per {row['source']}: {e}")
        return False


def _retry_plan(row):
    """
    Da dove riparte un file in errore: (coda, messaggio, spostamenti), None se i file
    non ci sono più. Se il testo è già stato ricavato si riparte dall'indicizzazione,
    senza rifare l'OCR.
    """
    folder, name, manifest = _location(row)
    rel_path = str(folder / manifest)

    failed = INGESTION_ERRORS / folder / manifest
    texts = [p for p in (failed.with_suffix(".md"), failed.with_suffix(".txt")) if p.exists()]
    if failed.exists() and texts:
        return (INGESTION_QUEUE, {"json_manifest_path": rel_path},
                [(p, WATCH_FOLDER / folder / p.name) for p in [*texts, failed]])

    to_converter = (CONVERTER_QUEUE, {"source_type": "json", "rel_path": rel_path})
    staging = STAGING_FOLDER / folder
    failed = CONVERTER_ERRORS / folder / manifest
    if failed.exists() and (staging / name).exists():
        return (*to_converter, [(failed, staging / manifest)])

    # Il converter archivia anche i file da cui non ha ricavato testo.
    archive = PROCESSED_FOLDER / folder
    if (archive / manifest).exists() and (archive / name).exists():
        return (*to_converter, [(archive / manifest, staging / manifest), (archive / name, staging / name)])
    return None


def _drop_dead_letters(row):
    """Toglie dalle DLQ i messaggi di questo file: dlq_converter e dlq_ingestion non lo rilancino."""
    folder, _, manifest = _location(row)
    rel_path = str(folder / manifest)
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(
            host=settings.broker_host, port=settings.broker_port,
            credentials=pika.PlainCredentials(settings.broker_username, settings.broker_password)))
        try:
            channel = connection.channel()
            for queue, key in (("da-convertire.dlq", "rel_path"), ("da-indicizzare.dlq", "json_manifest_path")):
                # Gli altri messaggi, senza ack, tornano in coda alla chiusura della connessione.
                while True:
                    method, _, body = channel.basic_get(queue=queue)
                    if method is None:
                        break
                    try:
                        mine = json.loads(body.decode()).get(key) == rel_path
                    except (ValueError, AttributeError):
                        mine = False
                    if mine:
                        channel.basic_ack(delivery_tag=method.delivery_tag)
        finally:
            connection.close()
    except Exception as e:
        log.warning(f"DLQ non ripulite per {row['source']}: {e}")


def _forget_failure(row):
    """Una nuova versione del file: i resti del tentativo fallito non devono poter ripartire."""
    folder, _, manifest = _location(row)
    stem = pathlib.Path(manifest).stem
    for path in (*(INGESTION_ERRORS / folder / f"{stem}{ext}" for ext in (".json", ".md", ".txt")),
                 CONVERTER_ERRORS / folder / f"{stem}.json"):
        path.unlink(missing_ok=True)
    _drop_dead_letters(row)


@bp.route("", methods=["GET"])
@require_permission(PERM_UPLOAD)
def list_uploads():
    """I caricamenti dell'utente: ?status=errori|in_lavorazione|indicizzati|tutti (predefinito tutti)."""
    wanted = request.args.get("status", "tutti")
    if wanted != "tutti" and wanted not in LIST_FILTERS:
        raise ApiError(400, f"status deve essere uno di: {', '.join([*LIST_FILTERS, 'tutti'])}")
    where = ["uploaded_by = %s", f"(status <> '{INDEXED}' OR updated_at >= NOW() - INTERVAL %s SECOND)"]
    params = [g.user["username"], INDEXED_DAYS * 86400]
    topics = g.user["topics"]
    if topics is not None:
        if not topics:
            return jsonify({"max_mb": settings.upload_max_mb, "indexed_days": INDEXED_DAYS,
                            "counts": dict.fromkeys(LIST_FILTERS, 0), "items": []})
        where.append(f"topic_id IN ({', '.join(['%s'] * len(topics))})")
        params += topics
    with Db() as db:
        totals = {r["status"]: r["n"] for r in db.all(
            f"SELECT status, COUNT(*) AS n FROM document_uploads WHERE {' AND '.join(where)} GROUP BY status", tuple(params))}
        if wanted != "tutti":
            where.append(f"status IN ({', '.join(['%s'] * len(LIST_FILTERS[wanted]))})")
            params += LIST_FILTERS[wanted]
        rows = db.all(f"SELECT * FROM document_uploads WHERE {' AND '.join(where)} "
                      f"ORDER BY updated_at DESC, id DESC LIMIT {LIST_LIMIT}", tuple(params))
    counts = {name: sum(totals.get(st, 0) for st in states) for name, states in LIST_FILTERS.items()}
    return jsonify({"max_mb": settings.upload_max_mb, "indexed_days": INDEXED_DAYS, "counts": counts, "items": [
        {"id": r["id"], "topic_id": r["topic_id"], "sub_topic_id": r["sub_topic_id"], "file_name": r["file_name"],
         "size_bytes": r["size_bytes"], "status": r["status"], "error": r["error"], "uploaded_by": r["uploaded_by"],
         "created_at": _iso(r["created_at"]), "updated_at": _iso(r["updated_at"])} for r in rows
    ]})


@bp.route("", methods=["POST"])
@require_permission(PERM_UPLOAD)
def create_upload():
    topic_id = (request.form.get("topic_id") or "").strip()
    sub_topic_id = (request.form.get("sub_topic_id") or "").strip()
    replace = request.form.get("replace") == "true"
    upload = request.files.get("file")
    if not upload:
        raise ApiError(422, "Nessun file ricevuto")
    _check_topic(g.user, topic_id)
    name = _safe_name(upload.filename)
    folder = _safe_folder(request.form.get("folder"))

    with Db() as db:
        if not db.one("SELECT 1 AS ok FROM sub_topics WHERE topic_id = %s AND sub_topic_id = %s",
                      (topic_id, sub_topic_id)):
            raise ApiError(422, "Sottocategoria inesistente per questo archivio")

    target_dir = STAGING_FOLDER / topic_id / sub_topic_id / folder
    target_dir.mkdir(parents=True, exist_ok=True)
    temp_path = target_dir / f".upload-{uuid.uuid4().hex}"
    written = []
    try:
        sha256 = _receive(upload, temp_path)
        final_name = _check_content(temp_path, name)
        rel_name = f"{folder}/{final_name}" if folder else final_name
        if len(rel_name) > MAX_FILE_NAME:
            raise ApiError(422, f"Percorso troppo lungo (massimo {MAX_FILE_NAME} caratteri): '{rel_name}'")
        source = f"direct://{topic_id}/{sub_topic_id}/{rel_name}"
        file_path = target_dir / final_name
        manifest_path = target_dir / f"{pathlib.Path(final_name).stem}.json"

        with Db() as db:
            known = _check_duplicates(db, topic_id, sub_topic_id, source, sha256, replace)
            if manifest_path.exists():
                raise ApiError(409, f"Un file di nome '{manifest_path.stem}' è ancora in lavorazione")
            size = temp_path.stat().st_size
            manifest = {"source": source, "files": [rel_name],
                        "metadati": {"percorso_originale": f"{topic_id}/{sub_topic_id}/{rel_name}"}}
            temp_manifest = target_dir / f".upload-{uuid.uuid4().hex}"
            written.append(temp_manifest)
            temp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            # Il converter trova solo file completi: prima si scrive, poi si rinomina.
            temp_path.replace(file_path)
            temp_manifest.replace(manifest_path)
            written = [file_path, manifest_path]
            values = (rel_name, size, sha256, QUEUED, g.user["username"])
            if known:
                db.execute("UPDATE document_uploads SET file_name = %s, size_bytes = %s, sha256 = %s, status = %s, "
                           "error = NULL, uploaded_by = %s WHERE source = %s", (*values, source))
            else:
                db.execute("INSERT INTO document_uploads (topic_id, sub_topic_id, source, file_name, size_bytes, "
                           "sha256, status, uploaded_by) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                           (topic_id, sub_topic_id, source, *values))
            db.commit()
            row = db.one("SELECT * FROM document_uploads WHERE source = %s", (source,))
            status = QUEUED
            if known:
                _forget_failure(row)
            if not _send(row):
                status = NOT_SENT
                db.execute("UPDATE document_uploads SET status = %s WHERE id = %s", (status, row["id"]))
                db.commit()
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    finally:
        temp_path.unlink(missing_ok=True)

    log.info(f"{g.user['username']}: caricato {source} ({size} byte), stato {status}")
    return jsonify({"id": row["id"], "file_name": rel_name, "status": status}), 201


@bp.route("/<int:upload_id>/send", methods=["POST"])
@require_permission(PERM_UPLOAD)
def resend_upload(upload_id):
    with Db() as db:
        row = db.one("SELECT * FROM document_uploads WHERE id = %s", (upload_id,))
        if not row:
            raise ApiError(404, "Caricamento non trovato")
        _check_topic(g.user, row["topic_id"])
        if row["status"] != NOT_SENT:
            raise ApiError(409, "Questo file non è da inviare")
        if not _send(row):
            raise ApiError(503, "Coda non raggiungibile: riprova tra poco")
        db.execute("UPDATE document_uploads SET status = %s WHERE id = %s", (QUEUED, upload_id))
        db.commit()
    return jsonify({"status": QUEUED})


@bp.route("/<int:upload_id>/retry", methods=["POST"])
@require_permission(PERM_UPLOAD)
def retry_upload(upload_id):
    with Db() as db:
        row = db.one("SELECT * FROM document_uploads WHERE id = %s", (upload_id,))
        if not row:
            raise ApiError(404, "Caricamento non trovato")
        _check_topic(g.user, row["topic_id"])
        if row["status"] != FAILED:
            raise ApiError(409, "Questo file non è in errore")
        plan = _retry_plan(row)
        if not plan:
            raise ApiError(409, "I file di questo documento non sono più disponibili: va caricato di nuovo")
        queue, payload, moves = plan
        if any(target.exists() for _, target in moves):
            raise ApiError(409, "Un file con lo stesso nome è in lavorazione: riprova tra poco")
        # Solo il primo di due clic contemporanei passa da qui.
        if not db.execute("UPDATE document_uploads SET status = %s, error = NULL WHERE id = %s AND status = %s",
                          (QUEUED, upload_id, FAILED)):
            raise ApiError(409, "Questo file non è in errore")
        db.commit()
        done = []
        try:
            for source, target in moves:
                target.parent.mkdir(parents=True, exist_ok=True)
                source.replace(target)
                done.append((source, target))
            publish_to_queue(queue, payload)
        except Exception as e:
            log.error(f"Nuovo tentativo fallito per {row['source']}: {e}")
            for source, target in reversed(done):
                target.replace(source)
            db.execute("UPDATE document_uploads SET status = %s, error = %s WHERE id = %s",
                       (FAILED, row["error"], upload_id))
            db.commit()
            raise ApiError(503, "Nuovo tentativo non riuscito: riprova tra poco")
    _drop_dead_letters(row)
    log.info(f"{g.user['username']}: nuovo tentativo per {row['source']} su {queue}")
    return jsonify({"status": QUEUED})
