"""Test del caricamento dei documenti: SQLite al posto di MySQL, coda simulata.

Uso:  python api/tests/test_upload.py      (esce con 1 se qualcosa fallisce)
"""
import io, json, os, pathlib, shutil, sqlite3, sys, tempfile, time

TMP = pathlib.Path(tempfile.mkdtemp())
os.environ.update(AUTH_SECRET_KEY="x" * 40, API_SECRET_KEY="k", DATA_FOLDER=str(TMP), DOCWS_RICERCA_ENDPOINT="u",
                  DOCWS_ATTI_ENDPOINT="u", DOCWS_CODICE_AMMINISTRAZIONE="u", DOCWS_CODICE_AOO="u", RUOLO_DOCWS="u",
                  LOG_LEVEL="CRITICAL", UPLOAD_MAX_MB="1")
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "api" / "src"), str(ROOT)]
DB_PATH = str(TMP / "db.sqlite")


class Cur:
    def __init__(self, c, d): self.c, self.d, self.lastrowid = c.cursor(), d, None
    def execute(self, sql, p=()):
        sql = sql.replace("NOW() - INTERVAL %s SECOND", "datetime(CURRENT_TIMESTAMP, '-' || %s || ' seconds')")
        self.c.execute(sql.replace("%s", "?").replace("NOW()", "CURRENT_TIMESTAMP"), p)
        self.rowcount, self.lastrowid = self.c.rowcount, self.c.lastrowid
    def _row(self, r): return dict(zip([x[0] for x in self.c.description], r)) if self.d else r
    def fetchone(self):
        r = self.c.fetchone(); return self._row(r) if r else None
    def fetchall(self): return [self._row(r) for r in self.c.fetchall()] if self.c.description else []
    def close(self): pass


class Conn:
    def __init__(self): self.c = sqlite3.connect(DB_PATH)
    def cursor(self, dictionary=False): return Cur(self.c, dictionary)
    def commit(self): self.c.commit()
    def rollback(self): self.c.rollback()
    def close(self): self.c.close()
    def is_connected(self): return True
    def ping(self, **k): pass


import common.db_logger as dbl
dbl.init_db_pool = lambda: None
dbl.get_db_connection = lambda: Conn()

c = sqlite3.connect(DB_PATH)
c.executescript("""
CREATE TABLE topics (topic_id TEXT PRIMARY KEY, description TEXT, public_access INT DEFAULT 0);
CREATE TABLE sub_topics (topic_id TEXT, sub_topic_id TEXT, description TEXT);
CREATE TABLE parent_documents (source TEXT, topic_id TEXT, sub_topic_id TEXT);
CREATE TABLE document_uploads (id INTEGER PRIMARY KEY, topic_id TEXT, sub_topic_id TEXT, source TEXT UNIQUE, file_name TEXT,
  size_bytes INT, sha256 TEXT, status TEXT, error TEXT, uploaded_by TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
INSERT INTO topics VALUES ('atti', 'Atti', 0), ('altro', 'Altro', 0);
INSERT INTO sub_topics VALUES ('atti', 'determine', 'Determine'), ('altro', 'x', 'X');
INSERT INTO parent_documents VALUES ('direct://atti/determine/vecchio.pdf', 'atti', 'determine');
""")
c.commit()

import app as api_app
import review_routes as rr
import upload_routes as ur
from common import upload_status as us

USERS = {"arch": (["archivista"], ["atti"]), "rev": (["revisore"], ["atti"]), "capo": (["admin"], [])}
c.executescript("""
CREATE TABLE review_users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, display_name TEXT, password_hash TEXT, active INT DEFAULT 1,
  token_version INT DEFAULT 1, created_at TIMESTAMP, last_login_at TIMESTAMP);
CREATE TABLE review_user_roles (user_id INT, role TEXT);
CREATE TABLE review_user_topics (user_id INT, topic_id TEXT);
""")
H = {}
for i, (name, (roles, topics)) in enumerate(USERS.items(), 1):
    c.execute("INSERT INTO review_users (id, username, password_hash) VALUES (?, ?, 'x')", (i, name))
    c.executemany("INSERT INTO review_user_roles VALUES (?, ?)", [(i, r) for r in roles])
    c.executemany("INSERT INTO review_user_topics VALUES (?, ?)", [(i, t) for t in topics])
    H[name] = {"Authorization": "Bearer " + rr._auth_serializer.dumps({"uid": i, "tv": 1})}
c.commit()

sent = []
send_fails = False


def fake_publish(queue, payload, *a, **k):
    if send_fails:
        raise OSError("broker giù")
    sent.append((queue, payload))


ur.publish_to_queue = fake_publish
dropped = []
ur._drop_dead_letters = lambda row: dropped.append(row["file_name"])
cl = api_app.app.test_client()
failed = 0


def check(cond, label):
    global failed
    failed += not cond
    print(("OK  " if cond else "FAIL"), label)


PDF = b"%PDF-1.4 contenuto di prova"
STAGING = TMP / "staging" / "atti" / "determine"


def post(data=PDF, name="atto.pdf", topic="atti", sub="determine", user="arch", replace=False, folder=""):
    return cl.post("/review/uploads", headers=H[user], content_type="multipart/form-data", data={
        "topic_id": topic, "sub_topic_id": sub, "replace": str(replace).lower(), "folder": folder,
        "file": (io.BytesIO(data), name)})


def finish(source="direct://atti/determine/atto.pdf", status=us.INDEXED):
    """Come farebbero converter e ingestione: consumano i file e aggiornano lo stato."""
    for f in list(STAGING.glob("*")):
        f.unlink()
    us.mark_upload(source, status, "guasto" if status == us.FAILED else None)


# --- permessi
check(post(user="rev").status_code == 403, "revisore: non carica")
check(post(topic="altro", sub="x").status_code == 403, "archivista: solo i suoi archivi")
check(cl.get("/review/uploads", headers=H["rev"]).status_code == 403, "revisore: niente elenco")
check(post(user="capo", topic="altro", sub="x", name="c.pdf", data=b"%PDF-capo").status_code == 201, "admin: tutti gli archivi")
sent.clear()
check([i["topic_id"] for i in cl.get("/review/uploads", headers=H["arch"]).json["items"]] == [], "elenco: solo i propri caricamenti")
check([i["file_name"] for i in cl.get("/review/uploads", headers=H["capo"]).json["items"]] == ["c.pdf"], "anche l'admin vede solo i suoi")

# --- controlli sul file
check(post(sub="boh").status_code == 422, "sottocategoria inesistente -> 422")
for bad, why in [(("x.exe", PDF), "estensione"), (("../x.pdf", PDF), "percorso"), ((".x.pdf", PDF), "nome nascosto"),
                 (("x.pdf", b"non un pdf"), "contenuto diverso dall'estensione"), (("x.pdf", b""), "vuoto"),
                 (("x.txt", b"\xff\xfe\x00"), "testo non UTF-8"), (("x.p7m", b"non una busta"), "busta non valida")]:
    check(post(data=bad[1], name=bad[0]).status_code == 422, f"rifiutato: {why}")
import zipfile as _zip


def office_zip(**members):
    buffer = io.BytesIO()
    with _zip.ZipFile(buffer, "w") as archive:
        for member, content in members.items():
            archive.writestr(member.replace("__", "/").replace("_dot_", "."), content)
    return buffer.getvalue()


DOCX = office_zip(word__document_dot_xml="<w/>", **{"[Content_Types]_dot_xml": "<t/>"})
ODS = office_zip(mimetype="application/vnd.oasis.opendocument.spreadsheet", content_dot_xml="<c/>")
check(post(data=DOCX, name="finto.xlsx").status_code == 422, "docx chiamato xlsx: rifiutato")
check(post(data=ODS, name="finto.odt").status_code == 422, "ods chiamato odt: rifiutato")
check(post(data=b"PK\x03\x04 rotto", name="rotto.docx").status_code == 422, "archivio rotto: rifiutato")
check(not list(STAGING.glob("*")), "i documenti d'ufficio rifiutati non lasciano tracce")
check(post(data=b"%PDF-" + b"x" * (1024 * 1024)).status_code == 413, "oltre il limite -> 413")
check(not list(STAGING.glob("*")), "i file rifiutati non lasciano tracce")
check(not sent, "nulla in coda")

# --- caricamento
r = post()
check(r.status_code == 201 and r.json["status"] == us.QUEUED, "caricato e in coda")
check(sent == [("da-convertire", {"source_type": "json", "rel_path": "atti/determine/atto.json"})], "messaggio sulla coda del converter")
manifest = json.loads((STAGING / "atto.json").read_text())
check((STAGING / "atto.pdf").read_bytes() == PDF and manifest["source"] == "direct://atti/determine/atto.pdf"
      and manifest["files"] == ["atto.pdf"], "file e manifest in staging")
check(not list(STAGING.glob(".upload-*")), "nessun file temporaneo")
item = cl.get("/review/uploads", headers=H["arch"]).json["items"][0]
check(item["uploaded_by"] == "arch" and item["status"] == us.QUEUED and item["size_bytes"] == len(PDF), "elenco con autore e stato")
check(post(data=PDF + b"2").status_code == 409, "stesso nome ancora in lavorazione -> 409")
check(post(data=b"ciao", name="atto.txt").status_code == 409, "stesso nome con altra estensione in lavorazione -> 409")

# --- duplicati e sostituzione
finish()
check(cl.get("/review/uploads", headers=H["arch"]).json["items"][0]["status"] == us.INDEXED, "stato aggiornato dalla pipeline")
check(post(name="copia.pdf").status_code == 409, "stesso contenuto con altro nome -> 409")
check(post(data=PDF + b"2").status_code == 409, "stesso nome già indicizzato, senza sostituzione -> 409")
check(post(data=PDF + b"2", replace=True).status_code == 201, "sostituzione esplicita")
check(post(name="vecchio.pdf", data=b"%PDF-v").status_code == 409, "nome di un documento non caricato da qui -> 409")
finish(status=us.FAILED)
item = cl.get("/review/uploads", headers=H["arch"]).json["items"][0]
check(item["status"] == us.FAILED and item["error"] == "guasto", "errore visibile")
check(post(data=PDF + b"2").status_code == 201, "un file in errore si ricarica")
finish()

# --- filtri dell'elenco
def listed(status, user="arch"):
    r = cl.get("/review/uploads", headers=H[user], query_string={"status": status})
    return r.json if r.status_code == 200 else r.status_code


check(listed("boh"), 400)
everything = listed("tutti")
check([i["status"] for i in listed("indicizzati")["items"]] == [us.INDEXED], "indicizzati")
check(listed("errori")["items"] == [] and listed("in_lavorazione")["items"] == [], "nessun errore né lavorazione")
check(everything["counts"] == {"errori": 0, "in_lavorazione": 0, "indicizzati": 1} and everything["indexed_days"] == 3, "conteggi per filtro")
post(name="lento.pdf", data=b"%PDF-lento")
us.mark_upload("direct://atti/determine/lento.pdf", us.PROCESSING)
check([i["file_name"] for i in listed("in_lavorazione")["items"]] == ["lento.pdf"], "in lavorazione")
us.mark_upload("direct://atti/determine/lento.pdf", us.FAILED, "guasto")
check([i["file_name"] for i in listed("errori")["items"]] == ["lento.pdf"], "errori")
c.execute("UPDATE document_uploads SET status = 'da_inviare' WHERE file_name = 'lento.pdf'")
c.commit()
check([i["file_name"] for i in listed("errori")["items"]] == ["lento.pdf"], "da inviare: fra gli errori, ha Riprova")
c.execute("UPDATE document_uploads SET updated_at = datetime(CURRENT_TIMESTAMP, '-10 days')")
c.commit()
check(listed("indicizzati")["items"] == [], "indicizzati più vecchi di 3 giorni: fuori")
check([i["file_name"] for i in listed("errori")["items"]] == ["lento.pdf"], "un errore vecchio resta in elenco")
check(listed("tutti")["counts"] == {"errori": 1, "in_lavorazione": 0, "indicizzati": 0}, "conteggi dopo 3 giorni")
c.execute("UPDATE document_uploads SET updated_at = CURRENT_TIMESTAMP")
c.commit()
c.execute("DELETE FROM document_uploads WHERE file_name = 'lento.pdf'")
c.commit()
for f in list(STAGING.glob("*")):
    f.unlink()
sent.clear()

for upload_name, upload_data in (("relazione.docx", DOCX), ("tabella.ods", ODS)):
    r = post(name=upload_name, data=upload_data)
    check(r.status_code == 201 and (STAGING / upload_name).read_bytes() == upload_data, f"caricato {upload_name}")
    finish(source=f"direct://atti/determine/{upload_name}")
sent.clear()

c.execute("DELETE FROM document_uploads WHERE file_name IN ('relazione.docx', 'tabella.ods')")
c.commit()

# --- coda non raggiungibile
send_fails = True
r = post(name="n.txt", data="testo è valido".encode())
check(r.status_code == 201 and r.json["status"] == us.NOT_SENT, "coda giù: resta da inviare")
upload_id = r.json["id"]
check(cl.post(f"/review/uploads/{upload_id}/send", headers=H["arch"]).status_code == 503, "invio fallito -> 503")
send_fails = False
check(cl.post(f"/review/uploads/{upload_id}/send", headers=H["arch"]).json["status"] == us.QUEUED, "nuovo invio riuscito")
check(cl.post(f"/review/uploads/{upload_id}/send", headers=H["arch"]).status_code == 409, "già inviato -> 409")
check(cl.post(f"/review/uploads/{upload_id}/send", headers=H["rev"]).status_code == 403, "invio: serve il permesso")

# --- nuovo tentativo
CONV_ERR = TMP / "converter" / "error" / "atti" / "determine"
ING_ERR = TMP / "ingestion" / "error" / "atti" / "determine"
ARCHIVE = TMP / "processed" / "atti" / "determine"
WATCH = TMP / "watch" / "atti" / "determine"
for folder in (CONV_ERR, ING_ERR, ARCHIVE):
    folder.mkdir(parents=True, exist_ok=True)


def failed_upload(name):
    """Un file caricato e fallito: restano da sistemare i suoi file, come farebbe la pipeline."""
    upload_id = post(name=name, data=f"%PDF-{name}".encode()).json["id"]
    (STAGING / f"{pathlib.Path(name).stem}.json").unlink()
    (STAGING / name).unlink()
    us.mark_upload(f"direct://atti/determine/{name}", us.FAILED, "guasto")
    sent.clear()
    return upload_id


def retry(upload_id, user="arch"):
    return cl.post(f"/review/uploads/{upload_id}/retry", headers=H[user])


def status_of(upload_id):
    return next(i for i in cl.get("/review/uploads", headers=H["arch"]).json["items"] if i["id"] == upload_id)


# Fermo al converter: il manifest è fra gli errori, il file ancora in staging.
rid = failed_upload("conv.pdf")
(CONV_ERR / "conv.json").write_text("{}")
(STAGING / "conv.pdf").write_bytes(PDF)
check(retry(rid, "rev").status_code == 403, "nuovo tentativo: serve il permesso")
check(retry(999).status_code == 404, "nuovo tentativo: file inesistente -> 404")
r = retry(rid)
check(r.status_code == 200 and r.json["status"] == us.QUEUED, "converter: file di nuovo in coda")
check(sent == [("da-convertire", {"source_type": "json", "rel_path": "atti/determine/conv.json"})], "converter: messaggio sulla sua coda")
check((STAGING / "conv.json").exists() and not (CONV_ERR / "conv.json").exists(), "converter: manifest tornato in staging")
check(status_of(rid)["error"] is None and dropped[-1] == "conv.pdf", "errore cancellato e DLQ ripulita")
check(retry(rid).status_code == 409, "un file non in errore non si riprova")
finish("direct://atti/determine/conv.pdf")

# Fermo all'indicizzazione: il testo c'è già e l'OCR non si rifà.
rid = failed_upload("ing.pdf")
(ING_ERR / "ing.json").write_text("{}")
(ING_ERR / "ing.md").write_text("testo")
(CONV_ERR / "ing.json").write_text("{}")
sent.clear()
r = retry(rid)
check(sent == [("da-indicizzare", {"json_manifest_path": "atti/determine/ing.json"})], "indicizzazione: riparte da lì")
check((WATCH / "ing.md").read_text() == "testo" and (WATCH / "ing.json").exists() and not (ING_ERR / "ing.md").exists(),
      "indicizzazione: testo e manifest tornati in watch")
(WATCH / "ing.md").unlink()
(WATCH / "ing.json").unlink()
(CONV_ERR / "ing.json").unlink()
us.mark_upload("direct://atti/determine/ing.pdf", us.INDEXED)

# Nessun testo ricavato: il converter ha archiviato tutto.
rid = failed_upload("arc.pdf")
(ARCHIVE / "arc.json").write_text("{}")
(ARCHIVE / "arc.pdf").write_bytes(PDF)
r = retry(rid)
check(r.status_code == 200 and sent[-1][0] == "da-convertire" and (STAGING / "arc.pdf").exists()
      and (STAGING / "arc.json").exists() and not (ARCHIVE / "arc.pdf").exists(), "archiviato: torna in staging")
finish("direct://atti/determine/arc.pdf")

# Casi in cui non si può ripartire: lo stato non cambia.
rid = failed_upload("via.pdf")
r = retry(rid)
check(r.status_code == 409 and "caricato di nuovo" in r.json["error"], "file spariti -> 409 con spiegazione")
check(status_of(rid)["status"] == us.FAILED and status_of(rid)["error"] == "guasto", "file spariti: errore invariato")
us.mark_upload("direct://atti/determine/via.pdf", us.INDEXED)

rid = failed_upload("doppio.pdf")
(CONV_ERR / "doppio.json").write_text("{}")
(STAGING / "doppio.pdf").write_bytes(PDF)
(STAGING / "doppio.json").write_text("altro")
check(retry(rid).status_code == 409 and (STAGING / "doppio.json").read_text() == "altro", "stesso nome in lavorazione: non sovrascrive")
(STAGING / "doppio.json").unlink()
send_fails = True
r = retry(rid)
send_fails = False
check(r.status_code == 503 and (CONV_ERR / "doppio.json").exists() and not (STAGING / "doppio.json").exists(), "coda giù: file rimessi a posto")
check(status_of(rid)["status"] == us.FAILED and status_of(rid)["error"] == "guasto", "coda giù: errore ancora visibile")

# Una nuova versione non deve far ripartire il testo vecchio.
rid = failed_upload("nuovo.pdf")
(ING_ERR / "nuovo.json").write_text("{}")
(ING_ERR / "nuovo.md").write_text("testo vecchio")
dropped.clear()
check(post(name="nuovo.pdf", data=b"%PDF-versione2").status_code == 201, "file in errore ricaricato")
check(not (ING_ERR / "nuovo.md").exists() and not (ING_ERR / "nuovo.json").exists() and dropped == ["nuovo.pdf"],
      "ricaricato: tolti i resti del tentativo fallito")

# --- sottocartelle di origine
FOLDER = "faldone 1/fascicolo.3"
r = post(name="scan.pdf", data=b"%PDF-scan", folder=FOLDER)
check(r.status_code == 201, "caricato in sottocartella")
check(r.json["file_name"] == f"{FOLDER}/scan.pdf", "percorso nella risposta")
fid = r.json["id"]
check(sent[-1] == ("da-convertire", {"source_type": "json", "rel_path": f"atti/determine/{FOLDER}/scan.json"}), "messaggio con il percorso")
manifest = json.loads((STAGING / FOLDER / "scan.json").read_text())
check(manifest["source"] == f"direct://atti/determine/{FOLDER}/scan.pdf" and manifest["files"] == [f"{FOLDER}/scan.pdf"]
      and manifest["metadati"]["percorso_originale"] == f"atti/determine/{FOLDER}/scan.pdf"
      and (STAGING / FOLDER / "scan.pdf").exists(), "manifest come direct.py, file nella sua cartella")
check(post(name="scan.pdf", data=b"%PDF-altro", folder="faldone 2").status_code == 201, "stesso nome in altra cartella: ammesso")
for bad in ("../fuori", "a/../../b", "a//b", ".nascosta", "a\\b", "a/ b", "x" * 101):
    check(post(name="v.pdf", data=b"%PDF-v2", folder=bad).status_code == 422, f"cartella rifiutata: {bad[:12]}")
check(post(name="v.pdf", data=b"%PDF-v3", folder="/".join(["y" * 90] * 3)).status_code == 422, "percorso oltre 255 caratteri")
check(not (TMP / "fuori").exists() and not list(TMP.rglob("v.pdf")), "le cartelle rifiutate non lasciano file")

(STAGING / FOLDER / "scan.json").unlink()
(STAGING / FOLDER / "scan.pdf").unlink()
us.mark_upload(f"direct://atti/determine/{FOLDER}/scan.pdf", us.FAILED, "guasto")
(ING_ERR / FOLDER).mkdir(parents=True)
(ING_ERR / FOLDER / "scan.json").write_text("{}")
(ING_ERR / FOLDER / "scan.md").write_text("testo")
sent.clear()
check(retry(fid).status_code == 200 and sent == [("da-indicizzare", {"json_manifest_path": f"atti/determine/{FOLDER}/scan.json"})]
      and (WATCH / FOLDER / "scan.md").exists(), "nuovo tentativo: riparte dalla sua cartella")

# --- OCR esterno su richiesta
import external_ocr_routes as vr
from common.utility import EXTERNAL_OCR_MARKER
vr._load_document = lambda db, s, t, st: ({"file_name": DOC_FILE}, None, None)
vr._require_lock = lambda *a: None
vr.publish_to_queue = fake_publish
DOC_FILE = "f/orig.pdf"
KEY = {"source": "direct://atti/determine/f/orig.pdf", "topic_id": "atti", "sub_topic_id": "determine"}
marker = TMP / EXTERNAL_OCR_MARKER


def vision(user="arch", key=KEY):
    return cl.post("/review/document/external-ocr", headers=H[user], json=key)


sent.clear()
r = vision()
check(r.status_code == 409 and "non è configurato" in r.json["error"] and not sent, "non configurato: nulla esce")
marker.parent.mkdir(parents=True, exist_ok=True)
marker.touch()
check(vision().status_code == 409, "file originale assente -> 409")
(ARCHIVE / "f").mkdir()
(ARCHIVE / "f" / "orig.pdf").write_bytes(PDF)
check(vision(user="capo").status_code == 202, "admin: avviato")


def ocr_state(user="arch", key=KEY):
    return cl.get("/review/document/external-ocr", headers=H[user], query_string=key)


def ocr_remove(user="arch", key=KEY):
    return cl.delete("/review/document/external-ocr", headers=H[user], query_string=key)


job_dir = vr._job_dir(KEY["source"], KEY["topic_id"], KEY["sub_topic_id"])
check(ocr_state(user="capo").json["status"] == "queued" and json.loads((job_dir / "request.json").read_text())["requested_by"], "lavoro dell'admin visibile")
sent.clear()
r = vision()
check(r.status_code == 200 and r.json["status"] == "queued" and not sent, "in corso: un altro clic non riparte")
request_data = json.loads((job_dir / "request.json").read_text())
check(request_data["external_ocr"] is True and (job_dir / request_data["input_file"]).read_bytes() == PDF, "copia dell'originale per il converter")
check((ARCHIVE / "f" / "orig.pdf").exists(), "l'originale resta in archivio")
check(ocr_remove().status_code == 409 and job_dir.exists(), "in corso: non si toglie")
(job_dir / "status.json").write_text(json.dumps({"status": "running"}))
os.utime(job_dir, (time.time() - vr.STALE_SECONDS - 1,) * 2)
check(ocr_state().json["status"] == "failed", "fermo da troppo: si può rifare")
sent.clear()
check(vision().status_code == 202 and len(sent) == 1, "lavoro fermo: si rifà")
check(sent[-1][1] == {"job_id": job_dir.name} and sent[-1][0] == "da-convertire-api", "lavoro in coda")
(job_dir / "result.md").write_text("testo esterno")
(job_dir / "status.json").write_text(json.dumps({"status": "completed"}))
os.utime(job_dir, (time.time() - vr.STALE_SECONDS - 1,) * 2)
r = ocr_state()
check(r.json["status"] == "completed" and r.json["content"] == "testo esterno", "rientrando il testo c'è, anche dopo ore")
sent.clear()
r = vision()
check(r.status_code == 200 and r.json["content"] == "testo esterno" and not sent, "pronto: un altro clic non ripaga")
OTHER = {**KEY, "source": "direct://atti/determine/f/altro.pdf"}
check(ocr_state(key=OTHER).json["status"] == "none", "altro documento: nessun lavoro")
check(ocr_state(key={**KEY, "topic_id": "altro", "sub_topic_id": "x"}).status_code == 403, "stato in archivio altrui -> 403")
check(ocr_remove(key={**KEY, "topic_id": "altro", "sub_topic_id": "x"}).status_code == 403 and job_dir.exists(), "togliere in archivio altrui -> 403")
check(vision(key={**KEY, "topic_id": "altro", "sub_topic_id": "x"}).status_code == 403, "archivio altrui -> 403")
check(ocr_remove().status_code == 200 and not job_dir.exists() and ocr_state().json["status"] == "none", "scartata")
(job_dir).mkdir()
(job_dir / "request.json").write_text("{}")
(job_dir / "status.json").write_text(json.dumps({"status": "failed", "error": "rifiutata"}))
check(ocr_state().json["error"] == "rifiutata", "errore del lavoro riportato")
sent.clear()
check(vision().status_code == 202 and len(sent) == 1, "fallita: si può riprovare")
check(ocr_remove().status_code == 409, "riprova in corso")
shutil.rmtree(job_dir)
DOC_FILE = "f/orig.txt"
(ARCHIVE / "f" / "orig.txt").write_text("x")
check(vision().status_code == 422, "testo: non si manda all'esterno")
send_fails = True
DOC_FILE = "f/orig.pdf"
before = set(vr.CONVERSIONS_FOLDER.iterdir())
check(vision().status_code == 503 and set(vr.CONVERSIONS_FOLDER.iterdir()) == before, "coda giù: nessun residuo")
send_fails = False

# --- errori da non ritentare
from common.utility import is_permanent_error


class StatusError(Exception):
    def __init__(self, status_code=None, code=None): self.status_code, self.code = status_code, code


check(all(is_permanent_error(StatusError(s)) for s in (400, 401, 403, 404, 422)), "4xx: permanente")
check(is_permanent_error(StatusError(code=404)), "4xx nel campo code (Gemini): permanente")
check(not any(is_permanent_error(StatusError(s)) for s in (408, 429, 500, 503)), "rate limit, timeout e 5xx: si ritenta")
wrapped = Exception("incapsulato")
wrapped.__cause__ = StatusError(code=404)
check(is_permanent_error(wrapped), "errore incapsulato: conta lo stato originale")
wrapped.__cause__ = StatusError(code=429)
check(not is_permanent_error(wrapped), "rate limit incapsulato: si ritenta")
check(not is_permanent_error(StatusError(code="rate_limit")) and not is_permanent_error(OSError("rete")), "senza stato numerico: si ritenta")

print("\nRISULTATO:", "TUTTO OK" if not failed else f"{failed} FALLITI")
sys.exit(1 if failed else 0)
