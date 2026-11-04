"""Test delle API per i sistemi esterni: SQLite al posto di MySQL, RabbitMQ intercettato.

Uso:  python api/tests/test_external.py      (esce con 1 se qualcosa fallisce)
"""
import hashlib, io, json, os, pathlib, re, sqlite3, sys, tempfile

TMP = pathlib.Path(tempfile.mkdtemp())
os.environ.update(AUTH_SECRET_KEY="x" * 40, API_SECRET_KEY="chiave-chat", DATA_FOLDER=str(TMP), DOCWS_RICERCA_ENDPOINT="u",
                  DOCWS_ATTI_ENDPOINT="u", DOCWS_CODICE_AMMINISTRAZIONE="u", DOCWS_CODICE_AOO="u", RUOLO_DOCWS="u",
                  CONVERT_MAX_UPLOAD_MB="1", LOG_LEVEL="CRITICAL")
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "api" / "src"), str(ROOT)]
DB_PATH = str(TMP / "db.sqlite")


def translate(sql):
    sql = sql.replace("%s", "?")
    sql = re.sub(r"NOW\(\) - INTERVAL \? SECOND", "datetime(CURRENT_TIMESTAMP, '-' || ? || ' seconds')", sql)
    sql = sql.replace("NOW() - INTERVAL 60 SECOND", "datetime(CURRENT_TIMESTAMP, '-60 seconds')")
    return sql.replace("NOW()", "CURRENT_TIMESTAMP").replace("active = TRUE", "active = 1")


class Cur:
    def __init__(self, c, d): self.c, self.d = c.cursor(), d
    def execute(self, sql, p=()): self.c.execute(translate(sql), p); self.rowcount = self.c.rowcount
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


import common.db_logger as dbl  # noqa: E402
dbl.init_db_pool = lambda: None
dbl.get_db_connection = lambda: Conn()

c = sqlite3.connect(DB_PATH)
c.executescript("""
CREATE TABLE topics (topic_id TEXT PRIMARY KEY, description TEXT, public_access INT DEFAULT 0);
CREATE TABLE sub_topics (topic_id TEXT, sub_topic_id TEXT, description TEXT);
CREATE TABLE parent_documents (id TEXT, topic_id TEXT, sub_topic_id TEXT, source TEXT, file_name TEXT,
  parent_index INT, content TEXT, metadata TEXT);
CREATE TABLE external_api_keys (id INTEGER PRIMARY KEY, name TEXT, key_hash TEXT, key_prefix TEXT, active INT DEFAULT 1,
  created_at TIMESTAMP, expires_at TIMESTAMP, last_used_at TIMESTAMP);
CREATE TABLE external_api_key_topics (key_id INT, topic_id TEXT);
INSERT INTO topics VALUES ('atti', 'Atti', 0), ('lettere', 'Lettere', 1), ('riservato', 'Altro', 0);
INSERT INTO sub_topics VALUES ('atti', 'determine', 'Determine'), ('lettere', 'corrispondenza', 'Corrispondenza');
""")
META = {"oggetto": "Lavori", "numero": "12", "data": "2025-03-01", "Header 1": "Determina", "id_sicraweb": "99",
        "source": "x", "_ingestion_id": "y", "vuoto": "", "dirigente_descrizione": "Rossi"}
c.executemany("INSERT INTO parent_documents VALUES (?,?,?,?,?,?,?,?)", [
    ("p0", "atti", "determine", "sicraweb://1::atto uno.pdf", "atto uno.pdf", 0, "Prima parte", json.dumps(META)),
    ("p1", "atti", "determine", "sicraweb://1::atto uno.pdf", "atto uno.pdf", 1, "Seconda parte", json.dumps(META)),
    ("x0", "riservato", "s", "direct://z", "z.pdf", 0, "Segreto", "{}"),
])


def add_key(i, name, topics=(), active=1, expires=None):
    c.execute("INSERT INTO external_api_keys (id, name, key_hash, key_prefix, active, expires_at) VALUES (?,?,?,?,?,?)",
              (i, name, hashlib.sha256(name.encode()).hexdigest(), name[:4], active, expires))
    c.executemany("INSERT INTO external_api_key_topics VALUES (?,?)", [(i, t) for t in topics])


add_key(1, "chiave-atti", ["atti", "lettere"])
add_key(2, "chiave-nessun-topic")
add_key(3, "chiave-revocata", ["atti"], active=0)
add_key(4, "chiave-scaduta", ["atti"], expires="2000-01-01 00:00:00")
c.commit()

import app as api_app  # noqa: E402
import external_routes as ex  # noqa: E402
published = []
ex.publish_to_queue = lambda q, payload, args=None: published.append((q, payload))
cl = api_app.app.test_client()
H = lambda k: {"Authorization": f"Bearer {k}"}
BASE = "/external/v1"
FAILED = []


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond: FAILED.append(msg)


# --- credenziali
check(cl.get(f"{BASE}/topics").status_code == 401, "senza credenziale -> 401")
check(cl.get(f"{BASE}/topics", headers=H("sbagliata")).status_code == 401, "credenziale inesistente -> 401")
check(cl.get(f"{BASE}/topics", headers=H("chiave-revocata")).status_code == 401, "credenziale revocata -> 401")
check(cl.get(f"{BASE}/topics", headers=H("chiave-scaduta")).status_code == 401, "credenziale scaduta -> 401")
check(cl.get(f"{BASE}/topics", headers=H("chiave-chat")).status_code == 401, "la chiave della chat non apre le API esterne")
check(cl.get("/config", headers=H("chiave-atti")).status_code in (401, 405), "la chiave esterna non apre le rotte della chat")

# --- topic
r = cl.get(f"{BASE}/topics", headers=H("chiave-atti"))
check([t["id"] for t in r.json["topics"]] == ["atti", "lettere"], "lista dei soli topic abilitati")
check(r.json["topics"][0]["sub_topics"] == [{"id": "determine", "description": "Determine"}], "topic con le sottocategorie")
check(cl.get(f"{BASE}/topics", headers=H("chiave-nessun-topic")).json["topics"] == [], "chiave senza topic: lista vuota")

# --- documenti
check(cl.get(f"{BASE}/topics/riservato/documents", headers=H("chiave-atti")).status_code == 403, "lista di un topic non abilitato -> 403")
r = cl.get(f"{BASE}/topics/atti/documents", headers=H("chiave-atti"))
check(r.json["total"] == 1 and len(r.json["items"]) == 1, "un documento anche se ha più parent")
item = r.json["items"][0]
check(item["metadata"] == {"oggetto": "Lavori", "numero": "12", "data": "2025-03-01", "dirigente_descrizione": "Rossi"},
      f"solo i metadati utili: {sorted(item['metadata'])}")
check(re.fullmatch(r"[A-Za-z0-9_-]+", item["id"]) is not None, "id utilizzabile in un indirizzo, anche con spazi e :: nel source")
check(cl.get(f"{BASE}/topics/atti/documents?page=x", headers=H("chiave-atti")).status_code == 400, "pagina non valida -> 400")
check(cl.get(f"{BASE}/topics/atti/documents?sub_topic_id=altro", headers=H("chiave-atti")).json["total"] == 0, "filtro per sottocategoria")

r = cl.get(f"{BASE}/topics/atti/documents/{item['id']}", headers=H("chiave-atti"))
check(r.json["content"] == "Prima parte\n\nSeconda parte", "testo del documento dal database, in ordine")
check(r.json["metadata"] == item["metadata"], "stessi metadati della lista")
check(cl.get(f"{BASE}/topics/atti/documents/nonvalido!", headers=H("chiave-atti")).status_code == 404, "id non valido -> 404")
other = ex._document_id("s", "direct://z")
check(cl.get(f"{BASE}/topics/riservato/documents/{other}", headers=H("chiave-atti")).status_code == 403, "documento di un topic non abilitato -> 403")
check(cl.get(f"{BASE}/topics/atti/documents/{other}", headers=H("chiave-atti")).status_code == 404,
      "un id di un altro topic non apre il documento attraverso un topic abilitato")


# --- conversione: basta una chiave valida
def upload(key="chiave-nessun-topic", name="prova.pdf", data=b"%PDF", **form):
    return cl.post(f"{BASE}/convert", headers=H(key), content_type="multipart/form-data",
                   data={"file": (io.BytesIO(data), name), **form})


r = upload()
check(r.status_code == 202 and published[-1][1]["job_id"] == r.json["job_id"], "chiave senza topic: conversione accettata e accodata")
job = r.json["job_id"]
req = json.loads((TMP / "conversions" / job / "request.json").read_text())
check(req["format"] == "markdown" and not req.get("external_ocr"), "default: Markdown, nessun OCR esterno")
r2 = upload(external_ocr="true", format="text")
req2 = json.loads((TMP / "conversions" / r2.json["job_id"] / "request.json").read_text())
check(not req2.get("external_ocr") and req2["format"] == "text", "l'OCR esterno non si chiede da /convert")
check(upload(format="html").status_code == 400, "formato non valido -> 400")
check(upload(name="virus.exe").status_code == 415, "tipo di file non gestito -> 415")
check(upload(data=b"x" * (2 * 1024 * 1024)).status_code == 413, "documento troppo grande -> 413")
r = upload(name="../../fuori.pdf")
check((TMP / "conversions" / r.json["job_id"] / "input.pdf").exists() and not (TMP / "fuori.pdf").exists(), "il nome del file non può uscire dalla cartella")
check(cl.post(f"{BASE}/convert", headers=H("chiave-revocata")).status_code == 401, "conversione con chiave revocata -> 401")

check(cl.get(f"{BASE}/convert/{job}", headers=H("chiave-nessun-topic")).json["status"] == "queued", "stato del lavoro")
check(cl.get(f"{BASE}/convert/{job}", headers=H("chiave-atti")).status_code == 404, "il lavoro di un'altra chiave non si vede")
check(cl.get(f"{BASE}/convert/..%2f..", headers=H("chiave-atti")).status_code == 404, "id del lavoro non valido -> 404")
d = TMP / "conversions" / job
(d / "result.md").write_text("# Titolo\n\n**testo**")
(d / "status.json").write_text(json.dumps({"status": "completed"}))
check(cl.get(f"{BASE}/convert/{job}", headers=H("chiave-atti")).status_code == 404 and d.exists(),
      "un'altra chiave non lo ritira né lo cancella")
check(cl.get(f"{BASE}/convert/{job}", headers=H("chiave-nessun-topic")).json["content"] == "# Titolo\n\n**testo**", "risultato in Markdown")
check(not d.exists(), "consegnato il risultato, il lavoro è cancellato")
check(cl.get(f"{BASE}/convert/{job}", headers=H("chiave-nessun-topic")).status_code == 404, "si ritira una volta sola")
d2 = TMP / "conversions" / r2.json["job_id"]
(d2 / "result.md").write_text("# Titolo\n\n**testo**")
(d2 / "status.json").write_text(json.dumps({"status": "completed"}))
check(cl.get(f"{BASE}/convert/{r2.json['job_id']}", headers=H("chiave-nessun-topic")).json["content"] == "Titolo\n\ntesto", "risultato in testo semplice")
r3 = upload()
d3 = TMP / "conversions" / r3.json["job_id"]
(d3 / "status.json").write_text(json.dumps({"status": "running"}))
check(cl.get(f"{BASE}/convert/{r3.json['job_id']}", headers=H("chiave-nessun-topic")).json["status"] == "running" and d3.exists(),
      "in corso: il lavoro resta")
(d3 / "status.json").write_text(json.dumps({"status": "failed", "code": "conversion_failed", "error": "x"}))
r = cl.get(f"{BASE}/convert/{r3.json['job_id']}", headers=H("chiave-nessun-topic"))
check(r.json["code"] == "conversion_failed" and "content" not in r.json and not d3.exists(), "errore riportato, poi cancellato")
r4 = upload(name="foglio.xlsx")
check(r4.status_code == 202, "fogli di calcolo accettati")
r5 = upload(name="lettera.odt")
check(r5.status_code == 202, "OpenDocument accettati")
d5 = TMP / "conversions" / r5.json["job_id"]
os.utime(d5, (1, 1))
ex._remove_expired_jobs()
check(not d5.exists() and (TMP / "conversions" / r4.json["job_id"]).exists(), "i lavori mai ritirati scadono")

print("\nRISULTATO:", "FALLITO" if FAILED else "TUTTO OK")
sys.exit(1 if FAILED else 0)
