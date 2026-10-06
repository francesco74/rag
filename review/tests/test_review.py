"""Harness di test del servizio di revisione: SQLite al posto di MySQL,
Qdrant in memoria, pubblicazione RabbitMQ intercettata.

Uso:  pip install -r review/requirements.txt && python review/tests/test_review.py
"""
import json, os, re, sqlite3, sys, tempfile, pathlib, time

TMP = pathlib.Path(tempfile.mkdtemp())
os.environ.update(REVIEW_SECRET_KEY="x" * 40, DATA_FOLDER=str(TMP), DOCWS_RICERCA_ENDPOINT="u",
                  DOCWS_ATTI_ENDPOINT="u", DOCWS_CODICE_AMMINISTRAZIONE="u", DOCWS_CODICE_AOO="u", RUOLO_DOCWS="u")
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "review" / "src"), str(ROOT)]

DB_PATH = str(TMP / "db.sqlite")


def translate(sql):
    sql = sql.replace("%s", "?").replace("NOW()", "CURRENT_TIMESTAMP")
    sql = sql.replace("AS CHAR)", "AS TEXT)").replace("CHAR_LENGTH(", "LENGTH(")
    sql = re.sub(r"ON DUPLICATE KEY UPDATE .*",
                 "ON CONFLICT(source, topic_id, sub_topic_id) DO UPDATE SET status = excluded.status, "
                 "note = excluded.note, updated_by = excluded.updated_by, updated_at = CURRENT_TIMESTAMP", sql)
    return sql


class Cur:
    def __init__(self, c, d): self.c, self.d = c.cursor(), d
    def execute(self, sql, p=()): self.c.execute(translate(sql), p); self.rowcount = self.c.rowcount
    def executemany(self, sql, p): self.c.executemany(translate(sql), p)
    def _row(self, r): return dict(zip([x[0] for x in self.c.description], r)) if self.d else r
    def fetchone(self):
        r = self.c.fetchone(); return self._row(r) if r else None
    def fetchall(self): return [self._row(r) for r in self.c.fetchall()] if self.c.description else []
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass


class Conn:
    def __init__(self):
        self.c = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
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
CREATE TABLE topics (topic_id TEXT PRIMARY KEY, description TEXT);
CREATE TABLE sub_topics (topic_id TEXT, sub_topic_id TEXT, description TEXT);
CREATE TABLE parent_documents (id TEXT PRIMARY KEY, topic_id TEXT, sub_topic_id TEXT, source TEXT,
  file_name TEXT, parent_index INT, content TEXT, metadata TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE review_users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, display_name TEXT,
  role TEXT DEFAULT 'revisore', password_hash TEXT, active INT DEFAULT 1, token_version INT DEFAULT 1,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, last_login_at TIMESTAMP);
CREATE TABLE review_status (source TEXT, topic_id TEXT, sub_topic_id TEXT, status TEXT, note TEXT,
  updated_by TEXT, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(source, topic_id, sub_topic_id));
CREATE TABLE review_audit (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  user_id INT, username TEXT, action TEXT, source TEXT, topic_id TEXT, sub_topic_id TEXT,
  old_value TEXT, new_value TEXT, note TEXT, details TEXT);
INSERT INTO topics VALUES ('attiprovincia', 'Atti');
INSERT INTO sub_topics VALUES ('attiprovincia', 'determine', 'Determine');
""")
from werkzeug.security import generate_password_hash
c.execute("INSERT INTO review_users (username, password_hash) VALUES ('mrossi', ?)",
          (generate_password_hash("passwordlunga"),))
# Documento Sicr@Web: md = <stemManifest>_<stemFile>.md accanto al pdf
meta = {"oggetto": "Affidamento lavori strada", "anno": "2024", "data": "2024-03-01", "numero": "12"}
SRC = "sicraweb://999::atto.pdf"
c.executemany("INSERT INTO parent_documents (id, topic_id, sub_topic_id, source, file_name, parent_index, content, metadata, created_at) VALUES (?,?,?,?,?,?,?,?, '2026-01-01 10:00:00')", [
    ("p0", "attiprovincia", "determine", SRC, "atto.pdf", 0, "# Determina\nTesto con erore OCR", json.dumps({"Header 1": "Determina", **meta})),
    ("p1", "attiprovincia", "determine", SRC, "atto.pdf", 1, "Seconda parte", json.dumps(meta)),
])
# Documento direct senza pacchetto su disco
c.execute("INSERT INTO parent_documents (id, topic_id, sub_topic_id, source, file_name, parent_index, content, metadata) VALUES ('d0','attiprovincia','determine','direct://attiprovincia/determine/sub/x.jpg','sub/x.jpg',0,'foto','{}')")
c.commit()

proc = TMP / "processed/attiprovincia/determine"
proc.mkdir(parents=True)
(proc / "atto.pdf").write_bytes(b"%PDF-1.4 finto")
(proc / "manifest_atto.md").write_text("# Determina\nTesto con erore OCR\n\nSeconda parte", encoding="utf-8")
(proc / "manifest_atto.json").write_text(json.dumps({"source": SRC, "files": ["atto.pdf"], "metadati": meta}))
# un altro pacchetto con lo stesso stem ma source diverso: non deve essere scelto
(proc / "altro_atto.md").write_text("SBAGLIATO")
(proc / "altro_atto.json").write_text(json.dumps({"source": "sicraweb://1::atto.pdf", "files": ["atto.pdf"]}))

import review_app as ra
from qdrant_client import QdrantClient, models
q = QdrantClient(":memory:")
q.create_collection("document_chunks", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
q.upsert("document_chunks", [
    models.PointStruct(id=i, vector=[1, 0], payload={"source": SRC, "topic_id": "attiprovincia",
                       "sub_topic_id": "determine", "content": "x", **meta}) for i in range(3)
] + [models.PointStruct(id=10, vector=[0, 1], payload={"source": "other", "topic_id": "attiprovincia",
                       "sub_topic_id": "determine", "oggetto": "intatto"})])
ra._qdrant = q
published = []
ra._publish_to_ingest = lambda rel: published.append(rel)

cl = ra.app.test_client()
KEY = dict(source=SRC, topic_id="attiprovincia", sub_topic_id="determine")


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond: check.failed = True
check.failed = False

r = cl.get("/documents"); check(r.status_code == 401, "senza token -> 401")
r = cl.post("/auth/login", json={"username": "mrossi", "password": "sbagliata"}); check(r.status_code == 401, "password errata -> 401")
r = cl.post("/auth/login", json={"username": "mrossi", "password": "passwordlunga"}); check(r.status_code == 200, "login")
H = {"Authorization": "Bearer " + r.json["token"]}

r = cl.get("/topics", headers=H); check(r.json["topics"][0]["sub_topics"][0]["id"] == "determine", "topics")
r = cl.get("/documents?topic_id=attiprovincia", headers=H)
check(r.json["total"] == 2, f"lista documenti total={r.json['total']}")
r = cl.get("/documents?q=strada", headers=H); check(r.json["total"] == 1 and r.json["items"][0]["title"] == meta["oggetto"], "ricerca nei metadati")

r = cl.get("/document", headers=H, query_string=KEY); d = r.json
check(r.status_code == 200 and d["content_origin"] == "file" and "Seconda parte" in d["content"], "testo dal pacchetto corretto")
check(d["metadata"] == meta, "metadati senza Header N")
check(d["original_file"]["mime_type"] == "application/pdf", "file originale trovato")

r = cl.get("/document/file", query_string={"token": d["original_file"]["token"]}); check(r.data.startswith(b"%PDF"), "download originale con token")
r = cl.get("/document/file", query_string={"token": "falso"}); check(r.status_code == 401, "token file falso -> 401")

r = cl.get("/document", headers=H, query_string=dict(KEY, source="direct://attiprovincia/determine/sub/x.jpg"))
check(r.json["content_origin"] == "parents" and r.json["original_file"] is None, "fallback sui parent senza pacchetto")

r = cl.get("/document", headers=H, query_string=dict(KEY, source="../../etc"))
check(r.status_code == 404, "documento inesistente -> 404")

# --- metadati ---
new_meta = {**meta, "oggetto": "Affidamento lavori strada provinciale", "responsabile": "Ing. Bianchi"}
del new_meta["numero"]
r = cl.put("/document/metadata", headers=H, json={**KEY, "metadata": {**new_meta, "content": "x"}})
check(r.status_code == 422, "chiave protetta rifiutata")
r = cl.put("/document/metadata", headers=H, json={**KEY, "metadata": {**new_meta, "data": "01/03/2024"}})
check(r.status_code == 422, "data non ISO rifiutata")
r = cl.put("/document/metadata", headers=H, json={**KEY, "metadata": new_meta, "note": "corretto oggetto"})
check(r.status_code == 200 and r.json["document"]["metadata"] == new_meta, "metadati salvati")
rows = sqlite3.connect(DB_PATH).execute("SELECT id, metadata FROM parent_documents WHERE source=? ORDER BY parent_index", (SRC,)).fetchall()
m0 = json.loads(rows[0][1]); check(m0.get("Header 1") == "Determina" and "numero" not in m0 and m0["responsabile"] == "Ing. Bianchi", "MySQL: header preservato, chiave rimossa, nuova aggiunta")
pts = q.scroll("document_chunks", limit=20)[0]
mine = [p.payload for p in pts if p.payload["source"] == SRC]
check(all(p["oggetto"] == new_meta["oggetto"] and "numero" not in p and p["responsabile"] == "Ing. Bianchi" for p in mine), "Qdrant: payload aggiornato su tutti i chunk")
check([p.payload for p in pts if p.id == 10][0]["oggetto"] == "intatto", "Qdrant: altri documenti intatti")
check(json.loads((proc / "manifest_atto.json").read_text())["metadati"] == new_meta, "manifest allineato")

# --- testo ---
d = cl.get("/document", headers=H, query_string=KEY).json
r = cl.put("/document/content", headers=H, json={**KEY, "content": "nuovo", "base_hash": "vecchio"})
check(r.status_code == 409, "conflitto di versione -> 409")
fixed = d["content"].replace("erore", "errore")
r = cl.put("/document/content", headers=H, json={**KEY, "content": fixed, "base_hash": d["content_hash"], "note": "OCR"})
check(r.status_code == 202, f"testo inviato a reindicizzazione ({r.status_code} {r.json.get('error')})")
check(published == ["attiprovincia/determine/manifest_atto.json"], f"messaggio ingest {published}")
w = TMP / "watch/attiprovincia/determine"
check((w / "manifest_atto.md").read_text() == fixed, "md scritto in watch")
wm = json.loads((w / "manifest_atto.json").read_text())
check(wm["source"] == SRC and wm["metadati"] == new_meta, "manifest in watch con source e metadati aggiornati")
doc = r.json["document"]
check(doc["reindex"]["state"] == "pending" and doc["content"] == fixed and doc["content_origin"] == "pending_review", "stato pending, si mostra il testo corretto")
r = cl.put("/document/metadata", headers=H, json={**KEY, "metadata": meta})
check(r.status_code == 409, "metadati bloccati durante la reindicizzazione")

# simulazione errore ingest
err = TMP / "ingestion/error/attiprovincia/determine"; err.mkdir(parents=True)
(err / "manifest_atto.json").write_text(json.dumps({**wm, "_ingestion_error": "boom"}))
r = cl.get("/document", headers=H, query_string=KEY)
check(r.json["reindex"]["state"] == "error" and r.json["reindex"]["error"] == "boom", "errore di ingest rilevato")
(err / "manifest_atto.json").unlink()

# simulazione ingest completato: parent rigenerati dopo l'audit
time.sleep(1.1)
cn = sqlite3.connect(DB_PATH)
cn.execute("DELETE FROM parent_documents WHERE source=?", (SRC,))
cn.execute("INSERT INTO parent_documents (id, topic_id, sub_topic_id, source, file_name, parent_index, content, metadata) VALUES ('n0','attiprovincia','determine',?,'atto.pdf',0,?,?)", (SRC, fixed, json.dumps(new_meta)))
cn.commit()
(proc / "manifest_atto.md").write_text(fixed)
r = cl.get("/document", headers=H, query_string=KEY)
check(r.json["reindex"] is None and r.json["content_origin"] == "file", "reindicizzazione completata")

# --- stato e storico ---
r = cl.put("/document/status", headers=H, json={**KEY, "status": "revisionato"}); check(r.status_code == 200, "stato revisionato")
r = cl.get("/documents?status=revisionato", headers=H); check(r.json["total"] == 1, "filtro per stato")
r = cl.get("/documents?status=da_revisionare", headers=H); check(r.json["total"] == 1, "filtro da revisionare")
h = cl.get("/document/history", headers=H, query_string=KEY).json["items"]
check([i["action"] for i in h] == ["status", "content", "metadata"], f"storico {[i['action'] for i in h]}")
e = cl.get(f"/document/history/{h[1]['id']}", headers=H).json
check(e["old_value"] != e["new_value"] and "errore" in e["new_value"], "voce di storico con prima/dopo")

# --- disattivazione utente invalida il token ---
cn.execute("UPDATE review_users SET active=0, token_version=token_version+1"); cn.commit()
check(cl.get("/topics", headers=H).status_code == 401, "utente disattivato -> token invalido")

print("\nRISULTATO:", "FALLITO" if check.failed else "TUTTO OK")
sys.exit(1 if check.failed else 0)
