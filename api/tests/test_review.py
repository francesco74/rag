"""Harness di test del servizio di revisione: SQLite al posto di MySQL,
Qdrant in memoria, pubblicazione RabbitMQ intercettata.

Uso:  pip install -r api/requirements.txt && python api/tests/test_review.py
"""
import json, os, re, sqlite3, sys, tempfile, pathlib, time

TMP = pathlib.Path(tempfile.mkdtemp())
os.environ.update(AUTH_SECRET_KEY="x" * 40, SESSION_COOKIE_SECURE="false", API_SECRET_KEY="chiave-chat", DATA_FOLDER=str(TMP), DOCWS_RICERCA_ENDPOINT="u",
                  DOCWS_ATTI_ENDPOINT="u", DOCWS_CODICE_AMMINISTRAZIONE="u", DOCWS_CODICE_AOO="u", RUOLO_DOCWS="u")
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "api" / "src"), str(ROOT)]

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
CREATE TABLE topics (topic_id TEXT PRIMARY KEY, description TEXT, public_access INT DEFAULT 0);
CREATE TABLE sub_topics (topic_id TEXT, sub_topic_id TEXT, description TEXT);
CREATE TABLE parent_documents (id TEXT PRIMARY KEY, topic_id TEXT, sub_topic_id TEXT, source TEXT,
  file_name TEXT, parent_index INT, content TEXT, metadata TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE review_users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, display_name TEXT,
  password_hash TEXT, active INT DEFAULT 1, token_version INT DEFAULT 1,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, last_login_at TIMESTAMP);
CREATE TABLE review_user_roles (user_id INT, role TEXT, PRIMARY KEY(user_id, role));
CREATE TABLE review_user_topics (user_id INT, topic_id TEXT, PRIMARY KEY(user_id, topic_id));
CREATE TABLE review_status (source TEXT, topic_id TEXT, sub_topic_id TEXT, status TEXT, note TEXT,
  updated_by TEXT, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(source, topic_id, sub_topic_id));
CREATE TABLE review_audit (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  user_id INT, username TEXT, action TEXT, source TEXT, topic_id TEXT, sub_topic_id TEXT,
  old_value TEXT, new_value TEXT, note TEXT, details TEXT);
INSERT INTO topics VALUES ('attiprovincia', 'Atti', 0);   -- riservato
INSERT INTO sub_topics VALUES ('attiprovincia', 'determine', 'Determine');
INSERT INTO topics VALUES ('delibere', 'Delibere', 1);    -- pubblico
INSERT INTO sub_topics VALUES ('delibere', 'giunta', 'Giunta');
""")
from werkzeug.security import generate_password_hash
USERS = {  # username -> (ruoli, archivi assegnati)
    "mrossi": (["revisore"], ["attiprovincia"]),
    "lettore1": (["lettore"], ["attiprovincia"]),
    "multi": (["correttore", "validatore"], ["attiprovincia"]),   # più ruoli: testo e stato, non metadati
    "admin1": (["admin"], []),                                    # admin: tutti gli archivi senza assegnazione
    "senzaruoli": ([], ["attiprovincia"]),
    "solodelibere": (["revisore"], ["delibere"]),
    "senzatopic": (["revisore"], []),
}
for i, (username, (roles, topics)) in enumerate(USERS.items(), start=1):
    c.execute("INSERT INTO review_users (id, username, password_hash) VALUES (?, ?, ?)",
              (i, username, generate_password_hash("passwordlunga")))
    c.executemany("INSERT INTO review_user_roles (user_id, role) VALUES (?, ?)", [(i, r) for r in roles])
    c.executemany("INSERT INTO review_user_topics (user_id, topic_id) VALUES (?, ?)", [(i, t) for t in topics])
# Documento Sicr@Web: md = <stemManifest>_<stemFile>.md accanto al pdf
meta = {"oggetto": "Affidamento lavori strada", "anno": "2024", "data": "2024-03-01", "numero": "12"}
SRC = "sicraweb://999::atto.pdf"
c.executemany("INSERT INTO parent_documents (id, topic_id, sub_topic_id, source, file_name, parent_index, content, metadata, created_at) VALUES (?,?,?,?,?,?,?,?, '2026-01-01 10:00:00')", [
    ("p0", "attiprovincia", "determine", SRC, "atto.pdf", 0, "# Determina\nTesto con erore OCR", json.dumps({"Header 1": "Determina", **meta})),
    ("p1", "attiprovincia", "determine", SRC, "atto.pdf", 1, "Seconda parte", json.dumps(meta)),
])
# Documento direct senza pacchetto su disco
c.execute("INSERT INTO parent_documents (id, topic_id, sub_topic_id, source, file_name, parent_index, content, metadata) VALUES ('d0','attiprovincia','determine','direct://attiprovincia/determine/sub/x.jpg','sub/x.jpg',0,'foto','{}')")
# Documento di un altro archivio
DKEY = dict(source="direct://delibere/giunta/y.jpg", topic_id="delibere", sub_topic_id="giunta")
c.execute("INSERT INTO parent_documents (id, topic_id, sub_topic_id, source, file_name, parent_index, content, metadata) VALUES ('g0','delibere','giunta',?,'y.jpg',0,'delibera','{}')", (DKEY["source"],))
c.commit()

proc = TMP / "processed/attiprovincia/determine"
proc.mkdir(parents=True)
(proc / "atto.pdf").write_bytes(b"%PDF-1.4 finto")
(TMP / "processed/delibere/giunta").mkdir(parents=True)
(TMP / "processed/delibere/giunta/y.jpg").write_bytes(b"\xff\xd8 jpeg finto")
(proc / "manifest_atto.md").write_text("# Determina\nTesto con erore OCR\n\nSeconda parte", encoding="utf-8")
(proc / "manifest_atto.json").write_text(json.dumps({"source": SRC, "files": ["atto.pdf"], "metadati": meta}))
# un altro pacchetto con lo stesso stem ma source diverso: non deve essere scelto
(proc / "altro_atto.md").write_text("SBAGLIATO")
(proc / "altro_atto.json").write_text(json.dumps({"source": "sicraweb://1::atto.pdf", "files": ["atto.pdf"]}))

import app as api_app
import review_routes as ra
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

_cl = api_app.app.test_client()


class _Prefixed:
    """Client che antepone /review a ogni percorso."""
    def __getattr__(self, method):
        return lambda path, **kw: getattr(_cl, method)("/review" + path, **kw)


cl = _Prefixed()
KEY = dict(source=SRC, topic_id="attiprovincia", sub_topic_id="determine")


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond: check.failed = True
check.failed = False

r = cl.get("/documents"); check(r.status_code == 401, "senza token -> 401")
check(_cl.post("/config", json={"topic_id": "x"}).status_code == 401, "chat: senza API key resta protetta")
r = cl.post("/auth/login", json={"username": "mrossi", "password": "x"}, headers={"Authorization": "Bearer chiave-chat"})
check(r.status_code == 401, "la API key della chat non apre /review")
r = cl.post("/auth/login", json={"username": "mrossi", "password": "sbagliata"}); check(r.status_code == 401, "password errata -> 401")
r = cl.post("/auth/login", json={"username": "mrossi", "password": "passwordlunga"}); check(r.status_code == 200, "login")
H = {"Authorization": "Bearer " + r.json["token"]}
check(r.json["user"]["roles"] == ["revisore"] and "documenti.testo" in r.json["user"]["permissions"]
      and "utenti.gestione" not in r.json["user"]["permissions"], "login restituisce ruoli e permessi")


def login(username):
    r = cl.post("/auth/login", json={"username": username, "password": "passwordlunga"})
    return {"Authorization": "Bearer " + r.json["token"]}, r.json["user"]

r = cl.get("/topics", headers=H); check(r.json["topics"][0]["sub_topics"][0]["id"] == "determine", "topics")
r = cl.get("/documents?topic_id=attiprovincia", headers=H)
check(r.json["total"] == 2, f"lista documenti total={r.json['total']}")
r = cl.get("/documents?q=strada", headers=H); check(r.json["total"] == 1 and r.json["items"][0]["title"] == meta["oggetto"], "ricerca nei metadati")

r = cl.get("/document", headers=H, query_string=KEY); d = r.json
check(r.status_code == 200 and d["content_origin"] == "file" and "Seconda parte" in d["content"], "testo dal pacchetto corretto")
check(d["metadata"] == meta, "metadati senza Header N")
check(d["original_file"]["mime_type"] == "application/pdf", "file originale trovato")

check("token" not in d["original_file"], "niente più token per il file: si scarica da /files")

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

# --- ruoli e permessi ---
check(cl.get("/users", headers=H).status_code == 403, "revisore: /users -> 403")
HA, _ = login("admin1")
u = {i["username"]: i["roles"] for i in cl.get("/users", headers=HA).json["items"]}
check(u["multi"] == ["correttore", "validatore"] and u["senzaruoli"] == [], "admin: /users con i ruoli")

HL, ul = login("lettore1")
check(ul["permissions"] == ["documenti.lettura"], "lettore: solo lettura")
check(cl.get("/documents", headers=HL).status_code == 200, "lettore: elenco documenti")
dl = cl.get("/document", headers=HL, query_string=KEY).json
check(cl.get("/document/history", headers=HL, query_string=KEY).status_code == 200, "lettore: storico")
check(cl.put("/document/content", headers=HL, json={**KEY, "content": "x", "base_hash": dl["content_hash"]}).status_code == 403,
      "lettore: correzione testo -> 403")
check(cl.put("/document/metadata", headers=HL, json={**KEY, "metadata": dl["metadata"]}).status_code == 403,
      "lettore: metadati -> 403")
check(cl.put("/document/status", headers=HL, json={**KEY, "status": "in_revisione"}).status_code == 403,
      "lettore: stato -> 403")

HM, um = login("multi")
check(um["permissions"] == ["documenti.lettura", "documenti.stato", "documenti.testo"], f"più ruoli: unione dei permessi {um['permissions']}")
check(cl.put("/document/status", headers=HM, json={**KEY, "status": "in_revisione"}).status_code == 200, "più ruoli: stato consentito")
check(cl.put("/document/metadata", headers=HM, json={**KEY, "metadata": dl["metadata"]}).status_code == 403,
      "più ruoli: metadati -> 403")

cn.execute("DELETE FROM review_user_roles WHERE user_id = 3 AND role = 'validatore'"); cn.commit()
check(cl.put("/document/status", headers=HM, json={**KEY, "status": "revisionato"}).status_code == 403,
      "ruolo revocato: vale subito, senza nuovo login")
r = cl.put("/document/content", headers=HM, json={**KEY, "content": "testo nuovo", "base_hash": dl["content_hash"], "mark_reviewed": True})
check(r.status_code == 403, "correttore senza stato: 'segna come revisionato' -> 403")
check(len(published) == 1, "nessuna re-indicizzazione avviata dalle richieste rifiutate")

HN, un = login("senzaruoli")
check(un["permissions"] == [] and cl.get("/auth/me", headers=HN).status_code == 200, "senza ruoli: login e /auth/me")
check(cl.get("/topics", headers=HN).status_code == 403, "senza ruoli: /topics -> 403")

ATTO = "/files/attiprovincia/determine/atto.pdf"   # archivio riservato
GIUNTA = "/files/delibere/giunta/y.jpg"           # archivio pubblico


def browser(username=None):
    """Client con il proprio cookie jar, come un browser; con login se username."""
    b = api_app.app.test_client()
    if username:
        r = b.post("/review/auth/login", json={"username": username, "password": "passwordlunga"})
        assert r.status_code == 200, username
    return b


bl = browser("lettore1")
check(bl.get(ATTO).status_code == 200, "lettore: file dell'archivio riservato con il cookie di sessione")
cn.execute("DELETE FROM review_user_roles WHERE user_id = 2"); cn.commit()
check(bl.get(ATTO).status_code == 403, "file non più accessibile per chi perde la lettura")

# --- archivi (topic) assegnati ---
check([t["id"] for t in cl.get("/topics", headers=H).json["topics"]] == ["attiprovincia"], "topics: solo gli archivi assegnati")
check(cl.get("/documents?topic_id=delibere", headers=H).status_code == 403, "elenco di un archivio non assegnato -> 403")
items = cl.get("/documents", headers=H).json["items"]
check(items and all(i["topic_id"] == "attiprovincia" for i in items), "elenco senza filtro: solo archivi assegnati")

HD, ud = login("solodelibere")
check(ud["topics"] == ["delibere"], "login restituisce gli archivi assegnati")
r = cl.get("/documents", headers=HD).json
check(r["total"] == 1 and r["items"][0]["topic_id"] == "delibere", "elenco del proprio archivio")
check(cl.get("/document", headers=HD, query_string=DKEY).status_code == 200, "documento del proprio archivio")
check(cl.put("/document/status", headers=HD, json={**DKEY, "status": "in_revisione"}).status_code == 200,
      "stato sul proprio archivio")
check(cl.get("/document", headers=HD, query_string=KEY).status_code == 403, "documento di un altro archivio -> 403")
check(cl.get("/document/history", headers=HD, query_string=KEY).status_code == 403, "storico di un altro archivio -> 403")
check(cl.get(f"/document/history/{h[0]['id']}", headers=HD).status_code == 403, "voce di storico di un altro archivio -> 403")
check(cl.put("/document/content", headers=HD, json={**KEY, "content": "x"}).status_code == 403, "testo di un altro archivio -> 403")
check(cl.put("/document/metadata", headers=HD, json={**KEY, "metadata": {}}).status_code == 403, "metadati di un altro archivio -> 403")
check(cl.put("/document/status", headers=HD, json={**KEY, "status": "in_revisione"}).status_code == 403, "stato di un altro archivio -> 403")

HT, ut = login("senzatopic")
check(ut["topics"] == [] and cl.get("/topics", headers=HT).json["topics"] == []
      and cl.get("/documents", headers=HT).json["total"] == 0, "senza archivi: nessun documento")
_, ua = login("admin1")
check(ua["topics"] is None and len(cl.get("/topics", headers=HA).json["topics"]) == 2, "admin: tutti gli archivi")
u = {i["username"]: i["topics"] for i in cl.get("/users", headers=HA).json["items"]}
check(u["solodelibere"] == ["delibere"] and u["admin1"] is None, "/users con gli archivi")

bm = browser("mrossi")
check(bm.get(ATTO).status_code == 200, "revisore: file del proprio archivio riservato")
cn.execute("DELETE FROM review_user_topics WHERE user_id = 1"); cn.commit()
check(cl.get("/document", headers=H, query_string=KEY).status_code == 403, "archivio tolto: vale subito, senza nuovo login")
check(bm.get(ATTO).status_code == 403, "file di un archivio tolto -> 403")
cn.execute("INSERT INTO review_user_topics VALUES (1, 'attiprovincia')"); cn.commit()

# --- download dei documenti (/files): pubblico o riservato per archivio ---
anon = browser()
r = anon.get(GIUNTA)
check(r.status_code == 200 and r.data.startswith(b"\xff\xd8") and r.headers["Cache-Control"].startswith("public"),
      "archivio pubblico: download anonimo")
r = anon.get(ATTO, headers={"Accept": "text/html"})
check(r.status_code == 401 and b"Accedi" in r.data and b"/revisione/" in r.data,
      "archivio riservato, anonimo dal browser -> 401 con link al login")
r = anon.get(ATTO, headers={"Accept": "application/json"})
check(r.status_code == 401 and "riservato" in r.json["error"], "archivio riservato, anonimo da client -> 401 JSON")
check(anon.get("/files/attiprovincia/determine/inesistente.pdf").status_code == 401,
      "archivio riservato: un anonimo non scopre quali file esistono")
check(_cl.get(ATTO, headers={"Authorization": "Bearer chiave-chat"}).status_code == 401,
      "la API key della chat non apre gli archivi riservati")
check(anon.get(ATTO, headers={"Cookie": "rag_session=falso"}).status_code == 401, "cookie falso -> 401")

bm = browser("mrossi")
r = bm.get(ATTO)
check(r.status_code == 200 and r.data.startswith(b"%PDF") and r.headers["Cache-Control"] == "private, no-store"
      and r.headers["Content-Disposition"].startswith("inline"), "riservato con cookie: visualizzazione")
check(bm.get(ATTO + "?download=1").headers["Content-Disposition"].startswith("attachment"), "?download=1 -> allegato")
check(_cl.get(ATTO, headers=H).status_code == 200, "riservato con header Authorization")
check(browser("solodelibere").get(ATTO).status_code == 403, "utente senza l'archivio -> 403")
check(browser("admin1").get(ATTO).status_code == 200, "admin: tutti gli archivi")

check(bm.get("/files/attiprovincia/determine/manifest_atto.md").status_code == 404,
      "testo estratto (.md) non scaricabile")
check(bm.get("/files/attiprovincia/determine/manifest_atto.json").status_code == 404,
      "manifest (.json) non scaricabile")
check(bm.get("/files/attiprovincia/determine/../../../db.sqlite").status_code in (400, 404),
      "percorso fuori dall'archivio rifiutato")
check(anon.get("/files/inesistente/x/y.pdf").status_code == 404, "archivio inesistente -> 404")
check(bm.get("/files/attiprovincia/determine/sub/x.jpg").status_code == 404,
      "documento indicizzato ma file assente su disco -> 404")

# --- metadati "sporchi" (parent migrati da Qdrant con le chiavi di sistema) ---
from common.utility import split_protected_metadata
clean, dropped = split_protected_metadata({"oggetto": "x", "content": "t", "source": "s"}, ra.settings.protected_keys)
check(clean == {"oggetto": "x"} and dropped == ["content", "source"], "split_protected_metadata separa le chiavi di sistema")

DIRTY = dict(source="direct://attiprovincia/determine/lettera.jpg", topic_id="attiprovincia", sub_topic_id="determine")
dirty_meta = {"source": DIRTY["source"], "content": "testo duplicato", "topic_id": "attiprovincia",
              "sub_topic_id": "determine", "parent_index": 0, "oggetto": "Lettera"}
cn.executemany("INSERT INTO parent_documents (id, topic_id, sub_topic_id, source, file_name, parent_index, content, metadata) "
               "VALUES (?,?,?,?,?,?,?,?)", [
    ("z0", "attiprovincia", "determine", DIRTY["source"], "lettera.jpg", 0, "Prima parte", json.dumps(dirty_meta)),
    ("z1", "attiprovincia", "determine", DIRTY["source"], "lettera.jpg", 1, "Seconda parte",
     json.dumps({**dirty_meta, "parent_index": 1, "Header 1": "Intestazione"})),
])
cn.commit()
q.upsert("document_chunks", [models.PointStruct(id=100 + i, vector=[1, 0], payload={
    "source": DIRTY["source"], "topic_id": "attiprovincia", "sub_topic_id": "determine",
    "content": f"chunk {i}", "oggetto": "Lettera"}) for i in range(2)])

d = cl.get("/document", headers=H, query_string=DIRTY).json
check(d["metadata"] == {"oggetto": "Lettera"}, f"metadati sporchi: la revisione mostra solo quelli veri {d['metadata']}")
r = cl.put("/document/metadata", headers=H, json={**DIRTY, "metadata": {"oggetto": "Lettera corretta", "mittente": "Mario"}})
check(r.status_code == 200, f"salvataggio dei metadati di un documento sporco ({r.status_code})")
rows = {i: json.loads(m) for i, m in cn.execute("SELECT id, metadata FROM parent_documents WHERE source = ?", (DIRTY["source"],))}
check(rows["z0"] == {"oggetto": "Lettera corretta", "mittente": "Mario"}
      and rows["z1"] == {"Header 1": "Intestazione", "oggetto": "Lettera corretta", "mittente": "Mario"},
      f"MySQL ripulito dalle chiavi di sistema, Header del parent conservato {rows}")
pts = q.scroll("document_chunks", scroll_filter=models.Filter(must=[models.FieldCondition(
    key="source", match=models.MatchValue(value=DIRTY["source"]))]), limit=10)[0]
check(len(pts) == 2 and all(pt.payload.get("content", "").startswith("chunk") and pt.payload.get("source") == DIRTY["source"]
                            and pt.payload.get("topic_id") == "attiprovincia" for pt in pts),
      "Qdrant: content, source e topic_id dei chunk intatti")
check(all(pt.payload.get("oggetto") == "Lettera corretta" and pt.payload.get("mittente") == "Mario" for pt in pts),
      "Qdrant: metadati veri aggiornati")

# correzione del testo: il manifest per l'ingest non porta chiavi di sistema
cn.execute("UPDATE parent_documents SET metadata = ? WHERE id = 'z0'", (json.dumps(dirty_meta),)); cn.commit()
d = cl.get("/document", headers=H, query_string=DIRTY).json
r = cl.put("/document/content", headers=H, json={**DIRTY, "content": "Testo corretto", "base_hash": d["content_hash"]})
check(r.status_code == 202, f"correzione del testo di un documento sporco ({r.status_code})")
wj = json.loads((TMP / "watch" / published[-1]).read_text())
check(not (set(wj["metadati"]) & ra.settings.protected_keys) and wj["metadati"].get("oggetto") == "Lettera",
      f"manifest per l'ingest senza chiavi di sistema {sorted(wj['metadati'])}")

# --- logout ---
ba = browser("admin1")
check(ba.get(ATTO).status_code == 200, "admin dal browser: file riservato")
HA, _ = login("admin1")
HA2, _ = login("admin1")  # seconda sessione dello stesso utente
check(cl.post("/auth/logout", headers=HA).status_code == 200, "logout")
check(cl.get("/auth/me", headers=HA).status_code == 401, "dopo il logout il token non vale più")
check(cl.get("/auth/me", headers=HA2).status_code == 401, "il logout chiude anche le altre sessioni dell'utente")
check(ba.get(ATTO).status_code == 401, "dopo il logout il cookie non apre più i documenti riservati")
check(browser().post("/review/auth/logout").status_code == 401, "logout senza token né cookie -> 401")
HA, _ = login("admin1")
check(cl.get("/auth/me", headers=HA).status_code == 200, "nuovo login dopo il logout")
check(cl.get("/topics", headers=H).status_code == 200, "il logout non tocca le sessioni degli altri utenti")

# --- login dalla chat: rotte /auth, solo cookie (nessun token in JavaScript) ---
chat = browser()
check(chat.get("/auth/me").status_code == 401, "chat anonima: /auth/me -> 401")
check(chat.head(ATTO, headers={"Accept": "application/json"}).status_code == 401,
      "chat anonima: HEAD su documento riservato -> 401 (la chat propone il login)")
r = chat.post("/auth/login", json={"username": "mrossi", "password": "passwordlunga"})
sc = r.headers.get("Set-Cookie", "")
check(r.status_code == 200 and "rag_session=" in sc and "HttpOnly" in sc and "SameSite=Lax" in sc,
      "login da /auth: cookie HttpOnly e SameSite=Lax")
check(chat.get("/auth/me").json["user"]["username"] == "mrossi", "/auth/me con il solo cookie")
check(chat.head(ATTO).status_code == 200 and chat.get(ATTO).data.startswith(b"%PDF"),
      "dopo il login la chat apre il documento riservato")
check(chat.get("/review/documents").status_code == 401,
      "il cookie non basta per le rotte della revisione (protezione CSRF)")
check(chat.head(GIUNTA).status_code == 200, "HEAD su documento pubblico -> 200")
r = chat.post("/auth/logout")
check(r.status_code == 200 and "rag_session=;" in r.headers.get("Set-Cookie", ""), "logout dalla chat: cookie cancellato")
check(chat.get("/auth/me").status_code == 401 and chat.get(ATTO).status_code == 401,
      "dopo il logout dalla chat: niente sessione né documenti riservati")
check(browser().post("/auth/login", json={"username": "mrossi", "password": "sbagliata"}).status_code == 401,
      "/auth/login con password errata -> 401")

# --- disattivazione utente invalida il token ---
cn.execute("UPDATE review_users SET active=0, token_version=token_version+1"); cn.commit()
check(cl.get("/topics", headers=H).status_code == 401, "utente disattivato -> token invalido")

print("\nRISULTATO:", "FALLITO" if check.failed else "TUTTO OK")
sys.exit(1 if check.failed else 0)
