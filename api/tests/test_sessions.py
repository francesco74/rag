"""Test delle sessioni di documenti (API esterne): SQLite al posto di MySQL, Qdrant in
memoria, embedding finto, RabbitMQ intercettato. L'indicizzazione (converter e
ingestione) si simula scrivendo sezioni e pezzi come farebbe ingest.py.

Uso:  python api/tests/test_sessions.py      (esce con 1 se qualcosa fallisce)
"""
import hashlib, io, json, os, pathlib, re, sqlite3, sys, tempfile, uuid
from datetime import timedelta

TMP = pathlib.Path(tempfile.mkdtemp())
os.environ.update(AUTH_SECRET_KEY="x" * 40, API_SECRET_KEY="chiave-chat", DATA_FOLDER=str(TMP), DOCWS_RICERCA_ENDPOINT="u",
                  DOCWS_ATTI_ENDPOINT="u", DOCWS_CODICE_AMMINISTRAZIONE="u", DOCWS_CODICE_AOO="u", RUOLO_DOCWS="u",
                  CONVERT_MAX_UPLOAD_MB="1", LOG_LEVEL="CRITICAL", RERANK_PROVIDER="lessicale",
                  SESSION_MAX_DOCUMENTS="3", SESSION_MAX_TOKENS="5000")
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "api" / "src"), str(ROOT)]
DB_PATH = str(TMP / "db.sqlite")


def translate(sql):
    sql = sql.replace("%s", "?")
    sql = sql.replace("NOW() - INTERVAL 60 SECOND", "datetime(CURRENT_TIMESTAMP, '-60 seconds')")
    return sql.replace("NOW()", "CURRENT_TIMESTAMP").replace("active = TRUE", "active = 1")


class Cur:
    def __init__(self, c, d): self.c, self.d = c.cursor(), d
    def execute(self, sql, p=()):
        p = tuple(x.isoformat(sep=" ") if hasattr(x, "isoformat") else x for x in p)
        self.c.execute(translate(sql), p); self.rowcount = self.c.rowcount
    def executemany(self, sql, rows): self.c.executemany(translate(sql), rows)
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


import common.db_logger as dbl  # noqa: E402
dbl.init_db_pool = lambda: None
dbl.get_db_connection = lambda: Conn()

c = sqlite3.connect(DB_PATH)
c.executescript("""
CREATE TABLE topics (topic_id TEXT PRIMARY KEY, description TEXT, public_access INT DEFAULT 0);
CREATE TABLE sub_topics (topic_id TEXT, sub_topic_id TEXT, description TEXT);
CREATE TABLE external_api_keys (id INTEGER PRIMARY KEY, name TEXT, key_hash TEXT, key_prefix TEXT, active INT DEFAULT 1,
  created_at TIMESTAMP, expires_at TIMESTAMP, last_used_at TIMESTAMP);
CREATE TABLE external_api_key_topics (key_id INT, topic_id TEXT);
CREATE TABLE sessions (id TEXT PRIMARY KEY, key_id INT, owner TEXT, created_at TEXT, expires_at TEXT);
CREATE TABLE session_documents (id TEXT PRIMARY KEY, session_id TEXT, file_name TEXT, status TEXT, error TEXT,
  tokens INT, sections INT, created_at TEXT);
CREATE TABLE session_parents (session_id TEXT, doc_id TEXT, parent_index INT, content TEXT, metadata TEXT);
""")
for i, name in ((1, "chiave-atti-pa"), (2, "chiave-altra")):
    c.execute("INSERT INTO external_api_keys (id, name, key_hash, key_prefix) VALUES (?,?,?,?)",
              (i, name, hashlib.sha256(name.encode()).hexdigest(), name[:4]))
c.commit()

from qdrant_client import QdrantClient, models  # noqa: E402
import app as api_app  # noqa: E402
import external_routes as ex  # noqa: E402
import external_sessions as es  # noqa: E402
import session_search  # noqa: E402
from common import document_sessions as ds  # noqa: E402

DIM = 32
q = QdrantClient(":memory:")
q.create_collection(ds.SESSION_COLLECTION, vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE))


def fake_vector(text):
    """Embedding finto: parole in un vettore di DIM posizioni (testi con parole comuni sono vicini)."""
    v = [0.0] * DIM
    for w in re.findall(r"\w+", text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
    return v if any(v) else [1.0] + [0.0] * (DIM - 1)


session_search._embed_query = fake_vector
es._qdrant = lambda: q
published = []
ex.publish_to_queue = lambda queue, payload, args=None: published.append((queue, payload))
cl = api_app.app.test_client()
H = lambda k: {"Authorization": f"Bearer {k}"}
A, B = H("chiave-atti-pa"), H("chiave-altra")
BASE = "/external/v1/sessions"
FAILED = []


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond: FAILED.append(msg)


def upload(sid, name="delibera.pdf", data=b"%PDF-1.4 finto", headers=A):
    return cl.post(f"{BASE}/{sid}/documents", headers=headers,
                   data={"file": (io.BytesIO(data), name)}, content_type="multipart/form-data")


def index(sid, did, sections, tokens=1000):
    """Quello che fanno converter e ingestione: sezioni in MySQL, pezzi in Qdrant, stato ready."""
    row = c.execute("SELECT created_at FROM sessions WHERE id = ?", (sid,)).fetchone()
    from datetime import datetime
    expires = ds.epoch(ds.max_expiry(datetime.fromisoformat(row[0])))
    ds.insert_parents([(sid, did, i, text, json.dumps({"Header 1": f"Sezione {i}"})) for i, text in enumerate(sections)])
    q.upsert(ds.SESSION_COLLECTION, points=[
        models.PointStruct(id=str(uuid.uuid4()), vector=fake_vector(text), payload={
            "session_id": sid, "key_id": 1, "doc_id": did, "file_name": "x", "parent_index": i,
            "chunk_index": 0, "content": text, "expires_at": expires})
        for i, text in enumerate(sections)])
    ds.set_status(did, ds.READY, None, tokens, len(sections))
    import shutil
    shutil.rmtree(ds.document_folder(sid, did), ignore_errors=True)


def points(**match):
    flt = models.Filter(must=[models.FieldCondition(key=k, match=models.MatchValue(value=v)) for k, v in match.items()])
    return q.count(ds.SESSION_COLLECTION, count_filter=flt, exact=True).count


# --- creazione e proprietà
r = cl.post(BASE, headers=A, json={"owner": "ente-1:utente-7"})
check(r.status_code == 201 and re.fullmatch(r"[0-9a-f]{32}", r.json["session_id"]), "sessione creata")
check(r.json["limits"]["max_documents"] == 3, "limiti restituiti alla creazione")
sid = r.json["session_id"]
check(cl.post(BASE, json={}).status_code == 401, "senza credenziale -> 401")
check(cl.post(BASE, headers=A, json={"owner": "x" * 200}).status_code == 400, "owner troppo lungo -> 400")
check(cl.get(f"{BASE}/{sid}", headers=B).status_code == 404, "sessione di un'altra chiave -> 404")
check(cl.get(f"{BASE}/{'0' * 32}", headers=A).status_code == 404, "sessione inesistente -> 404")
check(cl.get(f"{BASE}/non-valido", headers=A).status_code == 404, "id non valido -> 404")
r = cl.get(f"{BASE}/{sid}", headers=A)
check(r.status_code == 200 and r.json["owner"] == "ente-1:utente-7" and r.json["documents"] == [], "stato della sessione vuota")

# --- caricamento
check(upload(sid, "virus.exe").status_code == 415, "formato non gestito -> 415")
check(upload(sid, "grande.pdf", b"x" * (2 * 1024 * 1024)).status_code == 413, "documento troppo grande -> 413")
check(upload(sid, headers=B).status_code == 404, "caricamento in una sessione di un'altra chiave -> 404")
r = upload(sid, "Delibera 45.pdf")
check(r.status_code == 202 and r.json["status"] == "queued", "documento caricato -> 202 queued")
d1 = r.json["doc_id"]
check(published[-1] == ("da-convertire-api", {"session_id": sid, "doc_id": d1}), "pubblicato sulla coda delle conversioni")
folder = ds.document_folder(sid, d1)
check((folder / "input.pdf").is_file() and json.loads((folder / "request.json").read_text())["file_name"] == "Delibera 45.pdf",
      "file e richiesta salvati nella cartella del documento")
r = cl.get(f"{BASE}/{sid}/documents/{d1}", headers=A)
check(r.json == {"doc_id": d1, "file_name": "Delibera 45.pdf", "status": "queued"}, "stato del documento in coda")
r = cl.post(f"{BASE}/{sid}/search", headers=A, json={"query": "contributo"})
check(r.status_code == 409, "ricerca senza documenti pronti -> 409")

index(sid, d1, [
    "Premesse. La provincia ha ricevuto la domanda di contributo dal comune.",
    "Ai sensi dell'art. 35 comma 2 il contributo è concesso nella misura di 15.000 euro.",
    "Il termine di scadenza per la rendicontazione è fissato al 30 giugno 2026.",
], tokens=1200)
r = upload(sid, "Legge.pdf")
d2 = r.json["doc_id"]
index(sid, d2, ["Articolo 1. Oggetto della legge regionale sul turismo.",
                "Articolo 2. Le strutture ricettive devono comunicare le presenze."], tokens=800)

r = cl.get(f"{BASE}/{sid}/documents/{d1}", headers=A)
check(r.json["status"] == "ready" and r.json["tokens"] == 1200 and r.json["sections"] == 3, "documento pronto con token e sezioni")
r = cl.get(f"{BASE}/{sid}", headers=A)
check(len(r.json["documents"]) == 2 and r.json["tokens"] == 2000, "sessione: due documenti, token totali")

# --- ricerca
r = cl.post(f"{BASE}/{sid}/search", headers=A, json={"query": "Quanto è il contributo dell'art. 35 comma 2?"})
check(r.status_code == 200 and r.json["passages"], "ricerca: passaggi trovati")
top = r.json["passages"][0]
check(top["doc_id"] == d1 and top["parent_index"] == 1 and "15.000" in top["content"],
      f"ricerca: per primo il passaggio con articolo e comma ({top['parent_index']})")
check(top["file_name"] == "Delibera 45.pdf" and top["headings"] == {"Header 1": "Sezione 1"}, "ricerca: nome del file e intestazioni")
check(r.json["reranker"] == "lessicale", "ricerca: riordino lessicale indicato")
check(len({(p["doc_id"], p["parent_index"]) for p in r.json["passages"]}) == len(r.json["passages"]), "ricerca: una voce per sezione")
r = cl.post(f"{BASE}/{sid}/search", headers=A, json={"query": "strutture ricettive", "doc_ids": [d2], "top_k": 1})
check([p["doc_id"] for p in r.json["passages"]] == [d2], "ricerca limitata ai documenti scelti, top_k rispettato")
check(cl.post(f"{BASE}/{sid}/search", headers=A, json={"query": "x", "doc_ids": ["0" * 32]}).status_code == 404,
      "ricerca su un documento non della sessione -> 404")
check(cl.post(f"{BASE}/{sid}/search", headers=A, json={}).status_code == 400, "ricerca senza domanda -> 400")
check(cl.post(f"{BASE}/{sid}/search", headers=B, json={"query": "contributo"}).status_code == 404,
      "ricerca con un'altra chiave -> 404")

# Isolamento in Qdrant: punti di un'altra sessione con lo stesso testo non escono.
r2 = cl.post(BASE, headers=B, json={"owner": "altro"}).json["session_id"]
dx = upload(r2, "altro.pdf", headers=B).json["doc_id"]
index(r2, dx, ["Ai sensi dell'art. 35 comma 2 il contributo è concesso."])
r = cl.post(f"{BASE}/{sid}/search", headers=A, json={"query": "art. 35 comma 2 contributo"})
check(all(p["doc_id"] in (d1, d2) for p in r.json["passages"]), "ricerca: nessun passaggio di altre sessioni")

# --- sezioni in ordine
r = cl.get(f"{BASE}/{sid}/documents/{d1}/sections?from=1&limit=5", headers=A)
check(r.status_code == 200 and [s["index"] for s in r.json["sections"]] == [1, 2] and r.json["total"] == 3,
      "sezioni in ordine, a partire da from")
check(cl.get(f"{BASE}/{sid}/documents/{d1}/sections?from=x", headers=A).status_code == 400, "sezioni: parametro non valido -> 400")

# --- limiti
d3 = upload(sid, "terzo.pdf").json["doc_id"]
r = upload(sid, "quarto.pdf")
check(r.status_code == 409 and r.json["code"] == "max_documents", "oltre il numero massimo di documenti -> 409")
ds.set_status(d3, ds.FAILED, "Nessun testo estratto dal documento.")
check(upload(sid, "quarto.pdf").status_code == 202, "i documenti falliti non contano nel limite")
check(ds.session_tokens(sid) == 2000, "token usati: solo i documenti non falliti")

# --- rimozione di un documento
check(cl.delete(f"{BASE}/{sid}/documents/{d2}", headers=B).status_code == 404, "togliere un documento con un'altra chiave -> 404")
check(cl.delete(f"{BASE}/{sid}/documents/{d2}", headers=A).status_code == 204, "documento tolto -> 204")
check(points(doc_id=d2) == 0 and not c.execute("SELECT 1 FROM session_parents WHERE doc_id=?", (d2,)).fetchone(),
      "documento tolto: niente pezzi né sezioni")
check(cl.get(f"{BASE}/{sid}/documents/{d2}", headers=A).status_code == 404, "documento tolto -> 404")

# --- scadenza scorrevole e pulizia
before = c.execute("SELECT expires_at FROM sessions WHERE id=?", (sid,)).fetchone()[0]
c.execute("UPDATE sessions SET expires_at = ? WHERE id = ?", (str(ds.utcnow() + timedelta(minutes=1)), sid)); c.commit()
cl.get(f"{BASE}/{sid}", headers=A)
after = c.execute("SELECT expires_at FROM sessions WHERE id=?", (sid,)).fetchone()[0]
check(after > str(ds.utcnow() + timedelta(minutes=100)), "ogni chiamata rinnova la scadenza")
created = ds.utcnow() - timedelta(hours=11, minutes=30)
c.execute("UPDATE sessions SET created_at = ? WHERE id = ?", (str(created), r2)); c.commit()
cl.get(f"{BASE}/{r2}", headers=B)
exp = c.execute("SELECT expires_at FROM sessions WHERE id=?", (r2,)).fetchone()[0]
check(exp <= str(created + timedelta(hours=12)), "la scadenza non supera il tetto dalla creazione")

c.execute("UPDATE sessions SET expires_at = ? WHERE id = ?", (str(ds.utcnow() - timedelta(minutes=1)), r2)); c.commit()
check(cl.get(f"{BASE}/{r2}", headers=B).status_code == 404, "sessione scaduta -> 404")
orphan = TMP / "sessions" / ("f" * 32)
orphan.mkdir(parents=True)
n = ds.prune_expired(q)
check(n == 1 and points(session_id=r2) == 0 and not c.execute("SELECT 1 FROM sessions WHERE id=?", (r2,)).fetchone(),
      "pulizia: sessione scaduta cancellata (punti e righe)")
check(points(session_id=sid) > 0, "pulizia: le sessioni attive restano")
check(not orphan.exists(), "pulizia: cartelle senza sessione cancellate")

# Punti rimasti senza sessione con expires_at passato.
q.upsert(ds.SESSION_COLLECTION, points=[models.PointStruct(id=str(uuid.uuid4()), vector=fake_vector("x"), payload={
    "session_id": "e" * 32, "key_id": 1, "doc_id": "e" * 32, "parent_index": 0, "content": "x",
    "expires_at": ds.epoch(ds.utcnow()) - 60})])
ds.prune_expired(q)
check(points(session_id="e" * 32) == 0, "pulizia: punti scaduti senza sessione cancellati")

# --- chiusura
check(cl.delete(f"{BASE}/{sid}", headers=B).status_code == 404, "chiudere la sessione di un'altra chiave -> 404")
check(cl.delete(f"{BASE}/{sid}", headers=A).status_code == 204, "sessione chiusa -> 204")
check(points(session_id=sid) == 0 and not c.execute("SELECT 1 FROM session_documents WHERE session_id=?", (sid,)).fetchone(),
      "sessione chiusa: niente pezzi né documenti")
check(not (TMP / "sessions" / sid).exists(), "sessione chiusa: cartella cancellata")
check(cl.get(f"{BASE}/{sid}", headers=A).status_code == 404, "sessione chiusa -> 404")

print("\nRISULTATO:", "FALLITO" if FAILED else "TUTTO OK")
sys.exit(1 if FAILED else 0)
