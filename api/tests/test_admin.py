"""Test dei siti della chat e dell'amministrazione: SQLite al posto di MySQL.

Uso:  python api/tests/test_admin.py      (esce con 1 se qualcosa fallisce)
"""
import base64, os, pathlib, re, sqlite3, sys, tempfile

TMP = pathlib.Path(tempfile.mkdtemp())
os.environ.update(AUTH_SECRET_KEY="x" * 40, API_SECRET_KEY="chiave-chat", DATA_FOLDER=str(TMP), DOCWS_RICERCA_ENDPOINT="u",
                  DOCWS_ATTI_ENDPOINT="u", DOCWS_CODICE_AMMINISTRAZIONE="u", DOCWS_CODICE_AOO="u", RUOLO_DOCWS="u",
                  LOG_LEVEL="CRITICAL", SITE_CACHE_SECONDS="0")
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "api" / "src"), str(ROOT)]
DB_PATH = str(TMP / "db.sqlite")


def translate(sql):
    sql = sql.replace("%s", "?")
    sql = re.sub(r"NOW\(\) ([+-]) INTERVAL \? SECOND", r"datetime(CURRENT_TIMESTAMP, '\1' || ? || ' seconds')", sql)
    sql = sql.replace("NOW() - INTERVAL 60 SECOND", "datetime(CURRENT_TIMESTAMP, '-60 seconds')")
    sql = sql.replace("TIMESTAMPDIFF(SECOND, NOW(), locked_until)",
                      "CAST(ROUND((julianday(locked_until) - julianday(CURRENT_TIMESTAMP)) * 86400) AS INTEGER)")
    sql = re.sub(r"\bIF\(", "iif(", sql).replace("NOW()", "CURRENT_TIMESTAMP")
    if "INSERT INTO review_login_failures" in sql:
        sql = sql.replace("ON DUPLICATE KEY UPDATE", "ON CONFLICT(lkey) DO UPDATE SET")
    return re.sub(r"\b(active) = TRUE", r"\1 = 1", sql)


class Cur:
    def __init__(self, c, d): self.c, self.d, self.lastrowid = c.cursor(), d, None
    def execute(self, sql, p=()):
        self.c.execute(translate(sql), p); self.rowcount = self.c.rowcount; self.lastrowid = self.c.lastrowid
    def _row(self, r): return dict(zip([x[0] for x in self.c.description], r)) if self.d else r
    def fetchone(self):
        r = self.c.fetchone(); return self._row(r) if r else None
    def fetchall(self): return [self._row(r) for r in self.c.fetchall()] if self.c.description else []
    def close(self): pass


class Conn:
    def __init__(self):
        self.c = sqlite3.connect(DB_PATH)
        self.c.execute("PRAGMA foreign_keys = ON")
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
CREATE TABLE topics (topic_id TEXT PRIMARY KEY, description TEXT, aliases TEXT, prompt TEXT, public_access INT DEFAULT 0);
CREATE TABLE sub_topics (id INTEGER PRIMARY KEY, topic_id TEXT REFERENCES topics(topic_id) ON DELETE CASCADE, sub_topic_id TEXT,
  description TEXT, description_long TEXT, chunk_size INT, chunk_overlap INT, parent_chunk_size INT, use_markdown_splitter INT);
CREATE TABLE parent_documents (id TEXT, topic_id TEXT, sub_topic_id TEXT, source TEXT, file_name TEXT, parent_index INT, content TEXT, metadata TEXT);
CREATE TABLE chat_sites (hostname TEXT PRIMARY KEY, topic_id TEXT REFERENCES topics(topic_id), name TEXT, allow_date_filter INT DEFAULT 0, allow_subtopic_selection INT DEFAULT 1,
  active INT DEFAULT 1, welcome_it TEXT, welcome_en TEXT, home_buttons TEXT, icon BLOB);
CREATE TABLE review_users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, display_name TEXT, password_hash TEXT, active INT DEFAULT 1,
  token_version INT DEFAULT 1, created_at TIMESTAMP, last_login_at TIMESTAMP);
CREATE TABLE review_user_roles (user_id INT, role TEXT);
CREATE TABLE review_user_topics (user_id INT, topic_id TEXT);
CREATE TABLE review_login_failures (lkey TEXT PRIMARY KEY, failed_count INT, last_failed_at TIMESTAMP, locked_until TIMESTAMP);
CREATE TABLE external_api_keys (id INTEGER PRIMARY KEY, name TEXT, key_hash TEXT, key_prefix TEXT, active INT DEFAULT 1,
  created_at TIMESTAMP, expires_at TIMESTAMP, last_used_at TIMESTAMP);
CREATE TABLE external_api_key_topics (key_id INT REFERENCES external_api_keys(id) ON DELETE CASCADE, topic_id TEXT);
INSERT INTO topics (topic_id, description) VALUES ('atti', 'Atti'), ('lettere', 'Lettere');
INSERT INTO sub_topics (topic_id, sub_topic_id, description, description_long) VALUES ('atti', 'determine', 'Determine', '');
INSERT INTO parent_documents VALUES ('p', 'atti', 'determine', 's', 'f', 0, 'c', '{}');
INSERT INTO chat_sites (hostname, topic_id, name, welcome_it) VALUES ('atti.example.it', 'atti', 'Atti', '<p>Benvenuti</p>');
INSERT INTO chat_sites (hostname, topic_id, name, active) VALUES ('spento.example.it', 'atti', 'Spento', 0);
""")
from werkzeug.security import generate_password_hash  # noqa: E402
for i, name, role in [(1, "capo", "admin"), (2, "lettore", "revisore")]:
    c.execute("INSERT INTO review_users (id, username, password_hash) VALUES (?, ?, ?)", (i, name, generate_password_hash("password-lunga")))
    c.execute("INSERT INTO review_user_roles VALUES (?, ?)", (i, role))
c.commit()

import app as api_app  # noqa: E402
import review_routes as rr  # noqa: E402
cl = api_app.app.test_client()
FAILED = []


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond: FAILED.append(msg)


def token(name):
    r = cl.post("/auth/login", json={"username": name, "password": "password-lunga"})
    assert r.status_code == 200, r.data
    return r.json["token"]


A = {"Authorization": f"Bearer {token('capo')}"}
L = {"Authorization": f"Bearer {token('lettore')}"}
SITE = {"Host": "atti.example.it"}
ADM = "/review/admin"

# --- sito dal dominio
r = cl.get("/site", headers={"Host": "ATTI.example.it:8081"})
check(r.status_code == 200 and r.json["topic_id"] == "atti" and r.json["name"] == "Atti", "il dominio (senza porta, maiuscole ignorate) individua il sito")
check(r.json["welcome"]["it"] == "<p>Benvenuti</p>" and r.json["welcome"]["en"] is None, "home page del sito")
check(cl.get("/site", headers={"Host": "altro.example.it"}).status_code == 404, "dominio sconosciuto -> 404")
check(cl.get("/site", headers={"Host": "spento.example.it"}).status_code == 404, "sito disattivato -> 404")

# --- autorizzazione delle rotte della chat
check(cl.post("/config", json={"topic_id": "atti"}, headers=SITE).status_code != 401, "il sito apre /config sul proprio archivio")
check(cl.post("/config", json={"topic_id": "lettere"}, headers=SITE).status_code == 403, "il sito non apre un altro archivio")
check(cl.post("/chat", json={"topic_id": "lettere", "query": "x"}, headers=SITE).status_code == 403, "né su /chat")
check(cl.post("/config", json={"topic_id": "atti"}, headers={"Host": "altro.example.it"}).status_code == 401, "dominio sconosciuto -> 401")
check(cl.post("/config", json={"topic_id": "atti"}, headers={"Host": "spento.example.it"}).status_code == 401, "sito disattivato -> 401")
check(cl.post("/config", json={"topic_id": "lettere"}, headers={"Authorization": "Bearer chiave-chat"}).status_code != 401, "la chiave continua a valere per qualunque archivio")

# --- permessi
for method, path in [("get", "/topics"), ("get", "/sites"), ("get", "/keys"), ("get", "/roles")]:
    check(getattr(cl, method)(ADM + path, headers=L).status_code == 403, f"lettore: {path} -> 403")
    check(getattr(cl, method)(ADM + path).status_code == 401, f"senza sessione: {path} -> 401")

# --- utenti
r = cl.post(f"{ADM}/users", headers=A, json={"username": "nuovo", "password": "corta", "roles": ["revisore"]})
check(r.status_code == 422, "password corta -> 422")
r = cl.post(f"{ADM}/users", headers=A, json={"username": "nuovo", "password": "password-lunga", "roles": ["boh"]})
check(r.status_code == 422, "ruolo sconosciuto -> 422")
r = cl.post(f"{ADM}/users", headers=A, json={"username": "nuovo", "password": "password-lunga", "roles": ["revisore"], "topics": ["nessuno"]})
check(r.status_code == 422, "archivio inesistente -> 422")
r = cl.post(f"{ADM}/users", headers=A, json={"username": "nuovo", "password": "password-lunga", "roles": ["archivista"], "topics": ["atti"]})
check(r.status_code == 201, "utente creato")
uid = r.json["id"]
check(cl.post(f"{ADM}/users", headers=A, json={"username": "nuovo", "password": "password-lunga"}).status_code == 409, "nome già usato -> 409")
N = {"Authorization": f"Bearer {token('nuovo')}"}
me = cl.get("/auth/me", headers=N).json
check(me["user"]["topics"] == ["atti"] if "user" in me else me["topics"] == ["atti"], "il nuovo utente vede il suo archivio")
items = {u["username"]: u for u in cl.get("/review/users", headers=A).json["items"]}
check(items["nuovo"]["roles"] == ["archivista"], "elenco utenti aggiornato")

check(cl.post(f"{ADM}/users/{uid}/password", headers=A, json={"password": "un-altra-password"}).status_code == 200, "password cambiata")
check(cl.get("/auth/me", headers=N).status_code == 401, "il cambio password chiude le sessioni aperte")

r = cl.put(f"{ADM}/users/{uid}", headers=A, json={"roles": ["revisore"], "topics": [], "active": False})
check(r.status_code == 200, "utente disattivato")
check(cl.post("/auth/login", json={"username": "nuovo", "password": "un-altra-password"}).status_code == 401, "un utente disattivato non entra")
check(cl.put(f"{ADM}/users/1", headers=A, json={"roles": ["admin"], "active": False}).status_code == 409, "non ci si disattiva da soli")
check(cl.put(f"{ADM}/users/1", headers=A, json={"roles": ["revisore"], "active": True}).status_code == 409, "resta almeno un amministratore")
check(cl.get(f"{ADM}/roles", headers=A).status_code == 200, "dopo il rifiuto l'amministratore ha ancora il ruolo")
c.execute("INSERT INTO review_login_failures (lkey, failed_count) VALUES ('user:lettore', 3)"); c.commit()
check(cl.post(f"{ADM}/users/2/unlock", headers=A).status_code == 200 and not c.execute("SELECT 1 FROM review_login_failures WHERE lkey = 'user:lettore'").fetchone(), "sblocco del login")

# --- archivi
check(cl.post(f"{ADM}/topics", headers=A, json={"topic_id": "x", "description": "X"}).status_code == 405, "gli archivi non si creano dal web")
check(cl.delete(f"{ADM}/topics/atti", headers=A).status_code == 405, "gli archivi non si eliminano dal web")
c = sqlite3.connect(DB_PATH)
c.execute("INSERT INTO topics (topic_id, description, prompt, public_access) VALUES ('nuovo', 'Nuovo', 'general', 1)")
c.commit()
c.close()
check(cl.put(f"{ADM}/topics/nuovo", headers=A, json={"description": "", "prompt": "general"}).status_code == 422, "descrizione obbligatoria")
check(cl.put(f"{ADM}/topics/nuovo", headers=A, json={"description": "x", "prompt": "../segreto"}).status_code == 422, "prompt inesistente -> 422")
check(cl.put(f"{ADM}/topics/nuovo", headers=A, json={"description": "Nuovo", "prompt": "general", "public_access": True}).status_code == 200, "archivio modificato")
check(cl.post(f"{ADM}/topics/nuovo/subtopics", headers=A, json={"sub_topic_id": "serie", "description": "Serie", "chunk_size": 800}).status_code == 201, "sottocategoria creata")
check(cl.put(f"{ADM}/topics/nuovo/subtopics/serie", headers=A, json={"description": "Serie 2", "chunk_size": -1}).status_code == 422, "chunk negativo -> 422")
check(cl.put(f"{ADM}/topics/nuovo/subtopics/serie", headers=A, json={"description": "Serie 2", "chunk_size": None}).status_code == 200, "sottocategoria modificata")
data = cl.get(f"{ADM}/topics", headers=A).json
new = next(t for t in data["items"] if t["topic_id"] == "nuovo")
check(new["public_access"] is True and new["sub_topics"][0]["description"] == "Serie 2", "elenco archivi aggiornato")
check("general" in data["prompts"] and "query_rewriter" not in data["prompts"], "solo i prompt selezionabili")
check(cl.delete(f"{ADM}/topics/atti/subtopics/determine", headers=A).status_code == 409, "sottocategoria con documenti: non si elimina")

# --- siti
check(cl.post(f"{ADM}/sites", headers=A, json={"hostname": "x.it", "topic_id": "nuovo", "name": "X"}).status_code == 405, "i siti non si creano dal web")
c = sqlite3.connect(DB_PATH)
c.execute("INSERT INTO chat_sites (hostname, topic_id, name, allow_date_filter) VALUES ('nuovo.ente.it', 'nuovo', 'Nuovo', 1)")
c.commit()
c.close()
check(cl.get("/site", headers={"Host": "nuovo.ente.it"}).json["allow_date_filter"] is True, "il nuovo sito funziona subito")
check(cl.post("/config", json={"topic_id": "nuovo"}, headers={"Host": "nuovo.ente.it"}).status_code != 401, "e apre la chat del suo archivio")
sito = next(x for x in cl.get(f"{ADM}/sites", headers=A).json["items"] if x["hostname"] == "nuovo.ente.it")
check(sito["allow_subtopic_selection"] is True, "scelta delle serie attiva di default")
check(cl.put(f"{ADM}/sites/nuovo.ente.it", headers=A, json={"topic_id": "nuovo", "name": "Nuovo", "allow_subtopic_selection": False}).status_code == 200, "il sito stesso si può salvare")
sito = next(x for x in cl.get(f"{ADM}/sites", headers=A).json["items"] if x["hostname"] == "nuovo.ente.it")
check(sito["allow_subtopic_selection"] is False, "scelta delle serie disattivata sul sito")
check(cl.put(f"{ADM}/sites/nuovo.ente.it", headers=A, json={"topic_id": "atti", "name": "Nuovo"}).status_code == 200 and cl.get("/site", headers={"Host": "nuovo.ente.it"}).json["topic_id"] == "nuovo", "l'archivio del sito non si cambia dal web")
check(cl.put(f"{ADM}/sites/nuovo.ente.it", headers=A, json={"topic_id": "nuovo", "name": "Rinominato", "active": False}).status_code == 200, "sito modificato")
check(cl.get("/site", headers={"Host": "nuovo.ente.it"}).status_code == 404, "sito disattivato dall'amministrazione")
cl.put(f"{ADM}/sites/nuovo.ente.it", headers=A, json={"topic_id": "nuovo", "name": "Rinominato"})

png = b"\x89PNG\r\n\x1a\n" + b"0" * 20
check(cl.put(f"{ADM}/sites/nuovo.ente.it/icon", headers=A, json={"png": base64.b64encode(b"non png").decode()}).status_code == 422, "icona non PNG -> 422")
check(cl.put(f"{ADM}/sites/nuovo.ente.it/icon", headers=A, json={"png": base64.b64encode(png).decode()}).status_code == 200, "icona caricata")
r = cl.get("/site/icon", headers={"Host": "nuovo.ente.it"})
check(r.data == png and r.mimetype == "image/png", "icona servita al dominio giusto")
check(cl.get("/site", headers={"Host": "nuovo.ente.it"}).json["has_icon"] is True, "has_icon")
check(cl.delete(f"{ADM}/sites/nuovo.ente.it/icon", headers=A).status_code == 200 and cl.get("/site/icon", headers={"Host": "nuovo.ente.it"}).status_code == 404, "icona tolta")
# --- home: testo ripulito, pulsanti, traduzione
BTN = [{"label_it": "Sito", "label_en": "Site", "url": "https://ente.it/"}, {"label_it": "Albo", "url": "https://ente.it/albo"}]
r = cl.put(f"{ADM}/sites/nuovo.ente.it", headers=A, json={"name": "Nuovo", "buttons": BTN,
    "welcome_it": '<h1 onclick="x()">Ciao</h1><script>alert(1)</script><p><a href="javascript:x()">no</a> <a href="https://ente.it" target="_blank">sì</a> <img src=x onerror=y></p>'})
check(r.status_code == 200, "home salvata")
home = cl.get("/site", headers={"Host": "nuovo.ente.it"}).json
check(home["welcome"]["it"] == '<h1>Ciao</h1><p><a>no</a> <a href="https://ente.it">sì</a> </p>', "HTML ripulito da script, attributi e link pericolosi")
check([b["label_it"] for b in home["buttons"]] == ["Sito", "Albo"] and home["buttons"][1]["label_en"] is None, "pulsanti nell'ordine dato")
check(cl.put(f"{ADM}/sites/nuovo.ente.it", headers=A, json={"name": "N", "welcome_it": "<p> </p>"}).status_code == 200 and cl.get("/site", headers={"Host": "nuovo.ente.it"}).json["welcome"]["it"] is None, "testo vuoto: nessuna home")
check(next(x for x in cl.get(f"{ADM}/sites", headers=A).json["items"] if x["hostname"] == "nuovo.ente.it")["buttons"] == [], "elenco siti con i pulsanti")
for bad in ([{"label_it": "x", "url": "javascript:alert(1)"}], [{"label_it": "", "url": "https://a.it"}], "x", [{"label_it": "x", "url": "https://a.it"}] * 21):
    check(cl.put(f"{ADM}/sites/nuovo.ente.it", headers=A, json={"name": "N", "buttons": bad}).status_code == 422, f"pulsanti non validi -> 422: {str(bad)[:30]}")

import llm_provider
class FakeProvider:
    reply = '{"html": "<p onclick=\\"x\\">Hello</p>", "labels": ["Site"]}'
    def generate_json(self, model, prompt, **kw):
        FakeProvider.prompt = prompt
        return self.reply
llm_provider._PROVIDER_INSTANCE = FakeProvider()
r = cl.post(f"{ADM}/translate", headers=A, json={"html": "<p>Ciao</p>", "labels": ["Sito"]})
check(r.status_code == 200 and r.json == {"html": "<p>Hello</p>", "labels": ["Site"]}, "traduzione ripulita")
check("Ciao" in FakeProvider.prompt, "il testo arriva al modello")
check(cl.post(f"{ADM}/translate", headers=L, json={"html": "<p>Ciao</p>"}).status_code == 403, "traduzione: solo admin")
check(cl.post(f"{ADM}/translate", headers=A, json={"html": "", "labels": []}).status_code == 422, "niente da tradurre -> 422")
FakeProvider.reply = '{"html": "<p>Hello</p>", "labels": []}'
check(cl.post(f"{ADM}/translate", headers=A, json={"html": "<p>Ciao</p>", "labels": ["Sito"]}).status_code == 502, "etichette mancanti -> 502")
FakeProvider.reply = "non è JSON"
check(cl.post(f"{ADM}/translate", headers=A, json={"html": "<p>Ciao</p>"}).status_code == 502, "risposta non valida -> 502")
llm_provider._PROVIDER_INSTANCE = None

check(cl.delete(f"{ADM}/sites/nuovo.ente.it", headers=A).status_code == 405, "i siti non si eliminano dal web")

# --- credenziali esterne
check(cl.post(f"{ADM}/keys", headers=A, json={"name": "g", "topics": ["boh"]}).status_code == 422, "credenziale su archivio inesistente -> 422")
check(cl.post(f"{ADM}/keys", headers=A, json={"name": "g", "expires": "domani"}).status_code == 422, "scadenza non valida -> 422")
r = cl.post(f"{ADM}/keys", headers=A, json={"name": "gestionale", "topics": ["atti"], "expires": "2099-12-31"})
check(r.status_code == 201 and r.json["key"].startswith("rag_"), "credenziale creata, chiave restituita")
key, kid = r.json["key"], r.json["id"]
E = {"Authorization": f"Bearer {key}"}
check([t["id"] for t in cl.get("/external/v1/topics", headers=E).json["topics"]] == ["atti"], "la chiave funziona sulle API esterne")
listed = cl.get(f"{ADM}/keys", headers=A).json["items"][0]
check("key" not in listed and "key_hash" not in listed and listed["topics"] == ["atti"], "l'elenco non espone né chiave né hash")
check(cl.put(f"{ADM}/keys/{kid}", headers=A, json={"topics": ["lettere"], "active": True}).status_code == 200, "archivi cambiati")
check([t["id"] for t in cl.get("/external/v1/topics", headers=E).json["topics"]] == ["lettere"], "subito in vigore")
r = cl.post(f"{ADM}/keys/{kid}/rotate", headers=A)
check(cl.get("/external/v1/topics", headers=E).status_code == 401, "rigenerata: la vecchia non vale più")
E2 = {"Authorization": f"Bearer {r.json['key']}"}
check(cl.get("/external/v1/topics", headers=E2).status_code == 200, "la nuova sì")
check(cl.put(f"{ADM}/keys/{kid}", headers=A, json={"topics": ["lettere"], "active": False}).status_code == 200 and cl.get("/external/v1/topics", headers=E2).status_code == 401, "revoca")
check(cl.delete(f"{ADM}/keys/{kid}", headers=A).status_code == 200 and cl.get(f"{ADM}/keys", headers=A).json["items"] == [], "credenziale eliminata")

print(f"\n{'TUTTO OK' if not FAILED else f'{len(FAILED)} FALLITI'}")
sys.exit(1 if FAILED else 0)
