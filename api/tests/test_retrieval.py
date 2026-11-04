"""Test del motore di ricerca: Qdrant in memoria, SQLite al posto di MySQL,
reranker finto. Nessuna chiamata a servizi esterni.

Copre la ricerca per parole chiave a combinazioni, i concetti riconosciuti per
nome, la composizione del contesto (testo OCR ripulito, metadati, rimandi) e i
posti extra per pertinenza.

Uso:  python api/tests/test_retrieval.py            (esce con 1 se qualcosa fallisce)
      python api/tests/test_retrieval.py --snapshot FILE   salva l'output per confronti
"""
import json, os, sqlite3, sys, tempfile, pathlib

os.environ.update(DOCWS_RICERCA_ENDPOINT="u", DOCWS_ATTI_ENDPOINT="u", DOCWS_CODICE_AMMINISTRAZIONE="u",
                  DOCWS_CODICE_AOO="u", RUOLO_DOCWS="u", LOG_LEVEL="WARNING")
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "api" / "src"), str(ROOT)]

import worker  # noqa: E402  (registra anche i moduli del motore)
from common.config import settings  # noqa: E402
from rag.concepts import find_concepts, get_concepts_by_name  # noqa: E402
from rag.context import build_context  # noqa: E402
from rag.keywords import build_keyword_groups, keyword_search, plan_keyword_combinations  # noqa: E402
from rag.retrieval import retrieve_chunks  # noqa: E402
from qdrant_client import QdrantClient, models  # noqa: E402


def patch(name, value):
    """Sostituisce `name` ovunque sia usato: nel worker e nei moduli del motore."""
    for mod in [worker] + [m for k, m in sys.modules.items() if k == "rag" or k.startswith("rag.")]:
        if hasattr(mod, name):
            setattr(mod, name, value)


def set_qdrant(client):
    patch("qdrant_client", client)


def set_setting(**kw):
    for k, v in kw.items():
        object.__setattr__(settings, k, v)


FAILED = []


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        FAILED.append(msg)


SNAPSHOT = {}

# ---------------------------------------------------------------------------
# Database finto (parent_documents) e reranker finto
# ---------------------------------------------------------------------------
DB = tempfile.mktemp(suffix=".db")


class Cur:
    def __init__(s, c): s.c = c.cursor()
    def execute(s, sql, p=()): s.c.execute(sql.replace("%s", "?"), p)
    def fetchall(s): return [dict(zip([x[0] for x in s.c.description], r)) for r in s.c.fetchall()]
    def fetchone(s):
        r = s.c.fetchone()
        return dict(zip([x[0] for x in s.c.description], r)) if r else None
    def __enter__(s): return s
    def __exit__(s, *a): pass


class Conn:
    def __init__(s): s.c = sqlite3.connect(DB)
    def cursor(s, dictionary=False): return Cur(s.c)
    def close(s): s.c.close()
    def is_connected(s): return True
    def commit(s): s.c.commit()


patch("get_db_connection", lambda: Conn())
patch("load_active_boilerplate", lambda *a, **k: None)


class Res:
    def __init__(s, index, score): s.index, s.score = index, score


class KeywordReranker:
    """Logit alto ai testi che contengono una delle parole indicate."""
    words = ()

    def rerank(self, query, docs):
        return [Res(i, 3.0 if any(w in d.lower() for w in self.words) else -3.0) for i, d in enumerate(docs)]


reranker = KeywordReranker()
patch("get_reranker_pool", lambda: [reranker])

c = sqlite3.connect(DB)
c.execute("CREATE TABLE parent_documents (id TEXT, topic_id TEXT, sub_topic_id TEXT, source TEXT, "
          "file_name TEXT, content TEXT, metadata TEXT)")
c.commit()


def new_collection():
    q = QdrantClient(":memory:")
    q.create_collection("document_chunks", vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE))
    return q


# ===========================================================================
# 1. Ricerca per parole chiave a combinazioni
# ===========================================================================
q = new_collection()
pts, n = [], 0


def add(text, tag, k=1):
    global n
    for _ in range(k):
        n += 1
        pts.append(models.PointStruct(id=n, vector=[1, 0, 0], payload={
            "content": f"{text} {n}", "topic_id": "t", "content_hash": f"{tag}{n}", "parent_id": f"{tag}{n}"}))


add("atti amministrativi generici", "g", 300)
add("atti incarico a edoardo parisi", "pe", 2)
add("atti firmati da parisi", "p", 3)
add("viareggio spiagge turismo", "vt", 60)
add("dup elenco comuni viareggio pietrasanta camaiore", "dup", 10)
add("lavori scuole viareggio liceo", "sv", 30)
add("lavori scuole pietrasanta istituto", "sp", 3)
add("lavori scuole lucca", "sl", 80)
q.upsert("document_chunks", pts)
set_qdrant(q)
set_setting(qdrant_syntactic_size=20)

MUST = [models.FieldCondition(key="topic_id", match=models.MatchValue(value="t"))]
VERS = {"concept": "Versilia", "aliases": ["viareggio", "pietrasanta", "camaiore"], "category": "geografico"}
EDIL = {"concept": "Edilizia Scolastica", "aliases": ["pnrr scuole", "palestre"], "category": "amministrativo"}


def kw(keywords, max_hits, concepts=None):
    set_setting(keyword_max_hits=max_hits)
    return [h.payload["content_hash"] for h in keyword_search(keywords, concepts, MUST, False, "T")]


tags = lambda hs: [h.rstrip("0123456789") for h in hs]

g = build_keyword_groups(["Versilia", "viareggio", "lavori", "scuole", "Edilizia scolastica", "lavori", "x"],
                                [VERS, EDIL])
check([x["label"] for x in g] == ["Versilia", "Edilizia Scolastica", "lavori", "scuole"],
      "parole chiave: alias e nomi dei concetti nel loro gruppo, doppioni e parole corte via")

h = kw(["atti", "Parisi", "Edoardo Parisi"], 100)
SNAPSHOT["kw_parisi"] = h
check(set(tags(h[:2])) == {"pe"} and set(tags(h[2:5])) == {"p"} and "g" not in tags(h),
      "parole chiave: prima la combinazione più ampia, mai la parola troppo comune da sola")

h = kw(["Versilia", "lavori", "scuole"], 50, [VERS])
SNAPSHOT["kw_versilia"] = h
check(tags(h[:10]).count("sp") == 3 and "sv" in tags(h[:10]), "parole chiave: posti divisi fra i comuni")
check(not {"dup", "vt", "sl"} & set(tags(h)), "parole chiave: niente DUP, turismo o scuole di altri comuni")

h = kw(["lavori", "scuole"], 10)
check(len(h) == 20 and set(tags(h)) <= {"sv", "sp", "sl"}, "parole chiave: tutto oltre soglia, si usa la più specifica")
check(kw(["zzznessuno", "ab"], 100) == [], "parole chiave: nessun risultato")

calls = []
used, counts = plan_keyword_combinations(6, lambda combo: calls.append(combo) or 5, 100)
check(len(calls) == 63 and used[0][0] == (0, 1, 2, 3, 4, 5), "parole chiave: 6 gruppi, 63 conteggi")
used, counts = plan_keyword_combinations(4, lambda combo: 0 if combo == (0, 1) else 5, 100)
check((0, 1, 2) not in counts and (0, 2, 3) in counts, "parole chiave: superinsiemi di una coppia a zero non contati")

# ===========================================================================
# 2. Concetti riconosciuti per nome
# ===========================================================================
cq = QdrantClient(":memory:")
cq.create_collection("conceptual_dictionary", vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE))
cq.upsert("conceptual_dictionary", [
    models.PointStruct(id=1, vector=[1, 0, 0], payload={"concept": "Versilia", "aliases": ["viareggio"], "category": "geografico"}),
    models.PointStruct(id=2, vector=[0, 1, 0], payload={"concept": "Piana di Lucca", "aliases": ["lucca"], "category": "geografico"}),
    models.PointStruct(id=3, vector=[0, 0, 1], payload={"concept": "Smart Working", "aliases": ["lavoro agile"], "category": "amministrativo"}),
])
set_qdrant(cq)
names = lambda cs: [x["concept"] for x in cs]
check(names(find_concepts("gli atti sulle scuole della Versilia nel 2025?", [0, 0, 1], "T")) == ["Versilia", "Smart Working"],
      "concetti: per nome + per similarità, senza doppioni")
check(names(get_concepts_by_name("scuole di Viareggio", "T")) == [], "concetti: un alias da solo non basta")
check(names(get_concepts_by_name("atti della PIANA DI LUCCA", "T")) == ["Piana di Lucca"], "concetti: nome su più parole")
check(names(get_concepts_by_name("versiliana estate", "T")) == [], "concetti: solo parola intera")

# ===========================================================================
# 2bis. Periodo della domanda (date_range del rewriter)
# ===========================================================================
from rag.rewriter import parse_date_range  # noqa: E402
check(parse_date_range({"from": "2025-01-01", "to": "2025-12-31"}) == {"gte": "2025-01-01", "lte": "2025-12-31"},
      "date: anno intero")
check(parse_date_range({"from": "2023-01-01", "to": None}) == {"gte": "2023-01-01"}, "date: solo inizio")
check(parse_date_range({"from": None, "to": None}) is None and parse_date_range(None) is None, "date: nessun periodo")
check(parse_date_range({"from": "2025-12-31", "to": "2025-01-01"}) is None, "date: inizio dopo la fine, ignorato")
check(parse_date_range({"from": "marzo 2025"}) is None, "date: formato non valido, ignorato")

# ===========================================================================
# 3. retrieve_chunks + build_context: OCR ripulito, metadati, rimandi
# ===========================================================================
N = "&nbsp;"
letter = (f"165\nProv. di Napoli.\nDear Sirs,\n{N*8}I am enclosing the form which I have filled up.\n"
          + "Filler paragraph about the villa. " * 40 +
          f"\n{N*36}Yours faithfully,\n{N*36}Ian Greenlees.\nRent{N*8}£120\nRepairs{N*6}£45{N*6}£12\n")
meta = {"oggetto": "Lettera assicurazione villa", "anno": "1962", "data": "1962-08-29", "id_sicraweb": "x"}
c.execute("INSERT INTO parent_documents VALUES ('L1','greenlees','lettere','direct://l.jpg','l.jpg',?,?)",
          (letter, json.dumps(meta)))
c.commit()
q = new_collection()
pieces = [letter[i:i + 300] for i in range(0, len(letter), 300)]
q.upsert("document_chunks", [models.PointStruct(id=i + 1, vector=[1.0, 0.01 * i, 0.0], payload={
    "content": t, "parent_id": "L1", "topic_id": "greenlees", "sub_topic_id": "lettere", "source": "direct://l.jpg",
    "file_name": "l.jpg", "content_hash": f"h{i}", "child_index": i}) for i, t in enumerate(pieces)])
set_qdrant(q)
reranker.words = ("greenlees", "rent")
ctx, sources, stats = retrieve_chunks(["lettera Greenlees"], [[1.0, 0.0, 0.0]], [], "greenlees", [], {},
                                             task_id="t", semantic_size=40, semantic_threshold=0.0)
text, index_to_item, cstats = build_context(ctx, task_id="t")
SNAPSHOT["ctx_letter"] = text
SNAPSHOT["stats_letter"] = {k: v for k, v in stats.items() if isinstance(v, (int, float, str))}
check(len(ctx) == 1 and "&nbsp;" not in text, "contesto: testo OCR senza &nbsp;")
check("Yours faithfully,\nIan Greenlees." in text and "Repairs\t£45\t£12" in text,
      "contesto: firma senza rientro, colonne con tabulazione")
check("Metadati del documento:" in text and "Lettera assicurazione villa" in text and "id_sicraweb" not in text,
      "contesto: metadati del documento, campi tecnici esclusi")
check(len(sources) == 1 and sources[0]["file_name"] == "l.jpg", "contesto: fonte restituita")

# ===========================================================================
# 4. Posti extra per pertinenza
# ===========================================================================
c.execute("DELETE FROM parent_documents")
docs = [(f"capp{i}", f"DETERMINAZIONE n. {100 + i}: affidamento alla Dott.ssa Valentina CAPPELLINI, periodo {i}")
        for i in range(8)] + [(f"gen{i}", f"Clausole standard di affidamento n.{i}: tracciabilità") for i in range(10)]
for name, t in docs:
    c.execute("INSERT INTO parent_documents VALUES (?,?,?,?,?,?,?)",
              (f"P-{name}", "attiprovincia", "determine", f"sicraweb://{name}", f"{name}.pdf", t * 20,
               json.dumps({"oggetto": t[:50]})))
c.commit()
q = new_collection()
pts, i = [], 0
for name, t in docs:
    for k in range(2):
        i += 1
        pts.append(models.PointStruct(id=i, vector=[1.0, 0.02 * (i % 5), 0.01 * k], payload={
            "content": f"{t} (parte {k})", "parent_id": f"P-{name}", "topic_id": "attiprovincia",
            "sub_topic_id": "determine", "source": f"sicraweb://{name}", "file_name": f"{name}.pdf",
            "content_hash": f"c{i}", "child_index": k}))
q.upsert("document_chunks", pts)
set_qdrant(q)
reranker.words = ("cappellini",)


def run(**over):
    base = dict(extra_parents=True, extra_parents_min_prob=0.5, max_sub_queries=5, parents_per_query=4,
                context_score_source="rerank")
    set_setting(**{**base, **over})
    ctx, _, st = retrieve_chunks(["affidamenti a Valentina Cappellini"], [[1.0, 0.05, 0.0]], [],
                                        "attiprovincia", [], {}, task_id="t", semantic_size=40, semantic_threshold=0.0)
    return sorted(it["source"].split("//")[1] for it in ctx), st


nm, st = run(extra_parents=False)
check(len(nm) == 4, "posti extra: disattivati, solo la quota (4)")
nm, st = run()
SNAPSHOT["extra"] = nm
check(len(nm) == 8 and all(x.startswith("capp") for x in nm) and st["n_extra_parents"] == 4,
      "posti extra: 8 parent, tutti pertinenti")
nm, _ = run(max_sub_queries=1)
check(len(nm) == 4, "posti extra: tetto MAX_SUB_QUERIES × PARENTS_PER_QUERY")
nm, _ = run(extra_parents_min_prob=0.99)
check(len(nm) == 4, "posti extra: soglia di probabilità rispettata")

# ---------------------------------------------------------------------------
# Punteggi già normalizzati dal server (vLLM): niente seconda sigmoide. Con 0.4
# sotto EXTRA_PARENTS_MIN_PROB non scattano i posti extra; sigmoid(0.4) = 0.6 li aprirebbe.
class NormalizedReranker:
    def rerank(self, query, docs):
        out = []
        for i, _ in enumerate(docs):
            res = Res(i, 0.4)
            res.probability = 0.4
            out.append(res)
        return out


patch("get_reranker_pool", lambda: [NormalizedReranker()])
nm, st = run()
check(len(nm) == 4, "punteggi 0..1 del server: usati come probabilità, senza seconda sigmoide")

if "--snapshot" in sys.argv:
    pathlib.Path(sys.argv[sys.argv.index("--snapshot") + 1]).write_text(
        json.dumps(SNAPSHOT, ensure_ascii=False, indent=1, sort_keys=True))
print("\nRISULTATO:", "FALLITO" if FAILED else "TUTTO OK")
sys.exit(1 if FAILED else 0)
