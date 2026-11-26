"""Test del task della chat (process_rag_query) dall'inizio alla fine, con
Qdrant in memoria, SQLite al posto di MySQL, embedding, reranker e LLM finti.

Verifica i collegamenti fra worker.py e i moduli del pacchetto rag/: prompt
caricati dalla cartella giusta, concetti passati al rewriter, contesto con
metadati e testo ripulito, citazioni risolte, cache e metriche. Concetti
cercati sulle sole domande dell'utente; risposta che non rispetta le esclusioni
della domanda bocciata dal grader ("constraints") e rifatta. Grader del prompt del
topic (atti_grader); dati personali: filtro su codici fiscali, IBAN ed email, poi
pulizia con gli omissis della risposta bocciata e, solo se restano fino all'ultimo
tentativo, risposta non mostrata; prompt inesistente: general.

Uso:  python api/tests/test_task.py      (esce con 1 se qualcosa fallisce)
"""
import json, os, sqlite3, sys, tempfile, pathlib
os.environ.update(DOCWS_RICERCA_ENDPOINT="u", DOCWS_ATTI_ENDPOINT="u", DOCWS_CODICE_AMMINISTRAZIONE="u",
                  DOCWS_CODICE_AOO="u", RUOLO_DOCWS="u", LOG_LEVEL="WARNING")
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "api" / "src"), str(ROOT)]
import worker
from rag import state
from common.config import settings
from qdrant_client import QdrantClient, models

def patch(name, value):
    for m in [worker] + [m for k, m in sys.modules.items() if k.startswith("rag.")]:
        if hasattr(m, name): setattr(m, name, value)

DB = tempfile.mktemp()
c = sqlite3.connect(DB)
c.execute("CREATE TABLE parent_documents (id TEXT, topic_id TEXT, sub_topic_id TEXT, source TEXT, file_name TEXT, content TEXT, metadata TEXT)")
c.execute("CREATE TABLE topics (topic_id TEXT, prompt TEXT)")
c.execute("INSERT INTO topics VALUES ('attiprovincia', 'atti')")
for i in range(3):
    c.execute("INSERT INTO parent_documents VALUES (?, 'attiprovincia', 'determine', ?, ?, ?, ?)",
              (f"P{i}", f"sicraweb://{i}", f"det_{i}.pdf", f"Determina {i}: lavori scuola Viareggio liceo&nbsp;&nbsp;&nbsp;Carducci", json.dumps({"oggetto": f"Lavori {i}", "numero": str(i)})))
c.commit()
class Cur:
    def __init__(s, c): s.c = c.cursor()
    def execute(s, sql, p=()): s.c.execute(sql.replace("%s", "?"), p)
    def fetchall(s): return [dict(zip([x[0] for x in s.c.description], r)) for r in s.c.fetchall()]
    def fetchone(s):
        r = s.c.fetchone(); return dict(zip([x[0] for x in s.c.description], r)) if r else None
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

q = QdrantClient(":memory:")
q.create_collection("document_chunks", vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE))
DATES = {0: "2025-03-01", 1: "2023-05-10", 2: None}   # il terzo atto non ha data
q.upsert("document_chunks", [models.PointStruct(id=i + 1, vector=[1, 0, 0], payload={
    "content": f"Determina {i}: lavori scuola Viareggio liceo Carducci", "parent_id": f"P{i}", "topic_id": "attiprovincia",
    "sub_topic_id": "determine", "source": f"sicraweb://{i}", "file_name": f"det_{i}.pdf", "content_hash": f"h{i}",
    **({"data": DATES[i]} if DATES[i] else {})}) for i in range(3)])
q.create_collection("conceptual_dictionary", vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE))
q.upsert("conceptual_dictionary", [models.PointStruct(id=1, vector=[0, 1, 0], payload={"concept": "Versilia", "aliases": ["viareggio"], "category": "geografico"})])
state.qdrant_client = q

class Res:
    def __init__(s, i, sc): s.index, s.score = i, sc
class R:
    def rerank(self, query, docs): return [Res(i, 2.0) for i, _ in enumerate(docs)]
patch("get_reranker_pool", lambda: [R()])
vec = lambda *a, **k: [1.0, 0.0, 0.0]
patch("embed_query", vec); patch("embed_for_semantic_query", vec)
patch("embed_queries_batch", lambda qs: [[1.0, 0.0, 0.0] for _ in qs])
patch("check_semantic_cache", lambda *a, **k: None)
saved = []
patch("save_to_semantic_cache", lambda *a, **k: saved.append(a))
metrics = []
patch("log_rag_metrics", lambda **k: metrics.append(k))
feedback = []   # feedback negativi automatici (risposte non soddisfacenti)
patch("log_automatic_negative_feedback", lambda **k: feedback.append(k))
patch("get_all_sub_topics", lambda t: ["determine"])
for k, v in dict(query_rewriter_model_name="RW", answer_generator_model_name="GEN", grader_model_name="GR").items():
    object.__setattr__(settings, k, v)
prompts_seen = {}
REWRITER_DATE_RANGE = None   # periodo restituito dal rewriter finto
GRADER_VERDICTS = []         # verdetti da restituire in ordine prima del PASS
GEN_ANSWER = "<p>A Viareggio: lavori al liceo Carducci <i>[1]</i>.</p>"
redazioni = []               # richieste di pulizia con gli omissis
PULIZIA_RIESCE = True        # la pulizia finta toglie il nome (False: lo lascia)
class FakeLLM:
    def generate_json(self, model, prompt, with_meta=False, **kw):
        prompts_seen[model] = prompt
        if model == "RW":
            raw = json.dumps({"standalone_query": "Lavori sulle scuole della Versilia?", "search_queries": ["lavori scuole Viareggio"],
                              "coverage_targets": ["Viareggio"], "keywords": ["lavori", "scuola"],
                              "date_range": REWRITER_DATE_RANGE})
        elif model == "GEN":
            raw = json.dumps({"is_found": True, "is_general_knowledge": False,
                              "answer": GEN_ANSWER})
        elif "You remove personal data" in prompt:   # pulizia con gli omissis
            redazioni.append(prompt)
            raw = json.dumps({"answer": GEN_ANSWER.replace("Mario Rossi", "omissis") if PULIZIA_RIESCE else GEN_ANSWER})
        elif GRADER_VERDICTS:
            raw = json.dumps(GRADER_VERDICTS.pop(0))
        else:
            raw = json.dumps({"verdict": "PASS", "reason": "ok", "missing_targets": [], "note": ""})
        return (raw, {"truncated": False}) if with_meta else raw
patch("get_llm_provider", lambda: FakeLLM())

import logging
summaries, date_logs = [], []
class Grab(logging.Handler):
    def emit(self, r):
        m = r.getMessage()
        if "[SUMMARY]" in m: summaries.append(m)
        if "[DATE]" in m: date_logs.append(m)
worker.log.addHandler(Grab())
worker.log.setLevel(logging.INFO)

res = worker.process_rag_query.apply(args=["lavori sulle scuole della Versilia?", [], "attiprovincia"]).get()
ok = []
def check(c, m): ok.append(c); print(("OK   " if c else "FAIL ") + m)
check("<verified_concepts>" in prompts_seen.get("RW", "") and "You are an expert search query analyzer" in prompts_seen.get("RW", ""), "prompt del rewriter caricato dalla cartella prompts/")
check(res.get("status") == "success", f"task completato: {res.get('status')} {res.get('error','')}")
check("det_" in res.get("answer", "") and "[1]" not in res.get("answer", ""), f"citazione risolta nel nome file: {res.get('answer')}")
check(len(res.get("sources", [])) >= 1, f"fonti: {len(res.get('sources', []))}")
check("Versilia: viareggio" in prompts_seen.get("RW", ""), "concetto trovato per nome e passato al rewriter")
check("Metadati del documento" in prompts_seen.get("GEN", "") and "&nbsp;" not in prompts_seen.get("GEN", ""),
      "contesto al generatore: metadati presenti, testo ripulito")
check(saved and metrics and metrics[0]["is_satisfactory"], "risposta salvata in cache e metriche registrate")
check(len(summaries) == 1 and "esito=ok tentativi=1" in summaries[0]
      and all(f in summaries[0] for f in ("concetti", "riscrittura", "embedding", "ricerca", "generazione", "grader", "rerank")),
      f"riga di riepilogo con le fasi: {summaries[-1] if summaries else None}")
check("periodo=" not in summaries[0], "nessun periodo nella domanda: nessun filtro")

# --- periodo nella domanda: filtro sulla data, atti senza data inclusi
summaries.clear()
REWRITER_DATE_RANGE = {"from": "2025-01-01", "to": "2025-12-31"}
res = worker.process_rag_query.apply(args=["lavori sulle scuole della Versilia fatti nel 2025?", [], "attiprovincia"]).get()
files = sorted(src["file_name"] for src in res.get("sources", []))
check(files == ["det_0.pdf", "det_2.pdf"], f"periodo 2025: atto del 2025 e atto senza data, escluso quello del 2023 ({files})")
check(summaries and "periodo=2025-01-01→2025-12-31" in summaries[-1], "periodo riportato nel riepilogo")

# --- il periodo dei Filtri della chat ha la precedenza
summaries.clear(); date_logs.clear()
res = worker.process_rag_query.apply(args=["lavori fatti nel 2025?", [], "attiprovincia", None,
                                           {"data": {"gte": "2023-01-01", "lte": "2023-12-31"}}, False]).get()
files = sorted(src["file_name"] for src in res.get("sources", []))
check(files == ["det_1.pdf"], f"vale il periodo dei Filtri (2023, senza atti non datati): {files}")
check(date_logs and "non applicato" in date_logs[-1], "log: periodo della domanda non applicato")

# --- date non valide dal rewriter: nessun filtro
summaries.clear()
REWRITER_DATE_RANGE = {"from": "2025-12-31", "to": "2025-01-01"}
res = worker.process_rag_query.apply(args=["lavori?", [], "attiprovincia"]).get()
check(len(res.get("sources", [])) == 3 and "periodo=" not in summaries[-1], "periodo incoerente ignorato: nessun filtro")

# --- regole sui vincoli della domanda nel prompt di generazione e nel grader
check("QUERY CONSTRAINTS RULE" in prompts_seen.get("GEN", ""), "regola sui vincoli della domanda nel prompt di generazione")
check('"constraints"' in prompts_seen.get("GR", ""), "terzo motivo di bocciatura nel grader")

# --- concetti: solo le domande dell'utente, non le risposte del modello
from rag.concepts import concept_lookup_text
storia = [{"role": "user", "text": "revoche di incarichi?"}, {"role": "model", "text": "<p>" + "x" * 3000 + "</p>"},
          {"role": "user", "text": "solo affidamenti"}, {"role": "model", "text": "<ul><li>...</li></ul>"},
          {"role": "user", "text": "  "}]
check(concept_lookup_text("escludi le forniture", storia) == "revoche di incarichi? solo affidamenti escludi le forniture",
      "lookup dei concetti sulle ultime domande dell'utente")
check(concept_lookup_text("in Versilia?", None) == "in Versilia?", "senza storia: solo la domanda")

# --- grader "constraints": nuovo tentativo con l'istruzione di rispettare le esclusioni
summaries.clear(); saved.clear()
GRADER_VERDICTS.append({"verdict": "FAIL", "reason": "constraints", "missing_targets": [], "note": "forniture incluse"})
res = worker.process_rag_query.apply(args=["lavori sulle scuole della Versilia?", [], "attiprovincia"]).get()
check(res.get("status") == "success" and "tentativi=2" in summaries[-1], f"bocciata per i vincoli, rifatta: {summaries[-1:]}")
check("includeva documenti che la richiesta esclude" in prompts_seen.get("RW", ""), "istruzione del nuovo tentativo sui vincoli")

# --- grader del prompt del topic e prompt inesistente
from rag.prompts import get_topic_prompt, grader_name
check("omissis" in prompts_seen.get("GR", "") and '"privacy"' in prompts_seen.get("GR", ""),
      "topic con prompt atti: grader atti_grader")
check((grader_name("atti"), grader_name("greenlees"), grader_name("general"), grader_name(None))
      == ("atti_grader", "greenlees_grader", "grader", "grader"), "grader per prompt, altrimenti quello comune")
c = sqlite3.connect(DB); c.execute("INSERT INTO topics VALUES ('vecchio', 'protocollo')"); c.commit(); c.close()
check((get_topic_prompt("vecchio"), get_topic_prompt("attiprovincia"), get_topic_prompt("assente"))
      == ("general", "atti", "general"), "prompt inesistente o non indicato: general")

# --- atti: dati personali sostituiti da omissis nella risposta
from rag.generation import with_omissis
check(with_omissis("CF RSSMRA80A01E715X, IBAN IT31 T 05034 13701 000000006237, mail mario.rossi@pec.it; CIG 88794691A0")
      == "CF omissis, IBAN omissis, mail omissis; CIG 88794691A0", "codice fiscale, IBAN ed email in omissis, il CIG resta")
summaries.clear()
GEN_ANSWER = "<p>Incarico, codice fiscale RSSMRA80A01E715X <i>[1]</i>.</p>"
res = worker.process_rag_query.apply(args=["lavori sulle scuole della Versilia?", [], "attiprovincia"]).get()
check("RSSMRA80A01E715X" not in res.get("answer", "") and "omissis" in res.get("answer", ""), f"risposta con omissis: {res.get('answer')}")

# --- dati personali: la risposta si ripulisce con gli omissis, senza rifare la ricerca
summaries.clear(); saved.clear(); redazioni.clear()
GEN_ANSWER = "<p>Incarico all'ing. Mario Rossi <i>[1]</i>.</p>"
GRADER_VERDICTS.append({"verdict": "FAIL", "reason": "privacy", "missing_targets": [], "note": "nome"})
res = worker.process_rag_query.apply(args=["lavori sulle scuole della Versilia?", [], "attiprovincia"]).get()
check(res.get("answer", "").startswith("<p>Incarico all'ing. omissis") and "det_" in res.get("answer", ""),
      f"risposta ripulita con gli omissis e citazione risolta: {res.get('answer')}")
check(len(redazioni) == 1 and "Mario Rossi" in redazioni[0] and "[1]" in redazioni[0], "pulizia sulla risposta con i riferimenti numerici")
check("tentativi=1" in summaries[-1] and saved and "Mario Rossi" not in saved[-1][2], "nessun nuovo giro di ricerca; in cache la versione ripulita")

# --- dati personali anche dopo la pulizia, fino all'ultimo tentativo: la risposta non si mostra
summaries.clear(); saved.clear()
GEN_ANSWER = "<p>Incarico all'ing. Mario Rossi <i>[1]</i>.</p>"
PULIZIA_RIESCE = False
GRADER_VERDICTS.extend([{"verdict": "FAIL", "reason": "privacy", "missing_targets": [], "note": "nome"}] * 50)
res = worker.process_rag_query.apply(args=["lavori sulle scuole della Versilia?", [], "attiprovincia"]).get()
GRADER_VERDICTS.clear(); PULIZIA_RIESCE = True
GEN_ANSWER = "<p>A Viareggio: lavori al liceo Carducci <i>[1]</i>.</p>"
redazioni = []               # richieste di pulizia con gli omissis
PULIZIA_RIESCE = True        # la pulizia finta toglie il nome (False: lo lascia)
check("Mario Rossi" not in res.get("answer", "") and "dati personali" in res.get("answer", "") and not saved,
      f"risposta non mostrata né salvata in cache: {res.get('answer')}")
check(feedback and "Mario Rossi" not in feedback[-1]["answer"], "nemmeno nel feedback automatico")
check("riformulare" not in prompts_seen.get("RW", "") and "sostituiti con omissis" in prompts_seen.get("RW", ""),
      "istruzione del nuovo tentativo sui dati personali")

# --- errore: la riga di riepilogo c'è comunque
summaries.clear()
patch("get_all_sub_topics", lambda t: [])
res = worker.process_rag_query.apply(args=["x", [], "attiprovincia"]).get()
check(res["status"] == "failed" and summaries and "esito=errore (Configuration Error)" in summaries[-1],
      "anche su errore una riga di riepilogo")

# --- log HTTP silenziati qualunque sia LOG_LEVEL
worker.configure_worker_logging()
check(all(logging.getLogger(n).level == logging.WARNING for n in ("httpx", "httpcore")), "httpx e httpcore a WARNING")

print("\nRISULTATO:", "TUTTO OK" if all(ok) else "FALLITO")
sys.exit(0 if all(ok) else 1)
