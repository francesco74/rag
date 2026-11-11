"""Test del reranker vLLM contro un server finto.

Uso:  python api/tests/test_rerank.py      (esce con 1 se qualcosa fallisce)
"""
import math, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "api" / "src"), str(ROOT / "api" / "tests")]
from fake_llm_server import FakeLLM
from reranker import VLLMReranker

failures = 0


def check(label, got, expected=True):
    global failures
    ok = got == expected
    failures += not ok
    print(("OK  " if ok else "FAIL"), label, "" if ok else f"-> {got!r}, atteso {expected!r}")


fake = FakeLLM()
reranker = VLLMReranker(fake.url + "/v1/", "bge", api_key="segreta", max_length=100, timeout=5)


def reply(*pairs):
    fake.replies.append((200, {"results": [{"index": i, "relevance_score": s} for i, s in pairs]}))


reply((1, 0.9), (0, 0.1))
res = reranker.rerank("domanda", ["a", "b"])
path, body = fake.requests[-1]
check("endpoint", path, "/v1/rerank")
check("richiesta", (body["model"], body["query"], body["documents"], body["truncate_prompt_tokens"]),
      ("bge", "domanda", ["a", "b"], 100))
check("ordine e indici", [r.index for r in res], [1, 0])
check("punteggi 0..1: probabilità usate così come sono", [r.probability for r in res], [0.9, 0.1])

reply((0, -2.0), (1, 4.0))
res = reranker.rerank("domanda", ["a", "b"])
check("logit grezzi: conversione lasciata al chiamante", [(r.index, r.score, r.probability) for r in res],
      [(1, 4.0, None), (0, -2.0, None)])

check("nessun documento, nessuna chiamata", (reranker.rerank("q", []), len(fake.requests)), ([], 2))

for label, status, body in [("errore HTTP", 500, {"error": "x"}), ("risultati mancanti", 200, {"results": [{"index": 0, "relevance_score": 1}]})]:
    fake.replies.append((status, body))
    try:
        reranker.rerank("q", ["a", "b"])
        check(label + " propagato", False)
    except Exception:
        check(label + " propagato", True)

# --- Reranking locale (onnx) in un'immagine senza modelli locali: errore che dice cosa fare
import os, tempfile
os.environ.update({k: "u" for k in ("DOCWS_RICERCA_ENDPOINT", "DOCWS_ATTI_ENDPOINT", "DOCWS_CODICE_AMMINISTRAZIONE",
                                    "DOCWS_CODICE_AOO", "RUOLO_DOCWS")}, DATA_FOLDER=tempfile.mkdtemp(), LOG_LEVEL="CRITICAL",
                  RERANKER_MODEL_PATH="/non/esiste")
sys.path.insert(0, str(ROOT))
import reranker as reranker_module
from rag import retrieval

for label, deps, attesa in (("senza librerie", False, "mancano onnxruntime e transformers"),
                            ("senza modello", True, "manca il modello in /non/esiste")):
    reranker_module.DEPENDENCIES_AVAILABLE = deps
    try:
        retrieval._check_local_reranker()
        check(f"onnx {label}: errore", False)
    except RuntimeError as e:
        check(f"onnx {label}: errore chiaro", (attesa in str(e), "MODELLI_LOCALI=1" in str(e), "RERANK_PROVIDER=vllm" in str(e)),
              (True, True, True))

print("\nRISULTATO:", "TUTTO OK" if not failures else f"{failures} FALLITI")
sys.exit(1 if failures else 0)
