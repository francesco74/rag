"""Test del riordino lessicale (RERANK_PROVIDER=lessicale e ripiego dei reranker).

Uso:  python api/tests/test_lexical_rerank.py      (esce con 1 se qualcosa fallisce)
"""
import pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "api" / "src")]
from lexical_reranker import LexicalReranker, identifiers, terms  # noqa: E402

FAILED = []


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        FAILED.append(msg)


r = LexicalReranker()


def order(query, docs, vectors=None):
    return [res.index for res in r.rerank(query, docs, vectors)]


check(terms("Le delibere sui contributi") == terms("delibera contributo"), "radici: singolare e plurale coincidono")
check(set(identifiers("Cosa prevede l'art. 35 comma 2 del d.lgs. 36/2023 per l'ANAC?"))
      >= {"art 35", "comma 2", "36/2023", "anac"}, "identificativi: articolo, comma, numero, sigla")

docs = [
    "La provincia approva il bilancio di previsione e il documento unico di programmazione.",
    "Ai sensi dell'articolo 35, comma 2, l'affidamento diretto è consentito sotto soglia.",
    "L'articolo 36 disciplina i contratti sotto soglia e l'affidamento diretto.",
]
check(order("art. 35 comma 2 affidamento diretto", docs, [0.7, 0.7, 0.7])[0] == 1,
      "a parità di vettore vince il passaggio con l'articolo e il comma giusti")

docs = [
    "scadenza termine presentazione domande contributo",
    "il contributo è concesso ai comuni. La domanda va presentata. Il termine di scadenza è fissato in altra parte del bando, dopo molte righe di testo che parlano d'altro",
]
check(order("termine di scadenza per la domanda di contributo", docs, [0.6, 0.6])[0] == 0,
      "termini vicini e testo breve battono termini sparsi")

docs = ["testo che non c'entra nulla con la domanda", "altro testo diverso"]
res = r.rerank("affidamento lavori scuole", docs, [0.9, 0.5])
check(res[0].index == 0, "senza termini in comune decide la similarità vettoriale")
check(all(0.0 <= x.probability <= 1.0 for x in res), "punteggi tra 0 e 1, usati come probabilità")

res = r.rerank("affidamento lavori", ["affidamento lavori scuole", "nulla"], [None, None])
check(res[0].index == 0, "candidati della sola ricerca testuale (senza punteggio vettoriale)")
check(r.rerank("domanda", []) == [], "lista vuota")

print("\nRISULTATO:", "FALLITO" if FAILED else "TUTTO OK")
sys.exit(1 if FAILED else 0)
