"""
Riordino lessicale dei candidati, senza modelli (RERANK_PROVIDER=lessicale).

Serve in due casi:
  - come reranker scelto, dove un cross-encoder non c'è o è troppo lento;
  - come ripiego automatico quando il reranker configurato (vllm, onnx) va in
    errore o in timeout: la ricerca prosegue con questo ordine invece che con la
    sola similarità vettoriale.

Il punteggio di ogni candidato è una somma pesata, tra 0 e 1, di:
  vettore         similarità del coseno restituita da Qdrant (semantica dell'argomento);
  bm25            BM25 dei termini della domanda, con rarità e lunghezza calcolate
                  sull'insieme dei candidati (relativo al migliore);
  copertura       quota dei termini della domanda presenti, pesata per rarità;
  identificativi  quota di numeri, articoli, commi, date, sigle e codici della
                  domanda presenti nel passaggio;
  vicinanza       i termini trovati stanno vicini (finestra minima che li contiene).

Non coglie il senso della relazione tra domanda e passaggio (negazioni, chi fa
cosa): quello lo fa solo un cross-encoder. Ordina però per argomento e per
corrispondenza precisa, molto meglio della sola similarità vettoriale.
"""

import math
import re
import time
import logging
from collections import Counter
from typing import List, Optional, Sequence

from reranker import RerankResult

log = logging.getLogger(__name__)

WEIGHTS = {"vettore": 0.40, "bm25": 0.25, "copertura": 0.15, "identificativi": 0.10, "vicinanza": 0.10}
BM25_K1 = 1.2
BM25_B = 0.75

STOPWORDS = frozenset("""
a ad al allo ai agli all alla alle anche come con col coi da dal dallo dai dagli dall dalla dalle dei
del dello degli dell della delle di e ed è era essere fra gli i il in la le lo ma mi ne nei nel nello
negli nell nella nelle non o per più quale quali quando quanto quanti quanta quante questo questa questi
queste quello quella quelli quelle se si sia sono su sul sullo sui sugli sull sulla sulle tra un una
uno che chi cui cosa dove perché poi suo sua suoi sue loro ci vi ha hanno ho hai essere stato stata
stati state viene vengono deve devono può possono sono gli lo l d s c un dell nell all
""".split())

WORD_RE = re.compile(r"\w+", re.UNICODE)
FINAL_VOWELS = "aeiouàèéìòù"
# Riferimenti normativi: "art. 35", "articolo 35", "comma 2", "lett. b".
ARTICLE_RE = re.compile(r"\b(art(?:icol[oi])?|comma|commi|lett(?:era)?)\.?\s*(\d+[a-z]?(?:[-/]?(?:bis|ter|quater))?|[a-z])\b",
                        re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d[\d.,/-]*\d\b|\b\d\b")
# Sigle e codici scritti in maiuscolo nella domanda (ANAC, DURC, CIG, codici alfanumerici).
ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,}[A-Z0-9]*\b")


def _stem(word: str) -> str:
    """Radice grossolana per l'italiano: delibera/delibere, contributo/contributi."""
    if len(word) > 4:
        word = word.rstrip(FINAL_VOWELS) or word
    return word[:7]


def terms(text: str) -> List[str]:
    """Radici dei termini significativi, nell'ordine in cui compaiono."""
    out = []
    for w in WORD_RE.findall(text.lower()):
        if w in STOPWORDS or (len(w) < 3 and not w.isdigit()):
            continue
        out.append(w if w.isdigit() else _stem(w))
    return out


def _norm_ref(kind: str, value: str) -> str:
    kind = kind.lower()
    kind = "art" if kind.startswith("art") else "comma" if kind.startswith("comm") else "lett"
    return f"{kind} {value.lower()}"


def identifiers(query: str) -> List[str]:
    """Identificativi della domanda, in forma normalizzata (minuscolo, spazi singoli)."""
    found = {_norm_ref(k, v) for k, v in ARTICLE_RE.findall(query)}
    found |= {n.replace(",", ".") for n in NUMBER_RE.findall(query)}
    found |= {a.lower() for a in ACRONYM_RE.findall(query) if len(a) >= 2}
    return sorted(found)


def _doc_identifiers(text: str) -> set:
    found = {_norm_ref(k, v) for k, v in ARTICLE_RE.findall(text)}
    found |= {n.replace(",", ".") for n in NUMBER_RE.findall(text)}
    found |= {w.lower() for w in WORD_RE.findall(text)}
    return found


def _min_window(positions: dict) -> Optional[int]:
    """Larghezza della finestra minima che contiene almeno una occorrenza di ogni termine."""
    events = sorted((p, t) for t, ps in positions.items() for p in ps)
    need, have, best, left = len(positions), Counter(), None, 0
    for right, (pos, term) in enumerate(events):
        have[term] += 1
        while len(have) == need:
            width = pos - events[left][0] + 1
            best = width if best is None or width < best else best
            lt = events[left][1]
            have[lt] -= 1
            if not have[lt]:
                del have[lt]
            left += 1
    return best


class LexicalReranker:
    """Stessa interfaccia dei reranker con modello: rerank(query, documents) -> [RerankResult]."""

    def rerank(self, query: str, documents: List[str],
               vector_scores: Optional[Sequence[Optional[float]]] = None) -> List[RerankResult]:
        if not documents:
            return []
        started = time.perf_counter()
        q_terms = list(dict.fromkeys(terms(query)))
        q_ids = identifiers(query)
        docs_terms = [terms(d) for d in documents]
        n = len(documents)
        avgdl = (sum(len(t) for t in docs_terms) / n) or 1.0
        df = Counter(t for dt in docs_terms for t in set(dt) if t in q_terms)
        # Rarità sui candidati: un termine presente in tutti non distingue fra loro,
        # ma pesa comunque un poco nella copertura (+0.1).
        idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in q_terms}
        idf_total = sum(idf[t] + 0.1 for t in q_terms) or 1.0

        known = [s for s in (vector_scores or []) if s is not None]
        missing_vector = min(known) if known else None   # candidati della sola ricerca testuale

        weights = dict(WEIGHTS)
        if not q_ids:
            weights.pop("identificativi")
        if not known:
            weights.pop("vettore")
        if not q_terms:
            for k in ("bm25", "copertura", "vicinanza"):
                weights.pop(k)
        total_w = sum(weights.values()) or 1.0

        bm25 = []
        parts = []
        for i, dt in enumerate(docs_terms):
            tf = Counter(dt)
            dl = len(dt) or 1
            bm25.append(sum(
                idf[t] * tf[t] * (BM25_K1 + 1) / (tf[t] + BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl))
                for t in q_terms if tf[t]))
            present = [t for t in q_terms if tf[t]]
            coverage = sum(idf[t] + 0.1 for t in present) / idf_total
            proximity = 0.0
            if len(present) >= 2:
                positions = {t: [p for p, w in enumerate(dt) if w == t] for t in present}
                width = _min_window(positions)
                proximity = len(present) / width if width else 0.0
            ids_score = 0.0
            if q_ids:
                doc_ids = _doc_identifiers(documents[i])
                ids_score = sum(1 for x in q_ids if x in doc_ids) / len(q_ids)
            vec = vector_scores[i] if vector_scores and i < len(vector_scores) else None
            vec = missing_vector if vec is None else vec
            parts.append({"copertura": coverage, "vicinanza": min(1.0, proximity),
                          "identificativi": ids_score,
                          "vettore": max(0.0, min(1.0, vec)) if vec is not None else 0.0})

        best_bm25 = max(bm25) or 1.0
        results = []
        for i, p in enumerate(parts):
            p["bm25"] = bm25[i] / best_bm25
            score = sum(w * p[k] for k, w in weights.items()) / total_w
            results.append(RerankResult(index=i, score=score, probability=score))
        results.sort(key=lambda r: r.score, reverse=True)
        log.info(f"Riordino lessicale di {n} documenti: {time.perf_counter() - started:.3f}s")
        return results
