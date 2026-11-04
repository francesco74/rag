"""Composizione del contesto per il generatore: core, budget, finestre, metadati."""

import json
import logging
import re

from common.config import settings

log = logging.getLogger("WORKER")

# Tre stadi, in quest'ordine:
#   1. compute_core_spans()        — dove stanno, dentro il parent, i child che
#                                    hanno fatto match. Sono INCOMPRIMIBILI.
#   2. allocate_context_budget()   — quanti caratteri spettano a ciascun parent,
#                                    con equità garantita tra le sottoquery.
#   3. render_parent_window()      — ritaglio effettivo: i core sopravvivono
#                                    sempre, a cedere sono i margini.
#
# Il principio è che il budget si spende sui contenuti che il retrieval ha
# giudicato rilevanti, non sul testo di contorno; e che nessuna sottoquery può
# monopolizzare il contesto solo perché il suo comune ha più documenti a
# catalogo.


# Quanti child (per parent) teniamo come "ancore" per il windowing. Tenerne
# più di uno permette di coprire casi in cui più frammenti dello stesso
# documento sono stati rilevanti per query diverse.
SNIPPETS_PER_PARENT = 3


# Lunghezza dell'ancora usata per localizzare lo snippet dentro il parent:
# corta a sufficienza da tollerare piccole differenze di whitespace/normalizzazione
# tra come il child è stato salvato e come appare dentro il testo del parent.
SNIPPET_ANCHOR_LEN = 80


# Marcatore esplicito al posto del vecchio "[...]". NON è cosmetica: serve a
# impedire che il modello legga due passaggi distanti come contigui e ci
# costruisca sopra un nesso che nel documento non esiste. Su atti
# amministrativi, dove un comma può essere separato dalla sua deroga da venti
# pagine, il rischio è concreto.
OMISSION_MARKER = "\n[... porzione di documento omessa ...]\n"


# Se non riusciamo a localizzare nessuno snippet dentro il parent, questo è il
# minimo che gli garantiamo comunque (troncamento dall'inizio).
FALLBACK_MIN_CHARS = 600


# Terminatori di frase/paragrafo per lo "snap": arretriamo il taglio fino al
# confine più vicino DENTRO il budget, quindi a costo zero.
_SENTENCE_END_RE = re.compile(r"[.!?;:»)\]]\s")


_PARAGRAPH_END_RE = re.compile(r"\n\s*\n")


def _whitespace_tolerant_find(content: str, anchor: str):
    """
    Cerca `anchor` dentro `content` tollerando differenze di whitespace, e
    restituisce (start, end) come indici VALIDI SU `content`, oppure None.

    Sostituisce il vecchio fallback che cercava dentro
    `" ".join(content.split())` e poi usava l'indice trovato come se fosse un
    indice del testo originale: la normalizzazione collassa gli spazi, quindi
    lo scarto cresce con la quantità di whitespace e su un documento pieno di
    tabulazioni la finestra poteva slittare di centinaia di caratteri,
    centrandosi sul punto sbagliato.
    """
    tokens = anchor.split()
    if not tokens:
        return None
    pattern = r"\s+".join(re.escape(t) for t in tokens)
    try:
        m = re.search(pattern, content)
    except re.error:
        return None
    return (m.start(), m.end()) if m else None


def compute_core_spans(content: str, matched_snippets: list, task_id: str = "UNKNOWN") -> list:
    """
    Localizza dentro il parent i child che hanno fatto match e restituisce una
    lista ordinata e deduplicata di dict {start, end, score}: sono i "core",
    le porzioni di testo che NON devono mai essere sacrificate.

    Localizzazione a due livelli, in ordine di affidabilità:
      1. OFFSET dal payload (start_char/end_char, popolati dal backfill o
         dall'ingestion): esatti e gratuiti. Un sanity check (l'ancora deve
         comparire nel testo puntato dagli offset) protegge dal caso di parent
         modificato dopo il calcolo degli offset.
      2. Ricerca dell'ancora tollerante ai whitespace sul testo ORIGINALE.

    matched_snippets: lista di dict {content, score, start_char?, end_char?}.
    Per retrocompatibilità accetta anche semplici stringhe.
    """
    spans = []
    offset_hits = 0

    for snippet in matched_snippets or []:
        # Retrocompatibilità: snippet può essere una stringa o un dict
        if isinstance(snippet, str):
            snippet_text, s_char, e_char, score = snippet, None, None, 0.0
        else:
            snippet_text = snippet.get("content", "")
            s_char = snippet.get("start_char")
            e_char = snippet.get("end_char")
            score = snippet.get("score", 0.0) or 0.0

        if not snippet_text:
            continue
        anchor = " ".join(snippet_text.split())[:SNIPPET_ANCHOR_LEN]
        if not anchor:
            continue

        # Livello 1: offset dal payload, con sanity check
        if (
            s_char is not None and e_char is not None
            and 0 <= s_char < e_char <= len(content)
        ):
            window_text_norm = " ".join(content[s_char:e_char].split())
            if anchor[:40] in window_text_norm:
                spans.append({"start": s_char, "end": e_char, "score": score})
                offset_hits += 1
                continue
            log.debug(
                f"[{task_id}] [WINDOW] Offset ({s_char},{e_char}) non combaciano col contenuto "
                f"(parent modificato dopo il backfill?): fallback a ricerca testuale."
            )

        # Livello 2: ricerca tollerante ai whitespace, indici sempre coerenti
        found = _whitespace_tolerant_find(content, anchor)
        if found:
            start, _ = found
            # L'ancora è solo il PREFISSO dello snippet: il core si estende per
            # tutta la lunghezza dello snippet originale, non solo dell'ancora.
            end = min(len(content), start + len(snippet_text))
            spans.append({"start": start, "end": end, "score": score})

    if not spans:
        return []

    # Merge dei core sovrapposti: lo score del blocco unito è il massimo dei suoi.
    spans.sort(key=lambda s: s["start"])
    merged = [spans[0]]
    for s in spans[1:]:
        last = merged[-1]
        if s["start"] <= last["end"]:
            last["end"] = max(last["end"], s["end"])
            last["score"] = max(last["score"], s["score"])
        else:
            merged.append(s)

    log.debug(
        f"[{task_id}] [WINDOW] {len(merged)} core individuati "
        f"({offset_hits} via offset, {len(spans) - offset_hits} via ricerca testuale)."
    )
    return merged


def estimate_parent_needs(content: str, core_spans: list) -> tuple:
    """
    Restituisce (need_min, need_full) per un parent.

    need_min  = spazio sotto il quale si perde informazione che il retrieval
                aveva giudicato rilevante: la somma dei core più i marcatori
                di omissione che li separano.
    need_full = il documento intero.

    Sono i due numeri su cui lavora l'allocatore: garantisce need_min a tutti
    PRIMA di far crescere chiunque verso need_full.
    """
    need_full = len(content)
    if not core_spans:
        return min(need_full, FALLBACK_MIN_CHARS), need_full
    core_total = sum(s["end"] - s["start"] for s in core_spans)
    core_total += len(OMISSION_MARKER) * (len(core_spans) - 1)
    return min(core_total, need_full), need_full


def _water_fill(demands: list, budget: int) -> tuple:
    """
    Ripartizione max-min fair (water-filling) di `budget` tra richieste eterogenee.

    Tutti ricevono una quota uguale; chi chiede MENO della propria quota prende
    solo ciò che gli serve e RESTITUISCE l'avanzo, che viene ridistribuito a chi
    è ancora insoddisfatto. Si itera fino a convergenza.

    È il meccanismo che elimina lo spreco del vecchio calcolo a tetto fisso, in
    cui un parent lungo 900 caratteri con quota 1875 lasciava 975 caratteri che
    nessuno raccoglieva mentre altri parent venivano troncati.

    Restituisce (allocazioni, budget_residuo).
    """
    n = len(demands)
    alloc = [0] * n
    if n == 0 or budget <= 0:
        return alloc, max(0, budget)

    pending = set(range(n))
    remaining = budget

    while pending and remaining > 0:
        share = remaining // len(pending)
        if share <= 0:
            break
        satisfied = [i for i in pending if demands[i] <= share]
        if not satisfied:
            # Nessuno si accontenta della quota: la spartiamo e chiudiamo.
            for i in pending:
                alloc[i] += share
                remaining -= share
            break
        for i in satisfied:
            alloc[i] += demands[i]
            remaining -= demands[i]
            pending.discard(i)

    return alloc, max(0, remaining)


def _subquery_quotas(group_scores: list, total_budget: int, task_id: str = "UNKNOWN") -> list:
    """
    Quota di caratteri per ciascuna sottoquery, secondo la politica configurata
    (vedi la tabella delle tre politiche in common/config.py).

        quota = floor_equamente_diviso + resto_pesato_sulla_rilevanza
        quota = min(quota, cap)     # tetto anti-monopolio

    L'eccedenza tagliata dal cap viene ridistribuita alle sottoquery non ancora
    al tetto, così nessun carattere si perde per strada.
    """
    n = len(group_scores)
    if n == 0:
        return []

    floor_ratio = max(0.0, min(1.0, settings.context_floor_ratio))
    floor_pool = total_budget * floor_ratio
    weighted_pool = total_budget - floor_pool
    base = floor_pool / n

    if settings.context_score_weight == "none" or weighted_pool <= 0:
        weights = [1.0 / n] * n
    else:
        raw = [max(0.0, s) for s in group_scores]
        tot = sum(raw)
        weights = [r / tot for r in raw] if tot > 0 else [1.0 / n] * n

    quotas = [base + weighted_pool * w for w in weights]

    # Tetto anti-monopolio, con ridistribuzione iterativa dell'eccedenza.
    cap_ratio = settings.context_cap_ratio
    if 0 < cap_ratio < 1.0 and n > 1:
        cap = total_budget * cap_ratio
        for _ in range(4):
            excess = sum(q - cap for q in quotas if q > cap)
            if excess <= 1:
                break
            quotas = [min(q, cap) for q in quotas]
            free_idx = [i for i, q in enumerate(quotas) if q < cap - 1]
            if not free_idx:
                break
            free_weight = sum(weights[i] for i in free_idx) or len(free_idx)
            for i in free_idx:
                share = (weights[i] / free_weight) if free_weight else (1 / len(free_idx))
                quotas[i] = min(cap, quotas[i] + excess * share)

    quotas = [int(q) for q in quotas]
    log.debug(f"[{task_id}] [ALLOC] Quote per sottoquery: {quotas} (floor={floor_ratio}, cap={cap_ratio}, "
              f"rilevanza '{settings.context_score_source}': {[round(g, 3) for g in group_scores]}).")
    return quotas


def allocate_context_budget(items: list, total_budget: int, task_id: str = "UNKNOWN") -> tuple:
    """
    Assegna a ciascun parent il proprio budget di caratteri, garantendo che ogni
    sottoquery contribuisca al contesto finale.

    Cascata, nell'ordine deciso:
      FASE A — ogni parent riceve almeno need_min (i suoi core), dentro la quota
               della propria sottoquery.
      FASE B — l'avanzo del water-filling (il budget che le sottoquery povere non
               riescono a consumare) va a chi è ancora in deficit.
      FASE C — se serve ancora, si sfora max_context_chars fino a
               context_overflow_ratio, comunque entro context_hard_cap_chars.
      FASE D — solo se anche questo non basta, il parent riceve meno di need_min
               e render_parent_window() scarterà il core con lo score più basso,
               loggando un WARNING: è il segnale che SNIPPETS_PER_PARENT o il
               budget vanno ritarati per quel topic.
      FASE E — con quello che resta, i parent crescono verso need_full (margini
               di contesto attorno ai core).
      FASE F — completamento: un documento che sfora di poco (<= overflow_ratio)
               viene preso INTERO invece che finestrato. L'espansione non può
               mai servire ad aggiungere un parent nuovo, solo a completarne uno
               già selezionato.

    Restituisce (allocazioni, stats).
    """
    n = len(items)
    if n == 0:
        return [], {}

    # --- raggruppamento per sottoquery, preservando l'ordine di apparizione ---
    groups = {}
    for i, it in enumerate(items):
        groups.setdefault(it.get("query_idx", 0), []).append(i)
    group_keys = list(groups.keys())

    if settings.context_score_weight == "mean":
        group_scores = [
            sum(items[i].get("score", 0.0) for i in groups[g]) / len(groups[g])
            for g in group_keys
        ]
    else:  # 'best' (default) — meno sensibile alla numerosità dei documenti
        group_scores = [max(items[i].get("score", 0.0) for i in groups[g]) for g in group_keys]

    quotas = _subquery_quotas(group_scores, total_budget, task_id=task_id)

    need_min = [it["need_min"] for it in items]
    need_full = [it["need_full"] for it in items]
    alloc = [0] * n

    # --- FASE A: need_min dentro la quota di gruppo ---------------------------
    leftover_pool = 0
    for gi, g in enumerate(group_keys):
        idxs = groups[g]
        sub_alloc, leftover = _water_fill([need_min[i] for i in idxs], quotas[gi])
        for k, i in enumerate(idxs):
            alloc[i] = sub_alloc[k]
        leftover_pool += leftover

    # --- FASE B: avanzo globale ai parent ancora sotto need_min ---------------
    deficit_idx = [i for i in range(n) if alloc[i] < need_min[i]]
    if deficit_idx and leftover_pool > 0:
        deficits = [need_min[i] - alloc[i] for i in deficit_idx]
        got, leftover_pool = _water_fill(deficits, leftover_pool)
        for k, i in enumerate(deficit_idx):
            alloc[i] += got[k]
        log.info(
            f"[{task_id}] [ALLOC] Avanzo redistribuito a {len(deficit_idx)} parent sotto il minimo vitale."
        )

    # --- FASE C: overflow controllato, entro l'hard cap -----------------------
    ceiling = min(
        int(total_budget * (1 + settings.context_overflow_ratio)),
        settings.context_hard_cap_chars,
    )
    n_expanded = 0
    deficit_idx = [i for i in range(n) if alloc[i] < need_min[i]]
    if deficit_idx:
        headroom = ceiling - sum(alloc)
        if headroom > 0:
            deficits = [need_min[i] - alloc[i] for i in deficit_idx]
            got, _ = _water_fill(deficits, headroom)
            for k, i in enumerate(deficit_idx):
                if got[k] > 0:
                    alloc[i] += got[k]
                    n_expanded += 1
            log.info(
                f"[{task_id}] [ALLOC] Espansione oltre max_context_chars per garantire i core: "
                f"{sum(alloc)}/{total_budget} chars (tetto {ceiling})."
            )

    # --- FASE D: deficit residuo → sacrificio di uno snippet (rumoroso) -------
    still_short = [i for i in range(n) if alloc[i] < need_min[i]]
    if still_short:
        log.warning(
            f"[{task_id}] [ALLOC] {len(still_short)} parent restano sotto il minimo vitale anche "
            f"dopo espansione: verrà scartato lo snippet meno rilevante. Valutare di ridurre "
            f"SNIPPETS_PER_PARENT o di alzare MAX_CONTEXT_CHARS per questo topic."
        )

    # --- FASE E: crescita verso need_full con il budget residuo ---------------
    used = sum(alloc)
    growth_pool = max(0, total_budget - used)
    if growth_pool > 0:
        # Prima dentro il gruppo (equità), poi il residuo globalmente.
        for gi, g in enumerate(group_keys):
            idxs = groups[g]
            group_used = sum(alloc[i] for i in idxs)
            group_pool = max(0, quotas[gi] - group_used)
            group_pool = min(group_pool, growth_pool)
            if group_pool <= 0:
                continue
            extra = [max(0, need_full[i] - alloc[i]) for i in idxs]
            got, back = _water_fill(extra, group_pool)
            for k, i in enumerate(idxs):
                alloc[i] += got[k]
            growth_pool -= (group_pool - back)

        # Residuo globale, in due giri. Il primo rispetta il tetto
        # anti-monopolio: se più sottoquery hanno ancora fame, nessuna può
        # sfondare context_cap_ratio approfittando dell'avanzo altrui.
        # Il secondo giro serve solo quando NESSUN altro è in grado di
        # assorbire quel budget (tipicamente perché tutti gli altri hanno già
        # ricevuto i loro documenti INTERI): a quel punto lasciarlo inutilizzato
        # sarebbe uno spreco senza vittime, non un monopolio.
        if growth_pool > 0:
            cap = int(total_budget * settings.context_cap_ratio) if 0 < settings.context_cap_ratio < 1 else None
            for capped_round in (True, False):
                if growth_pool <= 0:
                    break
                extra = []
                for i in range(n):
                    want = max(0, need_full[i] - alloc[i])
                    if capped_round and cap is not None:
                        g = items[i].get("query_idx", 0)
                        group_used = sum(alloc[j] for j in groups[g])
                        want = min(want, max(0, cap - group_used))
                    extra.append(want)
                if not any(extra):
                    continue
                got, growth_pool = _water_fill(extra, growth_pool)
                for i in range(n):
                    alloc[i] += got[i]

    # --- FASE F: completamento dei documenti che sforano di poco --------------
    n_completed = 0
    used = sum(alloc)
    gaps = sorted(
        (i for i in range(n) if alloc[i] < need_full[i]),
        key=lambda i: need_full[i] - alloc[i],
    )
    for i in gaps:
        gap = need_full[i] - alloc[i]
        tolerance = max(1, int(need_full[i] * settings.context_overflow_ratio))
        if gap <= tolerance and used + gap <= ceiling:
            alloc[i] += gap
            used += gap
            n_completed += 1

    if n_completed:
        log.info(
            f"[{task_id}] [ALLOC] {n_completed} parent inclusi INTERI grazie all'overflow "
            f"(sforamento <= {settings.context_overflow_ratio:.0%}); totale {used} chars."
        )

    stats = {
        "n_groups": len(group_keys),
        "quotas": quotas,
        "allocated_total": sum(alloc),
        "n_expanded": n_expanded,
        "n_completed": n_completed,
        "n_below_min": len(still_short),
        "ceiling": ceiling,
    }
    return alloc, stats


def _snap_start(content: str, pos: int, limit: int) -> int:
    """
    Sposta l'inizio della finestra IN AVANTI fino al confine di frase/paragrafo
    più vicino, senza mai superare `limit` (l'inizio del core, che è intoccabile).
    Costo zero: si restringe, non si espande.
    """
    mode = settings.context_snap_boundary
    if mode == "none" or pos <= 0 or pos >= limit:
        return pos
    window = content[pos:limit]
    rx = _PARAGRAPH_END_RE if mode == "paragraph" else _SENTENCE_END_RE
    m = rx.search(window)
    if m:
        return pos + m.end()
    return pos


def _snap_end(content: str, pos: int, limit: int) -> int:
    """
    Arretra la fine della finestra fino all'ultimo confine di frase/paragrafo,
    senza mai scendere sotto `limit` (la fine del core).
    """
    mode = settings.context_snap_boundary
    if mode == "none" or pos >= len(content) or pos <= limit:
        return pos
    window = content[limit:pos]
    rx = _PARAGRAPH_END_RE if mode == "paragraph" else _SENTENCE_END_RE
    last = None
    for m in rx.finditer(window):
        last = m
    if last:
        return limit + last.end()
    return pos


def render_parent_window(content: str, core_spans: list, budget: int, task_id: str = "UNKNOWN") -> str:
    """
    Costruisce il testo effettivo del parent entro `budget` caratteri.

    Regola fondamentale: i CORE (i child che hanno fatto match) sono
    incomprimibili, i MARGINI attorno a loro sono comprimibili. Se un parent ha
    un chunk rilevante all'inizio e uno alla fine, teniamo entrambi separandoli
    con OMISSION_MARKER, invece di sacrificarne uno: quel documento è
    probabilmente il più informativo del lotto, e mutilarlo sarebbe il peggior
    uso possibile del budget.

    Non tronca MAI a metà di un blocco: se lo spazio non basta, scarta per
    intero il core con lo score più basso (ultima risorsa, già segnalata a
    WARNING dall'allocatore).
    """
    if len(content) <= budget:
        return content

    if not core_spans:
        log.debug(f"[{task_id}] [WINDOW] Nessun core localizzato: troncamento dall'inizio con snap.")
        cut = _snap_end(content, budget, 0)
        return content[:cut]

    # Se i soli core non ci stanno, sacrifichiamo i meno rilevanti — mai un
    # taglio a metà blocco.
    cores = sorted(core_spans, key=lambda s: s["start"])
    while len(cores) > 1:
        core_total = sum(c["end"] - c["start"] for c in cores)
        core_total += len(OMISSION_MARKER) * (len(cores) - 1)
        if core_total <= budget:
            break
        weakest = min(cores, key=lambda c: c["score"])
        cores.remove(weakest)
        log.warning(
            f"[{task_id}] [WINDOW] Budget insufficiente per tutti i core: scartato uno snippet "
            f"(score={weakest['score']:.4f}). Restano {len(cores)} core."
        )

    core_total = sum(c["end"] - c["start"] for c in cores)
    separators = len(OMISSION_MARKER) * (len(cores) - 1)

    # Il margine è ciò che AVANZA dopo aver messo al sicuro i core, distribuito
    # equamente sui blocchi (metà prima, metà dopo). Sostituisce il vecchio
    # margin = max(200, budget // (2 * len(spans))), che veniva calcolato prima
    # di sapere se i core ci stavano.
    margin_pool = max(0, budget - core_total - separators)

    # Espansione iterativa del margine. Un core a inizio o fine documento non
    # può crescere da entrambi i lati, e nella prima versione quello spazio
    # andava semplicemente perso (su un parent con core agli estremi si
    # arrivava a usare meno della metà del budget). Qui ricalcoliamo cosa è
    # stato davvero consumato e rioffriamo il resto a chi può ancora crescere.
    def _expand(pool):
        per_side = pool // (2 * len(cores)) if cores else 0
        wins = []
        for c in cores:
            start = _snap_start(content, max(0, c["start"] - per_side), c["start"])
            end = _snap_end(content, min(len(content), c["end"] + per_side), c["end"])
            wins.append((start, end))
        return wins

    windows = _expand(margin_pool)
    for _ in range(3):
        used = sum(e - s for s, e in windows) + separators
        slack = budget - used
        # Ci fermiamo quando l'avanzo è irrilevante o quando l'ultima
        # iterazione non ha prodotto alcun guadagno (tutti i core già al
        # confine del documento).
        if slack < 200:
            break
        candidate = _expand(margin_pool + slack)
        if sum(e - s for s, e in candidate) <= sum(e - s for s, e in windows):
            break
        margin_pool += slack
        windows = candidate

    # Merge dei blocchi che dopo l'espansione dei margini si toccano.
    merged = [windows[0]]
    for s, e in windows[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))

    pieces = [content[s:e] for s, e in merged]
    result = OMISSION_MARKER.join(pieces)

    # Se lo snap ha allargato oltre il previsto, rimuoviamo margine invece di
    # tagliare il risultato a caratteri (che spezzerebbe l'ultimo blocco).
    if len(result) > budget and len(merged) == 1:
        s, e = merged[0]
        core_end = max(c["end"] for c in cores)
        result = content[s:max(core_end, e - (len(result) - budget))]

    log.debug(
        f"[{task_id}] [WINDOW] {len(cores)} core, {len(merged)} blocchi dopo merge, "
        f"{len(result)}/{budget} chars usati (parent originale: {len(content)} chars)."
    )
    return result


# Campi data da cui si ricava "Date:" nell'header, in ordine di priorità
# (gli stessi indicizzati come DATETIME su Qdrant, vedi setup_qdrant.py).
_DATE_KEYS = ("data", "data_pubblicazione", "data_esecutivita")


# Intestazioni di sezione del MarkdownHeaderTextSplitter: proprie del singolo
# parent, non del documento.
_HEADER_KEY_RE = re.compile(r"^Header \d+$")


def document_context_metadata(raw_metadata):
    """
    Dai metadati di un parent (JSON salvato in MySQL) ricava:
      - la data del documento per "Date:" nell'header;
      - i metadati del documento da mostrare al generatore: tutti, esclusi le
        chiavi di sistema, le intestazioni "Header N", i campi con "_"
        iniziale (uso interno), i campi tecnici di CONTEXT_METADATA_EXCLUDE,
        i valori vuoti e il campo già mostrato come "Date:".
    Ritorna (data, metadati).
    """
    if not raw_metadata:
        return None, {}
    try:
        meta = raw_metadata if isinstance(raw_metadata, dict) else json.loads(raw_metadata)
    except (json.JSONDecodeError, TypeError):
        return None, {}
    if not isinstance(meta, dict):
        return None, {}

    date_key = next((k for k in _DATE_KEYS if meta.get(k)), None)
    doc_date = meta.get(date_key) if date_key else None
    if not settings.context_metadata:
        return doc_date, {}

    shown = {}
    for key, value in meta.items():
        if (key == date_key
                or key in settings.protected_keys
                or key in settings.context_metadata_exclude
                or key.startswith("_")
                or _HEADER_KEY_RE.match(key)):
            continue
        if value is None or value == [] or (isinstance(value, str) and not value.strip()):
            continue
        shown[key] = value
    return doc_date, shown


def format_document_metadata(metadata: dict) -> str:
    """
    Riga dei metadati del documento per il contesto, es.
    "Metadati del documento: numero: 375 | oggetto: Affidamento lavori...".
    Nomi resi leggibili (underscore -> spazio), valori lunghi accorciati.
    """
    limit = settings.context_metadata_max_value_chars
    parts = []
    for key, value in metadata.items():
        text = ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)
        text = " ".join(text.split())
        if len(text) > limit:
            text = text[:limit - 1].rstrip() + "…"
        parts.append(f"{key.replace('_', ' ')}: {text}")
    return ("Metadati del documento: " + " | ".join(parts) + "\n") if parts else ""


def build_context(rich_context, task_id="UNKNOWN"):
    """
    Assembla il contesto testuale per il generatore e restituisce
    (context_str, index_to_item, stats).

    L'ordine degli item è quello deciso da retrieve_chunks (round-robin tra
    sottoquery): se qualcosa deve cadere per esaurimento budget, cade la coda di
    TUTTE le sottoquery, non un comune intero.
    """
    n_parents = len(rich_context)
    if n_parents == 0:
        return "", {}, {}

    # --- 1. header: costruiti PRIMA, così il loro costo è sottratto dal budget
    # invece di sforare silenziosamente come accadeva prima (il vecchio codice
    # contava gli header solo nel break finale, non nel budget per parent).
    base_parts = []
    for idx, item in enumerate(rich_context, start=1):
        parts = [str(idx)]
        if item.get("sub_topic"):
            parts.append(f"Ambito: {item['sub_topic']}")
        # Provenienza della RICERCA, non del contenuto: il prompt istruisce il
        # modello a verificarla nel testo prima di attribuire il documento.
        if item.get("retrieved_by"):
            parts.append(f"Rif. ricerca: {'; '.join(item['retrieved_by'])}")
        parts.append(f"Date: {item.get('date') or 'unknown'}")
        base_parts.append(parts)

    # Metadati del documento (oggetto, numero, ufficio...): una volta per
    # documento, sul suo primo passaggio; gli altri passaggi dello stesso
    # documento li richiamano con "Stesso documento di: N". Un parent dal
    # centro di un atto non contiene né numero né oggetto: senza questa riga
    # il modello non saprebbe a quale atto appartiene.
    meta_lines = [format_document_metadata(item.get("metadata") or {}) for item in rich_context]

    def doc_key(pos, item):
        return item.get("source") or f"_senza_source_{pos}"

    def make_header(pos, same_as=None):
        if not settings.context_metadata:
            same_as = None  # funzione spenta: header identico a prima
        parts = base_parts[pos] + ([f"Stesso documento di: {same_as}"] if same_as else [])
        header = "[" + " | ".join(parts) + "]\n"
        return header if same_as else header + meta_lines[pos]

    # Stima per il budget: metadati sul primo passaggio di ogni documento. Se
    # quel passaggio verrà escluso, i metadati passano al successivo incluso
    # (vedi rendering): lo scarto è piccolo e l'hard cap resta il limite vero.
    planned = {}
    headers = []
    for pos, (idx, item) in enumerate(enumerate(rich_context, start=1)):
        key = doc_key(pos, item)
        headers.append(make_header(pos, same_as=planned.get(key)))
        planned.setdefault(key, idx)

    header_cost = sum(len(h) for h in headers) + 2 * n_parents  # "\n\n" tra i blocchi
    budget = max(1000, settings.max_context_chars - header_cost)

    # --- 2. core spans e fabbisogni
    for item in rich_context:
        content = item.get("content", "")
        spans = compute_core_spans(content, item.get("matched_snippets", []), task_id=task_id)
        item["_core_spans"] = spans
        need_min, need_full = estimate_parent_needs(content, spans)
        item["need_min"] = need_min
        item["need_full"] = need_full

    # --- 3. allocazione equa tra sottoquery
    alloc, alloc_stats = allocate_context_budget(rich_context, budget, task_id=task_id)

    # --- 4. rendering
    formatted_chunks = []
    index_to_item = {}
    curr_len = 0
    hard_cap = settings.context_hard_cap_chars
    # Documento -> numero del passaggio INCLUSO che ne porta i metadati: il
    # rimando "Stesso documento di" punta sempre a un passaggio presente.
    shown_docs = {}
    meta_chars = 0

    for pos, (idx, item) in enumerate(enumerate(rich_context, start=1)):
        content = item.get("content", "")
        item_budget = alloc[pos]
        if item_budget <= 0:
            log.debug(f"[{task_id}] Parent [{idx}] senza budget assegnato: escluso dal contesto.")
            continue

        rendered = render_parent_window(content, item.get("_core_spans", []), item_budget, task_id=task_id)
        key = doc_key(pos, item)
        chunk = f"{make_header(pos, same_as=shown_docs.get(key))}{rendered}\n\n"

        if curr_len + len(chunk) > hard_cap:
            log.warning(
                f"[{task_id}] Hard cap raggiunto ({curr_len}/{hard_cap} chars): "
                f"{n_parents - len(formatted_chunks)} parent scartati."
            )
            break

        formatted_chunks.append(chunk)
        curr_len += len(chunk)
        index_to_item[idx] = item
        if key not in shown_docs:
            shown_docs[key] = idx
            meta_chars += len(meta_lines[pos])
        # Caratteri REALMENTE finiti nel contesto per questo parent, header
        # incluso: è l'unico numero che serve per capire come il budget è stato
        # speso davvero. Va registrato qui perché dopo il windowing il dato non
        # è più ricostruibile dall'item.
        item["_rendered_chars"] = len(chunk)

    # --- 5. osservabilità: l'allocazione effettiva, per sottoquery
    # 'chars' contava need_full, cioè il fabbisogno TEORICO del parent, non i
    # caratteri effettivamente renderizzati: su un documento finestrato i due
    # numeri divergono di parecchio (9455 richiesti contro 7403 usati, nei log)
    # e la somma per sottoquery poteva superare il contesto reale, rendendo la
    # riga inservibile proprio quando serviva, cioè per capire chi si era preso
    # il budget.
    per_query = {}
    for idx, item in index_to_item.items():
        q = item.get("query_idx", 0)
        entry = per_query.setdefault(q, {"query": item.get("query_text", ""), "parents": 0, "chars": 0})
        entry["parents"] += 1
        entry["chars"] += item.get("_rendered_chars", 0)
    log.info(
        f"[{task_id}] [ALLOC] Contesto: {len(index_to_item)}/{n_parents} parent, "
        f"{curr_len} chars (budget {settings.max_context_chars}, tetto {alloc_stats.get('ceiling')}, "
        f"hard cap {hard_cap}), {len(shown_docs)} documenti, metadati {meta_chars} chars. "
        f"Ripartizione per sottoquery: "
        + " | ".join(
            f"#{q}({v['parents']} parent, {v['chars']} chars)"
            for q, v in sorted(per_query.items())
        )
    )

    stats = {**alloc_stats, "context_chars": curr_len, "n_parents_in_context": len(index_to_item),
             "n_documents_in_context": len(shown_docs), "metadata_chars": meta_chars}
    return "".join(formatted_chunks), index_to_item, stats
