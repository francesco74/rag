"""Generazione della risposta, citazioni e valutazione del Grader."""

import json
import logging
import re

from pydantic import BaseModel

from common.config import settings
from llm_provider import get_llm_provider
from rag.context import build_context
from rag.jsonparse import safe_json_parse
from rag.prompts import get_topic_prompt, grader_name, load_prompt_template

log = logging.getLogger("WORKER")


class AnswerSchema(BaseModel):
    """
    Schema strutturato per l'output di generate_answer, allineato ai 3 campi
    richiesti dal prompt (vedi prompts/atti: is_found, is_general_knowledge,
    answer). Passato come response_schema a generate_json: sui provider che
    lo supportano nativamente (oggi Gemini) vincola il decoding a livello di
    grammatica, non solo il "sapore" JSON del testo — elimina alla radice la
    classe di bug osservata in produzione (backslash orfano dentro "answer"
    che rompe json.loads pur essendo il resto del payload valido, vedi
    safe_json_parse Livello 2c/3b per la rete di sicurezza residua sui
    provider senza questo supporto).
    """
    is_found: bool
    is_general_knowledge: bool
    answer: str


def resolve_citations(answer_html: str, index_to_item: dict) -> str:
    """
    Sostituisce i riferimenti numerici scritti dal modello (es. <i>[3]</i> o
    <i>[2, 5]</i>) con il nome file CANONICO, preso direttamente dai metadati
    di retrieval (index_to_item), mai dal testo generato dal modello.
 
    Questo elimina alla radice il disallineamento tra la citazione mostrata
    nella risposta e il file_name reale usato per costruire i link di
    download (vedi CITATION RULE nel prompt: il modello ora cita SOLO indici
    numerici, mai filename).
    """
    def replace_match(m):
        raw_numbers = m.group(1)
        numbers = [n.strip() for n in raw_numbers.split(',')]
        file_names = []
        for n in numbers:
            try:
                idx = int(n)
            except ValueError:
                continue
            item = index_to_item.get(idx)
            if not item:
                log.warning(f"[CITATION] Riferimento [{idx}] generato dal modello non risolvibile: nessun chunk con questo indice.")
                continue
            fname = item["file_name"]
            if fname not in file_names:
                file_names.append(fname)
        if not file_names:
            # Nessun riferimento valido: rimuoviamo la citazione invece di mostrare un numero grezzo/errato
            return ""
        return "[" + ", ".join(file_names) + "]"
 
    # Cattura i pattern tipo [3] o [2, 5] ovunque compaiano nella risposta
    pattern = r'\[([\d,\s]+)\]'
    return re.sub(pattern, replace_match, answer_html)


# Prompt i cui archivi non devono mai restituire dati personali: oltre alla regola nel
# prompt (e al motivo "privacy" del suo grader), un filtro deterministico sostituisce con
# "omissis" i dati riconoscibili dalla forma: codice fiscale, IBAN, indirizzi email.
OMISSIS = "omissis"
PROMPTS_WITH_OMISSIS = {"atti"}
_PERSONAL_DATA = (
    re.compile(r"\b[A-Z]{6}\d{2}[ABCDEHLMPRST]\d{2}[A-Z]\d{3}[A-Z]\b", re.IGNORECASE),  # codice fiscale
    re.compile(r"\bIT\s?\d{2}\s?[A-Z](?:\s?\d){10}(?:\s?[0-9A-Z]){12}\b", re.IGNORECASE),  # IBAN italiano
    re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),  # email e PEC
)


def with_omissis(text):
    """Sostituisce con "omissis" i dati personali riconoscibili dalla forma."""
    for pattern in _PERSONAL_DATA:
        text = pattern.sub(OMISSIS, text)
    return text


# Passaggi di pulizia con gli omissis prima di arrendersi (vedi evaluate_generation).
REDACTION_PASSES = 2


class RedactedSchema(BaseModel):
    answer: str


def redact_personal_data(answer, task_id="UNKNOWN"):
    """La risposta con i dati personali sostituiti da omissis (prompts/redactor), con il
    modello del grader: lavora sul solo testo della risposta, quindi è una chiamata breve.
    I riferimenti [n] restano numerici. Solleva un'eccezione se il modello non risponde."""
    prompt = load_prompt_template("redactor").format(answer=answer)
    raw = get_llm_provider().generate_json(
        settings.grader_model_name, prompt,
        temperature=0.0,
        max_tokens=settings.answer_max_tokens,
        thinking_level=settings.grader_thinking_level,
        response_schema=RedactedSchema,
    )
    text = str(safe_json_parse(raw, task_id=task_id).get("answer") or "").strip()
    if not text:
        raise ValueError("risposta ripulita vuota")
    return with_omissis(text)


def generate_answer(query, rich_context, topic_id, search_queries=None,
                    coverage_targets=None, task_id="UNKNOWN"):
    """
    Genera la risposta finale.

    search_queries / coverage_targets: le facce in cui il rewriter ha scomposto
    la domanda e le entità da coprire. PRIMA non venivano passate, e questo
    rendeva la RULE FOR PARTIAL CONTEXT del prompt di fatto inapplicabile: la
    standalone_query è deliberatamente generica ("Sono previsti lavori nei
    comuni della Garfagnana?"), quindi il modello non aveva alcuna lista di
    entità da confrontare col contesto e non poteva dichiarare le assenze.

    Restituisce anche used_sources / used_parent_ids: le fonti REALMENTE entrate
    nel prompt, non tutte quelle recuperate.
    """
    context_str, index_to_item, ctx_stats = build_context(rich_context, task_id=task_id)

    # Fonti effettive: derivate da index_to_item, mai dal pool di retrieval.
    # Deduplicate per `source`, ma i parent_id sono raccolti TUTTI (vedi nota su
    # used_parent_ids: servono interi al fallback per escludere davvero ciò che
    # il generatore ha già visto).
    used_sources = {}
    used_parent_ids = set()
    for item in index_to_item.values():
        src = item.get("source", "Fonte_Sconosciuta")
        if item.get("parent_id"):
            used_parent_ids.add(item["parent_id"])
        if src not in used_sources:
            used_sources[src] = {
                "source": src,
                "sub_topic": item.get("sub_topic", ""),
                "file_name": item.get("file_name", ""),
                "date": item.get("date"),
                "parent_id": item.get("parent_id"),
            }

    facets = [q for q in (search_queries or []) if q]
    targets = [t for t in (coverage_targets or []) if t]

    prompt_file = get_topic_prompt(topic_id)
    prompt_tmpl = load_prompt_template(prompt_file)
    fmt_kwargs = {
        "context_str": context_str,
        "query": query,
        "search_facets_str": "\n".join(f"- {f}" for f in facets),
        "coverage_targets_str": ", ".join(targets),
    }
    try:
        prompt = prompt_tmpl.format(**fmt_kwargs)
    except KeyError as e:
        # Retrocompatibilità: prompt di topic che non conoscono i nuovi
        # placeholder continuano a funzionare invariati.
        log.debug(f"[{task_id}] Prompt '{prompt_file}' senza placeholder {e}: uso il formato legacy.")
        prompt = prompt_tmpl.format(context_str=context_str, query=query)

    log.info(f"[{task_id}] Generating structured answer for topic '{topic_id}'")
    log.debug(f"[{task_id}] Context size: {len(context_str)} chars | facets={len(facets)} | targets={targets}")

    # Budget di OUTPUT (distinto da quello di input): con settings.answer_max_tokens
    # a None il parametro non viene passato affatto al provider, quindi il tetto è
    # quello del modello per definizione — nessuna costante da tenere allineata.
    #
    # Qui NON c'è retry, ed è deliberato. Il vecchio raddoppio era già codice
    # morto in produzione (con ANSWER_MAX_TOKENS pari all'hard cap la guardia era
    # sempre falsa) e al soffitto del modello non esiste comunque nessun numero
    # più grande da chiedere: ritentare significherebbe solo ripagare per intero
    # un contesto da decine di migliaia di token per ottenere lo stesso esito.
    # Il troncamento smette quindi di essere un caso da gestire e diventa un
    # ALLARME, con in log i dati che servono a capire su quale leva agire.
    max_tokens = settings.answer_max_tokens
    raw, meta = get_llm_provider().generate_json(
        settings.answer_generator_model_name, prompt,
        max_tokens=max_tokens,
        thinking_level=settings.answer_thinking_level,
        with_meta=True,
        response_schema=AnswerSchema,
    )
    if meta.get("truncated"):
        thinking_tokens = meta.get("thinking_tokens")
        output_tokens = meta.get("output_tokens")
        # Le due cause richiedono interventi opposti: se a saturare il budget è
        # stato il thinking, la leva è ANSWER_THINKING_LEVEL; se invece l'output
        # visibile era davvero enorme, il problema sta a monte (troppi parent in
        # contesto, o un prompt che non impone sintesi) e abbassare il thinking
        # non servirebbe a niente.
        if thinking_tokens and output_tokens and thinking_tokens > output_tokens:
            causa = (
                f"il thinking ha prodotto più token dell'output visibile "
                f"({thinking_tokens} vs {output_tokens}): abbassare ANSWER_THINKING_LEVEL "
                f"(attuale: {settings.answer_thinking_level or 'off'})"
            )
        else:
            causa = (
                "l'output visibile ha saturato il budget del modello: ridurre il "
                "materiale da sintetizzare (PARENTS_PER_QUERY, MAX_CONTEXT_CHARS) "
                "o rendere il prompt più stringente sulla sintesi"
            )
        log.error(
            f"[{task_id}] Risposta TRONCATA (finish_reason={meta.get('finish_reason')}, "
            f"budget={'modello' if max_tokens is None else max_tokens}, "
            f"thinking_tokens={thinking_tokens}, output_tokens={output_tokens}, "
            f"contesto={len(context_str)} chars su {len(index_to_item)} parent). "
            f"Probabile causa: {causa}."
        )

    log.debug(f"[{task_id}] Raw response:\n{raw[:500]}...")

    if prompt_file in PROMPTS_WITH_OMISSIS:
        raw = with_omissis(raw)

    base_result = {
        "prompt_name": prompt_file,  # sceglie il grader (prompts/<prompt>_grader)
        "index_to_item": index_to_item,  # per risolvere le citazioni di una risposta ripulita
        "used_sources": list(used_sources.values()),
        "used_parent_ids": used_parent_ids,
        "context_str": context_str,
        "context_stats": ctx_stats,
    }

    try:
        result_data = safe_json_parse(raw, task_id=task_id)
        raw_answer = str(result_data.get("answer", ""))
        resolved_answer = resolve_citations(raw_answer, index_to_item)
        return {
            **base_result,
            "is_found": bool(result_data.get("is_found", True)),
            "is_general_knowledge": bool(result_data.get("is_general_knowledge", False)),
            "answer": resolved_answer,
            # Risposta PRIMA della risoluzione delle citazioni: conserva i
            # riferimenti numerici ([3]), gli stessi che numerano i blocchi del
            # contesto. È questa che va data al grader: la versione risolta
            # contiene filename, che nel contesto non compaiono da nessuna parte,
            # rendendo le citazioni non verificabili.
            "raw_answer": raw_answer,
        }
    except json.JSONDecodeError as e:
        log.error(f"[{task_id}] Generazione JSON fallita: {e}. Output grezzo: {raw}")
        return {**base_result, "is_found": True, "is_general_knowledge": False,
                "answer": raw.strip(), "raw_answer": raw.strip()}


def grade_answer(query, context_snippet, answer, coverage_targets=None,
                 is_general_knowledge=False, task_id="UNKNOWN", prompt_name=None):
    """
    Valuta la risposta e restituisce un dict:
        {"ok": bool, "reason": str, "missing_targets": list, "note": str}

    Il grader è quello del prompt del topic (prompts/<prompt>_grader, altrimenti
    prompts/grader). Boccia: "grounding" (afferma come tratto dai documenti
    qualcosa che il contesto non sostiene), "no_answer" (nessuna informazione
    utile), "constraints" (include ciò che la domanda esclude) e, per atti,
    "privacy" (dati personali al posto degli omissis). La copertura parziale NON è un fallimento: se per un
    comune non risultano documenti, dirlo è la risposta corretta, non un
    difetto. missing_targets viene raccolto per sola diagnostica e finisce in
    rag_metrics, dove sui volumi distingue un problema di ingestion (documenti
    mai indicizzati) da uno di soglie (presenti ma non agganciati).

    context_snippet DEVE essere il context_str realmente passato al generatore,
    non i content integrali: giudicare su un testo più ampio di quello che
    l'autore aveva davanti produce sia falsi negativi (retry inutili su
    informazioni che erano state finestrate) sia falsi positivi (il grader
    trova la prova che mancava al generatore e valida un'affermazione non
    ancorata — cioè è cieco proprio sul caso che dovrebbe intercettare).

    FAIL-OPEN: qualunque output non interpretabile vale PASS, coerentemente col
    trattamento delle eccezioni. Prima l'incoerenza era silenziosa e costosa:
    un'eccezione dava is_satisfactory=True, ma una stringa vuota (tipica del
    troncamento con max_tokens=32) falliva startswith("YES") e bruciava un
    tentativo di retry.
    """
    targets = [t for t in (coverage_targets or []) if t]
    grader_file = grader_name(prompt_name)
    log.debug(f"[{task_id}] [GRADER] Prompt '{prompt_name}': grader '{grader_file}'.")
    grader_tmpl = load_prompt_template(grader_file)

    fmt_kwargs = {
        "query": query,
        "context_snippet": context_snippet,
        "answer": answer,
        "coverage_targets_str": ", ".join(targets),
        "is_general_knowledge": "true" if is_general_knowledge else "false",
    }
    try:
        grader_prompt = grader_tmpl.format(**fmt_kwargs)
    except KeyError as e:
        # Retrocompatibilità con il grader.txt legacy (solo query/context/answer).
        log.debug(f"[{task_id}] [GRADER] Prompt senza placeholder {e}: uso il formato legacy.")
        grader_prompt = grader_tmpl.format(
            query=query, context_snippet=context_snippet, answer=answer
        )

    raw = get_llm_provider().generate_json(
        settings.grader_model_name,
        grader_prompt,
        temperature=0.0,
        max_tokens=settings.grader_max_tokens,
        thinking_level=settings.grader_thinking_level,
    )

    result = {"ok": True, "reason": "ok", "missing_targets": [], "note": ""}
    text = (raw or "").strip()

    if not text:
        log.warning(f"[{task_id}] [GRADER] Output vuoto: fail-open, risposta accettata.")
        result["note"] = "grader output vuoto"
        return result

    try:
        data = safe_json_parse(text, task_id=task_id)
        verdict = str(data.get("verdict", "PASS")).strip().upper()
        result["ok"] = verdict != "FAIL"
        result["reason"] = str(data.get("reason", "ok")).strip().lower() or "ok"
        raw_missing = data.get("missing_targets") or []
        if isinstance(raw_missing, list):
            result["missing_targets"] = [str(m) for m in raw_missing if m]
        result["note"] = str(data.get("note", ""))[:300]
    except Exception:
        # Fallback sul contratto testuale precedente (YES/NO), così un
        # grader.txt non aggiornato continua a funzionare.
        upper = text.upper()
        if upper.startswith("NO"):
            result.update(ok=False, reason="grounding", note="verdetto legacy NO")
        elif upper.startswith("YES"):
            result["note"] = "verdetto legacy YES"
        else:
            log.warning(f"[{task_id}] [GRADER] Output non interpretabile ({text[:120]}): fail-open.")
            result["note"] = "output non interpretabile"

    # La copertura è diagnostica: non può mai ribaltare il verdetto.
    if result["reason"] == "coverage":
        log.debug(f"[{task_id}] [GRADER] reason='coverage' ignorata: la copertura parziale non è un fallimento.")
        result.update(ok=True, reason="ok")

    log.info(
        f"[{task_id}] [GRADER] verdict={'PASS' if result['ok'] else 'FAIL'} "
        f"reason={result['reason']} missing={result['missing_targets']} note='{result['note']}'"
    )
    return result


def evaluate_generation(gen_result, query, coverage_targets=None, task_id="UNKNOWN"):
    """
    Valutazione unificata di una risposta generata. Restituisce
    (is_satisfactory, reason, missing_targets). Usata sia dal loop principale
    sia dal fallback finale, che prima ne avevano due copie divergenti.

    reason ∈ {"ok", "not_found", "grounding", "no_answer", "constraints", "privacy", "general_knowledge"}
    ("constraints": la risposta include documenti che la richiesta esclude; "privacy": dati
    personali al posto degli omissis, solo per i prompt che li vietano)
    e guida il messaggio iniettato nel retry: riformulare in modo ampio ha senso
    quando la ricerca ha fallito, non quando i documenti c'erano ma erano
    insufficienti su un aspetto specifico.

    NOTA sul flag is_general_knowledge: NON è più una scorciatoia che salta il
    grader. Il flag di `atti` si alza anche quando la risposta è ancorata al 95%
    ai documenti e contiene un solo esempio suggerito in fondo: far uscire
    l'INTERA risposta dal controllo di grounding lasciava senza verifica proprio
    le risposte miste — le più esposte al rischio di confondere ciò che il
    documento dice con ciò che il modello sa — e poi le metteva in cache
    semantica, dove venivano riproposte ad altri utenti senza essere mai state
    controllate. Ora il grader vede sempre la risposta, e sa distinguere un
    suggerimento dichiarato (legittimo) da un fatto attribuito ai documenti ma
    assente dal contesto (allucinazione).
    """
    if not gen_result.get("is_found", False):
        log.info(f"[{task_id}] Model explicitly flagged is_found=False.")
        return False, "not_found", []

    is_gk = bool(gen_result.get("is_general_knowledge"))

    def grade():
        try:
            return grade_answer(
                query,
                gen_result.get("context_str", ""),
                # Risposta grezza: i riferimenti numerici sono verificabili contro
                # il contesto numerato; i filename della versione risolta no.
                gen_result.get("raw_answer") or gen_result.get("answer", ""),
                coverage_targets=coverage_targets,
                is_general_knowledge=is_gk,
                task_id=task_id,
                prompt_name=gen_result.get("prompt_name"),
            )
        except Exception as e:
            log.error(f"[{task_id}] Grader LLM failed (non-blocking): {e}. Fail-open.")
            return {"ok": True, "reason": "ok", "missing_targets": [], "note": "grader exception"}

    verdict = grade()

    # Dati personali (grader di atti): invece di rifare tutta la ricerca si ripulisce la
    # risposta con gli omissis e la si rivaluta, fino a REDACTION_PASSES volte. La
    # risposta ripulita sostituisce quella generata (gen_result è del chiamante).
    passes = 0
    while not verdict["ok"] and verdict["reason"] == "privacy" and passes < REDACTION_PASSES:
        passes += 1
        try:
            redacted = redact_personal_data(gen_result.get("raw_answer") or gen_result.get("answer", ""), task_id)
        except Exception as e:
            log.warning(f"[{task_id}] [OMISSIS] Pulizia non riuscita ({e}): resta la bocciatura per dati personali.")
            break
        gen_result["raw_answer"] = redacted
        gen_result["answer"] = resolve_citations(redacted, gen_result.get("index_to_item") or {})
        log.info(f"[{task_id}] [OMISSIS] Dati personali sostituiti con omissis (passaggio {passes}): nuova valutazione.")
        verdict = grade()

    missing = verdict.get("missing_targets", [])

    if not verdict["ok"]:
        return False, verdict["reason"], missing

    # Il grader ha promosso la risposta. Resta la policy sulla conoscenza
    # generale: se non è ammessa si torna nel loop per un altro tentativo di
    # retrieval, con i profili progressivamente più permissivi (semantic_size
    # più ampio, soglia più bassa). Se il modello ha dovuto ricorrere a
    # conoscenza esterna perché il contesto era povero, allargare la recall è
    # esattamente la mossa giusta.
    if is_gk and not settings.allow_general_knowledge:
        log.warning(
            f"[{task_id}] Conoscenza generale non ammessa (ALLOW_GENERAL_KNOWLEDGE=false): nuovo tentativo."
        )
        return False, "general_knowledge", missing

    return True, "ok", missing
