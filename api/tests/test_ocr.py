"""Test dell'OCR e della conversione del converter (su pa-common) contro un server finto. Va
eseguito nell'immagine del converter (serve LibreOffice per i documenti d'ufficio).

I fornitori OCR con i veri SDK (Gemini, OpenAI, Anthropic, Mistral, vLLM, Ollama) e la
conversione dei singoli formati sono provati in pa-common; qui la configurazione del RAG, il
comando del revisore (modello esterno, Google Cloud Vision), i lavori di conversione, il
conteggio delle pagine OCR e la ricostruzione di document_ocr.

Uso:  python api/tests/test_ocr.py      (esce con 1 se qualcosa fallisce)
"""
import asyncio, os, pathlib, shutil, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "converter" / "src"), str(ROOT), str(ROOT / "api" / "tests")]
from fake_llm_server import FakeLLM

fake = FakeLLM()
os.environ.update(AUTH_SECRET_KEY="x" * 40, API_SECRET_KEY="k", DATA_FOLDER=tempfile.mkdtemp(), DOCWS_RICERCA_ENDPOINT="u",
                  DOCWS_ATTI_ENDPOINT="u", DOCWS_CODICE_AMMINISTRAZIONE="u", DOCWS_CODICE_AOO="u", RUOLO_DOCWS="u",
                  LOG_LEVEL="CRITICAL", API_LLM_KEY="k", OCR_PROVIDER="local", LOCAL_LLM_BACKEND="vllm",
                  LOCAL_VLLM_BASE_URL=fake.url + "/v1", OPENAI_BASE_URL=fake.url + "/v1", ANTHROPIC_BASE_URL=fake.url,
                  OCR_MODEL_NAME="qwen-vision", OCR_RATE_PER_MINUTE="100000", OCR_MAX_ATTEMPTS="3")

import io
from PIL import Image
from pa_common import processo
from pa_common.conversione import Statistiche
from pa_common.errori import OcrRifiutato, OcrTransitorio
from pa_common.ocr import FornitoreOcr, Ocr
import converter

# I processi separati ("spawn") rilancerebbero questo script: qui basta la funzione.
processo.esegui = lambda funzione, argomenti, timeout: funzione(*argomenti)

buffer = io.BytesIO()
Image.new("RGB", (4, 4), "white").save(buffer, format="PNG")
png = buffer.getvalue()
failures = 0


def check(label, got, expected=True):
    global failures
    if got != expected:
        failures += 1
        print(f"FALLITO {label}: {got!r} invece di {expected!r}")
    else:
        print(f"ok {label}")


def fails_with(fn):
    try:
        fn()
        return None
    except Exception as e:
        return type(e).__name__


def senza_attese(ocr):
    ocr._attesa = (0, 0)
    return ocr


from common.utility import clean_ocr_text, markdown_to_text, normalize_ws
check("pulizia: sottolineato nei titoli", clean_ocr_text("## **<u>DETERMINAZIONE DIRIGENZIALE</u>**"), "## **DETERMINAZIONE DIRIGENZIALE**")
check("pulizia: evidenziato e sottolineato, maiuscole", clean_ocr_text("_<U>pari a 695 mc.</U>_ <mark>x</mark>"), "_pari a 695 mc._ x")
check("pulizia: altri tag restano", clean_ocr_text("m<sup>2</sup>"), "m<sup>2</sup>")
check("pulizia: spazi", normalize_ws("a&nbsp;&nbsp; b"), "a b")
check("pulizia: Markdown in testo", markdown_to_text("| a | b |\n|---|---|\n| 1 | 2 |"), "a\tb\n\n1\t2")

# --- Configurazione: OCR_PROVIDER=local con vLLM, il modello di OCR_MODEL_NAME
ocr = senza_attese(converter.get_ocr())
check("configurazione: vLLM e modello", (ocr.fornitore.nome, ocr.fornitore.modello, ocr.al_minuto, ocr.tentativi),
      ("vllm", "qwen-vision", 100000, 3))
fake.replies = [(200, "# Titolo\ntesto")]
check("vllm: trascrizione", asyncio.run(ocr.trascrivi(png)), "# Titolo\ntesto")
path, body = fake.requests[-1]
check("vllm: endpoint e modello", (path, body["model"]), ("/v1/chat/completions", "qwen-vision"))
fake.replies = [(429, {"error": {"message": "lento"}})] * 2 + [(200, "dopo il rate limit")]
check("transitorio: ritentato", asyncio.run(ocr.trascrivi(png)), "dopo il rate limit")

# --- Rifiuto del modello in ingestione: documento in errore, nessun altro servizio
def must_not_leave(*a, **k):
    raise AssertionError("uscita verso l'esterno")
filtered = (200, {"id": "x", "object": "chat.completion", "created": 0, "model": "m", "choices": [
    {"index": 0, "finish_reason": "content_filter", "message": {"role": "assistant", "content": ""}}]})
image_dir = pathlib.Path(tempfile.mkdtemp())
(image_dir / "pagina.png").write_bytes(png)
saved = converter.get_external_fallback, converter._cloud_vision_transcribe
converter.get_external_fallback = converter._cloud_vision_transcribe = must_not_leave
fake.replies = [filtered]
check("ingestione: rifiuto, documento in errore",
      fails_with(lambda: asyncio.run(converter.convert_file_to_markdown(image_dir / "pagina.png", "pagina.png"))), "OcrRifiutato")
fake.replies = [(400, {"error": {"message": "modello senza immagini"}})]
check("ingestione: errore, documento in errore",
      fails_with(lambda: asyncio.run(converter.convert_file_to_markdown(image_dir / "pagina.png", "pagina.png"))), "ErroreOcr")
converter.get_external_fallback, converter._cloud_vision_transcribe = saved
fake.replies = []

# --- Comando del revisore: modello esterno, poi Vision se abilitato
class Scripted(FornitoreOcr):
    nome, modello = "scripted", "prova"

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), 0

    async def trascrivi(self, image, prompt):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def on_demand(fornitore, vision_on=True):
    used = []
    def vision(_png):
        used.append("vision")
        return "da vision"
    esterno = senza_attese(Ocr(fornitore)) if fornitore else None
    saved = converter.get_external_fallback, converter._cloud_vision_transcribe, converter.vision_enabled
    converter.get_external_fallback, converter._cloud_vision_transcribe, converter.vision_enabled = lambda: esterno, vision, lambda: vision_on
    try:
        try:
            return asyncio.run(converter._external_transcribe_page(png)), used
        except Exception as e:
            return f"{type(e).__name__}: {e}", used
    finally:
        converter.get_external_fallback, converter._cloud_vision_transcribe, converter.vision_enabled = saved


blocked, down = OcrRifiutato("Gemini: RECITATION"), OcrTransitorio("giù")
check("richiesta: modello esterno", on_demand(Scripted("da gemini")), ("da gemini", []))
gemini = Scripted(down, "da gemini")
check("richiesta: esterno ritentato se transitorio", (on_demand(gemini), gemini.calls), (("da gemini", []), 2))
gemini = Scripted(blocked, "mai")
check("richiesta: rifiuto, Vision senza ritentare", (on_demand(gemini), gemini.calls), (("da vision", ["vision"]), 1))
got, used = on_demand(Scripted(blocked), vision_on=False)
check("richiesta: rifiuto, Vision spento", (got, used), ("ValueError: Gemini: RECITATION; Google Cloud Vision non è abilitato.", []))
check("richiesta: esterno spento, Vision", on_demand(None), ("da vision", ["vision"]))
check("richiesta: tutto spento", on_demand(None, vision_on=False)[1], [])


def external(enabled=True, provider="openai", model="esterno-prova", key="chiave"):
    for name, value in (("fallback", enabled), ("provider", provider), ("model_name", model), ("api_key", key)):
        object.__setattr__(converter.settings, f"ocr_external_{name}", value)
    converter._EXTERNAL_FALLBACK = None
    return converter.get_external_fallback


check("esterno: spento, nessun ripiego", external(enabled=False)(), None)
check("esterno: fornitore sconosciuto", fails_with(external(provider="boh")), "ValueError")
check("esterno: fornitore mancante", fails_with(external(provider="")), "ValueError")
check("esterno: modello mancante", fails_with(external(model="")), "ValueError")
check("esterno: chiave mancante", fails_with(external(key="")), "ValueError")
for name, cls in (("gemini", "OcrGemini"), ("openai", "OcrOpenAI"), ("anthropic", "OcrAnthropic"), ("mistral", "OcrMistral")):
    built = external(provider=name)()
    check(f"esterno: {name}", (type(built.fornitore).__name__, built.fornitore.modello), (cls, "esterno-prova"))
fake.replies = [(200, "dal modello esterno")]
check("esterno: usa il suo modello", (asyncio.run(external(provider="openai")().trascrivi(png)), fake.requests[-1][1]["model"]),
      ("dal modello esterno", "esterno-prova"))
external(provider="boh")
vision_calls = []
saved = converter._cloud_vision_transcribe
converter._cloud_vision_transcribe = vision_calls.append
check("esterno: configurazione errata ferma il lavoro", (fails_with(lambda: asyncio.run(converter._external_transcribe_page(png))), vision_calls), ("ValueError", []))
converter._cloud_vision_transcribe = saved
external(enabled=False)

object.__setattr__(converter.settings, "ocr_model_name", "")
converter._OCR = None
try:
    converter.get_ocr()
    check("senza modello: errore", "nessun errore", "ValueError")
except ValueError as e:
    check("senza modello: errore chiaro", "OCR_MODEL_NAME" in str(e), True)
object.__setattr__(converter.settings, "ocr_model_name", "qwen-vision")
converter._OCR = None

# --- Lavoro del revisore: ogni pagina all'OCR esterno, senza passare dal modello locale
import json, pymupdf
calls = []


async def fake_external_page(_png):
    calls.append(1)
    return f"pagina {len(calls)}"


real_page, converter._external_transcribe_page = converter._external_transcribe_page, fake_external_page
job_id = "a" * 32
job_dir = converter.settings.data_folder / "conversions" / job_id
job_dir.mkdir(parents=True)
pdf = pymupdf.open()
pdf.new_page().insert_text((50, 50), "testo nativo")
pdf.new_page()
pdf.save(job_dir / "input.pdf")
(job_dir / "request.json").write_text(json.dumps({"input_file": "input.pdf", "file_name": "a.pdf", "external_ocr": True}))
asyncio.run(converter.process_conversion_job(job_id))
check("lavoro esterno: esito", json.loads((job_dir / "status.json").read_text())["status"], "completed")
check("lavoro esterno: ogni pagina", (job_dir / "result.md").read_text(), "pagina 1\n\n---\n\npagina 2")
converter._external_transcribe_page = real_page

# --- Documenti d'ufficio: testo e tabelle senza OCR (DOCX e ODT con LibreOffice)
import datetime, openpyxl
from odf.opendocument import OpenDocumentSpreadsheet
from odf.table import Table, TableCell, TableRow
from odf.text import P
from common.file_intake import detect_extension_from_content
office_dir = pathlib.Path(tempfile.mkdtemp())
if shutil.which("soffice"):
    (office_dir / "atto.html").write_text("<html><body><h1>Determina</h1><p>Il <b>dirigente</b> dispone.</p>"
                                          "<table><tr><td>Voce</td><td>Importo</td></tr>"
                                          "<tr><td>Pulizie</td><td>1.200,00</td></tr></table></body></html>")
    for ext, filtro in (("docx", "docx:MS Word 2007 XML"), ("odt", "odt")):
        subprocess.run(["soffice", "--headless", "--convert-to", filtro, "--outdir", str(office_dir), str(office_dir / "atto.html")],
                       check=True, capture_output=True, timeout=120)
        check(f"ufficio: tipo riconosciuto atto.{ext}", detect_extension_from_content((office_dir / f"atto.{ext}").read_bytes()), f".{ext}")
        text = asyncio.run(converter.convert_file_to_markdown(office_dir / f"atto.{ext}", f"atto.{ext}"))
        check(f"ufficio: {ext} con titolo, grassetto e tabella",
              ("# Determina" in text, "**dirigente**" in text, "| Pulizie | 1.200,00 |" in text), (True, True, True))
else:
    print("(LibreOffice non installato: DOCX e ODT non provati)")
workbook = openpyxl.Workbook()
workbook.active.title = "Spese"
for row in (["Data", "Voce", "Importo"], [datetime.date(2026, 3, 1), "Pulizie", 1200.5], [None, None, None],
            [datetime.date(2026, 4, 2), "a | b", 800]):
    workbook.active.append(row)
workbook.create_sheet("Vuoto")
workbook.save(office_dir / "conti.xlsx")
documento = OpenDocumentSpreadsheet()
tabella = Table(name="Comuni")
for valori in (["Comune", "Abitanti"], ["Lucca", "89000"]):
    riga = TableRow()
    for v in valori:
        cella = TableCell()
        cella.addElement(P(text=v))
        riga.addElement(cella)
    tabella.addElement(riga)
documento.spreadsheet.addElement(tabella)
documento.save(str(office_dir / "comuni.ods"))
for name in ("conti.xlsx", "comuni.ods"):
    check(f"ufficio: tipo riconosciuto {name}", detect_extension_from_content((office_dir / name).read_bytes()), pathlib.Path(name).suffix)
# date e numeri all'italiana; il nome del foglio come titolo se la cartella ne ha più di uno
check("ufficio: xlsx", asyncio.run(converter.convert_file_to_markdown(office_dir / "conti.xlsx", "conti.xlsx")),
      "## Spese\n\n| Data | Voce | Importo |\n|---|---|---|\n| 01/03/2026 | Pulizie | 1200,5 |\n| 02/04/2026 | a \\| b | 800 |\n")
check("ufficio: ods", asyncio.run(converter.convert_file_to_markdown(office_dir / "comuni.ods", "comuni.ods")),
      "| Comune | Abitanti |\n|---|---|\n| Lucca | 89000 |\n")
check("ufficio: zip qualsiasi non è un documento", detect_extension_from_content(b"PK\x03\x04 non un archivio"), None)
(office_dir / "ignoto.xyz").write_bytes(b"x")
check("formato non gestito: ignorato", asyncio.run(converter.convert_file_to_markdown(office_dir / "ignoto.xyz", "ignoto.xyz")), None)

# --- Conteggio delle pagine OCR
class Trascrive:
    async def trascrivi(self, png, *, markdown=True):
        return "trascritto: " + "parole " * 20


async def counted(path, name):
    stats = Statistiche()
    await converter.convert_file_to_markdown(path, name, stats)
    return stats.pagine_ocr, stats.pagine


converter.get_ocr = lambda: Trascrive()
image_path = job_dir / "pagina.png"
image_path.write_bytes(png)
check("pagine OCR: immagine", asyncio.run(counted(image_path, "pagina.png")), (1, 1))
pdf_path = job_dir / "misto.pdf"
with pymupdf.open() as misto:
    pagina = misto.new_page()
    for i in range(8):
        pagina.insert_text((50, 50 + 16 * i), f"Riga {i} del testo nativo della determina, abbastanza lunga.")
    misto.new_page()
    misto.save(pdf_path)
check("pagine OCR: pdf misto", asyncio.run(counted(pdf_path, "misto.pdf")), (1, 2))
text_path = job_dir / "nota.txt"
text_path.write_text("testo")
check("pagine OCR: testo", asyncio.run(counted(text_path, "nota.txt")), (0, 0))

# --- Ricostruzione di document_ocr dai file archiviati
import json, time
import rebuild_document_ocr as rebuild

archive = pathlib.Path(tempfile.mkdtemp())
series = archive / "arc" / "ser" / "sotto"
series.mkdir(parents=True)
with pymupdf.open() as doc:
    doc.new_page().insert_text((50, 50), "testo nativo " * 10)
    doc.new_page()
    doc.save(series / "misto.pdf")
Image.new("RGB", (4, 4), "white").save(series / "foto.jpg")
(series / "nota.txt").write_text("testo")
(series / "rotto.pdf").write_bytes(b"non un pdf")
docs = {"misto.pdf": "direct://arc/ser/sotto/misto.pdf", "foto.jpg": "direct://arc/ser/sotto/foto.jpg",
        "nota.txt": "direct://arc/ser/sotto/nota.txt", "rotto.pdf": "direct://arc/ser/sotto/rotto.pdf",
        "perso.pdf": "direct://arc/ser/sotto/perso.pdf", "noto.pdf": "direct://arc/ser/sotto/noto.pdf"}
for name, source in docs.items():
    (series / f"{name}.json").write_text(json.dumps({"source": source, "files": [f"sotto/{name}"]}))
(series / "atto.json").write_text(json.dumps({"source": "sicraweb://1", "files": ["a.jpg"]}))
Image.new("RGB", (4, 4), "white").save(series / "a.jpg")
(series / "atto_a.json").write_text(json.dumps({"source": "sicraweb://1::a.jpg", "files": ["a.jpg"]}))
(series / "estraneo.json").write_text(json.dumps({"source": "direct://altro.jpg", "files": ["a.jpg"]}))
(series / "guasto.json").write_text("{")

saved = {}
rebuild._save = lambda key, ocr_pages, total_pages: saved.__setitem__(key[0], (ocr_pages, total_pages))
keys = {(s, "arc", "ser") for s in docs.values()} | {("sicraweb://1::a.jpg", "arc", "ser"), ("direct://senza_manifest", "arc", "ser")}
known = {(docs["noto.pdf"], "arc", "ser")}
done, problems = rebuild.rebuild(archive, keys, known, False, 30)
check("ricostruzione: senza --apply non scrive", (saved, done), ({}, 4))
done, problems = rebuild.rebuild(archive, keys, known, True, 30)
check("ricostruzione: conteggi", saved, {docs["misto.pdf"]: (1, 2), docs["foto.jpg"]: (1, 1), docs["nota.txt"]: (0, 0),
                                         "sicraweb://1::a.jpg": (1, 1)})
problem_text = "\n".join(problems)
check("ricostruzione: problemi elencati", (len(problems), "guasto.json" in problem_text, "rotto.pdf" in problem_text,
                                           "perso.pdf" in problem_text, "senza_manifest" in problem_text), (4, True, True, True, True))
check("ricostruzione: riga già presente ignorata", docs["noto.pdf"] in saved or "noto.pdf" in problem_text, False)

original_inspect = rebuild.inspect_page
rebuild.inspect_page = lambda page: time.sleep(30)
started = time.time()
check("ricostruzione: file bloccato interrotto", fails_with(lambda: rebuild.count_pages(series / "misto.pdf", 1)), "TimeoutError")
check("ricostruzione: senza attendere", time.time() - started < 10)
rebuild.inspect_page = original_inspect

marker = converter.DATA_FOLDER / converter.EXTERNAL_OCR_MARKER
cred = converter.DATA_FOLDER / "cred.json"
object.__setattr__(converter.settings, "ocr_vision_fallback", True)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred)
converter.publish_external_ocr_availability()
check("credenziali assenti: niente Vision", marker.exists(), False)
cred.write_text("")
converter.publish_external_ocr_availability()
check("credenziali vuote: niente Vision", marker.exists(), False)
cred.write_text("{}")
converter.publish_external_ocr_availability()
check("credenziali presenti: Vision disponibile", marker.exists(), True)
object.__setattr__(converter.settings, "ocr_vision_fallback", False)
converter.publish_external_ocr_availability()
check("Vision disattivato: niente pulsante", marker.exists(), False)
external(provider="openai")
converter.publish_external_ocr_availability()
check("solo modello esterno: pulsante", marker.exists(), True)
external(enabled=False)
object.__setattr__(converter.settings, "ocr_vision_fallback", True)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""
converter.publish_external_ocr_availability()
check("variabile vuota: niente Vision", marker.exists(), False)

print(f"\n{'TUTTO OK' if not failures else f'{failures} FALLITI'}")
sys.exit(1 if failures else 0)
