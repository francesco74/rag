"""Test dell'OCR del converter con i veri SDK contro un server finto. Va eseguito nell'immagine del converter.

Uso:  python api/tests/test_ocr.py      (esce con 1 se qualcosa fallisce)
"""
import asyncio, os, pathlib, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "converter" / "src"), str(ROOT), str(ROOT / "api" / "tests")]
from fake_llm_server import FakeLLM

fake = FakeLLM()
os.environ.update(AUTH_SECRET_KEY="x" * 40, API_SECRET_KEY="k", DATA_FOLDER=tempfile.mkdtemp(), DOCWS_RICERCA_ENDPOINT="u",
                  DOCWS_ATTI_ENDPOINT="u", DOCWS_CODICE_AMMINISTRAZIONE="u", DOCWS_CODICE_AOO="u", RUOLO_DOCWS="u",
                  LOG_LEVEL="CRITICAL", API_LLM_KEY="k", OCR_PROVIDER="local", LOCAL_LLM_BACKEND="vllm",
                  LOCAL_VLLM_BASE_URL=fake.url + "/v1", OPENAI_BASE_URL=fake.url + "/v1", ANTHROPIC_BASE_URL=fake.url,
                  OCR_MODEL_NAME="qwen-vision", OCR_RATE_PER_MINUTE="100000", OCR_MAX_ATTEMPTS="3")

from PIL import Image
from tenacity import wait_none
import ocr_provider
import converter

converter.async_ocr_generate.retry.wait = wait_none()
converter.external_ocr_generate.retry.wait = wait_none()
image = Image.new("RGB", (4, 4), "white")
failures = 0


def check(label, got, expected=True):
    global failures
    if got != expected:
        failures += 1
        print(f"FALLITO {label}: {got!r} invece di {expected!r}")
    else:
        print(f"ok {label}")


def ocr(as_markdown=True):
    return asyncio.run(converter.async_ocr_generate(image, as_markdown=as_markdown))


def fails_with(fn):
    try:
        fn()
        return None
    except Exception as e:
        return type(e).__name__


from common.utility import clean_ocr_text
check("pulizia: sottolineato nei titoli", clean_ocr_text("## **<u>DETERMINAZIONE DIRIGENZIALE</u>**"), "## **DETERMINAZIONE DIRIGENZIALE**")
check("pulizia: evidenziato e sottolineato, maiuscole", clean_ocr_text("_<U>pari a 695 mc.</U>_ <mark>x</mark>"), "_pari a 695 mc._ x")
check("pulizia: altri tag restano", clean_ocr_text("m<sup>2</sup>"), "m<sup>2</sup>")
check("clean: recinto e sillabazione", ocr_provider.clean_transcription("<think>x</think>\n```markdown\nparo-\nla\n```"), "parola")

# --- vLLM: immagine inviata come data URL al modello scelto
fake.replies = [(200, "# Titolo\ntesto")]
check("vllm: trascrizione", ocr(), "# Titolo\ntesto")
path, body = fake.requests[-1]
check("vllm: endpoint e modello", (path, body["model"]), ("/v1/chat/completions", "qwen-vision"))
content = body["messages"][0]["content"]
check("vllm: immagine", content[1]["image_url"]["url"].startswith("data:image/png;base64,"))
check("vllm: Markdown richiesto", "Markdown" in content[0]["text"])

# --- Rate limit: si ritenta senza passare da Vision
fake.requests.clear()
fake.replies = [(429, {"error": {"message": "lento"}})] * 3 + [(200, "dopo il rate limit")]
check("transitorio: ritentato", ocr(), "dopo il rate limit")

fake.replies = [(500, {"error": {"message": "giù"}})] * 30
check("transitorio: tentativi esauriti", fails_with(ocr), "OcrTransientError")
fake.replies = []

# --- Errore definitivo: nessun tentativo in più
fake.requests.clear()
fake.replies = [(400, {"error": {"message": "modello senza immagini"}})]
check("definitivo: propagato", fails_with(ocr), "BadRequestError")
check("definitivo: una sola chiamata", len(fake.requests), 1)

# --- Risposta vuota: errore, non una pagina bianca nel documento
fake.replies = [(200, "   ")]
check("vuota: errore", fails_with(ocr), "ValueError")

# --- Rifiuto del modello e ingestione: errore, nessun altro servizio
filtered = (200, {"id": "x", "object": "chat.completion", "created": 0, "model": "m", "choices": [
    {"index": 0, "finish_reason": "content_filter", "message": {"role": "assistant", "content": ""}}]})
fake.replies = [filtered, filtered]
check("bloccata: errore del fornitore", fails_with(ocr), "OcrBlockedError")
def must_not_leave(*a, **k):
    raise AssertionError("uscita verso l'esterno")
for name in ("get_external_fallback", "_cloud_vision_transcribe", "external_ocr_generate"):
    setattr(converter, f"_saved_{name}", getattr(converter, name))
    setattr(converter, name, must_not_leave)
check("ingestione: rifiuto, documento in errore", fails_with(lambda: asyncio.run(converter._ocr_page(image))), "OcrBlockedError")
fake.replies = [(400, {"error": {"message": "x"}})]
check("ingestione: errore, documento in errore", fails_with(lambda: asyncio.run(converter._ocr_page(image))), "BadRequestError")
for name in ("get_external_fallback", "_cloud_vision_transcribe", "external_ocr_generate"):
    setattr(converter, name, getattr(converter, f"_saved_{name}"))

# --- Comando del revisore: modello esterno, poi Vision se abilitato
class Scripted:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), 0

    async def transcribe(self, image, prompt):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def on_demand(gemini, vision_on=True):
    used = []
    def vision(_image):
        used.append("vision")
        return "da vision"
    saved = converter.get_external_fallback, converter._cloud_vision_transcribe, converter.vision_enabled
    converter.get_external_fallback, converter._cloud_vision_transcribe, converter.vision_enabled = lambda: gemini, vision, lambda: vision_on
    try:
        try:
            return asyncio.run(converter._external_transcribe_page(image)), used
        except Exception as e:
            return f"{type(e).__name__}: {e}", used
    finally:
        converter.get_external_fallback, converter._cloud_vision_transcribe, converter.vision_enabled = saved


blocked, down = ocr_provider.OcrBlockedError("Gemini: RECITATION"), ocr_provider.OcrTransientError("giù")
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
    ocr_provider._EXTERNAL_FALLBACK = None
    return ocr_provider.get_external_fallback

check("esterno: spento, nessun ripiego", external(enabled=False)(), None)
check("esterno: fornitore sconosciuto", fails_with(external(provider="boh")), "ValueError")
check("esterno: fornitore mancante", fails_with(external(provider="")), "ValueError")
check("esterno: modello mancante", fails_with(external(model="")), "ValueError")
check("esterno: chiave mancante", fails_with(external(key="")), "ValueError")
for name, cls in (("gemini", "GeminiOcr"), ("openai", "OpenAICompatibleOcr"), ("anthropic", "AnthropicOcr"), ("mistral", "MistralOcr")):
    built = external(provider=name)()
    check(f"esterno: {name}", (type(built).__name__, built._model), (cls, "esterno-prova"))
fake.replies = [(200, "dal modello esterno")]
check("esterno: usa il suo modello", (asyncio.run(external(provider="openai")().transcribe(image, "p")), fake.requests[-1][1]["model"]),
      ("dal modello esterno", "esterno-prova"))
external(provider="boh")
vision_calls = []
saved = converter._cloud_vision_transcribe
converter._cloud_vision_transcribe = vision_calls.append
check("esterno: configurazione errata ferma il lavoro", (fails_with(lambda: asyncio.run(converter._external_transcribe_page(image))), vision_calls), ("ValueError", []))
converter._cloud_vision_transcribe = saved
external(enabled=False)

# --- Gemini con il vero SDK: un rifiuto, anche con testo parziale, non è una trascrizione
os.environ["GOOGLE_GEMINI_BASE_URL"] = fake.url
real_gemini = ocr_provider.GeminiOcr(api_key="k", model="gemini-prova")
def gemini_reply(reason, parts):
    content = {"role": "model", "parts": parts} if parts else {}
    fake.replies = [(200, {"candidates": [{"finishReason": reason, "content": content}]})]
    return fails_with(lambda: asyncio.run(real_gemini.transcribe(image, "p")))
check("gemini: rifiuto senza testo", gemini_reply("RECITATION", None), "OcrBlockedError")
check("gemini: rifiuto con testo parziale", gemini_reply("RECITATION", [{"text": "parziale"}]), "OcrBlockedError")
check("gemini: risposta normale", gemini_reply("STOP", [{"text": "ok"}]), None)
check("gemini: modello del ripiego", "gemini-prova" in fake.requests[-1][0], True)
del os.environ["GOOGLE_GEMINI_BASE_URL"]

# --- Altri fornitori
os.environ.setdefault("x", "y")
provider = ocr_provider.OpenAICompatibleOcr()
fake.replies = [(200, "openai ok")]
check("openai", asyncio.run(provider.transcribe(image, "p")), "openai ok")

provider = ocr_provider.AnthropicOcr()
fake.replies = [(200, "claude ok")]
check("anthropic", asyncio.run(provider.transcribe(image, "p")), "claude ok")
check("anthropic: immagine", fake.requests[-1][1]["messages"][0]["content"][0]["source"]["media_type"], "image/png")

provider = ocr_provider.MistralOcr()
provider._client.sdk_configuration.server_url = fake.url
fake.replies = [(200, "pixtral ok")]
check("mistral: chat con immagine", asyncio.run(provider.transcribe(image, "p")), "pixtral ok")
object.__setattr__(ocr_provider.settings, "ocr_model_name", "mistral-ocr-latest")
provider = ocr_provider.MistralOcr()
provider._client.sdk_configuration.server_url = fake.url
fake.replies = [(200, {"model": "m", "pages": [
    {"index": 0, "markdown": "pagina", "images": [], "dimensions": None}], "usage_info": {"pages_processed": 1}})]
check("mistral: API OCR", asyncio.run(provider.transcribe(image, "p")), "pagina")
check("mistral: endpoint OCR", fake.requests[-1][0], "/v1/ocr")

check("ollama: si crea", type(ocr_provider.OllamaOcr()).__name__, "OllamaOcr")

object.__setattr__(ocr_provider.settings, "ocr_provider", "local")
object.__setattr__(ocr_provider.settings, "ocr_model_name", "")
try:
    ocr_provider._build_provider()
    check("senza modello: errore", "nessun errore", "ValueError")
except ValueError as e:
    check("senza modello: errore chiaro", "OCR_MODEL_NAME" in str(e), True)

# --- Lavoro del revisore: ogni pagina all'OCR esterno, senza passare dal modello locale
import json, pymupdf
calls = []


async def fake_external_page(image):
    calls.append(1)
    return f"pagina {len(calls)}"


real_page, converter._external_transcribe_page = converter._external_transcribe_page, fake_external_page
# Il processo separato rilancerebbe questo script: qui basta la funzione.
converter._run_killable = lambda target, args, timeout: target(*args)
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

# --- Documenti d'ufficio: testo e tabelle senza OCR
import datetime, openpyxl, pandas, pypandoc
from common.file_intake import detect_extension_from_content
office_dir = pathlib.Path(tempfile.mkdtemp())
source_md = "# Determina\n\nIl **dirigente** dispone.\n\n| Voce | Importo |\n|---|---|\n| Pulizie | 1.200,00 |\n"
for ext in ("docx", "odt"):
    pypandoc.convert_text(source_md, ext, format="md", outputfile=str(office_dir / f"atto.{ext}"))
workbook = openpyxl.Workbook()
workbook.active.title = "Spese"
for row in (["Data", "Voce", "Importo"], [datetime.date(2026, 3, 1), "Pulizie", 1200.5], [None, None, None],
            [datetime.date(2026, 4, 2), "a | b", 800]):
    workbook.active.append(row)
workbook.create_sheet("Vuoto")
workbook.save(office_dir / "conti.xlsx")
pandas.DataFrame([["Comune", "Abitanti"], ["Lucca", 89000]]).to_excel(
    office_dir / "comuni.ods", engine="odf", header=False, index=False, sheet_name="Comuni")
for name in ("atto.docx", "atto.odt", "conti.xlsx", "comuni.ods"):
    check(f"ufficio: tipo riconosciuto {name}", detect_extension_from_content((office_dir / name).read_bytes()),
          pathlib.Path(name).suffix)
for ext in ("docx", "odt"):
    text = asyncio.run(converter.convert_file_to_markdown(office_dir / f"atto.{ext}", f"atto.{ext}"))
    check(f"ufficio: {ext} con titolo, grassetto e tabella",
          ("# Determina" in text, "**dirigente**" in text, "| Pulizie" in text and "1.200,00" in text), (True, True, True))
check("ufficio: xlsx", asyncio.run(converter.convert_file_to_markdown(office_dir / "conti.xlsx", "conti.xlsx")),
      "## Spese\n\n| Data | Voce | Importo |\n|---|---|---|\n| 2026-03-01 | Pulizie | 1200.5 |\n| 2026-04-02 | a \\| b | 800 |")
check("ufficio: ods", asyncio.run(converter.convert_file_to_markdown(office_dir / "comuni.ods", "comuni.ods")),
      "## Comuni\n\n| Comune | Abitanti |\n|---|---|\n| Lucca | 89000 |")
check("ufficio: zip qualsiasi non è un documento", detect_extension_from_content(b"PK\x03\x04 non un archivio"), None)

# --- Conteggio delle pagine OCR
async def counted(path, name):
    stats = {"ocr": 0, "total": 0}
    converter.PAGE_STATS.set(stats)
    await converter.convert_file_to_markdown(path, name)
    return stats


object.__setattr__(converter.settings, "ocr_vision_fallback", True)
converter.async_ocr_generate = lambda image, as_markdown=False: asyncio.sleep(0, "trascritto")
image_path = job_dir / "pagina.png"
image.save(image_path)
check("pagine OCR: immagine", asyncio.run(counted(image_path, "pagina.png")), {"ocr": 1, "total": 1})
converter._run_killable = lambda target, args, timeout: [{"type": "text", "content": "nativo"}, {"type": "image", "content": image_path.read_bytes()}]
pdf_path = job_dir / "misto.pdf"
pdf.save(pdf_path)
check("pagine OCR: pdf misto", asyncio.run(counted(pdf_path, "misto.pdf")), {"ocr": 1, "total": 2})
text_path = job_dir / "nota.txt"
text_path.write_text("testo")
check("pagine OCR: testo", asyncio.run(counted(text_path, "nota.txt")), {"ocr": 0, "total": 0})

# --- Ricostruzione di document_ocr dai file archiviati
import json, time
import pymupdf
import rebuild_document_ocr as rebuild

archive = pathlib.Path(tempfile.mkdtemp())
series = archive / "arc" / "ser" / "sotto"
series.mkdir(parents=True)
with pymupdf.open() as doc:
    doc.new_page().insert_text((50, 50), "testo nativo " * 10)
    doc.new_page()
    doc.save(series / "misto.pdf")
image.save(series / "foto.jpg")
(series / "nota.txt").write_text("testo")
(series / "rotto.pdf").write_bytes(b"non un pdf")
docs = {"misto.pdf": "direct://arc/ser/sotto/misto.pdf", "foto.jpg": "direct://arc/ser/sotto/foto.jpg",
        "nota.txt": "direct://arc/ser/sotto/nota.txt", "rotto.pdf": "direct://arc/ser/sotto/rotto.pdf",
        "perso.pdf": "direct://arc/ser/sotto/perso.pdf", "noto.pdf": "direct://arc/ser/sotto/noto.pdf"}
for name, source in docs.items():
    (series / f"{name}.json").write_text(json.dumps({"source": source, "files": [f"sotto/{name}"]}))
(series / "atto.json").write_text(json.dumps({"source": "sicraweb://1", "files": ["a.jpg"]}))
image.save(series / "a.jpg")
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
