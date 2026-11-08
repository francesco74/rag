"""Test dei provider LLM ed embedding con i veri SDK contro un server finto.

Uso:  python api/tests/test_llm.py      (esce con 1 se qualcosa fallisce)
"""
import asyncio, json, os, pathlib, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "api" / "src"), str(ROOT), str(ROOT / "api" / "tests")]
from fake_llm_server import FakeLLM

fake = FakeLLM()
os.environ.update(AUTH_SECRET_KEY="x" * 40, API_SECRET_KEY="k", DATA_FOLDER=tempfile.mkdtemp(), DOCWS_RICERCA_ENDPOINT="u",
                  DOCWS_ATTI_ENDPOINT="u", DOCWS_CODICE_AMMINISTRAZIONE="u", DOCWS_CODICE_AOO="u", RUOLO_DOCWS="u",
                  LOG_LEVEL="CRITICAL", API_LLM_KEY="k", LLM_PROVIDER="local", LOCAL_LLM_BACKEND="vllm",
                  LOCAL_VLLM_BASE_URL=fake.url + "/v1", OPENAI_BASE_URL=fake.url + "/v1", ANTHROPIC_BASE_URL=fake.url,
                  QUERY_REWRITER_MODEL_NAME="q", ANSWER_GENERATOR_MODEL_NAME="a", GRADER_MODEL_NAME="g",
                  EMBEDDING_PROVIDER="vllm", LOCAL_VLLM_EMBEDDING_MODEL="BAAI/bge-m3", EMBEDDING_DIMENSION="2")

import llm_provider
from common import embedding
from common.config import settings
from rag.generation import AnswerSchema

failures = 0


def check(label, got, expected=True):
    global failures
    if got != expected:
        failures += 1
        print(f"FALLITO {label}: {got!r} invece di {expected!r}")
    else:
        print(f"ok {label}")


# --- vLLM: risposta strutturata, thinking, nomi dei parametri
provider = llm_provider.init_llm_provider()
check("LLM_PROVIDER=local + vllm", type(provider).__name__, "VLLMProvider")

fake.replies = [(200, '<think>ragiono</think>\n{"is_found": true, "is_general_knowledge": false, "answer": "x"}')]
raw, meta = provider.generate_json("a", "domanda", max_tokens=100, thinking_level="LOW", with_meta=True,
                                   response_schema=AnswerSchema)
path, body = fake.requests[-1]
check("vllm: endpoint", path, "/v1/chat/completions")
check("vllm: <think> tolto", json.loads(raw)["answer"], "x")
check("vllm: meta", (meta["truncated"], meta["output_tokens"]), (False, 2))
schema = body["response_format"]["json_schema"]["schema"]
check("vllm: schema Pydantic convertito", isinstance(schema, dict) and schema["additionalProperties"] is False)
check("vllm: required", sorted(schema["required"]), ["answer", "is_found", "is_general_knowledge"])
check("vllm: max_tokens", body.get("max_tokens"), 100)
check("vllm: thinking acceso", body["chat_template_kwargs"], {"enable_thinking": True})

provider.generate_json("g", "domanda")
body = fake.requests[-1][1]
check("vllm: thinking spento", body["chat_template_kwargs"], {"enable_thinking": False})
check("vllm: json mode", body["response_format"], {"type": "json_object"})

provider.generate_text("g", "domanda", max_tokens=None)
check("vllm: max_tokens omesso con None", "max_tokens" in fake.requests[-1][1], False)

# --- OpenAI: parametro nuovo e modelli che rifiutano temperature
provider = llm_provider.OpenAIProvider()
provider.generate_text("gpt", "domanda", max_tokens=10)
body = fake.requests[-1][1]
check("openai: max_completion_tokens", (body.get("max_completion_tokens"), "max_tokens" in body), (10, False))
check("openai: nessun campo vLLM", "chat_template_kwargs" in body, False)

fake.replies = [(400, {"error": {"message": "Unsupported value: 'temperature' does not support 0.1"}}), (200, "bene")]
check("openai: temperature rifiutata", provider.generate_text("gpt", "domanda", temperature=0.1), "bene")
check("openai: secondo tentativo senza temperature", "temperature" in fake.requests[-1][1], False)

fake.replies = [(400, {"error": {"message": "altro errore"}})]
try:
    provider.generate_text("gpt", "domanda")
    check("openai: altri 400 propagati", False)
except Exception as e:
    check("openai: altri 400 propagati", type(e).__name__, "BadRequestError")

# --- Anthropic: blocchi di testo, senza temperature
provider = llm_provider.AnthropicProvider()
fake.replies = [(200, '{"a": 1}')]
check("anthropic: testo", provider.generate_json("c", "domanda", response_schema=AnswerSchema), '{"a": 1}')
check("anthropic: temperature non inviata", "temperature" in fake.requests[-1][1], False)

# --- Mistral: l'SDK 2.x si importa da mistralai.client
provider = llm_provider.MistralProvider()
provider._client.sdk_configuration.server_url = fake.url
check("mistral: risposta", provider.generate_text("m", "domanda"), "ok")

# --- Modelli Gemini con un altro provider: errore chiaro all'avvio
saved = settings.grader_model_name
object.__setattr__(settings, "grader_model_name", "gemini-3.1-flash-lite")
try:
    llm_provider.init_llm_provider()
    check("modello Gemini con vLLM rifiutato", False)
except ValueError as e:
    check("modello Gemini con vLLM rifiutato", "GRADER_MODEL_NAME" in str(e))
object.__setattr__(settings, "grader_model_name", saved)

# --- Embedding su vLLM
provider = embedding.init_embedding()
check("embedding: provider", (type(provider).__name__, provider.dimension), ("VLLMEmbeddingProvider", 2))
check("embedding: query", embedding.embed_query("testo"), [0.5, 0.5])
path, body = fake.requests[-1]
check("embedding: richiesta", (path, body["model"], "dimensions" in body), ("/v1/embeddings", "BAAI/bge-m3", False))
check("embedding: documenti", asyncio.run(provider.embed_documents_batch(["a", "b"])), [[0.5, 0.5], [0.5, 0.5]])
check("embedding: senza istruzione", body["input"], ["testo"])

# --- Istruzione per le domande (Qwen3-Embedding): solo sulle domande della ricerca
object.__setattr__(settings, "embedding_query_instruction", "Given a question, retrieve passages")
provider = embedding.init_embedding()
embedding.embed_query("chi firma?")
check("istruzione: domanda", fake.requests[-1][1]["input"], ["Instruct: Given a question, retrieve passages\nQuery:chi firma?"])
embedding.embed_queries_batch(["a", "b"])
check("istruzione: più domande", [t.split("Query:")[1] for t in fake.requests[-1][1]["input"]], ["a", "b"])
embedding.embed_for_semantic_query("concetto")
check("istruzione: non nei confronti tra frasi", fake.requests[-1][1]["input"], ["concetto"])
asyncio.run(provider.embed_documents_batch(["doc"]))
check("istruzione: non nei documenti", fake.requests[-1][1]["input"], ["doc"])
object.__setattr__(settings, "embedding_query_instruction", None)

# --- Vettori di dimensione diversa da EMBEDDING_DIMENSION: errore chiaro e permanente
from common.utility import is_permanent_error
object.__setattr__(settings, "embedding_dimension", 3)
provider = embedding.init_embedding()
for label, call in (("query", lambda: embedding.embed_query("x")),
                    ("documenti", lambda: asyncio.run(provider.embed_documents_batch(["x"])))):
    try:
        call()
        check(f"dimensione errata: {label}", False)
    except embedding.EmbeddingDimensionError as e:
        check(f"dimensione errata: {label}", ("2 dimensioni" in str(e), is_permanent_error(e)), (True, True))
object.__setattr__(settings, "embedding_dimension", 2)

print(f"\n{'TUTTO OK' if not failures else f'{failures} FALLITI'}")
sys.exit(1 if failures else 0)
