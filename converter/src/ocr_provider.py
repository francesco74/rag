"""
ocr_provider.py — trascrizione di una pagina-immagine con il modello scelto da
OCR_PROVIDER (OCR_MODEL_NAME è il nome del modello per quel fornitore).

  gemini    — Google Gemini (default)
  openai    — OpenAI
  anthropic — Anthropic Claude
  mistral   — Mistral: API OCR dedicata per i modelli "mistral-ocr-*",
              altrimenti chat con immagine (es. pixtral)
  local     — self-hosted, con LOCAL_LLM_BACKEND: vllm (stesso protocollo di
              OpenAI, su LOCAL_VLLM_BASE_URL) oppure ollama (LOCAL_LLM_HOST).
              Il modello deve essere multimodale.

Il chiamante vede due sole eccezioni oltre agli errori definitivi:
OcrTransientError (rate limit, sovraccarico, rete: si ritenta) e OcrBlockedError
(il modello rifiuta la pagina per policy: inutile ritentare).

I client sono creati al primo uso: gli SDK dei fornitori non attivi non vengono
importati e un converter senza credenziali parte comunque.
"""

import base64
import io
import logging
import re
from abc import ABC, abstractmethod

from PIL import Image

from common.config import settings

log = logging.getLogger("OCR")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*\n(.*?)\n?```\s*$", re.DOTALL)


class OcrTransientError(Exception):
    """Rate limit, sovraccarico o errore di rete: la chiamata si può ritentare."""


class OcrBlockedError(Exception):
    """Il modello ha rifiutato la pagina per policy: ritentare non cambierebbe l'esito."""


def clean_transcription(text):
    """Toglie ciò che i modelli aggiungono attorno alla trascrizione e le sillabazioni a fine riga."""
    text = _THINK_RE.sub("", text or "").strip()
    if m := _FENCE_RE.match(text):
        text = m[1].strip()
    return text.replace("-\n", "")


def image_to_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def image_to_data_url(image: Image.Image) -> str:
    return "data:image/png;base64," + base64.b64encode(image_to_png(image)).decode()


class OcrProvider(ABC):
    @abstractmethod
    async def transcribe(self, image: Image.Image, prompt: str) -> str:
        """Testo della pagina, grezzo. Solleva OcrTransientError o OcrBlockedError."""


class GeminiOcr(OcrProvider):
    def __init__(self, api_key=None, model=None):
        from google import genai
        from google.genai import errors, types
        self._model = model or settings.ocr_model_name
        self._client = genai.Client(api_key=api_key or settings.api_llm_key)
        self._types = types
        self._errors = errors
        categories = (types.HarmCategory.HARM_CATEGORY_HARASSMENT, types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                      types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT)
        self._safety = [types.SafetySetting(category=c, threshold=types.HarmBlockThreshold.BLOCK_NONE) for c in categories]

    async def transcribe(self, image, prompt):
        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=[prompt, image],
                config=self._types.GenerateContentConfig(safety_settings=self._safety),
            )
        except self._errors.ServerError as e:
            raise OcrTransientError(str(e)) from e
        except self._errors.APIError as e:
            if getattr(e, "code", None) == 429:
                raise OcrTransientError(str(e)) from e
            raise

        candidate = response.candidates[0] if response.candidates else None
        reason = candidate.finish_reason.name if candidate and candidate.finish_reason else "Unknown"
        # Un rifiuto può arrivare con un testo parziale: non va preso per una trascrizione.
        if reason in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "3", "4"):
            raise OcrBlockedError(f"Gemini: {reason}")
        if not candidate or not candidate.content or not candidate.content.parts:
            raise ValueError(f"Gemini API Blocked Page. Exact Finish Reason: {reason}")
        return response.text


class OpenAICompatibleOcr(OcrProvider):
    """OpenAI e vLLM: stesso protocollo, cambia solo l'indirizzo (vedi VLLMProvider in llm_provider.py)."""

    def __init__(self, base_url=None, api_key=None, model=None):
        from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
        self._model = model or settings.ocr_model_name
        self._client = AsyncOpenAI(api_key=api_key or settings.api_llm_key, base_url=base_url)
        self._transient = (RateLimitError, APIConnectionError)
        self._status_error = APIStatusError

    async def transcribe(self, image, prompt):
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
                ]}],
            )
        except self._transient as e:
            raise OcrTransientError(str(e)) from e
        except self._status_error as e:
            if e.status_code >= 500:
                raise OcrTransientError(str(e)) from e
            raise
        choice = response.choices[0]
        if choice.finish_reason == "content_filter":
            raise OcrBlockedError(f"finish_reason={choice.finish_reason}")
        return choice.message.content


class AnthropicOcr(OcrProvider):
    # max_tokens è obbligatorio nella Messages API (vedi llm_provider.py).
    MAX_TOKENS = 8192

    def __init__(self, api_key=None, model=None):
        import anthropic
        self._model = model or settings.ocr_model_name
        self._client = anthropic.AsyncAnthropic(api_key=api_key or settings.api_llm_key)
        self._transient = (anthropic.RateLimitError, anthropic.APIConnectionError)
        self._status_error = anthropic.APIStatusError

    async def transcribe(self, image, prompt):
        data = base64.b64encode(image_to_png(image)).decode()
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=self.MAX_TOKENS,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
                    {"type": "text", "text": prompt},
                ]}],
            )
        except self._transient as e:
            raise OcrTransientError(str(e)) from e
        except self._status_error as e:
            if e.status_code >= 500:
                raise OcrTransientError(str(e)) from e
            raise
        if response.stop_reason == "refusal":
            raise OcrBlockedError("Anthropic: refusal")
        return "".join(block.text for block in response.content if block.type == "text")


class MistralOcr(OcrProvider):
    def __init__(self, api_key=None, model=None):
        from mistralai.client import Mistral
        from mistralai.client.errors import SDKError
        self._model = model or settings.ocr_model_name
        self._client = Mistral(api_key=api_key or settings.api_llm_key)
        self._sdk_error = SDKError

    async def transcribe(self, image, prompt):
        url = image_to_data_url(image)
        try:
            if "ocr" in self._model:
                # L'API OCR ha un formato suo e non riceve il prompt: restituisce sempre Markdown.
                response = await self._client.ocr.process_async(
                    model=self._model, document={"type": "image_url", "image_url": url},
                )
                return "\n\n".join(page.markdown for page in response.pages)
            response = await self._client.chat.complete_async(
                model=self._model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": url},
                ]}],
            )
            return response.choices[0].message.content
        except self._sdk_error as e:
            status_code = getattr(e, "status_code", None)
            if status_code == 429 or (status_code or 0) >= 500:
                raise OcrTransientError(str(e)) from e
            raise


class OllamaOcr(OcrProvider):
    def __init__(self):
        import ollama
        self._client = ollama.AsyncClient(host=settings.local_llm_host)
        self._response_error = ollama.ResponseError

    async def transcribe(self, image, prompt):
        try:
            response = await self._client.chat(
                model=settings.ocr_model_name,
                messages=[{"role": "user", "content": prompt, "images": [image_to_png(image)]}],
            )
        except ConnectionError as e:
            raise OcrTransientError(str(e)) from e
        except self._response_error as e:
            if e.status_code == 429 or e.status_code >= 500:
                raise OcrTransientError(str(e)) from e
            raise
        return response.message.content


EXTERNAL_PROVIDERS = {"gemini": GeminiOcr, "openai": OpenAICompatibleOcr, "anthropic": AnthropicOcr, "mistral": MistralOcr}


def _build_provider() -> OcrProvider:
    provider = settings.ocr_provider
    if not settings.ocr_model_name:
        raise ValueError(f"OCR_PROVIDER='{provider}' richiede OCR_MODEL_NAME: nessun modello impostato.")
    if provider == "local":
        if settings.local_llm_backend == "vllm":
            return OpenAICompatibleOcr(base_url=settings.local_vllm_base_url,
                                       api_key=settings.local_vllm_api_key or "not-needed")
        if settings.local_llm_backend == "ollama":
            return OllamaOcr()
        raise ValueError(f"Unknown LOCAL_LLM_BACKEND='{settings.local_llm_backend}'. Valid options: ['ollama', 'vllm']")
    cls = EXTERNAL_PROVIDERS.get(provider)
    if cls is None:
        raise ValueError(f"Unknown OCR_PROVIDER='{provider}'. Valid options: {list(EXTERNAL_PROVIDERS) + ['local']}")
    return cls()


_INSTANCE = None
_EXTERNAL_FALLBACK = None


def get_ocr_provider() -> OcrProvider:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = _build_provider()
        log.info(f"OCR provider: {settings.ocr_provider} (modello {settings.ocr_model_name})")
    return _INSTANCE


def get_external_fallback() -> OcrProvider | None:
    """Il modello esterno per il comando del revisore, None se OCR_EXTERNAL_FALLBACK non è attivo. Attivo ma incompleto è un errore."""
    global _EXTERNAL_FALLBACK
    if not settings.ocr_external_fallback:
        return None
    if _EXTERNAL_FALLBACK is None:
        cls = EXTERNAL_PROVIDERS.get(settings.ocr_external_provider)
        if cls is None:
            raise ValueError(f"OCR_EXTERNAL_ALTERNATIVE_PROVIDER='{settings.ocr_external_provider}' non valido. "
                             f"Valori ammessi: {list(EXTERNAL_PROVIDERS)}")
        if not settings.ocr_external_model_name or not settings.ocr_external_api_key:
            raise ValueError("OCR_EXTERNAL_FALLBACK=true richiede OCR_EXTERNAL_ALTERNATIVE_MODEL_NAME e OCR_EXTERNAL_API_KEY.")
        _EXTERNAL_FALLBACK = cls(api_key=settings.ocr_external_api_key, model=settings.ocr_external_model_name)
        log.info(f"Ripiego OCR esterno: {settings.ocr_external_provider} (modello {settings.ocr_external_model_name})")
    return _EXTERNAL_FALLBACK
