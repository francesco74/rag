"""
llm_provider.py — Provider-agnostic LLM adapter.

Supported providers (set via LLM_PROVIDER env var):
  gemini    — Google Gemini (default)
  openai    — OpenAI
  anthropic — Anthropic Claude
  mistral   — Mistral AI
  local     — self-hosted, backend scelto da LOCAL_LLM_BACKEND:
                ollama — sviluppo/basso-concorrenza (default)
                vllm   — produzione multi-utente (continuous batching)

Each provider exposes two operations used by worker.py:
  generate_json(model_name, prompt, temperature, max_tokens) -> str   (raw JSON string)
  generate_text(model_name, prompt, temperature, max_tokens) -> str   (plain text)

max_tokens=None significa "nessun tetto esplicito: usa il massimo del modello".
Serve alla generazione della risposta, dove la lunghezza è imprevedibile e un
tetto scritto a mano è solo una costante di vendor che diverge in silenzio al
primo cambio di modello. NON va usato dove l'output è strutturalmente corto
(grader, rewriter): lì il tetto non è una capability da inseguire ma un
circuit breaker contro i loop di ripetizione, e resta impostato.
Su Gemini e OpenAI il parametro viene semplicemente omesso dalla richiesta.
Su Anthropic max_tokens è OBBLIGATORIO nella Messages API, quindi lì si ripiega
su _ANTHROPIC_MAX_TOKENS_FALLBACK: è un vincolo dell'SDK, non una policy.

Entrambe accettano with_meta=True: in quel caso restituiscono la tupla
(text, meta) dove meta è un dict:
    {"truncated": bool, "finish_reason": str,
     "thinking_tokens": int|None, "output_tokens": int|None}
Serve a generate_answer per accorgersi che il modello ha esaurito il budget di
output (su Gemini CONDIVISO col thinking) e per capirne il MOTIVO: sapere solo
"è troncata" non dice se la risposta era genuinamente lunga o se il thinking si
è mangiato il budget, due situazioni che richiedono interventi opposti. Il flag
di troncamento è normalizzato qui: ogni SDK lo espone in modo diverso
(MAX_TOKENS su Gemini, 'length' su OpenAI, 'max_tokens' su Anthropic), ma
worker.py non deve saperlo.

Retry logic is provider-aware: each provider raises different exceptions
on rate-limit/overload, and we normalise them into a single LLMRateLimitError
so worker.py never needs to know which SDK is in use.
"""

import os
import logging
import time
from abc import ABC, abstractmethod
from common.config import settings

log = logging.getLogger("rag_queue")

# Anthropic è l'unico dei tre provider in cui max_tokens è obbligatorio nella
# Messages API: non esiste un modo di dire "usa il massimo del modello". Quando
# il chiamante passa None si ripiega su questo valore. È un dettaglio di
# implementazione dell'SDK, non una scelta di deployment: per questo è una
# costante qui e non una variabile d'ambiente — non c'è nessuna decisione
# operativa da prendere, solo un buco dell'API da tappare.
_ANTHROPIC_MAX_TOKENS_FALLBACK = 8192

# ==============================================================================
# PROVIDER REGISTRY
# ==============================================================================
_PROVIDER_INSTANCE = None  # set once in init_llm_provider()


# ==============================================================================
# RETRY HELPERS
# ==============================================================================
class LLMRateLimitError(Exception):
    """Raised by any provider when the API signals rate-limit or overload."""


def _retry_with_backoff(fn, max_attempts=6, base_wait=4, max_wait=60):
    """
    Simple exponential backoff with jitter.
    Retries only on LLMRateLimitError; all other exceptions propagate immediately.
    """
    import random
    attempt = 0
    while True:
        try:
            return fn()
        except LLMRateLimitError as e:
            attempt += 1
            if attempt >= max_attempts:
                raise
            wait = min(base_wait * (2 ** (attempt - 1)), max_wait)
            wait = wait * (0.5 + random.random() * 0.5)  # jitter
            log.warning(f"Rate limit hit ({attempt}/{max_attempts}). Retrying in {wait:.1f}s... [{e}]")
            time.sleep(wait)


# ==============================================================================
# BASE PROVIDER
# ==============================================================================
class LLMProvider(ABC):

    @abstractmethod
    def generate_json(self, model_name: str, prompt: str,
                      temperature: float = 0.1, max_tokens: int | None = 2048,
                      thinking_level: str | None = None,
                      with_meta: bool = False,
                      response_schema=None):
        """
        Call the model asking for a JSON response.
        Returns the raw JSON string, oppure (str, meta) se with_meta=True.

        max_tokens=None -> nessun tetto esplicito (massimo del modello).
        Il default resta 2048: chi non specifica nulla è un chiamante con
        output corto e strutturato (rewriter), che il tetto lo vuole.

        response_schema=None -> comportamento invariato (JSON mode "libero":
        l'API garantisce solo che il testo SIA JSON valido, non che rispetti
        una forma precisa). Se valorizzato, i provider che lo supportano
        nativamente (Gemini, OpenAI) applicano un decoding vincolato alla
        grammatica dello schema — che oltre a validare i campi elimina anche
        la classe di bug "backslash orfano dentro una stringa" perché il
        modello non può più emettere un token di escape non valido. I
        provider senza equivalente nativo (Anthropic, Mistral, local) lo
        ignorano silenziosamente e restano sul JSON mode "libero" esistente:
        l'interfaccia resta unica, il comportamento no-op è esplicito e
        documentato qui, non una sorpresa da scoprire in produzione.
        Formato atteso: una classe Pydantic (BaseModel) per Gemini, oppure
        un JSON Schema (dict) per OpenAI — vedi le rispettive implementazioni.
        """

    @abstractmethod
    def generate_text(self, model_name: str, prompt: str,
                      temperature: float = 0.0, max_tokens: int | None = 16,
                      thinking_level: str | None = None,
                      with_meta: bool = False):
        """
        Call the model asking for a plain-text response.
        Returns the text string, oppure (str, meta) se with_meta=True.
        """


# ==============================================================================
# GEMINI PROVIDER
# ==============================================================================
class GeminiProvider(LLMProvider):

    def __init__(self):
        try:
            # Nuovo import pulito come desideravi
            from google import genai
            from google.genai.errors import APIError
            
            # Il nuovo SDK centralizza tutto in un oggetto Client.
            # Viene istanziato una sola volta usando la chiave dei tuoi settings.
            self._client = genai.Client(api_key=settings.api_llm_key)
            self._rate_limit_exceptions = (APIError,)
            
            log.info("✓ GeminiProvider initialized with new google-genai SDK.")
        except ImportError:
            raise RuntimeError(
                "google-genai package not installed. "
            )

    def _call(self, model_name: str, prompt: str, temperature: float, max_tokens: int | None, json_mode: bool = False,
              thinking_level: str | None = None, with_meta: bool = False, response_schema=None):
        from google.genai import types

        # Nel nuovo SDK le configurazioni di generazione passano da GenerateContentConfig
        config_kwargs = {
            "temperature": temperature,
        }
        # max_output_tokens OMESSO quando None: il modello usa il proprio
        # massimo. Passare un numero scritto a mano significherebbe mantenere
        # una copia locale di un dato che appartiene al vendor e che cambia a
        # ogni cambio di ANSWER_GENERATOR_MODEL_NAME.
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens

        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        # response_schema attiva il "controlled generation" di Gemini: il SDK
        # accetta direttamente una classe Pydantic (BaseModel) e la traduce
        # nello schema interno. A differenza del solo response_mime_type,
        # questo vincola la grammatica di decoding token-per-token, quindi
        # non solo i campi rispettano tipi/nomi attesi, ma il testo delle
        # stringhe non può più contenere un escape JSON invalido: il
        # tokenizer non ha la possibilità di produrre una sequenza illegale.
        if response_schema is not None:
            config_kwargs["response_schema"] = response_schema

        if thinking_level:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=thinking_level
            )
            
        config = types.GenerateContentConfig(**config_kwargs)

        def _attempt():
            try:
                # La chiamata passa attraverso il gestore dei modelli del client
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config
                )
                self._log_usage(model_name, response, max_tokens)
                if not with_meta:
                    return response.text
                # finish_reason normalizzato: MAX_TOKENS significa che il budget
                # di output (condiviso col thinking) si è esaurito prima che il
                # modello finisse di scrivere.
                candidates = getattr(response, "candidates", None)
                fr = getattr(candidates[0], "finish_reason", None) if candidates else None
                fr_str = str(fr) if fr is not None else "UNKNOWN"
                usage = getattr(response, "usage_metadata", None)
                return response.text, {
                    "truncated": "MAX_TOKENS" in fr_str,
                    "finish_reason": fr_str,
                    # Servono a distinguere "risposta genuinamente lunga" da
                    # "thinking che ha divorato il budget": senza questi due
                    # numeri il troncamento resta un evento non diagnosticabile.
                    "thinking_tokens": getattr(usage, "thoughts_token_count", None) if usage else None,
                    "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
                }
            except self._rate_limit_exceptions as e:
                # Il nuovo APIError espone l'attributo 'code' (lo status HTTP).
                # Intercettiamo i codici 429 (Rate Limit) e 503 (Servizio Non Disponibile/Overload).
                status_code = getattr(e, "code", None)
                if status_code in (429, 503):
                    raise LLMRateLimitError(str(e)) from e
                raise

        return _retry_with_backoff(_attempt)

    def _log_usage(self, model_name: str, response, max_tokens: int) -> None:
        """
        Logga finish_reason e ripartizione token (thinking vs output) per ogni
        chiamata Gemini. Nel google-genai SDK, max_output_tokens è un budget
        CONDIVISO tra token di thinking e token di output visibile (comportamento
        confermato da più issue upstream, es. googleapis/python-genai#782,#2062):
        se il thinking (che può essere dinamico e imprevedibile) consuma quasi
        tutto il budget, il testo visibile viene troncato con finish_reason
        MAX_TOKENS — esattamente il sintomo osservato nei log di generate_answer.

        Non solleva mai eccezioni: è pura osservabilità, un suo fallimento non
        deve mai far cadere la chiamata principale.
        """
        try:
            usage = getattr(response, "usage_metadata", None)
            candidates = getattr(response, "candidates", None)
            finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
            finish_reason_str = str(finish_reason) if finish_reason is not None else "UNKNOWN"

            thinking_tokens = getattr(usage, "thoughts_token_count", None) if usage else None
            output_tokens = getattr(usage, "candidates_token_count", None) if usage else None
            total_tokens = getattr(usage, "total_token_count", None) if usage else None

            # Con max_tokens=None il budget è quello del modello: stamparlo come
            # "None" farebbe sembrare un dato mancante invece di una scelta.
            budget_str = str(max_tokens) if max_tokens is not None else "model_default"

            base_msg = (
                f"[GEMINI_USAGE] model={model_name} finish_reason={finish_reason_str} "
                f"thinking_tokens={thinking_tokens} output_tokens={output_tokens} "
                f"total_tokens={total_tokens} max_output_tokens_budget={budget_str}"
            )

            if "MAX_TOKENS" in finish_reason_str:
                # La percentuale ha senso solo rispetto a un tetto che abbiamo
                # imposto noi. Se il budget è quello del modello, la quota di
                # thinking si legge dai valori assoluti già presenti nel messaggio.
                thinking_pct = (
                    round(thinking_tokens / max_tokens * 100)
                    if thinking_tokens and max_tokens else None
                )
                log.warning(
                    f"{base_msg} — ⚠ Risposta troncata (MAX_TOKENS). Il thinking ha "
                    f"consumato {f'{thinking_pct}%' if thinking_pct is not None else 'N/D'} "
                    f"del budget totale, lasciando poco/nessun margine per l'output visibile."
                )
            else:
                log.debug(base_msg)
        except Exception as e:
            # La telemetria non deve mai bloccare il flusso principale.
            log.debug(f"[GEMINI_USAGE] Impossibile leggere usage_metadata: {e}")

    def generate_json(self, model_name: str, prompt: str, temperature: float = 0.1, max_tokens: int | None = 2048,
                      thinking_level: str | None = None, with_meta: bool = False, response_schema=None):
        return self._call(model_name, prompt, temperature, max_tokens, json_mode=True,
                          thinking_level=thinking_level, with_meta=with_meta, response_schema=response_schema)

    def generate_text(self, model_name: str, prompt: str, temperature: float = 0.0, max_tokens: int | None = 16,
                      thinking_level: str | None = None, with_meta: bool = False):
        return self._call(model_name, prompt, temperature, max_tokens, json_mode=False,
                          thinking_level=thinking_level, with_meta=with_meta)


# ==============================================================================
# OPENAI PROVIDER
# ==============================================================================
class OpenAIProvider(LLMProvider):
    """
    base_url=None (default) -> API OpenAI ufficiali. Parametrico apposta:
    VLLMProvider sotto eredita questa classe passando solo un base_url/api_key
    diversi, perché vLLM espone lo STESSO protocollo (endpoint
    /v1/chat/completions, stesso response_format per il JSON mode) — non
    serve duplicare _call/retry/normalizzazione finish_reason per un backend
    che parla già la lingua che questa classe sa già parlare.
    """

    def __init__(self, base_url: str | None = None, api_key: str | None = None, label: str = "OpenAIProvider"):
        try:
            from openai import OpenAI, RateLimitError, APIStatusError
            self._client = OpenAI(api_key=api_key or settings.api_llm_key, base_url=base_url)
            self._RateLimitError = RateLimitError
            self._APIStatusError = APIStatusError
            log.info(f"✓ {label} initialized" + (f" (base_url={base_url})" if base_url else "."))
        except ImportError:
            raise RuntimeError("openai package not installed.")

    def _call(self, model_name, prompt, temperature, max_tokens, json_mode=False, thinking_level: str | None = None,
              with_meta: bool = False, response_schema=None):
        if thinking_level:
            log.warning ("OpenAIProvider thinking level not used")

        kwargs = dict(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        # Omesso quando None: l'API usa il massimo del modello.
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_schema is not None:
            # OpenAI vuole un JSON Schema (dict), non una classe Pydantic:
            # "strict": True attiva il constrained decoding vero e proprio
            # (non solo validazione a posteriori).
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": response_schema,
                    "strict": True,
                },
            }
        elif json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        def _attempt():
            try:
                response = self._client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if not with_meta:
                    return content
                fr_str = str(getattr(response.choices[0], "finish_reason", None) or "UNKNOWN")
                usage = getattr(response, "usage", None)
                # OpenAI non espone i token di reasoning per i modelli standard:
                # None è il valore corretto, non un dato mancante.
                return content, {
                    "truncated": fr_str == "length",
                    "finish_reason": fr_str,
                    "thinking_tokens": None,
                    "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                }
            except self._RateLimitError as e:
                raise LLMRateLimitError(str(e)) from e
            except self._APIStatusError as e:
                if e.status_code in (429, 503):
                    raise LLMRateLimitError(str(e)) from e
                raise

        return _retry_with_backoff(_attempt)

    def generate_json(self, model_name, prompt, temperature=0.1, max_tokens: int | None = 2048,
                      thinking_level: str | None = None, with_meta: bool = False, response_schema=None):
        return self._call(model_name, prompt, temperature, max_tokens, json_mode=True,
                          thinking_level=thinking_level, with_meta=with_meta, response_schema=response_schema)

    def generate_text(self, model_name, prompt, temperature=0.0, max_tokens: int | None = 16,
                      thinking_level: str | None = None, with_meta: bool = False):
        return self._call(model_name, prompt, temperature, max_tokens, json_mode=False,
                          thinking_level=thinking_level, with_meta=with_meta)


# ==============================================================================
# ANTHROPIC PROVIDER
# ==============================================================================
class AnthropicProvider(LLMProvider):
    """
    Anthropic does not have a native JSON mode.
    We inject a system prompt that instructs the model to reply with JSON only,
    then rely on safe_json_parse in worker.py to extract it robustly.
    """

    def __init__(self):
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=settings.api_llm_key)
            self._RateLimitError = anthropic.RateLimitError
            self._APIStatusError = anthropic.APIStatusError
            log.info("✓ AnthropicProvider initialized.")
        except ImportError:
            raise RuntimeError("anthropic package not installed.")

    def _call(self, model_name, prompt, temperature, max_tokens, json_mode=False, thinking_level: str | None = None,
              with_meta: bool = False):
        if thinking_level:
            log.warning ("AnthropicProvider thinking level not used")

        # Unico provider dei tre in cui max_tokens è obbligatorio: qui None non
        # può essere propagato all'SDK, va risolto. Il log è a livello INFO e
        # non WARNING perché non è un errore né una configurazione da correggere:
        # è il comportamento previsto quando si chiede "usa il massimo" a un'API
        # che quel concetto non lo ha.
        if max_tokens is None:
            max_tokens = _ANTHROPIC_MAX_TOKENS_FALLBACK
            log.info(
                f"Anthropic richiede max_tokens esplicito: uso il fallback "
                f"{_ANTHROPIC_MAX_TOKENS_FALLBACK} (modello {model_name})."
            )

        system = (
            "You must reply with valid JSON only. No explanation, no markdown fences."
            if json_mode else None
        )

        def _attempt():
            try:
                kwargs = dict(
                    model=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                if system:
                    kwargs["system"] = system
                response = self._client.messages.create(**kwargs)
                content = response.content[0].text
                if not with_meta:
                    return content
                fr_str = str(getattr(response, "stop_reason", None) or "UNKNOWN")
                usage = getattr(response, "usage", None)
                return content, {
                    "truncated": fr_str == "max_tokens",
                    "finish_reason": fr_str,
                    "thinking_tokens": None,
                    "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
                }
            except self._RateLimitError as e:
                raise LLMRateLimitError(str(e)) from e
            except self._APIStatusError as e:
                if e.status_code in (429, 503, 529):
                    raise LLMRateLimitError(str(e)) from e
                raise

        return _retry_with_backoff(_attempt)

    def generate_json(self, model_name, prompt, temperature=0.1, max_tokens: int | None = 2048,
                      thinking_level: str | None = None, with_meta: bool = False, response_schema=None):
        if response_schema is not None:
            # Nessun equivalente nativo di constrained decoding su schema per
            # questo provider: il parametro è accettato per compatibilità con
            # l'interfaccia comune ma ignorato, restando sul JSON mode
            # "libero" già esistente (validato poi da safe_json_parse lato
            # worker.py). No-op esplicito e loggato, non un fallimento silenzioso.
            log.debug(f"{self.__class__.__name__}: response_schema richiesto ma non supportato, ignorato.")
        return self._call(model_name, prompt, temperature, max_tokens, json_mode=True,
                          thinking_level=thinking_level, with_meta=with_meta)

    def generate_text(self, model_name, prompt, temperature=0.0, max_tokens: int | None = 16,
                      thinking_level: str | None = None, with_meta: bool = False):
        return self._call(model_name, prompt, temperature, max_tokens, json_mode=False,
                          thinking_level=thinking_level, with_meta=with_meta)


# ==============================================================================
# MISTRAL PROVIDER
# ==============================================================================
class MistralProvider(LLMProvider):
    """
    Mistral ha un JSON mode nativo (response_format={"type": "json_object"}),
    analogo a quello "legacy" di OpenAI: va comunque istruito via prompt/system
    a produrre solo JSON, l'API garantisce solo che il testo SIA JSON valido,
    non che rispetti uno schema.
    """

    def __init__(self):
        try:
            from mistralai import Mistral
            from mistralai.models import SDKError
            self._client = Mistral(api_key=settings.api_llm_key)
            self._SDKError = SDKError
            log.info("✓ MistralProvider initialized.")
        except ImportError:
            raise RuntimeError("mistralai package not installed.")

    def _call(self, model_name, prompt, temperature, max_tokens, json_mode=False, thinking_level: str | None = None,
              with_meta: bool = False):
        if thinking_level:
            log.warning("MistralProvider thinking level not used")

        kwargs = dict(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        # Omesso quando None: come Gemini/OpenAI, l'API usa il massimo del modello.
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        def _attempt():
            try:
                response = self._client.chat.complete(**kwargs)
                content = response.choices[0].message.content
                if not with_meta:
                    return content
                fr_str = str(getattr(response.choices[0], "finish_reason", None) or "UNKNOWN")
                usage = getattr(response, "usage", None)
                # Mistral non espone token di reasoning: None è corretto, non mancante.
                return content, {
                    "truncated": fr_str in ("length", "model_length"),
                    "finish_reason": fr_str,
                    "thinking_tokens": None,
                    "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                }
            except self._SDKError as e:
                # SDKError copre sia i 429 (rate limit) sia i 503/overload; lo
                # status code non è sempre esposto con lo stesso nome a seconda
                # della versione dell'SDK, quindi si controllano entrambi.
                status_code = getattr(e, "status_code", None) or getattr(e, "status", None)
                if status_code in (429, 503):
                    raise LLMRateLimitError(str(e)) from e
                raise

        return _retry_with_backoff(_attempt)

    def generate_json(self, model_name, prompt, temperature=0.1, max_tokens: int | None = 2048,
                      thinking_level: str | None = None, with_meta: bool = False, response_schema=None):
        if response_schema is not None:
            # Nessun equivalente nativo di constrained decoding su schema per
            # questo provider: il parametro è accettato per compatibilità con
            # l'interfaccia comune ma ignorato, restando sul JSON mode
            # "libero" già esistente (validato poi da safe_json_parse lato
            # worker.py). No-op esplicito e loggato, non un fallimento silenzioso.
            log.debug(f"{self.__class__.__name__}: response_schema richiesto ma non supportato, ignorato.")
        return self._call(model_name, prompt, temperature, max_tokens, json_mode=True,
                          thinking_level=thinking_level, with_meta=with_meta)

    def generate_text(self, model_name, prompt, temperature=0.0, max_tokens: int | None = 16,
                      thinking_level: str | None = None, with_meta: bool = False):
        return self._call(model_name, prompt, temperature, max_tokens, json_mode=False,
                          thinking_level=thinking_level, with_meta=with_meta)


# ==============================================================================
# VLLM PROVIDER
# ==============================================================================
class VLLMProvider(OpenAIProvider):
    """
    vLLM espone un server OpenAI-compatible (stesso endpoint
    /v1/chat/completions, stesso response_format per il JSON mode): non serve
    un SDK/protocollo diverso, basta puntare il client OpenAI ufficiale a
    settings.local_vllm_base_url invece che alle API OpenAI vere. Eredita
    quindi INTERAMENTE _call/retry/normalizzazione finish_reason da
    OpenAIProvider, senza duplicare nulla.

    Perché questa NON è la stessa scelta fatta per LocalLLMProvider (Ollama):
    lì il protocollo è diverso (SDK ollama, non OpenAI-compatible), quindi la
    duplicazione era necessaria. Qui invece il protocollo è identico — usare
    un adapter con base_url configurabile invece di una classe parallela è
    la controparte diretta di quella scelta, non un'incoerenza.

    api_key: vLLM in locale tipicamente non richiede autenticazione reale,
    ma il client OpenAI pretende comunque una stringa non vuota — da cui il
    placeholder quando LOCAL_VLLM_API_KEY non è impostata.
    """

    def __init__(self):
        super().__init__(
            base_url=settings.local_vllm_base_url,
            api_key=settings.local_vllm_api_key or "not-needed",
            label="VLLMProvider",
        )


# ==============================================================================
# LOCAL PROVIDER (Ollama, self-hosted)
# ==============================================================================
class LocalLLMProvider(LLMProvider):
    """
    LLM locale via Ollama. A differenza degli altri tre, `model_name` non
    seleziona un modello "hosted" dietro una API key: dev'essere già stato
    scaricato sull'host Ollama (`ollama pull <model>`, es. `qwen3:14b`) — se
    manca, la prima chiamata fallisce con un errore esplicito di Ollama, non
    silenziosamente. Il nome va impostato negli stessi env var già esistenti
    (QUERY_REWRITER_MODEL_NAME, ANSWER_GENERATOR_MODEL_NAME, GRADER_MODEL_NAME),
    con un tag Ollama al posto di un nome Gemini/OpenAI/Anthropic/Mistral.

    NESSUN retry-con-backoff su rate limit: un'istanza Ollama locale non ha
    il concetto di quota/rate limit. Un errore di connessione (host non
    raggiungibile, GPU OOM, modello non scaricato) è un errore reale e deve
    propagarsi subito — stessa filosofia di LocalEmbeddingProvider in
    embedding.py, per lo stesso motivo: ritentare alla cieca un errore
    strutturale (non transitorio) nasconde il problema invece di segnalarlo.

    JSON mode: Ollama supporta `format="json"`, che vincola il decoding a
    produrre JSON sintatticamente valido (constrained decoding, non un
    prompt "per favore rispondi in JSON") — garantisce la stessa cosa del
    JSON mode di OpenAI/Mistral: JSON valido, non conformità a uno schema.

    Thinking: i modelli "hybrid thinking" supportati da Ollama (es. Qwen3)
    espongono un toggle booleano (`think`), non livelli come Gemini —
    qualunque thinking_level truthy lo attiva.
    """

    def __init__(self):
        try:
            import ollama
        except ImportError:
            raise RuntimeError("ollama package not installed.\npip install ollama")

        self._client = ollama.Client(host=settings.local_llm_host)
        self._ResponseError = ollama.ResponseError
        log.info(f"✓ LocalLLMProvider initialized (host={settings.local_llm_host}).")

    @staticmethod
    def _options(temperature: float, max_tokens: int | None) -> dict:
        # num_predict=-1: comportamento equivalente a "omesso" su
        # Gemini/OpenAI/Mistral quando max_tokens=None — nessun tetto
        # esplicito, usa il default/contesto del modello.
        return {
            "temperature": temperature,
            "num_predict": max_tokens if max_tokens is not None else -1,
        }

    def _call(self, model_name, prompt, temperature, max_tokens, json_mode=False,
              thinking_level: str | None = None, with_meta: bool = False):
        kwargs = dict(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options=self._options(temperature, max_tokens),
        )
        if json_mode:
            kwargs["format"] = "json"
        if thinking_level:
            kwargs["think"] = True

        try:
            response = self._client.chat(**kwargs)
        except self._ResponseError as e:
            # Errore reale (modello non scaricato, host irraggiungibile, OOM):
            # propaga così com'è, niente normalizzazione a LLMRateLimitError —
            # qui non esiste il concetto di rate limit da ritentare.
            raise RuntimeError(f"Ollama error per modello '{model_name}': {e}") from e

        content = response["message"]["content"]
        if not with_meta:
            return content

        done_reason = response.get("done_reason", "unknown")
        return content, {
            "truncated": done_reason == "length",
            "finish_reason": done_reason,
            # Ollama non espone un conteggio separato dei token di thinking
            # (a differenza di Gemini): None è corretto, non un dato mancante
            # — stessa convenzione già usata per Anthropic/Mistral sopra.
            "thinking_tokens": None,
            "output_tokens": response.get("eval_count"),
        }

    def generate_json(self, model_name, prompt, temperature=0.1, max_tokens: int | None = 2048,
                      thinking_level: str | None = None, with_meta: bool = False, response_schema=None):
        if response_schema is not None:
            # Nessun equivalente nativo di constrained decoding su schema per
            # questo provider: il parametro è accettato per compatibilità con
            # l'interfaccia comune ma ignorato, restando sul JSON mode
            # "libero" già esistente (validato poi da safe_json_parse lato
            # worker.py). No-op esplicito e loggato, non un fallimento silenzioso.
            log.debug(f"{self.__class__.__name__}: response_schema richiesto ma non supportato, ignorato.")
        return self._call(model_name, prompt, temperature, max_tokens, json_mode=True,
                          thinking_level=thinking_level, with_meta=with_meta)

    def generate_text(self, model_name, prompt, temperature=0.0, max_tokens: int | None = 16,
                      thinking_level: str | None = None, with_meta: bool = False):
        return self._call(model_name, prompt, temperature, max_tokens, json_mode=False,
                          thinking_level=thinking_level, with_meta=with_meta)


# ==============================================================================
# PUBLIC API
# ==============================================================================
def init_llm_provider() -> LLMProvider:
    """
    Instantiate the provider selected by the LLM_PROVIDER env var.
    Must be called once inside init_worker_process() (post-fork).
    Returns the provider instance and also stores it in the module-level singleton.
    """
    global _PROVIDER_INSTANCE
    providers = {
        "gemini":    GeminiProvider,
        "openai":    OpenAIProvider,
        "anthropic": AnthropicProvider,
        "mistral":   MistralProvider,
    }

    if settings.llm_provider == "local":
        # "local" è un ombrello per due backend con caratteristiche opposte
        # (vedi docstring di LocalLLMProvider e VLLMProvider sopra): Ollama
        # per basso-concorrenza/sviluppo, vLLM per produzione multi-utente.
        # LOCAL_LLM_BACKEND sceglie quale dei due, senza toccare LLM_PROVIDER.
        local_backends = {"ollama": LocalLLMProvider, "vllm": VLLMProvider}
        cls = local_backends.get(settings.local_llm_backend)
        if cls is None:
            raise ValueError(
                f"Unknown LOCAL_LLM_BACKEND='{settings.local_llm_backend}'. "
                f"Valid options: {list(local_backends.keys())}"
            )
        _PROVIDER_INSTANCE = cls()
        log.info(f"LLM provider set to: local (backend={settings.local_llm_backend})")
        return _PROVIDER_INSTANCE

    cls = providers.get(settings.llm_provider)
    if cls is None:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{settings.llm_provider}'. "
            f"Valid options: {list(providers.keys()) + ['local']}"
        )

    _PROVIDER_INSTANCE = cls()
    log.info(f"LLM provider set to: {settings.llm_provider}")
    return _PROVIDER_INSTANCE


def get_llm_provider() -> LLMProvider:
    """Return the already-initialized provider. Raises if init_llm_provider() was not called."""
    if _PROVIDER_INSTANCE is None:
        raise RuntimeError("LLM provider not initialized. Call init_llm_provider() first.")
    return _PROVIDER_INSTANCE