"""
llm_provider.py — Provider-agnostic LLM adapter.

Supported providers (set via LLM_PROVIDER env var):
  gemini    — Google Gemini (default)
  openai    — OpenAI
  anthropic — Anthropic Claude

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
                      with_meta: bool = False):
        """
        Call the model asking for a JSON response.
        Returns the raw JSON string, oppure (str, meta) se with_meta=True.

        max_tokens=None -> nessun tetto esplicito (massimo del modello).
        Il default resta 2048: chi non specifica nulla è un chiamante con
        output corto e strutturato (rewriter), che il tetto lo vuole.
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
              thinking_level: str | None = None, with_meta: bool = False):
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
                      thinking_level: str | None = None, with_meta: bool = False):
        return self._call(model_name, prompt, temperature, max_tokens, json_mode=True,
                          thinking_level=thinking_level, with_meta=with_meta)

    def generate_text(self, model_name: str, prompt: str, temperature: float = 0.0, max_tokens: int | None = 16,
                      thinking_level: str | None = None, with_meta: bool = False):
        return self._call(model_name, prompt, temperature, max_tokens, json_mode=False,
                          thinking_level=thinking_level, with_meta=with_meta)


# ==============================================================================
# OPENAI PROVIDER
# ==============================================================================
class OpenAIProvider(LLMProvider):

    def __init__(self):
        try:
            from openai import OpenAI, RateLimitError, APIStatusError
            self._client = OpenAI(api_key=settings.api_llm_key)
            self._RateLimitError = RateLimitError
            self._APIStatusError = APIStatusError
            log.info("✓ OpenAIProvider initialized.")
        except ImportError:
            raise RuntimeError("openai package not installed.")

    def _call(self, model_name, prompt, temperature, max_tokens, json_mode=False, thinking_level: str | None = None,
              with_meta: bool = False):
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
        if json_mode:
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
                      thinking_level: str | None = None, with_meta: bool = False):
        return self._call(model_name, prompt, temperature, max_tokens, json_mode=True,
                          thinking_level=thinking_level, with_meta=with_meta)

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
                      thinking_level: str | None = None, with_meta: bool = False):
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
    }

    cls = providers.get(settings.llm_provider)
    if cls is None:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{settings.llm_provider}'. "
            f"Valid options: {list(providers.keys())}"
        )

    _PROVIDER_INSTANCE = cls()
    log.info(f"LLM provider set to: {settings.llm_provider}")
    return _PROVIDER_INSTANCE


def get_llm_provider() -> LLMProvider:
    """Return the already-initialized provider. Raises if init_llm_provider() was not called."""
    if _PROVIDER_INSTANCE is None:
        raise RuntimeError("LLM provider not initialized. Call init_llm_provider() first.")
    return _PROVIDER_INSTANCE