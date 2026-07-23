"""
llm_provider.py — Provider-agnostic LLM adapter.

Supported providers (set via LLM_PROVIDER env var):
  gemini    — Google Gemini (default)
  openai    — OpenAI
  anthropic — Anthropic Claude

Each provider exposes two operations used by worker.py:
  generate_json(model_name, prompt, temperature, max_tokens) -> str   (raw JSON string)
  generate_text(model_name, prompt, temperature, max_tokens) -> str   (plain text)

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
                      temperature: float = 0.1, max_tokens: int = 2048, thinking_level: bool = None) -> str:
        """Call the model asking for a JSON response. Returns the raw JSON string."""

    @abstractmethod
    def generate_text(self, model_name: str, prompt: str,
                      temperature: float = 0.0, max_tokens: int = 16, thinking_level: bool = None) -> str:
        """Call the model asking for a plain-text response. Returns the text string."""


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

    def _call(self, model_name: str, prompt: str, temperature: float, max_tokens: int, json_mode: bool = False, thinking_level: bool = None) -> str:
        from google.genai import types

        # Nel nuovo SDK le configurazioni di generazione passano da GenerateContentConfig
        config_kwargs = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
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
                return response.text
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

            base_msg = (
                f"[GEMINI_USAGE] model={model_name} finish_reason={finish_reason_str} "
                f"thinking_tokens={thinking_tokens} output_tokens={output_tokens} "
                f"total_tokens={total_tokens} max_output_tokens_budget={max_tokens}"
            )

            if "MAX_TOKENS" in finish_reason_str:
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

    def generate_json(self, model_name: str, prompt: str, temperature: float = 0.1, max_tokens: int = 2048, thinking_level: bool = None) -> str:
        return self._call(model_name, prompt, temperature, max_tokens, json_mode=True, thinking_level=thinking_level)

    def generate_text(self, model_name: str, prompt: str, temperature: float = 0.0, max_tokens: int = 16, thinking_level: bool = None) -> str:
        return self._call(model_name, prompt, temperature, max_tokens, json_mode=False, thinking_level=thinking_level)


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

    def _call(self, model_name, prompt, temperature, max_tokens, json_mode=False, thinking_level: bool = None):
        if thinking_level:
            log.warning ("OpenAIProvider thinking level not used")

        kwargs = dict(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        def _attempt():
            try:
                response = self._client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            except self._RateLimitError as e:
                raise LLMRateLimitError(str(e)) from e
            except self._APIStatusError as e:
                if e.status_code in (429, 503):
                    raise LLMRateLimitError(str(e)) from e
                raise

        return _retry_with_backoff(_attempt)

    def generate_json(self, model_name, prompt, temperature=0.1, max_tokens=2048, thinking_level: bool = None):
        return self._call(model_name, prompt, temperature, max_tokens, json_mode=True, thinking_level = thinking_level)

    def generate_text(self, model_name, prompt, temperature=0.0, max_tokens=16, thinking_level: bool = None):
        return self._call(model_name, prompt, temperature, max_tokens, json_mode=False, thinking_level = thinking_level)


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

    def _call(self, model_name, prompt, temperature, max_tokens, json_mode=False, thinking_level: bool = None):
        if thinking_level:
            log.warning ("AnthropicProvider thinking level not used")

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
                return response.content[0].text
            except self._RateLimitError as e:
                raise LLMRateLimitError(str(e)) from e
            except self._APIStatusError as e:
                if e.status_code in (429, 503, 529):
                    raise LLMRateLimitError(str(e)) from e
                raise

        return _retry_with_backoff(_attempt)

    def generate_json(self, model_name, prompt, temperature=0.1, max_tokens=2048, thinking_level: bool = None):
        return self._call(model_name, prompt, temperature, max_tokens, json_mode=True, thinking_level=thinking_level)

    def generate_text(self, model_name, prompt, temperature=0.0, max_tokens=16, thinking_level: bool = None):
        return self._call(model_name, prompt, temperature, max_tokens, json_mode=False, thinking_level=thinking_level)


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