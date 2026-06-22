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
                      temperature: float = 0.1, max_tokens: int = 2048) -> str:
        """Call the model asking for a JSON response. Returns the raw JSON string."""

    @abstractmethod
    def generate_text(self, model_name: str, prompt: str,
                      temperature: float = 0.0, max_tokens: int = 16) -> str:
        """Call the model asking for a plain-text response. Returns the text string."""


# ==============================================================================
# GEMINI PROVIDER
# ==============================================================================
class GeminiProvider(LLMProvider):

    def __init__(self):
        try:
            import google.generativeai as genai
            from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            self._genai = genai
            self._rate_limit_exceptions = (ResourceExhausted, ServiceUnavailable)
            log.info("✓ GeminiProvider initialized.")
        except ImportError:
            raise RuntimeError("google-generativeai package not installed.")

    def _call(self, model_name, prompt, temperature, max_tokens, json_mode=False):
        import google.generativeai as genai

        config_kwargs = dict(temperature=temperature, max_output_tokens=max_tokens)
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        model = self._genai.GenerativeModel(model_name)

        def _attempt():
            try:
                return model.generate_content(
                    prompt,
                    generation_config=self._genai.types.GenerationConfig(**config_kwargs)
                ).text
            except self._rate_limit_exceptions as e:
                raise LLMRateLimitError(str(e)) from e

        return _retry_with_backoff(_attempt)

    def generate_json(self, model_name, prompt, temperature=0.1, max_tokens=2048):
        return self._call(model_name, prompt, temperature, max_tokens, json_mode=True)

    def generate_text(self, model_name, prompt, temperature=0.0, max_tokens=16):
        return self._call(model_name, prompt, temperature, max_tokens, json_mode=False)


# ==============================================================================
# OPENAI PROVIDER
# ==============================================================================
class OpenAIProvider(LLMProvider):

    def __init__(self):
        try:
            from openai import OpenAI, RateLimitError, APIStatusError
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            self._RateLimitError = RateLimitError
            self._APIStatusError = APIStatusError
            log.info("✓ OpenAIProvider initialized.")
        except ImportError:
            raise RuntimeError("openai package not installed.")

    def _call(self, model_name, prompt, temperature, max_tokens, json_mode=False):
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

    def generate_json(self, model_name, prompt, temperature=0.1, max_tokens=2048):
        return self._call(model_name, prompt, temperature, max_tokens, json_mode=True)

    def generate_text(self, model_name, prompt, temperature=0.0, max_tokens=16):
        return self._call(model_name, prompt, temperature, max_tokens, json_mode=False)


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
            self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            self._RateLimitError = anthropic.RateLimitError
            self._APIStatusError = anthropic.APIStatusError
            log.info("✓ AnthropicProvider initialized.")
        except ImportError:
            raise RuntimeError("anthropic package not installed.")

    def _call(self, model_name, prompt, temperature, max_tokens, json_mode=False):
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

    def generate_json(self, model_name, prompt, temperature=0.1, max_tokens=2048):
        return self._call(model_name, prompt, temperature, max_tokens, json_mode=True)

    def generate_text(self, model_name, prompt, temperature=0.0, max_tokens=16):
        return self._call(model_name, prompt, temperature, max_tokens, json_mode=False)


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
    provider_name = os.environ.get("LLM_PROVIDER", "gemini").lower()

    providers = {
        "gemini":    GeminiProvider,
        "openai":    OpenAIProvider,
        "anthropic": AnthropicProvider,
    }

    cls = providers.get(provider_name)
    if cls is None:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{provider_name}'. "
            f"Valid options: {list(providers.keys())}"
        )

    _PROVIDER_INSTANCE = cls()
    log.info(f"LLM provider set to: {provider_name}")
    return _PROVIDER_INSTANCE


def get_llm_provider() -> LLMProvider:
    """Return the already-initialized provider. Raises if init_llm_provider() was not called."""
    if _PROVIDER_INSTANCE is None:
        raise RuntimeError("LLM provider not initialized. Call init_llm_provider() first.")
    return _PROVIDER_INSTANCE