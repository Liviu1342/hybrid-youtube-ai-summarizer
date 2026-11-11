"""LLM providers: one common contract, multiple interchangeable implementations."""

from typing import Protocol

import ollama
import httpx2  # HTTP client used internally by google-genai
import httpx  # HTTP client used internally by ollama
from google import genai
from google.genai import errors, types
from collections.abc import Iterator


class LLMError(Exception):
    """Error with a user-friendly message, raised by any provider."""


class LLMProvider(Protocol):
    """The contract every AI provider must fulfil (local or cloud)."""

    name: str

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Send the prompts to the model and return its text answer."""
        ...

    def stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """Send the prompts and yield the answer piece by piece."""
        ...


class OllamaProvider:
    """Runs a model locally through the Ollama server (free and private)."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        host: str = "http://localhost:11434",
        context_window: int = 8192,
    ) -> None:
        self.name = f"Ollama ({model})"
        self.model = model
        self.context_window = context_window
        self._client = ollama.Client(host=host)  # no network call happens here yet

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.chat(
                model=self.model,
                messages=self._messages(system_prompt, user_prompt),
                # Ollama's default context is 4096 tokens; long transcripts need more
                options={"num_ctx": self.context_window},
            )
        except (ConnectionError, ollama.ResponseError) as e:
            raise self._friendly_error(e) from None

        content = response.message.content
        if not content:  # content is "str | None": the model may return nothing
            raise LLMError("The model returned an empty response. Try again.")
        return content
    
    def stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """Yield the answer piece by piece, as the model writes it."""
        try:
            chunks = self._client.chat(
                model=self.model,
                messages=self._messages(system_prompt, user_prompt),
                options={"num_ctx": self.context_window},
                stream=True,  # receive the answer in small pieces
            )
            for chunk in chunks:
                if chunk.message.content:  # some chunks carry no text
                    yield chunk.message.content
        # In streaming mode, ollama lets httpx's own ConnectError through
        except (ConnectionError, httpx.ConnectError, ollama.ResponseError) as e:
            raise self._friendly_error(e) from None

    def _messages(self, system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
        """Build the chat messages in the format Ollama expects."""
        return [
            {"role": "system", "content": system_prompt},  # how the model should behave
            {"role": "user", "content": user_prompt},      # what we actually ask
        ]

    def _friendly_error(self, error: Exception) -> LLMError:
        """Translate a technical Ollama error into a message the user understands."""
        if isinstance(error, (ConnectionError, httpx.ConnectError)):
            return LLMError("Ollama is not running. Start it and try again.")
        if isinstance(error, ollama.ResponseError) and error.status_code == 404:
            return LLMError(f"Model '{self.model}' not found. Run: ollama pull {self.model}")
        return LLMError(f"Ollama returned an error: {error}")

class GeminiProvider:
    """Runs a model in Google's cloud through the Gemini API (fast, needs an API key)."""

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            # Fail early with a clear message instead of a cryptic 401 later
            raise LLMError("GEMINI_API_KEY is missing. Add it to your .env file.")
        self.name = f"Gemini ({model})"
        self.model = model
        # Retry transient errors (429, 5xx) automatically, with growing pauses
        retry = types.HttpRetryOptions(attempts=3)
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(retry_options=retry),
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=self._config(system_prompt),
            )
        except (errors.APIError, httpx2.HTTPError) as e:
            raise self._friendly_error(e) from None

        content = response.text
        if not content:  # e.g. the answer was blocked by safety filters
            raise LLMError("Gemini returned an empty response. Try again.")
        return content

    def stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """Yield the answer piece by piece, as the model writes it."""
        try:
            chunks = self._client.models.generate_content_stream(
                model=self.model,
                contents=user_prompt,
                config=self._config(system_prompt),
            )
            for chunk in chunks:
                if chunk.text:  # some chunks carry no text
                    yield chunk.text
        except (errors.APIError, httpx2.HTTPError) as e:
            raise self._friendly_error(e) from None

    def _config(self, system_prompt: str) -> types.GenerateContentConfig:
        """Gemini receives the system prompt as a config option, not as a message."""
        return types.GenerateContentConfig(system_instruction=system_prompt)

    def _friendly_error(self, error: Exception) -> LLMError:
        """Translate a technical Gemini error into a message the user understands."""
        if isinstance(error, errors.ClientError):
            # 4xx: our side is wrong (invalid key, unknown model, quota exceeded)
            return LLMError(f"Gemini rejected the request ({error.code}): {error.message}")
        if isinstance(error, errors.ServerError):
            # 5xx: Google's side is having problems; retrying later usually helps
            return LLMError(
                f"Gemini is temporarily unavailable ({error.code}): {error.message}. Try again."
            )
        if isinstance(error, errors.APIError):
            return LLMError(f"Gemini returned an error ({error.code}): {error.message}")
        return LLMError("Could not connect to Gemini. Check your internet connection.")