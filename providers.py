"""LLM providers: one common contract, multiple interchangeable implementations."""

from typing import Protocol

import ollama
import httpx2  # HTTP client used internally by google-genai
from google import genai
from google.genai import errors, types


class LLMError(Exception):
    """Error with a user-friendly message, raised by any provider."""


class LLMProvider(Protocol):
    """The contract every AI provider must fulfil (local or cloud)."""

    name: str

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Send the prompts to the model and return its text answer."""
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
        messages = [
            {"role": "system", "content": system_prompt},  # how the model should behave
            {"role": "user", "content": user_prompt},      # what we actually ask
        ]
        try:
            response = self._client.chat(
                model=self.model,
                messages=messages,
                # Ollama's default context is 4096 tokens; long transcripts need more
                options={"num_ctx": self.context_window},
            )
        except (ConnectionError, ollama.ResponseError) as e:
            raise self._friendly_error(e) from None

        content = response.message.content
        if not content:  # content is "str | None": the model may return nothing
            raise LLMError("The model returned an empty response. Try again.")
        return content

    def _friendly_error(self, error: Exception) -> LLMError:
        """Translate a technical Ollama error into a message the user understands."""
        if isinstance(error, ConnectionError):
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
        self._client = genai.Client(api_key=api_key)  # the key stays inside the client

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                # Gemini receives the system prompt as a config option, not as a message
                config=types.GenerateContentConfig(system_instruction=system_prompt),
            )
        except errors.ClientError as e:
            # 4xx: our side is wrong (invalid key, unknown model, quota exceeded)
            raise LLMError(f"Gemini rejected the request ({e.code}): {e.message}") from None
        except errors.ServerError as e:
            # 5xx: Google's side is having problems; retrying later usually helps
            raise LLMError(f"Gemini is temporarily unavailable ({e.code}): {e.message}. Try again.") from None
        except httpx2.HTTPError:
            raise LLMError("Could not connect to Gemini. Check your internet connection.") from None

        content = response.text
        if not content:  # e.g. the answer was blocked by safety filters
            raise LLMError("Gemini returned an empty response. Try again.")
        return content