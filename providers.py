"""LLM providers: one common contract, multiple interchangeable implementations."""

from typing import Protocol

import ollama


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
        except ConnectionError:
            raise LLMError("Ollama is not running. Start it and try again.") from None
        except ollama.ResponseError as e:
            if e.status_code == 404:
                raise LLMError(
                    f"Model '{self.model}' not found. Run: ollama pull {self.model}"
                ) from None
            raise LLMError(f"Ollama returned an error: {e.error}") from e

        content = response.message.content
        if not content:  # content is "str | None": the model may return nothing
            raise LLMError("The model returned an empty response. Try again.")
        return content