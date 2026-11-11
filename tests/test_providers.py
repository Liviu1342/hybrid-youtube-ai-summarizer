"""Tests for provider error handling, with the network calls replaced by fakes."""

from types import SimpleNamespace

import ollama
import pytest

from providers import GeminiProvider, LLMError, OllamaProvider


def fake_chat_response(content: str | None) -> SimpleNamespace:
    """Mimic the shape of ollama's ChatResponse: response.message.content."""
    return SimpleNamespace(message=SimpleNamespace(content=content))


def test_ollama_returns_model_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider()
    monkeypatch.setattr(provider._client, "chat", lambda **kwargs: fake_chat_response("Hi!"))
    assert provider.generate("system", "user") == "Hi!"


def test_ollama_not_running(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_connection_error(**kwargs):
        raise ConnectionError("connection refused")

    provider = OllamaProvider()
    monkeypatch.setattr(provider._client, "chat", raise_connection_error)
    with pytest.raises(LLMError, match="Ollama is not running"):
        provider.generate("system", "user")


def test_ollama_missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_not_found(**kwargs):
        raise ollama.ResponseError("model not found", status_code=404)

    provider = OllamaProvider(model="missing-model")
    monkeypatch.setattr(provider._client, "chat", raise_not_found)
    with pytest.raises(LLMError, match="ollama pull missing-model"):
        provider.generate("system", "user")


def test_ollama_empty_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider()
    monkeypatch.setattr(provider._client, "chat", lambda **kwargs: fake_chat_response(None))
    with pytest.raises(LLMError, match="empty response"):
        provider.generate("system", "user")


def test_gemini_requires_api_key() -> None:
    with pytest.raises(LLMError, match="GEMINI_API_KEY is missing"):
        GeminiProvider(api_key="", model="any-model")