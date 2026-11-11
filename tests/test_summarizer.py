"""Tests for prompt building, using a fake provider instead of a real AI."""

from summarizer import MAX_TRANSCRIPT_CHARS, generate_summary


class FakeProvider:
    """Records what it receives and returns a fixed answer (no network, no GPU)."""

    name = "Fake"

    def __init__(self) -> None:
        self.system_prompt = ""
        self.user_prompt = ""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return "## TL;DR\nFake summary"


def test_returns_the_provider_answer() -> None:
    provider = FakeProvider()
    assert generate_summary("Some transcript", provider) == "## TL;DR\nFake summary"


def test_transcript_is_wrapped_in_tags() -> None:
    provider = FakeProvider()
    generate_summary("Hello world", provider)
    assert provider.user_prompt == "<transcript>\nHello world\n</transcript>"


def test_requested_language_is_in_system_prompt() -> None:
    provider = FakeProvider()
    generate_summary("Hello world", provider, language="Romanian")
    assert "Write your answer in Romanian" in provider.system_prompt


def test_long_transcript_is_truncated() -> None:
    provider = FakeProvider()
    generate_summary("x" * (MAX_TRANSCRIPT_CHARS + 5_000), provider)
    assert provider.user_prompt.count("x") == MAX_TRANSCRIPT_CHARS