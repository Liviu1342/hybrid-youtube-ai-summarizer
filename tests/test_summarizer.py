"""Tests for prompt building, using a fake provider instead of a real AI."""

from collections.abc import Iterator
from summarizer import (
    MAX_TRANSCRIPT_CHARS,
    generate_summary,
    split_into_chunks,
    stream_summary,
    summarize_chunks,
)


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

    def stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        yield "## TL;DR\n"
        yield "Fake summary"


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

def test_stream_summary_yields_the_provider_pieces() -> None:
    provider = FakeProvider()
    pieces = list(stream_summary("Hello world", provider, language="Romanian"))
    assert pieces == ["## TL;DR\n", "Fake summary"]
    assert provider.user_prompt == "<transcript>\nHello world\n</transcript>"
    assert "Write your answer in Romanian" in provider.system_prompt

def test_short_text_is_a_single_chunk() -> None:
    assert split_into_chunks("one two three", max_chars=100) == ["one two three"]


def test_empty_text_has_no_chunks() -> None:
    assert split_into_chunks("   ", max_chars=100) == []


def test_long_text_is_split_between_words() -> None:
    text = " ".join(f"word{i}" for i in range(1_000))
    chunks = split_into_chunks(text, max_chars=50)
    assert len(chunks) > 1
    assert all(len(chunk) <= 50 for chunk in chunks)
    assert " ".join(chunks) == text  # nothing lost, nothing duplicated

def test_summarize_chunks_makes_notes_for_each_chunk() -> None:
    provider = FakeProvider()
    notes = summarize_chunks(["first part", "second part"], provider)
    assert notes == ["## TL;DR\nFake summary", "## TL;DR\nFake summary"]
    assert provider.user_prompt == "<transcript>\nsecond part\n</transcript>"
    assert "notes" in provider.system_prompt